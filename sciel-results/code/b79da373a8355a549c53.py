"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    try:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        # Check all valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check a sample of invalid triples
        rng = np.random.RandomState(42)
        num_samples = 500
        for _ in range(num_samples):
            a = rng.randint(0, n*m-1)
            b = rng.randint(0, m*p-1)
            c = rng.randint(0, p*n-1)
            
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            
            if j_a == j_b and k_b == k_c and i_c == i_a:
                continue
            
            val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
            if abs(val) > tol:
                return False
        
        return True
    except Exception:
        return False

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _replicate_strassen(n=2, m=2, p=16):
    """
    Replicate the 2x2x2 Strassen decomposition 8 times for 2x2x16.
    Rank = 7 * 8 = 56
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            U[:, col] = U2[:, r]
            for j in range(m):
                for k in range(2):
                    b_idx = j * p + (block * 2 + k)
                    v_idx = j * 2 + k
                    V[b_idx, col] = V2[v_idx, r]
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    return U, V, W

def _try_circulant_strassen(n=2, m=2, p=16):
    """
    Try to use circulant structure to share Strassen factors across k blocks.
    
    Idea: Use a 2x2 Strassen for the (i,j) part, and for the k dimension,
    use a circulant/FFT-based decomposition that shares the U factors.
    
    For each of the 7 Strassen terms, we need to handle 16 k values.
    If we can use a rank-r decomposition for the k part that's shared,
    we can reduce the total rank.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # For each Strassen term r (0..6), we have:
    # U factor: depends on (i,j)
    # V factor: depends on (j,k) 
    # W factor: depends on (k,i)
    
    # The key insight: for fixed r and j, V[j,k,r] and W[k,i,r] need to
    # satisfy: sum_k V[j,k,r] * W[k,i,r] * U[i,j,r] = 1 for all i,j
    
    # Actually, the condition is: for each valid (i,j,k):
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Let's try: use a 16-point DFT to decompose the k dimension
    # For each (i,j,r) triple from Strassen, we need V[j,k,r] and W[k,i,r]
    # such that sum_k V[j,k,r] * W[k,i,r] = 1/U[i,j,r] (if U[i,j,r] != 0)
    
    # This is a rank-1 constraint in k for each (i,j,r)
    # So for each (i,j,r), we need V[j,*,r] and W[*,i,r] to be rank-1 in k
    
    # A rank-1 decomposition of the 16-dim identity-like structure:
    # We can use: V[j,k,r] = alpha_r * f_j(k), W[k,i,r] = beta_r * g_i(k)
    # such that sum_k f_j(k) * g_i(k) = 1/U[i,j,r]
    
    # For simplicity, let's use: f_j(k) = 1 for all k, g_i(k) = c_i for all k
    # Then sum_k f_j(k) * g_i(k) = 16 * c_i
    # We need 16 * c_i = 1/U[i,j,r] for all j, so this only works if U[i,j,r] is same for all j
    
    # This doesn't work directly. Let's try a different approach.
    
    # Alternative: Use the fact that for fixed r, the matrices V_r (2x16) and W_r (16x2)
    # need to satisfy: U_r * V_r[k] * W_r[k] summed over k gives the right result
    
    # Let me try a direct construction using 8 blocks of 2, but with shared U
    
    # Actually, let's just try to find a lower rank decomposition using optimization
    # starting from the replicated Strassen
    
    return None

def _try_optimize_from_strassen(n=2, m=2, p=16, target_rank=None, time_budget=30, seed=42):
    """
    Try to optimize the replicated Strassen decomposition to find a lower rank.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U0, V0, W0 = _replicate_strassen(n, m, p)
    rank0 = U0.shape[1]
    
    if target_rank is None:
        target_rank = rank0 - 4  # Try to reduce by 4
    
    if target_rank < 1:
        return None
    
    # Initialize with random projection of the Strassen solution
    if target_rank < rank0:
        # Use random projection
        P = rng.randn(rank0, target_rank)
        U = U0 @ P
        V = V0 @ P
        W = W0 @ P
    else:
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
    
    # Collect valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    valid_triples = np.array(valid_triples)
    
    # Sample invalid triples
    invalid_triples = []
    for _ in range(1000):
        a = rng.randint(0, n*m-1)
        b = rng.randint(0, m*p-1)
        c = rng.randint(0, p*n-1)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if not (j_a == j_b and k_b == k_c and i_c == i_a):
            invalid_triples.append((a, b, c))
    invalid_triples = np.array(invalid_triples) if invalid_triples else np.empty((0, 3), dtype=int)
    
    all_triples = np.vstack([valid_triples, invalid_triples])
    targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
    a_idx = all_triples[:, 0]
    b_idx = all_triples[:, 1]
    c_idx = all_triples[:, 2]
    
    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    t = 0
    lr = 0.05
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        t += 1
        
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        residuals = preds - targets
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        UVW_U = residuals[:, None] * V_sub * W_sub
        UVW_V = residuals[:, None] * U_sub * W_sub
        UVW_W = residuals[:, None] * U_sub * V_sub
        
        np.add.at(grad_U, a_idx, UVW_U)
        np.add.at(grad_V, b_idx, UVW_V)
        np.add.at(grad_W, c_idx, UVW_W)
        
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U**2
        U -= lr_t * m_U / (np.sqrt(v_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V**2
        V -= lr_t * m_V / (np.sqrt(v_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W**2
        W -= lr_t * m_W / (np.sqrt(v_W) + eps)
        
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        loss = np.sum((preds - targets) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_loss < 1e-10:
            break
    
    # Try rounding to half-integers
    def _round(arr):
        rounded = np.round(arr * 2) / 2
        if np.max(np.abs(arr - rounded)) < 0.01:
            return rounded
        rounded_int = np.round(arr)
        if np.max(np.abs(arr - rounded_int)) < 0.01:
            return rounded_int
        return arr
    
    U_r = _round(best_U)
    V_r = _round(best_V)
    W_r = _round(best_W)
    
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-4):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-4):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140
    
    def _schoolbook():
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    try:
        # Start with replicated Strassen decomposition (rank 56)
        U_str, V_str, W_str = _replicate_strassen(n, m, p)
        best_result = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
        best_rank = 56
        
        if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p, tol=1e-6):
            best_result = _schoolbook()
            best_rank = n * m * p
        
        # Try to find lower rank via optimization
        for target_rank in [52, 48, 44, 40, 36, 32, 28]:
            if time.time() > deadline - 5:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            
            result = _try_optimize_from_strassen(n, m, p, target_rank, remaining_time * 0.4, seed=42 + target_rank)
            
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
        
        return best_result
    except Exception:
        return _schoolbook()

# EVOLVE-BLOCK-END
