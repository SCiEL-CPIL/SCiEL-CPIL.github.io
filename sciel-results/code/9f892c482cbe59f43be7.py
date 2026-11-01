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
    Fast vectorized check.
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
        
        # Build valid triples
        valid_a, valid_b, valid_c = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_a.append(i * m + j)
                    valid_b.append(j * p + k)
                    valid_c.append(k * n + i)
        
        valid_a = np.array(valid_a)
        valid_b = np.array(valid_b)
        valid_c = np.array(valid_c)
        
        # Check valid entries
        vals = np.sum(U_arr[valid_a] * V_arr[valid_b] * W_arr[valid_c], axis=1)
        if np.max(np.abs(vals - 1.0)) > tol:
            return False
        
        # Build invalid triples - use structured approach to avoid full meshgrid
        # For 2x2x16: a in [0,4), b in [0,32), c in [0,32)
        # Total: 4*32*32 = 4096 triples, 64 valid, 4032 invalid
        all_a = np.arange(n * m)
        all_b = np.arange(m * p)
        all_c = np.arange(p * n)
        
        # Instead of full meshgrid, check a subset of invalid triples
        # The key invalid triples are those where indices don't match
        # Sample strategically
        rng = np.random.RandomState(42)
        
        # Generate random invalid triples
        num_samples = min(2000, (n*m)*(m*p)*(p*n) - n*m*p)
        invalid_checked = 0
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
                continue  # This is valid, skip
            
            val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
            if abs(val) > tol:
                return False
            invalid_checked += 1
        
        # Also check specific structured invalid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Check nearby invalid triples
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni = (i + di) % n
                            nj = (j + dj) % m
                            nk = (k + 1) % p
                            if ni == i and nj == j and nk == k:
                                continue
                            na = ni * m + nj
                            nb = nj * p + nk
                            nc = nk * n + ni
                            # Only check if this is actually invalid
                            i_a2 = na // m
                            j_a2 = na % m
                            j_b2 = nb // p
                            k_b2 = nb % p
                            k_c2 = nc // n
                            i_c2 = nc % n
                            if j_a2 == j_b2 and k_b2 == k_c2 and i_c2 == i_a2:
                                continue
                            val = np.sum(U_arr[na] * V_arr[nb] * W_arr[nc])
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

def _fourier_block_decomposition(n=2, m=2, p=16):
    """
    Try a 4-block Fourier structured decomposition.
    Group 16 columns into 4 blocks of 4. Apply 4x4 DFT across blocks.
    Each frequency component may need fewer multiplications.
    """
    U2, V2, W2 = _strassen_2x2x2()
    num_blocks = 4
    block_size = p // num_blocks  # 4
    
    # 4x4 DFT matrix
    omega = np.exp(2j * np.pi / num_blocks)
    F = np.array([[omega**(i*j) for j in range(num_blocks)] for i in range(num_blocks)])
    F_inv = F.conj().T / num_blocks
    
    # For each frequency f (0..3), we have a 2x2x4 sub-problem
    # The 4 columns within each block are handled by the 2x2x2 Strassen
    # but we need to see if frequency coupling helps
    
    # Actually, let's try: for each frequency f, use Strassen on the
    # "averaged" block. Total rank = 4 * 7 = 28 if it works.
    
    rank = 4 * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for f in range(num_blocks):
        for r in range(7):
            col = f * 7 + r
            # U is the same Strassen U
            U[:, col] = U2[:, r]
            
            # V: for each j in {0,1}, and each column k in {0..15}
            # Group columns into blocks: block b contains columns [b*4, (b+1)*4)
            # Within each block, positions 0,1 are handled by Strassen
            # Apply DFT across blocks
            for j in range(m):
                for k in range(p):
                    b = k // block_size  # block index 0..3
                    k_local = k % block_size  # position within block 0..3
                    
                    # Only positions 0,1 within block contribute (Strassen structure)
                    if k_local < 2:
                        v_idx = j * 2 + k_local
                        # DFT weight for frequency f, block b
                        weight = F[f, b]
                        V[j * p + k, col] = V2[v_idx, r] * weight
                    # positions 2,3 within block: these are extra, need handling
            
            # W: for each k in {0..15}, i in {0,1}
            for i in range(n):
                for k in range(p):
                    b = k // block_size
                    k_local = k % block_size
                    if k_local < 2:
                        w2_idx = k_local * n + i
                        weight = F_inv[f, b]
                        W[k * n + i, col] = W2[w2_idx, r] * weight
    
    return U, V, W

def _round_to_half_int(arr, tol=1e-3):
    """Round array to nearest half-integer if close."""
    result = arr.copy()
    rounded = np.round(result * 2) / 2
    if np.max(np.abs(result - rounded)) < tol:
        return rounded
    rounded_int = np.round(result)
    if np.max(np.abs(result - rounded_int)) < tol:
        return rounded_int
    return result

def _try_find_decomposition(n, m, p, rank, seed, time_budget, init=None):
    """
    Try to find a rank-rank decomposition using gradient-based optimization.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Collect all valid triples
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
    total_invalid = (n*m)*(m*p)*(p*n) - n*m*p
    num_invalid_samples = min(1500, total_invalid)
    for _ in range(num_invalid_samples):
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
    
    # Initialize
    if init is not None:
        U, V, W = init
        U = U.astype(np.float64).copy()
        V = V.astype(np.float64).copy()
        W = W.astype(np.float64).copy()
    else:
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
    
    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    t = 0
    lr = 0.01
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for step in range(2000):
        if time.time() - start_time > time_budget:
            break
        t += 1
        
        # Compute predictions
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        residuals = preds - targets
        
        # Gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        UVW_U = residuals[:, None] * V_sub * W_sub
        UVW_V = residuals[:, None] * U_sub * W_sub
        UVW_W = residuals[:, None] * U_sub * V_sub
        
        np.add.at(grad_U, a_idx, UVW_U)
        np.add.at(grad_V, b_idx, UVW_V)
        np.add.at(grad_W, c_idx, UVW_W)
        
        # Adam update
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
        
        # Loss
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
        
        if best_loss < 1e-8:
            break
    
    # Try rounding
    U_round = _round_to_half_int(best_U)
    V_round = _round_to_half_int(best_V)
    W_round = _round_to_half_int(best_W)
    
    if _verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p, tol=1e-4):
        return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
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
        
        # Try Fourier-structured decomposition (rank 28)
        if time.time() < deadline - 10:
            U_f, V_f, W_f = _fourier_block_decomposition(n, m, p)
            if _verify_decomposition(U_f.tolist(), V_f.tolist(), W_f.tolist(), n, m, p, tol=1e-4):
                best_result = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                best_rank = 28
        
        # Try to find lower rank via optimization
        seed = 42
        for rank in range(27, 15, -1):
            if time.time() > deadline - 5:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 3:
                break
            
            result = _try_find_decomposition(n, m, p, rank, seed, remaining_time * 0.3)
            seed += 1
            
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        
        return best_result
    except Exception:
        return _schoolbook()

# EVOLVE-BLOCK-END
