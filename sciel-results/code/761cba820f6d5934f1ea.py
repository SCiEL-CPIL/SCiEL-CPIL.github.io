"""Matrix multiplication 2x2x16: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Valid triple: (i*m+j, j*p+k, k*n+i) should give sum_r U[a,r]*V[b,r]*W[c,r] = 1
    Invalid triples should give 0.
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
                    val = np.dot(U_arr[a] * V_arr[b], W_arr[c])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check invalid triples
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    if j_a == j_b and k_b == k_c and i_c == i_a:
                        continue  # valid triple, already checked
                    
                    val = np.dot(U_arr[a] * V_arr[b], W_arr[c])
                    if abs(val) > tol:
                        return False
        
        return True
    except Exception:
        return False


def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b21-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b11-b12)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
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
                for k_local in range(2):
                    b_idx = j * p + (block * 2 + k_local)
                    v_idx = j * 2 + k_local
                    V[b_idx, col] = V2[v_idx, r]
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    return U, V, W


def _try_k_dimension_optimization(n=2, m=2, p=16, time_budget=20, seed=42):
    """
    Try to find a lower rank decomposition by exploiting the k-dimension structure.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U0, V0, W0 = _replicate_strassen(n, m, p)
    rank0 = U0.shape[1]  # 56
    
    # Try various target ranks
    target_ranks = [48, 44, 40, 36, 34, 32]
    
    # Collect all constraints
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    # For invalid triples, we need to sample since there are many
    invalid_triples = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    invalid_triples.append((a, b, c))
    
    # Use all valid + subset of invalid for speed
    num_invalid_sample = min(len(invalid_triples), 500)
    if num_invalid_sample < len(invalid_triples):
        indices = rng.choice(len(invalid_triples), num_invalid_sample, replace=False)
        invalid_triples = [invalid_triples[i] for i in indices]
    
    all_triples = np.array(valid_triples + invalid_triples)
    targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
    a_idx = all_triples[:, 0]
    b_idx = all_triples[:, 1]
    c_idx = all_triples[:, 2]
    
    for target_rank in target_ranks:
        if time.time() - start_time > time_budget:
            break
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        # Initialize with random projection of the Strassen solution
        P = rng.randn(rank0, target_rank)
        U = U0 @ P
        V = V0 @ P
        W = W0 @ P
        
        # Adam optimizer
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0
        lr = 0.1
        
        best_loss = float('inf')
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        max_steps = 3000
        step_start = time.time()
        
        for step in range(max_steps):
            if time.time() - step_start > remaining * 0.8:
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
            
            np.add.at(grad_U, a_idx, residuals[:, None] * V_sub * W_sub)
            np.add.at(grad_V, b_idx, residuals[:, None] * U_sub * W_sub)
            np.add.at(grad_W, c_idx, residuals[:, None] * U_sub * V_sub)
            
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
            
            # Compute loss periodically
            if step % 50 == 0:
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
        
        # Final loss check
        U_sub = best_U[a_idx, :]
        V_sub = best_V[b_idx, :]
        W_sub = best_W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        final_loss = np.sum((preds - targets) ** 2)
        
        if final_loss < 1e-6:
            # Try rounding to half-integers
            def _round(arr):
                rounded = np.round(arr * 2) / 2
                if np.max(np.abs(arr - rounded)) < 0.05:
                    return rounded
                rounded_int = np.round(arr)
                if np.max(np.abs(arr - rounded_int)) < 0.05:
                    return rounded_int
                return arr
            
            U_r = _round(best_U)
            V_r = _round(best_V)
            W_r = _round(best_W)
            
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-4):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            if _verify_decomposition(best_U.tolist(), best_V.tolist(), W_r.tolist(), n, m, p, tol=1e-4):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None


def _try_kronecker_structure(n=2, m=2, p=16, time_budget=10, seed=42):
    """
    Try a Kronecker-product-based structure.
    """
    return None


def _try_direct_search(n=2, m=2, p=16, time_budget=15, seed=42):
    """
    Try a more direct search for low-rank decompositions.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Start from the 56-rank Strassen solution and try to reduce
    U0, V0, W0 = _replicate_strassen(n, m, p)
    
    # Try to find a rank-48 solution by merging some columns
    return None


def _schoolbook():
    """Schoolbook decomposition with rank = n*m*p = 64."""
    n, m, p = DIMENSIONS
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


def _try_block_split(n=2, m=2, p=16, time_budget=30, seed=42):
    """
    Try splitting the 2x16 matrix B into blocks and using different
    decompositions for each block, then combining.
    
    This is a structural change from the simple replication approach.
    We try:
    1. Split B into 2x8 + 2x8, use rank-28 for each, total rank 56 (same as baseline)
    2. Try to find if we can do better with overlapping or shared structure
    
    Actually, let's try a different approach: use the fact that the problem
    can be viewed as computing C = A @ B where A is 2x2 and B is 2x16.
    
    We can write this as:
    C[:, 0:8] = A @ B[:, 0:8]
    C[:, 8:16] = A @ B[:, 8:16]
    
    Each is a 2x2x8 multiplication. If we can find a rank-r decomposition
    for 2x2x8, then the total rank would be 2*r.
    
    For 2x2x8, we can replicate Strassen 4 times to get rank 28.
    But maybe we can do better?
    
    Let's try to find a rank-24 or rank-20 decomposition for 2x2x8.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # First, build a rank-28 solution for 2x2x8 by replicating Strassen 4 times
    def _strassen_2x2x8():
        U2, V2, W2 = _strassen_2x2x2()
        num_blocks = 8 // 2
        rank = 7 * num_blocks
        
        U = np.zeros((2 * 2, rank))
        V = np.zeros((2 * 8, rank))
        W = np.zeros((8 * 2, rank))
        
        for block in range(num_blocks):
            for r in range(7):
                col = block * 7 + r
                U[:, col] = U2[:, r]
                for j in range(2):
                    for k_local in range(2):
                        b_idx = j * 8 + (block * 2 + k_local)
                        v_idx = j * 2 + k_local
                        V[b_idx, col] = V2[v_idx, r]
                for i in range(2):
                    for k_local in range(2):
                        k = block * 2 + k_local
                        w_idx = k * 2 + i
                        w2_idx = k_local * 2 + i
                        W[w_idx, col] = W2[w2_idx, r]
        
        return U, V, W
    
    U8, V8, W8 = _strassen_2x2x8()
    
    # Now build the 2x2x16 solution by using two copies of the 2x2x8 solution
    # with appropriate index shifts
    U16 = np.zeros((2 * 2, 2 * 28))
    V16 = np.zeros((2 * 16, 2 * 28))
    W16 = np.zeros((16 * 2, 2 * 28))
    
    # First half: k = 0..7
    U16[:, 0:28] = U8
    V16[0:16, 0:28] = V8  # B[0, 0:8] and B[1, 0:8]
    W16[0:16, 0:28] = W8  # C[0:8, 0:2]
    
    # Second half: k = 8..15
    U16[:, 28:56] = U8
    V16[0:16, 28:56] = 0
    V16[16:32, 28:56] = V8  # B[0, 8:16] and B[1, 8:16]
    W16[16:32, 28:56] = W8  # C[8:16, 0:2]
    
    # Verify
    if _verify_decomposition(U16.tolist(), V16.tolist(), W16.tolist(), n, m, p, tol=1e-6):
        # This gives rank 56, same as baseline. Not an improvement, but valid.
        # Let's try to optimize this structure to see if we can reduce rank.
        pass
    
    # Try to find a lower-rank solution for 2x2x8
    target_ranks_8 = [24, 22, 20, 18]
    
    valid_triples_8 = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                valid_triples_8.append((a, b, c))
    
    invalid_triples_8 = []
    for a in range(4):
        for b in range(16):
            for c in range(16):
                i_a = a // 2
                j_a = a % 2
                j_b = b // 8
                k_b = b % 8
                k_c = c // 2
                i_c = c % 2
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    invalid_triples_8.append((a, b, c))
    
    num_invalid_sample = min(len(invalid_triples_8), 200)
    if num_invalid_sample < len(invalid_triples_8):
        indices = rng.choice(len(invalid_triples_8), num_invalid_sample, replace=False)
        invalid_triples_8 = [invalid_triples_8[i] for i in indices]
    
    all_triples_8 = np.array(valid_triples_8 + invalid_triples_8)
    targets_8 = np.concatenate([np.ones(len(valid_triples_8)), np.zeros(len(invalid_triples_8))])
    a_idx_8 = all_triples_8[:, 0]
    b_idx_8 = all_triples_8[:, 1]
    c_idx_8 = all_triples_8[:, 2]
    
    for target_rank in target_ranks_8:
        if time.time() - start_time > time_budget:
            break
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        # Initialize with random projection
        P = rng.randn(28, target_rank)
        U = U8 @ P
        V = V8 @ P
        W = W8 @ P
        
        # Adam optimizer
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0
        lr = 0.1
        
        best_loss = float('inf')
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        max_steps = 2000
        step_start = time.time()
        
        for step in range(max_steps):
            if time.time() - step_start > remaining * 0.8:
                break
            t += 1
            
            U_sub = U[a_idx_8, :]
            V_sub = V[b_idx_8, :]
            W_sub = W[c_idx_8, :]
            preds = np.sum(U_sub * V_sub * W_sub, axis=1)
            residuals = preds - targets_8
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            np.add.at(grad_U, a_idx_8, residuals[:, None] * V_sub * W_sub)
            np.add.at(grad_V, b_idx_8, residuals[:, None] * U_sub * W_sub)
            np.add.at(grad_W, c_idx_8, residuals[:, None] * U_sub * V_sub)
            
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
            
            if step % 50 == 0:
                U_sub = U[a_idx_8, :]
                V_sub = V[b_idx_8, :]
                W_sub = W[c_idx_8, :]
                preds = np.sum(U_sub * V_sub * W_sub, axis=1)
                loss = np.sum((preds - targets_8) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_loss < 1e-8:
                    break
        
        U_sub = best_U[a_idx_8, :]
        V_sub = best_V[b_idx_8, :]
        W_sub = best_W[c_idx_8, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        final_loss = np.sum((preds - targets_8) ** 2)
        
        if final_loss < 1e-6:
            # Build the 2x2x16 solution using this 2x2x8 solution
            U16 = np.zeros((4, 2 * target_rank))
            V16 = np.zeros((32, 2 * target_rank))
            W16 = np.zeros((32, 2 * target_rank))
            
            U16[:, 0:target_rank] = best_U
            V16[0:16, 0:target_rank] = best_V
            W16[0:16, 0:target_rank] = best_W
            
            U16[:, target_rank:2*target_rank] = best_U
            V16[16:32, target_rank:2*target_rank] = best_V
            W16[16:32, target_rank:2*target_rank] = best_W
            
            if _verify_decomposition(U16.tolist(), V16.tolist(), W16.tolist(), n, m, p, tol=1e-5):
                return U16.tolist(), V16.tolist(), W16.tolist()
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def _safe_return(result):
        """Return result, ensuring it's valid."""
        if result is None:
            return _schoolbook()
        if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-5):
            return _schoolbook()
        return result
    
    try:
        # Baseline: replicated Strassen (rank 56)
        U_str, V_str, W_str = _replicate_strassen(n, m, p)
        baseline = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
        
        if not _verify_decomposition(baseline["U"], baseline["V"], baseline["W"], n, m, p, tol=1e-6):
            return _schoolbook()
        
        best_result = baseline
        best_rank = 56
        
        # Try block split optimization
        if time.time() < deadline - 20:
            result = _try_block_split(n, m, p, time_budget=30, seed=42)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                    new_rank = len(U[0])
                    if new_rank < best_rank:
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = new_rank
        
        # Try k-dimension optimization
        if time.time() < deadline - 15 and best_rank > 40:
            result = _try_k_dimension_optimization(n, m, p, time_budget=20, seed=42)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                    new_rank = len(U[0])
                    if new_rank < best_rank:
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = new_rank
        
        return best_result
    except Exception:
        return _schoolbook()

# EVOLVE-BLOCK-END
