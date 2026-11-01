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
        
        # Check invalid triples (all of them since dimensions are small)
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
    
    Key insight: For 2x2x16, we can view this as:
    - 4 independent (i,j) pairs (since n=2, m=2)
    - Each needs to handle 16 k-values
    
    The tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For fixed (i,j), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    
    This means for each (i,j), the vectors {U[i*m+j, r]}_r, {V[j*p+k, r]}_r, {W[k*n+i, r]}_r
    must satisfy the constraint.
    
    Strategy: Try to share rank across different (i,j) pairs by using a common
    k-decomposition structure.
    
    Alternative: Use the fact that the problem is 2x2x16 = 2x2x(8*2).
    We can try to find a decomposition where the k-dimension uses a structured
    (e.g., Kronecker) factorization.
    
    Let's try: rank = 7 * 8 = 56 is the baseline.
    Can we do better? The lower bound for 2x2x16 is at least 2*2*16/2 = 32
    (from the trivial bound rank >= n*m*p / max(n,m,p) ... actually that's not right).
    
    Actually, for matrix multiplication (n,m,p), the rank is at least n*m*p / (n+m+p-2)
    by the Lander-Szekeres bound... no, that's not a standard bound.
    
    The trivial lower bound is rank >= max(n*m, m*p, p*n) = max(4, 32, 32) = 32.
    But actually for matrix multiplication, rank >= n*m*p / (n+m+p) is not standard.
    
    Let me think differently. The schoolbook rank is 64. Strassen gives 56.
    Can we do 48, 42, 36, 32?
    
    One approach: use a rank-r decomposition of the 16-dim "identity" in k,
    shared across the 4 (i,j) pairs, combined with the 2x2x2 structure.
    
    For the 2x2x2 case, we need 7 rank-1 terms. For 2x2x16, if we could use
    the same 7 terms but with the k-dimension handled by a rank-s decomposition
    of a 16x16 identity (or similar), we'd get rank = 7*s.
    
    But the k-dimension isn't an identity - it's more like: for each (i,j),
    we need V[j,*,r] and W[*,i,r] to be "dual" vectors in the k-space.
    
    Let me try a direct optimization approach.
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
    # Total triples: 4 * 32 * 32 = 4096
    # Valid: 4 * 16 = 64
    # Invalid: 4096 - 64 = 4032
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
    
    2x2x16 = 2x2x(2*8). We can try:
    - Use Strassen for the 2x2x2 part (rank 7)
    - Use a rank-r decomposition for the remaining 8-fold k-replication
    
    But the k-dimension isn't simply replicated - each k is independent.
    
    Alternative: Think of 2x2x16 as computing C = A * B where A is 2x2, B is 2x16.
    C is 2x16. C[i,k] = sum_j A[i,j] * B[j,k].
    
    This is 4 independent dot products of length 16... no wait, it's:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    So for each (i,k), we compute a sum of 2 products. Total 4*16 = 64 products.
    With Strassen-like tricks on the 2x2 part, we reduce to 7*16 = 112? No...
    
    Actually, the standard approach: treat it as 8 independent 2x2x2 multiplications
    (grouping k in pairs), each with rank 7, total rank 56.
    
    Can we do better by sharing across the k-pairs? The answer is generally no for
    exact decomposition, because the k-dimension is "independent" in the tensor sense.
    
    However, there might be a clever construction. Let's try a different approach:
    use the fact that rank(2x2x16) >= rank(2x2x2) = 7, and the "slice" argument
    gives rank >= 4*16/2 = 32 (using the fact that each of the 4 (i,j) slices
    must be nonzero, and each rank-1 term can cover at most 2 k-values per slice... 
    actually this isn't rigorous).
    
    Let me just try to verify that 56 is achievable and see if we can find anything lower.
    """
    # This is a placeholder for future exploration
    return None


def _try_direct_search(n=2, m=2, p=16, time_budget=15, seed=42):
    """
    Try a more direct search for low-rank decompositions.
    
    Key observation: For 2x2x16, the tensor has 4*32*32 = 4096 entries,
    of which 64 are 1 (valid) and 4032 are 0 (invalid).
    
    A rank-r decomposition means we have r "gates" (U_col, V_col, W_col)
    such that the product tensor matches.
    
    Let's try to construct a solution with rank < 56 by exploiting
    symmetries or structure.
    
    One idea: For each of the 4 (i,j) pairs, we need 16 k-values to work.
    If we use a rank-s decomposition for the k-part (where s < 8), we'd save rank.
    
    For a single (i,j) pair, the constraint is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k = 0..15
    
    If U[i*m+j, r] is nonzero for r in some set S_{ij}, then we need:
    sum_{r in S_{ij}} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a system of 16 equations (one per k) in the unknowns V[j*p+k, r] and W[k*n+i, r].
    
    If we fix U, then for each (i,j), we need V[j,*,r] and W[*,i,r] to satisfy
    a bilinear constraint in k. This is like a matrix factorization problem.
    
    Let's try: for each (i,j), define M_{ij}[k, r] = U[i*m+j, r] (constant in k).
    Then we need: sum_r M_{ij}[k,r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    If we set V[j*p+k, r] = v_r * f_{j,r}(k) and W[k*n+i, r] = w_r * g_{i,r}(k),
    then: sum_r M_{ij}[k,r] * v_r * w_r * f_{j,r}(k) * g_{i,r}(k) = 1.
    
    This is getting complex. Let me just try the optimization approach with more care.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Start from the 56-rank Strassen solution and try to reduce
    U0, V0, W0 = _replicate_strassen(n, m, p)
    
    # Try to find a rank-48 solution by merging some columns
    # Idea: if two columns are "similar" in their contribution, we might merge them
    
    # For now, just return None to indicate no improvement found
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
            # Shouldn't happen, but fallback
            return _schoolbook()
        
        best_result = baseline
        best_rank = 56
        
        # Try optimization to find lower rank
        if time.time() < deadline - 30:
            result = _try_k_dimension_optimization(n, m, p, time_budget=30, seed=42)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = len(U[0])
        
        # Try direct search with different seeds
        if time.time() < deadline - 15 and best_rank > 40:
            for seed in [123, 456, 789]:
                if time.time() > deadline - 10:
                    break
                result = _try_k_dimension_optimization(n, m, p, time_budget=10, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                        new_rank = len(U[0])
                        if new_rank < best_rank:
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = new_rank
                        break
        
        return best_result
    except Exception:
        return _schoolbook()

# EVOLVE-BLOCK-END
