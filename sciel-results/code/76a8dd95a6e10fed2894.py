"""Matrix multiplication 3x3 by 3x7 - rank minimization via structural factorization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8, trials=8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    rng = np.random.default_rng(42)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for trial in range(trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        
        # Vectorized verification
        # A_flat has shape (n*m,), B_flat has shape (m*p,)
        A_flat = A.flatten()  # row-major: A[i,j] at index i*m+j
        B_flat = B.flatten()  # row-major: B[j,k] at index j*p+k
        
        # For each rank element c:
        # a_form = U[:, c] @ A_flat  (but U is (n*m, rank), so U[:, c] is (n*m,))
        # Actually: a_form[i] = sum_j U[i*m+j, c] * A[i,j]
        # This is U[:, c] dot A_flat
        # b_form[k] = sum_j V[j*p+k, c] * B[j,k]
        # This is V[:, c] dot B_flat
        # C_comp[i,k] = sum_c W[k*n+i, c] * a_form[i] * b_form[k]
        
        # Let's do it more efficiently
        # U_T = U.T has shape (rank, n*m)
        # a_forms = U_T @ A_flat has shape (rank, n*m) -> reshape to (rank, n, m)
        # But we need a_form[i] = sum_j U[i*m+j, c] * A[i,j] for each c
        # So a_forms[c, i] = sum_j U[i*m+j, c] * A[i,j]
        # This is: for each c, a_forms[c, i] = dot(U[i*m:(i+1)*m, c], A[i,:])
        # 
        # Let's use einsum or reshape
        # U is (n*m, rank). U.reshape(n, m, rank)
        # a_forms = np.einsum('ijc,ij->ci', U_arr.reshape(n, m, rank), A)  # shape (rank, n)
        # V is (m*p, rank). V.reshape(m, p, rank)
        # b_forms = np.einsum('jk c, jk -> c k', V_arr.reshape(m, p, rank), B)  # shape (rank, p)
        # W is (p*n, rank). W.reshape(p, n, rank)
        # C_comp = np.einsum('c i, c k, c k i -> i k', a_forms, b_forms, W_arr.reshape(p, n, rank))
        # Wait, W[k*n+i, c] so W.reshape(p, n, rank) gives W[k, i, c]
        # C_comp[i,k] = sum_c W[k, i, c] * a_forms[c, i] * b_forms[c, k]
        # C_comp = np.einsum('cik, ci, ck -> ik', W_arr.reshape(p, n, rank), a_forms, b_forms)
        
        a_forms = np.einsum('ijc,ij->ci', U_arr.reshape(n, m, rank), A)
        b_forms = np.einsum('jkc,jk->ck', V_arr.reshape(m, p, rank), B)
        C_comp = np.einsum('kic,ci,ck->ik', W_arr.reshape(p, n, rank), a_forms, b_forms)
        
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduced_rank(n, m, p, rank, time_budget):
    """Try to find a valid decomposition of given rank using random search."""
    start = time.time()
    
    # For each rank element c, we have:
    # U_c: vector in R^(n*m) - coefficients for A
    # V_c: vector in R^(m*p) - coefficients for B
    # W_c: vector in R^(p*n) - coefficients for C
    #
    # The constraint is: for all (i,k),
    # sum_j A[i,j] B[j,k] = sum_c W_c[k*n+i] * (sum_j U_c[i*m+j] A[i,j]) * (sum_j V_c[j*p+k] B[j,k])
    #
    # This must hold for all A, B.
    #
    # Let's think about this differently. The tensor T[i,j,k] = 1 for all i,j,k (in the sense that
    # C[i,k] = sum_j A[i,j] B[j,k]).
    #
    # A rank-r decomposition means we can write the multiplication as r "multiplications" of
    # linear forms. The key insight is that we need:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'} ... 
    # This is getting complicated. Let me just try random initializations and verify.
    
    rng = np.random.default_rng(123)
    
    # Strategy: start with schoolbook and try to merge columns
    # Or: try random rank-r decompositions and verify
    
    # For small ranks, we can try to find exact solutions
    # The minimum rank for 3x3x7 is unknown but likely around 40-50
    
    # Let's try a structured approach:
    # For each of the m=3 "middle" indices j, we have n*p = 3*7 = 21 terms
    # Total schoolbook = 3 * 21 = 63
    
    # Can we share computations across j values?
    # If we allow U_c and V_c to be arbitrary (not just unit vectors), we might be able to
    # combine terms.
    
    # Try random search for the given rank
    if rank < n * m:  # Not enough terms to cover all A entries
        return None
    if rank < m * p:  # Not enough terms to cover all B entries
        return None
    
    # Random initialization
    for attempt in range(50):
        if time.time() - start > time_budget:
            break
            
        U_arr = rng.standard_normal((n * m, rank))
        V_arr = rng.standard_normal((m * p, rank))
        W_arr = rng.standard_normal((p * n, rank))
        
        # Verify
        if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p, tol=1e-6, trials=3):
            # Round to clean values if close to integers/half-integers
            U_clean = np.round(U_arr * 2) / 2
            V_clean = np.round(V_arr * 2) / 2
            W_clean = np.round(W_arr * 2) / 2
            
            if _verify(U_clean.tolist(), V_clean.tolist(), W_clean.tolist(), n, m, p, tol=1e-8, trials=8):
                return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
    
    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution (rank 63)
    candidate = _schoolbook(n, m, p)
    
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        best = candidate
        best_rank = n * m * p
    else:
        best = candidate
        best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also rank >= n*p = 21 (from output dimension)
    # Practical lower bound is higher
    
    # Try ranks from just above lower bound up to schoolbook rank
    # But be smart about it - start from promising values
    
    # Known results: R(3,3,3) = 23 (Strassen-like), R(2,2,2) = 7
    # For 3x3x7, let's try a range of ranks
    
    # First, try to see if we can do better than 63
    # Try ranks in decreasing order of likelihood
    ranks_to_try = [55, 50, 45, 40, 35, 30, 28, 25, 23, 22, 21]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        result = _try_reduced_rank(n, m, p, rank, time_budget=8.0)
        if result is not None:
            best = result
            best_rank = rank
            # If we found a good one, try slightly lower
            break
    
    # Also try a few more specific ranks if time allows
    if time.time() < deadline - 15:
        for rank in [48, 42, 38, 32, 26]:
            if time.time() > deadline - 5:
                break
            if rank >= best_rank:
                continue
            result = _try_reduced_rank(n, m, p, rank, time_budget=5.0)
            if result is not None:
                best = result
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
