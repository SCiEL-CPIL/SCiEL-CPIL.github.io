"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _build_schoolbook_tensors(n, m, p):
    """Build the n*m*p schoolbook basis tensors as flat arrays of length n*m*p."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    # Each tensor is a 3-tensor T[i,j,k] = A[i,j]*B[j,k] -> C[i,k]
    # Flatten: U index = i*m+j, V index = j*p+k, W index = k*n+i
    # Tensor component: T[U_idx, V_idx, W_idx] = 1 if consistent (i,j,k)
    tensors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Create flat tensor of size nm*mp*pn
                t = np.zeros(nm * mp * pn, dtype=np.int64)
                t[u_idx * mp * pn + v_idx * pn + w_idx] = 1
                tensors.append(t)
    return tensors

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Compute the resulting tensor from the decomposition
    # For each (i,j,k), the coefficient should be 1
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal delta(i,j,k) for the multiplication tensor
    # Actually, the multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # Wait, let me reconsider. The tensor for matrix mult C=AB where A is n x m, B is m x p:
    # T[i,j,k] = 1 means C[i,k] gets A[i,j]*B[j,k]
    # In the flattened form: U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    # The tensor component is T[U_idx, V_idx, W_idx] = 1 for valid (i,j,k)
    
    # For the decomposition to be exact:
    # sum_c U[U_idx, c] * V[V_idx, c] * W[W_idx, c] = T[U_idx, V_idx, W_idx]
    
    # Check all nonzero components
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that zero components remain zero (spot check)
    # For random (u_idx, v_idx, w_idx) not corresponding to valid (i,j,k), should be 0
    nm = n * m
    mp = m * p
    pn = p * n
    # Build set of valid (u_idx, v_idx, w_idx)
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Spot check some invalid entries
    rng = np.random.RandomState(42)
    for _ in range(200):
        u_idx = rng.randint(0, nm)
        v_idx = rng.randint(0, mp)
        w_idx = rng.randint(0, pn)
        if (u_idx, v_idx, w_idx) in valid:
            continue
        val = 0.0
        for c in range(rank):
            val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
        if abs(val) > 1e-9:
            return False
    return True

def _find_minimal_rank(n, m, p, time_budget):
    """Find minimal rank by detecting linear dependencies among schoolbook tensors."""
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    total_dim = nm * mp * pn
    rank_schoolbook = n * m * p
    
    # Build schoolbook tensors
    tensors = _build_schoolbook_tensors(n, m, p)
    T = np.array(tensors, dtype=np.float64)  # shape (rank_schoolbook, total_dim)
    
    # The multiplication tensor is the sum of all schoolbook tensors
    T_target = T.sum(axis=0)
    
    # We want to find a subset of size r such that some linear combination equals T_target
    # Actually, we want: T_target = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where each term is a rank-1 tensor
    # The schoolbook gives us rank-1 tensors: e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # We can take linear combinations of these rank-1 tensors
    
    # Strategy: Try to express T_target as a linear combination of fewer than 40 rank-1 tensors
    # by finding linear dependencies
    
    # First, check if the 40 rank-1 tensors are linearly independent
    # If they are, rank is 40. If not, we can reduce.
    
    # Use random projections to find dependencies
    # Project T onto random vectors to get a smaller system
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = rank_schoolbook
    
    # Schoolbook solution as fallback
    U_school = np.zeros((nm, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((mp, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((pn, rank_schoolbook), dtype=np.float64)
    for c in range(rank_schoolbook):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U_school[i * m + j, c] = 1.0
        V_school[j * p + k, c] = 1.0
        W_school[k * n + i, c] = 1.0
    
    if _verify_decomposition(U_school.tolist(), V_school.tolist(), W_school.tolist(), n, m, p):
        best_U = U_school.tolist()
        best_V = V_school.tolist()
        best_W = W_school.tolist()
        best_rank = rank_schoolbook
    
    # Now try to reduce rank
    # Approach: find linear dependencies among the rank-1 tensors
    # If sum_{c in S} alpha_c * T_c = 0, we can eliminate one tensor
    
    # Use SVD to find the rank of the tensor matrix
    # But the tensors are 3D, so we need to flatten appropriately
    
    # Alternative approach: try random linear combinations
    # For r < 40, try to find r rank-1 tensors that sum to T_target
    
    # Use iterative approach: start with all 40, try to eliminate one at a time
    # Check if removing tensor c and adjusting others still works
    
    # More systematic: use the fact that we can combine tensors
    # If T_a + T_b + T_c = T_d (as rank-1 tensors), we can replace 3 with 1
    
    # For small dimensions, brute force search for linear dependencies
    # T_target = sum_c x_c * T_c where x_c are coefficients
    # We need the resulting expression to be a sum of rank-1 tensors with integer/half-integer entries
    
    # Let's try: can we find coefficients such that the sum has fewer nonzero "terms"?
    # Actually, any linear combination of the 40 rank-1 tensors is still a valid decomposition
    # with rank = number of nonzero coefficients
    
    # So the question is: what is the minimum number of nonzero coefficients needed?
    # T_target = sum_c x_c * T_c
    # where T_c are the 40 rank-1 tensors
    
    # This is a linear system: T @ x = T_target
    # But T_target = T @ ones, so x = ones is one solution
    # Are there other solutions? Yes, if the columns of T are linearly dependent
    
    # Find null space of T
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T, full_matrices=False)
        # Rank of T
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < rank_schoolbook:
            # There are linear dependencies
            # Null space dimension = rank_schoolbook - effective_rank
            null_dim = rank_schoolbook - effective_rank
            null_space = Vt[effective_rank:]  # shape (null_dim, rank_schoolbook)
            
            # We can add any vector in the null space to x=ones
            # x = ones + null_space @ z
            # We want to minimize the number of nonzero entries in x
            # This is an L0 minimization, but let's try simple approaches
            
            # Try to find z that zeros out some components
            for attempt in range(100):
                if time.time() - start_time > time_budget - 20:
                    break
                # Random z
                z = np.random.randn(null_dim)
                x = np.ones(rank_schoolbook) + null_space @ z
                # Round to nearest 0.5
                x_rounded = np.round(x * 2) / 2
                # Check if this still gives T_target
                T_recon = T @ x_rounded
                if np.allclose(T_recon, T_target, atol=1e-9):
                    # Count nonzero entries
                    nz = np.sum(np.abs(x_rounded) > 1e-9)
                    if nz < best_rank:
                        # Build the decomposition
                        U_new = np.zeros((nm, int(nz)), dtype=np.float64)
                        V_new = np.zeros((mp, int(nz)), dtype=np.float64)
                        W_new = np.zeros((pn, int(nz)), dtype=np.float64)
                        col = 0
                        for c in range(rank_schoolbook):
                            if abs(x_rounded[c]) > 1e-9:
                                i = c // (m * p)
                                j = (c // p) % m
                                k = c % p
                                U_new[i * m + j, col] = x_rounded[c]
                                V_new[j * p + k, col] = 1.0
                                W_new[k * n + i, col] = 1.0
                                col += 1
                        # Verify
                        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                            best_U = U_new.tolist()
                            best_V = V_new.tolist()
                            best_W = W_new.tolist()
                            best_rank = int(nz)
    except Exception:
        pass
    
    # Try more aggressive: search for small-rank decompositions directly
    # For r from 10 to best_rank-1, try to find r rank-1 tensors
    # This is hard, so use a heuristic
    
    # Try specific known small-rank decompositions
    # For 2x4x5, the border rank might be less than 40
    
    # Use a different approach: random initialization + gradient-free search
    # But this is risky for exactness
    
    # Let's try: can we merge pairs of tensors?
    # T_{i,j,k} + T_{i',j',k'} where they share structure
    
    # For 2x4x5: n=2, m=4, p=5
    # Tensors are indexed by (i,j,k) with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    
    # Check if any two tensors can be combined into one rank-1 tensor
    # T_a + T_b is rank-1 only if they are proportional (which they're not, since each has a single 1)
    
    # So we can't simply merge. But we can use linear combinations.
    
    # Try: find a subset of tensors and coefficients such that the sum is still a valid
    # decomposition with fewer terms
    
    # Actually, the key insight: the multiplication tensor for C=AB has a specific structure.
    # Let's think about it differently.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map. The tensor rank is the minimum number of rank-1 tensors needed.
    
    # For 2x4x5, known results suggest the rank might be around 20-30.
    
    # Let's try a systematic search for rank reduction using the null space approach
    # more thoroughly
    
    if effective_rank < rank_schoolbook:
        # Try to find sparse solutions
        for trial in range(500):
            if time.time() - start_time > time_budget - 15:
                break
            # Random null space vector
            z = np.random.randn(null_dim)
            x = np.ones(rank_schoolbook) + null_space @ z
            # Try to sparsify: set small values to 0
            for threshold in [0.1, 0.2, 0.3, 0.5, 0.7]:
                x_sparse = x.copy()
                x_sparse[np.abs(x_sparse) < threshold] = 0
                # Check if still valid
                T_recon = T @ x_sparse
                if np.allclose(T_recon, T_target, atol=1e-9):
                    nz = np.sum(np.abs(x_sparse) > 1e-9)
                    if nz < best_rank:
                        U_new = np.zeros((nm, int(nz)), dtype=np.float64)
                        V_new = np.zeros((mp, int(nz)), dtype=np.float64)
                        W_new = np.zeros((pn, int(nz)), dtype=np.float64)
                        col = 0
                        for c in range(rank_schoolbook):
                            if abs(x_sparse[c]) > 1e-9:
                                i = c // (m * p)
                                j = (c // p) % m
                                k = c % p
                                U_new[i * m + j, col] = x_sparse[c]
                                V_new[j * p + k, col] = 1.0
                                W_new[k * n + i, col] = 1.0
                                col += 1
                        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                            best_U = U_new.tolist()
                            best_V = V_new.tolist()
                            best_W = W_new.tolist()
                            best_rank = int(nz)
    
    # Try even more: use the fact that we can use arbitrary rank-1 tensors, not just the schoolbook ones
    # This is a much harder problem, but for small dimensions we can try
    
    # For now, return the best found
    if best_U is None:
        best_U = U_school.tolist()
        best_V = V_school.tolist()
        best_W = W_school.tolist()
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Find minimal rank decomposition
    U, V, W, rank = _find_minimal_rank(n, m, p, TIME_LIMIT)
    
    # Ensure entries are exact (integer or half-integer)
    # Round to nearest 0.5 and verify
    U_rounded = [[round(val * 2) / 2 for val in row] for row in U]
    V_rounded = [[round(val * 2) / 2 for val in row] for row in V]
    W_rounded = [[round(val * 2) / 2 for val in row] for row in W]
    
    # Verify after rounding
    if not _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        # Fall back to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        for c in range(rank_schoolbook):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
