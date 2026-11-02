"""Combined: structural baseline + ALS-based rank reduction search for 3x4x8 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix.
    
    The tensor T has entries T[i, j, k] for i in [0,n), j in [0,m), k in [0,p).
    A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    
    Strategy:
    1. Start with schoolbook decomposition (rank 96) as a valid baseline.
    2. Use alternating least squares (ALS) to find lower-rank decompositions.
    3. For each candidate rank r, randomly initialize and iterate ALS until convergence.
    4. If a valid decomposition is found, return it.
    5. Also try rank-1 peeling: iteratively extract rank-1 components from the residual.
    """
    deadline = time.time() + 115  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Precompute the target tensor indices
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Build index arrays for efficient tensor operations
    # For each (i,j,k) triple, we have a constraint
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    # Schoolbook baseline
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for (u_idx, v_idx, w_idx) in triples:
        U_school[u_idx, col] = 1
        V_school[v_idx, col] = 1
        W_school[w_idx, col] = 1
        col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        for (u_idx, v_idx, w_idx) in triples:
            s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            if abs(s - 1.0) > tol:
                return False
        return True
    
    def tensor_norm_residual(U, V, W):
        """Compute max absolute residual."""
        max_res = 0.0
        for (u_idx, v_idx, w_idx) in triples:
            s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            res = abs(s - 1.0)
            if res > max_res:
                max_res = res
        return max_res
    
    # ALS solver for a given rank
    def solve_als(target_rank, max_iters=500, tol=1e-8, seed=42):
        """
        Alternating least squares to find U, V, W of rank target_rank
        such that T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
        
        For fixed V and W, each row of U is independent:
          U[u_idx, :] must satisfy: for all (v_idx, w_idx) paired with u_idx,
          sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c] = 1
        
        Actually, the constraints couple all three matrices. Let me reformulate.
        
        For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        
        Fix V, W. Then for each u_idx = i*m+j, we have constraints for each k:
          sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        This is a linear system in U[u_idx,:]. Similarly for V and W.
        """
        rng = np.random.RandomState(seed)
        
        # Random initialization
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Precompute for each u_idx, which (v_idx, w_idx) pairs constrain it
        # u_idx = i*m + j, for each k: v_idx = j*p+k, w_idx = k*n+i
        u_constraints = {}  # u_idx -> list of (v_idx, w_idx)
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                pairs = []
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    pairs.append((v_idx, w_idx))
                u_constraints[u_idx] = pairs
        
        # Similarly for v_idx = j*p + k, constrained by i:
        v_constraints = {}  # v_idx -> list of (u_idx, w_idx)
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                pairs = []
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    pairs.append((u_idx, w_idx))
                v_constraints[v_idx] = pairs
        
        # For w_idx = k*n + i, constrained by j:
        w_constraints = {}  # w_idx -> list of (u_idx, v_idx)
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                pairs = []
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    pairs.append((u_idx, v_idx))
                w_constraints[w_idx] = pairs
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            prev_norm = tensor_norm_residual(U, V, W)
            
            # Update U: for each u_idx, solve linear system
            for u_idx in range(n * m):
                pairs = u_constraints[u_idx]
                # Build system: for each (v_idx, w_idx), 
                # sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c] = 1
                # This is: (elementwise product of V[v_idx,:] and W[w_idx,:]) dot U[u_idx,:] = 1
                if len(pairs) == 0:
                    continue
                A_rows = []
                for (v_idx, w_idx) in pairs:
                    A_rows.append(V[v_idx] * W[w_idx])
                A_mat = np.array(A_rows)  # shape (num_constraints, target_rank)
                b_vec = np.ones(len(pairs))
                
                # Solve A_mat @ U[u_idx,:] = b_vec
                # Use least squares
                try:
                    U[u_idx], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Update V
            for v_idx in range(m * p):
                pairs = v_constraints[v_idx]
                if len(pairs) == 0:
                    continue
                A_rows = []
                for (u_idx, w_idx) in pairs:
                    A_rows.append(U[u_idx] * W[w_idx])
                A_mat = np.array(A_rows)
                b_vec = np.ones(len(pairs))
                try:
                    V[v_idx], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Update W
            for w_idx in range(p * n):
                pairs = w_constraints[w_idx]
                if len(pairs) == 0:
                    continue
                A_rows = []
                for (u_idx, v_idx) in pairs:
                    A_rows.append(U[u_idx] * V[v_idx])
                A_mat = np.array(A_rows)
                b_vec = np.ones(len(pairs))
                try:
                    W[w_idx], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            new_norm = tensor_norm_residual(U, V, W)
            if abs(prev_norm - new_norm) < tol:
                break
        
        return U, V, W, tensor_norm_residual(U, V, W)
    
    # Try to find lower-rank decompositions
    # Start with reasonable ranks and go down
    candidate_ranks = list(range(90, 60, -2))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            seed = 42 + attempt * 1000 + target_rank
            U_try, V_try, W_try, resid = solve_als(
                target_rank, max_iters=300, tol=1e-9, seed=seed
            )
            
            if resid < 1e-6:
                # Valid decomposition found!
                best_result = {
                    "U": U_try.tolist(),
                    "V": V_try.tolist(),
                    "W": W_try.tolist()
                }
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we couldn't find at this rank, try slightly higher
            # but don't go above schoolbook
            if target_rank < best_rank:
                break
    
    # Final verification
    U_final = np.array(best_result["U"])
    V_final = np.array(best_result["V"])
    W_final = np.array(best_result["W"])
    
    if not verify(U_final, V_final, W_final, tol=1e-5):
        # Fallback to schoolbook
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return best_result
# EVOLVE-BLOCK-END
