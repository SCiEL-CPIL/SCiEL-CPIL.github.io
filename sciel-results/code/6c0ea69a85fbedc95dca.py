import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Target rank 15
    target_rank = 15
    
    # Generate candidate patterns
    # Indices for U: 0..n*m-1, V: 0..m*p-1, W: 0..p*n-1
    # We generate patterns with support size <= 2
    # A pattern is (supp_u, supp_v, supp_w)
    
    def get_supports(dim, max_size):
        supports = []
        # Size 1
        for i in range(dim):
            supports.append((i,))
        # Size 2
        for i in range(dim):
            for j in range(i+1, dim):
                supports.append((i, j))
        return supports

    supp_u_opts = get_supports(n*m, 2)
    supp_v_opts = get_supports(m*p, 2)
    supp_w_opts = get_supports(p*n, 2)
    
    # Filter patterns to those that are "potentially useful"
    # A pattern is useful if it has non-zero overlap with T_MM
    # T_MM has 1s at (i,j,k) -> idx_u=i*m+j, idx_v=j*p+k, idx_w=k*n+i
    
    t_mm = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_mm[i*m+j, j*p+k, k*n+i] = 1.0

    candidates = []
    for su in supp_u_opts:
        for sv in supp_v_opts:
            for sw in supp_w_opts:
                # Construct the tensor for this pattern
                # It is 1 at all (a,b,c) in su x sv x sw
                # Check if it overlaps with T_MM
                overlap = 0
                for a in su:
                    for b in sv:
                        for c in sw:
                            if t_mm[a,b,c] != 0:
                                overlap += 1
                if overlap > 0:
                    # Store the pattern as a flattened vector
                    vec = np.zeros(n*m * m*p * p*n)
                    for a in su:
                        for b in sv:
                            for c in sw:
                                vec[a * m*p*p*n + b * p*n + c] = 1.0
                    candidates.append(vec)
    
    # Remove duplicates (vectors might be identical)
    # And remove zero vectors
    unique_candidates = []
    seen = set()
    for v in candidates:
        key = tuple(v)
        if key not in seen and np.any(v):
            seen.add(key)
            unique_candidates.append(v)
            
    candidates = np.array(unique_candidates)
    
    # Search for a set of 15 patterns
    # We will use a simple random search with a fixed seed
    np.random.seed(42)
    
    t_mm_flat = t_mm.flatten()
    
    best_solution = None
    num_trials = 1000
    
    for _ in range(num_trials):
        # Pick 15 random candidates
        if len(candidates) < target_rank:
            break
        idx = np.random.choice(len(candidates), target_rank, replace=False)
        M = candidates[idx].T # Shape (15, 324)
        
        # Solve M x = t_mm_flat
        # Using least squares
        x, residuals, rank, s = np.linalg.lstsq(M, t_mm_flat, rcond=None)
        
        # Check if the solution is exact
        if np.allclose(M @ x, t_mm_flat, atol=1e-9):
            # Check if x is close to integer
            if np.allclose(x, np.round(x), atol=1e-6):
                x_int = np.round(x).astype(int)
                # Verify exact integer solution
                if np.array_equal(M @ x_int, t_mm_flat):
                    # Construct U, V, W
                    U = np.zeros((n*m, target_rank))
                    V = np.zeros((m*p, target_rank))
                    W = np.zeros((p*n, target_rank))
                    
                    for r in range(target_rank):
                        # Reconstruct supports from the candidate vector?
                        # We lost the support info. We need to store it.
                        pass
                    # This approach is flawed because we don't know the supports.
                    # We need to store the supports along with the vectors.
                    
                    best_solution = (idx, x_int)
                    break
                    
    # If we found a solution, we need to reconstruct U, V, W
    # This requires knowing the supports.
    # Let's redo the storage to include supports.
    
    # Re-generate candidates with supports
    candidates_with_supports = []
    for su in supp_u_opts:
        for sv in supp_v_opts:
            for sw in supp_w_opts:
                overlap = 0
                for a in su:
                    for b in sv:
                        for c in sw:
                            if t_mm[a,b,c] != 0:
                                overlap += 1
                if overlap > 0:
                    vec = np.zeros(n*m * m*p * p*n)
                    for a in su:
                        for b in sv:
                            for c in sw:
                                vec[a * m*p*p*n + b * p*n + c] = 1.0
                    candidates_with_supports.append((vec, su, sv, sw))
    
    # Unique-ify
    unique_cws = []
    seen = set()
    for v, su, sv, sw in candidates_with_supports:
        key = (tuple(v), su, sv, sw)
        if key not in seen and np.any(v):
            seen.add(key)
            unique_cws.append((v, su, sv, sw))
            
    # Search
    found = False
    for _ in range(num_trials):
        if len(unique_cws) < target_rank:
            break
        idx = np.random.choice(len(unique_cws), target_rank, replace=False)
        
        M = np.array([unique_cws[i][0] for i in idx]).T
        x, _, _, _ = np.linalg.lstsq(M, t_mm_flat, rcond=None)
        
        if np.allclose(M @ x, t_mm_flat, atol=1e-9):
            if np.allclose(x, np.round(x), atol=1e-6):
                x_int = np.round(x).astype(int)
                if np.array_equal(M @ x_int, t_mm_flat):
                    # Build U, V, W
                    U = np.zeros((n*m, target_rank))
                    V = np.zeros((m*p, target_rank))
                    W = np.zeros((p*n, target_rank))
                    
                    for r, i in enumerate(idx):
                        _, su, sv, sw = unique_cws[i]
                        coeff = x_int[r]
                        for a in su:
                            U[a, r] = coeff
                        for b in sv:
                            V[b, r] = coeff
                        for c in sw:
                            W[c, r] = coeff
                            
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return fallback
# EVOLVE-BLOCK-END
