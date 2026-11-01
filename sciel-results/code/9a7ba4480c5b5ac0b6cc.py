"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Known rank-11 solution (Strassen + schoolbook for col 2)
    rank_known = 11
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    U_known[0][0] = 1; U_known[3][0] = 1
    V_known[0][0] = 1; V_known[4][0] = 1
    U_known[2][1] = 1; U_known[3][1] = 1
    V_known[0][1] = 1
    U_known[0][2] = 1
    V_known[1][2] = 1; V_known[4][2] = -1
    U_known[3][3] = 1
    V_known[3][3] = 1; V_known[0][3] = -1
    U_known[0][4] = 1; U_known[1][4] = 1
    V_known[4][4] = 1
    U_known[2][5] = 1; U_known[0][5] = -1
    V_known[0][5] = 1; V_known[1][5] = 1
    U_known[1][6] = 1; U_known[3][6] = -1
    V_known[3][6] = 1; V_known[4][6] = 1
    W_known[0][0] = 1; W_known[0][3] = 1; W_known[0][4] = -1; W_known[0][6] = 1
    W_known[2][2] = 1; W_known[2][4] = 1
    W_known[1][1] = 1; W_known[1][3] = 1
    W_known[3][0] = 1; W_known[3][1] = -1; W_known[3][2] = 1; W_known[3][5] = 1
    U_known[0][7] = 1
    V_known[2][7] = 1
    U_known[1][8] = 1
    V_known[5][8] = 1
    U_known[2][9] = 1
    V_known[2][9] = 1
    U_known[3][10] = 1
    V_known[5][10] = 1
    W_known[4][7] = 1; W_known[4][8] = 1
    W_known[5][9] = 1; W_known[5][10] = 1
    
    # Build the multiplication tensor as a 3-tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # We'll represent it as a list of (a,b,c) tuples with value 1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    num_entries = len(tensor_entries)
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition exactly reproduces the tensor."""
        for (a, b, c) in tensor_entries:
            s = 0.0
            for r in range(rank):
                s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def verify_decomposition_fast(U_arr, V_arr, W_arr, rank):
        """Fast verification using numpy."""
        # For each (a,b,c) entry, compute sum_r U[a,r]*V[b,r]*W[c,r]
        # We can do this with einsum
        # result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        for (a, b, c) in tensor_entries:
            if abs(result[a, b, c] - 1.0) > 1e-9:
                return False
        return True
    
    # Try to find a rank-10 decomposition
    # Strategy: use the known rank-11 solution and try to eliminate one term
    # by finding linear dependencies
    
    # Alternative: randomized search for rank-10
    # The tensor has 24 entries (2*2*3=12, but each appears once)
    # Actually n*m*p = 2*2*3 = 12 entries
    
    # Let's try a direct approach: assume rank R and set up the equations
    # For rank R, we need U (4xR), V (6xR), W (6xR)
    # 12 equations, 4R + 6R + 6R = 16R unknowns
    # For R=10: 160 unknowns, 12 equations -> very underdetermined
    # But the equations are nonlinear (products of three variables)
    
    # Let's try a different approach: start from rank 11 and try to merge terms
    # or find a valid rank-10 via numerical optimization
    
    # Method: Randomized initialization + iterative refinement
    def try_rank(R, max_iter=5000, seed=42):
        """Try to find a rank-R decomposition."""
        if time.time() > _deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U_arr = rng.randn(n*m, R) * 0.5
        V_arr = rng.randn(m*p, R) * 0.5
        W_arr = rng.randn(p*n, R) * 0.5
        
        # Use a simple gradient-free approach: 
        # For each tensor entry (a,b,c), we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # We can try to satisfy these one at a time
        
        # Actually, let's use a smarter approach:
        # Fix U and V, then W is determined linearly for each c
        # For a fixed c, the equation for each (a,b) with that c is:
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # This is linear in W[c,:]
        
        # But we need to be careful about which (a,b) pairs correspond to which c
        
        # Let's use a different strategy: 
        # Use the known rank-11 solution and try to find a linear combination
        # that eliminates one rank term
        
        # Rank-11 solution as arrays
        U11 = np.array(U_known, dtype=float)
        V11 = np.array(V_known, dtype=float)
        W11 = np.array(W_known, dtype=float)
        
        # Try to find vectors u, v, w such that for all (a,b,c):
        # sum_{r=0}^{10} U11[a,r]*V11[b,r]*W11[c,r] + u[a]*v[b]*w[c] = 1
        # This means the "residual" tensor R[a,b,c] = 1 - sum_{r=0}^{10} ... should be rank 1
        # But the rank-11 solution already gives exactly 1, so R = 0
        # We need to modify the solution to have rank 10
        
        # Better approach: try all possible rank-10 structures
        # by removing one column from U, V, W and adjusting
        
        # For each column r to remove, we need to redistribute its contribution
        # This is complex. Let's try a numerical optimization approach.
        
        # Use scipy minimize if available
        try:
            from scipy.optimize import minimize
        except ImportError:
            return None
        
        def objective(params):
            """Flatten U, V, W and compute error."""
            nU = n*m*R
            nV = m*p*R
            nW = p*n*R
            U_flat = params[:nU]
            V_flat = params[nU:nU+nV]
            W_flat = params[nU+nV:]
            
            U_m = U_flat.reshape(n*m, R)
            V_m = V_flat.reshape(m*p, R)
            W_m = W_flat.reshape(p*n, R)
            
            err = 0.0
            for (a, b, c) in tensor_entries:
                val = np.dot(U_m[a], V_m[b] * W_m[c])
                err += (val - 1.0) ** 2
            return err
        
        # Initial guess: use rank-11 solution truncated to R columns + random
        if R < 11:
            U_init = np.zeros((n*m, R))
            V_init = np.zeros((m*p, R))
            W_init = np.zeros((p*n, R))
            # Use first R columns of the rank-11 solution
            for r in range(R):
                U_init[:, r] = U11[:, r]
                V_init[:, r] = V11[:, r]
                W_init[:, r] = W11[:, r]
            # Add small random perturbation
            U_init += rng.randn(n*m, R) * 0.1
            V_init += rng.randn(m*p, R) * 0.1
            W_init += rng.randn(p*n, R) * 0.1
        else:
            U_init = rng.randn(n*m, R) * 0.5
            V_init = rng.randn(m*p, R) * 0.5
            W_init = rng.randn(p*n, R) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try multiple random restarts
        best_result = None
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > _deadline:
                break
            
            if restart == 0:
                p0 = params_init
            else:
                p0 = rng.randn(len(params_init)) * 0.5
            
            try:
                res = minimize(objective, p0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < 1e-10:
                    U_m = res.x[:nU].reshape(n*m, R)
                    V_m = res.x[nU:nU+nV].reshape(m*p, R)
                    W_m = res.x[nU+nV:].reshape(p*n, R)
                    # Convert to list of lists with rounded values
                    U_list = [[round(U_m[i][r], 4) for r in range(R)] for i in range(n*m)]
                    V_list = [[round(V_m[i][r], 4) for r in range(R)] for i in range(m*p)]
                    W_list = [[round(W_m[i][r], 4) for r in range(R)] for i in range(p*n)]
                    if verify_decomposition(U_list, V_list, W_list, R):
                        return (U_list, V_list, W_list)
            except Exception:
                pass
        
        return None
    
    # Try rank 10
    result = try_rank(10)
    if result is not None:
        U_list, V_list, W_list = result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to rank 11
    return {"U": U_known, "V": V_known, "W": W_known}
# EVOLVE-BLOCK-END
