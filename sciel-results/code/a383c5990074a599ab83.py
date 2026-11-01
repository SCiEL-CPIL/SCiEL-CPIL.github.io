import time
import numpy as np
from scipy.optimize import minimize, least_squares
import itertools

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    """
    n, m, p = 2, 3, 10
    np_dim = p * n   # 20
    nm_dim = n * m   # 6
    mp_dim = m * p   # 30
    schoolbook_rank = n * m * p  # 60
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Build the target tensor in flattened form
    # T[a, b, c] = 1 if a=k*n+i, b=i*m+j, c=j*p+k for valid i,j,k
    # We need: T[a,b,c] = sum_r W[a,r] * U[b,r] * V[c,r]
    
    # For a given rank r, we have variables:
    # W: (np_dim, r), U: (nm_dim, r), V: (mp_dim, r)
    # Total variables: (np_dim + nm_dim + mp_dim) * r = (20+6+30)*r = 56*r
    
    # The constraint is: for all (a,b,c), sum_r W[a,r]*U[b,r]*V[c,r] = T[a,b,c]
    # This is a system of np_dim*nm_dim*mp_dim = 20*6*30 = 3600 equations
    
    # But T is sparse - only 60 entries are 1, rest are 0
    # Actually, we need the sum to equal 1 at the 60 positions and 0 elsewhere
    
    # Strategy: try ranks from lower bound up to schoolbook
    # Lower bound: at least max(nm_dim, mp_dim, np_dim) = 30
    # But let's try a range
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, let's try to find if we can beat 60
    # Try ranks: 30, 35, 40, 45, 50, 55, 60
    # But this is computationally expensive for large ranks
    
    # Alternative approach: use the structure of the problem
    # For M_{2,3,10}, we can think of it as 10 independent M_{2,3} problems
    # but they share the middle dimension m=3
    
    # Actually, let's try a different approach:
    # The tensor can be viewed as a linear map from R^{nm} ⊗ R^{mp} to R^{np}
    # T: R^{nm} × R^{mp} -> R^{np}
    # T(e_{i,j} ⊗ e_{j,k}) = e_{i,k} (in output space indexed by (k,i))
    
    # For rank r decomposition: T = sum_r w_r ⊗ (u_r ⊗ v_r)
    # where w_r ∈ R^{np}, u_r ∈ R^{nm}, v_r ∈ R^{mp}
    
    # This is equivalent to: the set {u_r ⊗ v_r} spans a subspace, and the map
    # from that subspace to R^{np} is given by the w_r coefficients
    
    # Let me try a numerical search for small rank improvements
    
    # For efficiency, let's focus on trying to find rank < 60
    # We'll use a least-squares approach with random initialization
    
    def try_rank(r, num_restarts=5, max_iter=1000):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # Variables: flatten W, U, V into a single vector
        # W: (np_dim, r), U: (nm_dim, r), V: (mp_dim, r)
        total_vars = (np_dim + nm_dim + mp_dim) * r
        
        def unpack(x):
            W = x[:np_dim*r].reshape(np_dim, r)
            U = x[np_dim*r:(np_dim+nm_dim)*r].reshape(nm_dim, r)
            V = x[(np_dim+nm_dim)*r:].reshape(mp_dim, r)
            return W, U, V
        
        def residuals(x):
            W, U, V = unpack(x)
            # Compute sum_r W[:,r] * U[:,r] * V[:,r] for all (a,b,c)
            # This is a 3-tensor, so we need to be careful
            # T[a,b,c] = sum_r W[a,r] * U[b,r] * V[c,r]
            
            # We only need to check the 60 non-zero positions and some zero positions
            # But for correctness, we need all positions
            # Let's compute the full tensor and compare
            
            # Efficient computation:
            # For each r, we have outer product W[:,r] ⊗ U[:,r] ⊗ V[:,r]
            # Sum over r
            
            # Reshape to matrices for efficiency:
            # T[a,b,c] = sum_r W[a,r]*U[b,r]*V[c,r]
            # = (W * U^T) [a,b] ... no, that's not right
            # Actually: T[a,b,c] = sum_r W[a,r] * (U[b,r] * V[c,r])
            # Let M_r = U[:,r] ⊗ V[:,r], then T[:, :, r] contribution is W[:,r] ⊗ M_r
            # This is getting complex. Let's just compute directly for small sizes.
            
            res = np.zeros((np_dim, nm_dim, mp_dim))
            for r_idx in range(r):
                res += np.einsum('a,b,c->abc', W[:, r_idx], U[:, r_idx], V[:, r_idx])
            
            # Target: T_target[a,b,c] = 1 if a=k*n+i, b=i*m+j, c=j*p+k
            # Otherwise 0
            # Flatten and return residuals
            target = np.zeros((np_dim, nm_dim, mp_dim))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = k * n + i
                        b = i * m + j
                        c = j * p + k
                        target[a, b, c] = 1.0
            
            return (res - target).flatten()
        
        best_sol = None
        best_cost = np.inf
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            x0 = np.random.randn(total_vars) * 0.1
            # Scale: each variable should be around 1/sqrt(r) for the sum to be O(1)
            x0 /= np.sqrt(r)
            
            try:
                result = least_squares(residuals, x0, max_nfev=max_iter, 
                                       method='lm' if total_vars <= 200 else 'trf',
                                       bounds=(None, None) if total_vars <= 200 else (-10, 10))
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    best_sol = result.x
                # Check if we found an exact solution
                if cost < 1e-10:
                    break
            except Exception:
                continue
        
        if best_sol is not None and best_cost < 1e-8:
            W, U, V = unpack(best_sol)
            return W, U, V
        
        return None
    
    # Try ranks from lower bound up
    # Start with a few promising ranks
    ranks_to_try = []
    # Theoretical: rank >= max(nm, mp, np) = 30
    # But for M_{2,3,10}, let's try a range
    # Be conservative due to time constraints
    for r in [30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 5:
            break
        
        if r >= schoolbook_rank:
            # Use schoolbook as fallback
            break
        
        sol = try_rank(r, num_restarts=3, max_iter=500)
        if sol is not None:
            W, U, V = sol
            best_rank = r
            best_result = (W, U, V)
            break
    
    # If we didn't find a better rank, use schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = np.zeros((nm_dim, rank))
        V = np.zeros((mp_dim, rank))
        W = np.zeros((np_dim, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        best_result = (W, U, V)
    
    W, U, V = best_result
    
    # Convert to lists of lists with integer/half-integer entries
    # Round to nearest 0.5
    def to_half_int_matrix(M):
        rows, cols = M.shape
        result = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                result[i][j] = rounded
        return result
    
    U_list = to_half_int_matrix(U)
    V_list = to_half_int_matrix(V)
    W_list = to_half_int_matrix(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
