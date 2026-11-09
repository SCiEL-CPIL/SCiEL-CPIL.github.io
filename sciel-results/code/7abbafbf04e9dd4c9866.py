import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor: for each (i,j,k), T[i,j,k] = 1
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,k] * (sum over j of A_ij * B_jk)
    # Actually the tensor is: T_{i,j,k} = 1 for all i,j,k (it's the identity in the middle index)
    # We need: for all i,k: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if we're computing (AB)_{i,k}
    # Wait, let me reconsider. The multiplication tensor T has entries T[i,j,k] = 1.
    # A rank-r decomposition means: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # The constraint is: for each (i,j,k) in [0,n)x[0,m)x[0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 24 polynomial equations in r* (nm+mp+pn) variables.
    
    # Schoolbook solution: rank = 24
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find lower rank
    # For each candidate rank r, we need to solve:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # 
    # Strategy: Fix U and V randomly (integer/half-integer), then solve for W linearly.
    # For fixed U and V, the equations become linear in W:
    # For each (i,j,k): sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is a linear system in W variables.
    
    # Let's try: for each r from best_rank-1 down to some minimum,
    # try random U, V and solve for W.
    
    def try_rank(r, max_attempts=200):
        """Try to find a rank-r decomposition by fixing U,V and solving for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random integer/half-integer U and V
            # Use values from {-1, 0, 0.5, 1, 1.5, 2} for variety
            vals = [-1, 0, 0.5, 1, 1.5, 2]
            
            U = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(nm)], dtype=float)
            V = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(mp)], dtype=float)
            
            # Build linear system for W
            # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Total unknowns: p*n*r = 8*r
            # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Number of equations: n*m*p = 24
            
            num_unknowns = pn * r
            num_eqs = n * m * p
            
            # Build A matrix (num_eqs x num_unknowns) and b vector (num_eqs,)
            A = np.zeros((num_eqs, num_unknowns))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            # W unknown index: (k*n+i)*r + c
                            w_idx = (k*n+i)*r + c
                            A[eq_idx, w_idx] += coeff
                        eq_idx += 1
            
            # Solve A * W_flat = b
            # Use least squares or check if exact solution exists
            try:
                # First check if the system is consistent
                # Use np.linalg.lstsq
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact (residuals close to 0)
                if len(residuals) == 0 or np.max(np.abs(residuals)) < 1e-9:
                    # Check if W entries are integer or half-integer
                    W_mat = W_flat.reshape(pn, r)
                    # Check if all entries are close to integer or half-integer
                    is_valid = True
                    for val in W_mat:
                        nearest = round(val * 2) / 2
                        if abs(val - nearest) > 1e-7:
                            is_valid = False
                            break
                    if is_valid:
                        # Round to half-integers
                        W_mat = np.round(W_mat * 2) / 2
                        U_mat = np.round(U * 2) / 2
                        V_mat = np.round(V * 2) / 2
                        
                        # Verify the solution
                        if _verify(U_mat, V_mat, W_mat, n, m, p):
                            return U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), r
            except:
                pass
        
        return None
    
    def _verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try ranks from 23 down
    for r in range(23, 15, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_attempts=50)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break  # Found lower rank, return it
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
