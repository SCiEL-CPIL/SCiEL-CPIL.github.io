import time
import numpy as np
import sys

def construct():
    """
    Finds the rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x5 matrix.
    
    Strategy:
    1. The tensor decomposition of rank R corresponds to R rank-1 tensors summing to the multiplication tensor.
    2. We use a "Linearized" approach: 
       - Fix U and V to random half-integer matrices of shape (dim_u, R) and (dim_v, R).
       - The tensor equation T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] is LINEAR in the entries of W.
       - We construct the linear system A * W_flat = T_flat and solve for W.
       - If the solution W exists, is unique (or we pick a solution), and has half-integer entries, 
         we have found a valid decomposition of rank R.
    3. We iterate R from a lower bound (15) upwards.
    4. We use multiple random restarts for U and V to increase the chance of finding a solvable 
       system with half-integer W.
    5. Fallback to Schoolbook (Rank 30) if no lower rank is found within time limits.
    """
    
    n, m, p = 2, 3, 5
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Precompute the target tensor T[a, b, c]
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    def is_half_integer(arr, tol=1e-6):
        """Check if all entries in arr are close to half-integers (k/2)."""
        doubled = 2.0 * arr
        rounded = np.round(doubled)
        return np.all(np.abs(doubled - rounded) < tol)
        
    def round_to_half_integer(arr):
        """Round entries to the nearest half-integer."""
        return np.round(2.0 * arr) / 2.0

    def try_rank_linear(R, num_restarts=100):
        """
        Attempts to find a decomposition of rank R by fixing U, V randomly 
        and solving the linear system for W.
        """
        # We need to build the matrix A such that A * W_flat = T_flat
        # W_flat has length dim_w * R
        # Equation: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This can be written as: 
        # For each (a,b,c), sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
        
        # Let's structure the linear system.
        # Unknowns: W[0,0], W[0,1], ..., W[0,R-1], W[1,0], ..., W[dim_w-1, R-1]
        # Total unknowns: dim_w * R
        
        # We can build A of shape (dim_u * dim_v * dim_w, dim_w * R)
        # But 900 x (10*R) is small.
        
        # A[i, j] where i is index of (a,b,c) and j is index of (c, r)
        # i = (a * dim_v + b) * dim_w + c
        # j = c * R + r
        # A[i, j] = U[a, r] * V[b, r] if the c matches, else 0.
        
        # Actually, for a fixed (a,b), the equation involves all c? No.
        # The equation is for each (a,b,c).
        # For a specific (a,b,c), the unknowns involved are W[c, 0], ..., W[c, R-1].
        # So the system is block-diagonal with respect to c?
        # NO. The sum is over r. The term is U[a,r] V[b,r] W[c,r].
        # For a fixed c, the variables are W[c, r] for r=0..R-1.
        # The equations for different c are independent?
        # Let's check: T[a,b,c] depends on c. The RHS is T[a,b,c].
        # The LHS for a specific (a,b,c) only involves W[c, :].
        # It does NOT involve W[c', :] for c' != c.
        # Therefore, the system DECOUPLES into dim_w independent systems!
        
        # For each c in 0..dim_w-1:
        #   We have dim_u * dim_v equations (for all a,b).
        #   We have R unknowns (W[c, 0], ..., W[c, R-1]).
        #   The coefficient matrix M_c has shape (dim_u * dim_v, R).
        #   M_c[a*dim_v + b, r] = U[a, r] * V[b, r]
        #   RHS vector t_c has shape (dim_u * dim_v).
        #   t_c[a*dim_v + b] = T[a, b, c]
        
        # This is MUCH faster. We solve dim_w (10) small linear systems of size (90, R).
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Random half-integer initialization for U and V
            # Values in {-1, -0.5, 0, 0.5, 1}
            # To ensure full rank potential, avoid all zeros.
            np.random.seed(12345 + restart * 7 + R)
            
            # Generate random half-integers
            def rand_half_int(shape):
                # 2 * rand_int / 2
                # Use integers from -2 to 2 (i.e., -1 to 1 in half-ints)
                # Or -1 to 1? Let's use -1 to 1 for simplicity.
                vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=shape)
                return vals
            
            U = rand_half_int((dim_u, R))
            V = rand_half_int((dim_v, R))
            
            # Check if U and V are "generic" enough (rank R)
            # If rank(U) < R or rank(V) < R, we might not be able to solve.
            # But we can still try.
            
            W = np.zeros((dim_w, R))
            success = True
            
            for c in range(dim_w):
                # Build M_c: (dim_u * dim_v, R)
                # M_c[idx, r] = U[a, r] * V[b, r] where idx = a*dim_v + b
                M_c = np.zeros((dim_u * dim_v, R))
                t_c = np.zeros(dim_u * dim_v)
                
                for a in range(dim_u):
                    for b in range(dim_v):
                        idx = a * dim_v + b
                        t_c[idx] = T[a, b, c]
                        # Vector of U[a, :] * V[b, :]
                        M_c[idx, :] = U[a, :] * V[b, :]
                
                # Solve M_c * W_c = t_c
                # W_c is a vector of length R
                try:
                    # Use least squares if overdetermined or singular
                    # Since we want exact solution, we check residual.
                    W_c, residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                    
                    # Check if the solution is exact (within tolerance)
                    # Residual should be close to 0
                    if len(residuals) > 0 and residuals[0] > 1e-8:
                        # Not an exact solution for this c
                        success = False
                        break
                    
                    W[c, :] = W_c
                    
                except np.linalg.LinAlgError:
                    success = False
                    break
            
            if success:
                # Check if W has half-integer entries
                if is_half_integer(W):
                    # Round to exact half-integers to remove floating point noise
                    W = round_to_half_integer(W)
                    
                    # Final Verification
                    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T_calc, T, atol=1e-9):
                        return U, V, W
        
        return None

    # Fallback to schoolbook
    def _schoolbook_decomp(n, m, p):
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
        return U, V, W

    best_U, best_V, best_W = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p

    # Search for lower rank
    # Lower bound is 15. 
    # Let's try from 15 to 29.
    for R in range(15, 30):
        if time.time() > deadline - 10:
            break
            
        # Increase restarts for lower ranks as they are harder?
        # Or keep constant.
        res = try_rank_linear(R, num_restarts=50)
        
        if res is not None:
            U, V, W = res
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            best_rank = R
            break # Found minimal rank in this sequence
            
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
