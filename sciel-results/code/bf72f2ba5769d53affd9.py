import numpy as np
from scipy.optimize import least_squares
import time
from itertools import product
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Build target tensor indices
    targets_alpha = []
    targets_beta = []
    targets_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_alpha.append(a)
                targets_beta.append(b)
                targets_gamma.append(c)
                
    targets_alpha = np.array(targets_alpha, dtype=int)
    targets_beta = np.array(targets_beta, dtype=int)
    targets_gamma = np.array(targets_gamma, dtype=int)
    
    num_targets = len(targets_alpha)
    
    def verify_solution(U, V, W):
        """Verify if the solution is exact (within tolerance)"""
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_alpha, targets_beta, targets_gamma] = 1.0
        max_err = np.max(np.abs(T_full - T_target))
        return max_err < 1e-9, max_err

    def solve_with_fixed_W(W_fixed, R):
        """
        Given a fixed W matrix (size_W x R), solve for U and V.
        T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        For each c: T[:,:,c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
        This is linear in the entries of U and V if we treat them as unknowns.
        Actually, it's bilinear. But we can fix U and solve for V, or vice versa.
        
        Better approach: For fixed W, the constraint becomes:
        For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        This is a system of equations in the 6*R + 15*R variables.
        It's quadratic (bilinear). We can linearize by fixing one set.
        
        Alternative: If W is fixed, then for each pair (b,c), we have:
        sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c] for all a
        
        Let's define M_b = V[b,:] (length R) and W_c = W[c,:] (length R)
        Then the constraint is: for each (a,b,c), sum_r U[a,r] * M_b[r] * W_c[r] = T[a,b,c]
        
        This can be written as: U * (M_b ⊙ W_c)^T = T[:,b,c] where ⊙ is element-wise product
        
        Actually, let me think differently. Fix W. Then the tensor equation is:
        T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        For a fixed (b,c), this is: T[:,b,c] = U @ (V[b,:] * W[c,:])
        where V[b,:] * W[c,:] is element-wise product of two length-R vectors.
        
        So if we define for each (b,c) the vector z_{b,c} = V[b,:] ⊙ W[c,:], then
        T[:,b,c] = U @ z_{b,c}
        
        This means U @ [z_{b,c} for all b,c] = [T[:,b,c] for all b,c]
        But z_{b,c} depends on V, so it's still nonlinear.
        
        Let's try a different approach: enumerate sparse patterns for W.
        """
        pass

    def try_structured_rank(R, time_limit=30):
        """
        Try to find a rank-R decomposition by fixing W to have a structured form.
        
        Key insight: If we fix W to be a matrix where each column has exactly one
        non-zero entry, then the problem becomes much simpler.
        
        For W with column r having non-zero entry at position c_r with value w_r:
        T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        Only the term where c_r = c contributes to T[a,b,c]
        
        So for each c, let R_c = {r : c_r = c}
        Then T[a,b,c] = sum_{r in R_c} U[a,r]*V[b,r]*w_r
        
        This means T[:,:,c] = sum_{r in R_c} w_r * (U[:,r] ⊗ V[:,r])
        
        If |R_c| = 1 for all c, then T[:,:,c] = w_{r_c} * (U[:,r_c] ⊗ V[:,r_c])
        which means each T[:,:,c] must be rank 1.
        
        If |R_c| = 2, then T[:,:,c] is a sum of 2 rank-1 matrices.
        
        Let's enumerate assignments of columns to c-values.
        """
        start_time = time.time()
        
        # Try different patterns for W
        # Pattern 1: Each column of W has exactly one non-zero entry
        # This means R >= size_W = 10 (need at least 10 columns, one per c)
        # Actually, multiple columns can map to the same c
        
        # For each c in 0..p*n-1, assign a subset of columns
        # Let's try: each c gets exactly ceil(R / (p*n)) columns
        
        # Simpler: try all ways to assign R columns to p*n positions
        # This is (p*n)^R which is too large
        
        # Instead, use a greedy approach:
        # For each c, find the minimum number of rank-1 matrices needed to span T[:,:,c]
        
        # Compute the "rank" of each slice T[:,:,c]
        T = np.zeros((size_U, size_V, size_W))
        T[targets_alpha, targets_beta, targets_gamma] = 1.0
        
        # For each c, T[:,:,c] is a 6x15 matrix
        # Its rank tells us the minimum number of rank-1 terms needed
        
        slice_ranks = []
        for c in range(size_W):
            slice_ranks.append(np.linalg.matrix_rank(T[:,:,c]))
        
        # The total rank needed is at least sum of slice_ranks if columns are disjoint
        # But columns can be shared across c values
        
        # Let's try: assign columns to c values to minimize total rank
        # This is a combinatorial optimization
        
        # Heuristic: try to use as few columns as possible per c
        # Start with R = max(slice_ranks) and increase
        
        for R_try in range(max(slice_ranks), R + 1):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find an assignment of R_try columns to c values
            # such that each c gets at least slice_ranks[c] columns
            
            # Simple greedy: assign columns round-robin
            cols_per_c = [0] * size_W
            for r in range(R_try):
                # Assign column r to the c with the highest remaining need
                need = [slice_ranks[c] - cols_per_c[c] for c in range(size_W)]
                c_best = np.argmax(need)
                if need[c_best] > 0:
                    cols_per_c[c_best] += 1
                else:
                    cols_per_c[r % size_W] += 1
            
            # Now try to solve for U, V, W with this column assignment
            # W[c, r] is non-zero only if r is assigned to c
            
            # Build the linear system
            # Unknowns: U (6 x R_try), V (15 x R_try), and the non-zero entries of W
            # For each (a,b,c): sum_{r assigned to c} U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # This is still nonlinear. Let's fix W values to 1 and solve for U, V
            # W[c,r] = 1 if r is assigned to c, else 0
            
            # Let me redo with explicit column-to-c assignment
            col_to_c = []
            for c in range(size_W):
                for _ in range(cols_per_c[c]):
                    col_to_c.append(c)
            
            # Pad if needed
            while len(col_to_c) < R_try:
                col_to_c.append(0)
            
            W_fixed = np.zeros((size_W, R_try))
            for r in range(R_try):
                W_fixed[col_to_c[r], r] = 1.0
            
            # Now solve: T[a,b,c] = sum_{r: col_to_c[r]=c} U[a,r]*V[b,r]
            # For each c: T[:,:,c] = sum_{r in R_c} U[:,r] ⊗ V[:,r]
            
            # This is a linear system in the variables U[a,r]*V[b,r] for r in R_c
            # But U and V are shared across different c values
            
            # Let's set up the system:
            # Variables: U[0..5, 0..R_try-1] and V[0..14, 0..R_try-1]
            # Equations: for each (a,b,c), sum_{r in R_c} U[a,r]*V[b,r] = T[a,b,c]
            
            # This is bilinear. To linearize, fix U and solve for V, or vice versa.
            
            # Try: fix U to be sparse (one non-zero per column) and solve for V
            
            # For each column r, U[:,r] has one non-zero entry at position a_r with value 1
            # Then the equation becomes: for each (a,b,c) with a = a_r and r in R_c:
            # V[b,r] = T[a,b,c]
            
            # This is over-determined in general. Let's try a different approach.
            
            # Use least squares to solve the bilinear system
            num_vars_U = size_U * R_try
            num_vars_V = size_V * R_try
            num_vars = num_vars_U + num_vars_V
            
            def residuals(x):
                U = x[:num_vars_U].reshape(size_U, R_try)
                V = x[num_vars_U:].reshape(size_V, R_try)
                T_full = np.einsum('ar,br,cr->abc', U, V, W_fixed)
                return (T_full - T).flatten()
            
            # Initialize
            np.random.seed(42)
            x0 = np.random.randn(num_vars) * 0.1
            
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if res.cost < 1e-10:
                    U = res.x[:num_vars_U].reshape(size_U, R_try)
                    V = res.x[num_vars_U:].reshape(size_V, R_try)
                    
                    # Snap to half-integers
                    U_snapped = np.round(U * 2) / 2.0
                    V_snapped = np.round(V * 2) / 2.0
                    W_snapped = np.round(W_fixed * 2) / 2.0
                    
                    is_valid, max_err = verify_solution(U_snapped, V_snapped, W_snapped)
                    if is_valid:
                        return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                    
                    is_valid, max_err = verify_solution(U, V, W_fixed)
                    if is_valid:
                        return U.tolist(), V.tolist(), W_fixed.tolist()
                        
            except Exception:
                pass
        
        return None

    def try_rank_R(R, num_restarts=5, max_nfev=300):
        """Try to find a decomposition of rank R using optimization"""
        num_vars = (size_U + size_V + size_W) * R
        
        T = np.zeros((size_U, size_V, size_W))
        T[targets_alpha, targets_beta, targets_gamma] = 1.0
        
        def residuals_from_flat(x):
            U = x[:size_U*R].reshape(size_U, R)
            V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
            W = x[(size_U+size_V)*R:].reshape(size_W, R)
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_full - T).flatten()
        
        for restart in range(num_restarts):
            if restart == 0:
                x0 = np.random.randint(-2, 3, num_vars).astype(float)
            elif restart == 1:
                x0 = np.random.uniform(-1, 1, num_vars)
            else:
                scale = 0.5 + restart * 0.1
                x0 = np.random.uniform(-scale, scale, num_vars)
            
            try:
                res = least_squares(residuals_from_flat, x0, method='trf', max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                
                if res.cost < 1e-8:
                    x_sol = res.x
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
        
        return None

    start_time = time.time()
    deadline = 110  # Leave 10s safety margin
    
    # Compute lower bound from unfoldings
    T = np.zeros((size_U, size_V, size_W))
    T[targets_alpha, targets_beta, targets_gamma] = 1.0
    T_flat = T.reshape(size_U * size_V, size_W)
    rank_T = np.linalg.matrix_rank(T_flat)
    T_flat2 = T.reshape(size_U, size_V * size_W)
    rank_T2 = np.linalg.matrix_rank(T_flat2)
    T_flat3 = T.reshape(size_U * size_W, size_V)
    rank_T3 = np.linalg.matrix_rank(T_flat3)
    lower_bound = max(rank_T, rank_T2, rank_T3)
    
    # Try ranks from lower_bound to 30
    for R in range(max(10, lower_bound), 31):
        if time.time() > deadline:
            break
        
        # Try structured approach first
        result = try_structured_rank(R, time_limit=15)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            break
        
        # Try optimization-based approach
        result = try_rank_R(R, num_restarts=5, max_nfev=200)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
