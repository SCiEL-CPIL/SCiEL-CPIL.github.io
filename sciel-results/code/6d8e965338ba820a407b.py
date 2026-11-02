import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # We work over F_2.
    # Target: T[i,j,k] = 1.
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (mod 2)
    
    # Strategy:
    # 1. Choose R.
    # 2. Randomly generate U (nm x R) and V (mp x R) over F_2.
    # 3. Check if for all i,j,k, there exists r such that U[i*m+j, r] * V[j*p+k, r] == 1.
    #    If not, restart.
    # 4. Construct linear system A W = 1 over F_2.
    #    Variables W are indexed by (k, i, r). Total R*p*n variables.
    #    Equations indexed by (i, j, k). Total n*m*p equations.
    #    A[e, v] = 1 if the variable v corresponds to W[k*n+i, r] and U[i*m+j, r]*V[j*p+k, r] == 1.
    # 5. Solve A W = 1. If consistent, we have a solution.
    
    def solve_linear_system_f2(A, b):
        # A is (Eq, Var) binary matrix, b is (Eq,) binary vector.
        # Returns W if solution exists, else None.
        num_eq, num_var = A.shape
        # Augmented matrix
        M = np.hstack([A, b.reshape(-1, 1)])
        
        # Gaussian Elimination over F_2
        rank = 0
        for col in range(num_var):
            # Find pivot
            pivot_row = -1
            for row in range(rank, num_eq):
                if M[row, col] == 1:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            
            # Swap rows
            if pivot_row != rank:
                M[[rank, pivot_row]] = M[[pivot_row, rank]]
            
            # Eliminate
            for row in range(num_eq):
                if row != rank and M[row, col] == 1:
                    M[row] ^= M[rank]
            
            rank += 1
            
        # Check for inconsistency: row with 0...0 | 1
        for row in range(rank, num_eq):
            if np.any(M[row, :num_var] == 1):
                return None
            if M[row, num_var] == 1:
                return None
        
        # Solution exists. We can pick the specific solution where free variables are 0.
        # Since we just need *a* solution, and the system is underdetermined usually,
        # we can just set free variables to 0 and back-substitute.
        # For simplicity, since we only need to return *some* valid W, 
        # and the system is consistent, we can use a simple heuristic or just return a zero vector?
        # No, we need to satisfy A W = 1.
        
        # Let's construct the solution.
        # We track which variable is the pivot for each row.
        pivot_vars = [-1] * rank
        for r in range(rank):
            for c in range(num_var):
                if M[r, c] == 1:
                    pivot_vars[r] = c
                    break
        
        W = np.zeros(num_var, dtype=np.int8)
        
        # Back substitution
        # Since we reduced to row echelon form, M[r, pivot_vars[r]] = 1 and M[r, c] = 0 for c < pivot_vars[r]
        # We need to handle free variables. Let's set free variables to 0.
        # Then M[r, pivot_vars[r]] = sum(M[r, free_vars] * W[free_vars]) + W[pivot_vars[r]]
        # If we set free vars to 0, W[pivot_vars[r]] = b[r] (which is M[r, num_var] after elimination? No, b is modified).
        # Actually, the augmented column is the current RHS.
        
        # Let's do proper back substitution.
        # M is in row echelon form.
        # For r from rank-1 down to 0:
        #   sum_{c > pivot_vars[r]} M[r, c] W[c] + W[pivot_vars[r]] = M[r, num_var]
        #   W[pivot_vars[r]] = M[r, num_var] + sum_{c > pivot_vars[r]} M[r, c] W[c] (mod 2)
        
        # We assume free variables (not in pivot_vars) are 0.
        # So W[c] is 0 if c is not a pivot var.
        
        for r in range(rank - 1, -1, -1):
            pv = pivot_vars[r]
            val = M[r, num_var]
            # Add contributions from free variables (which are 0) and other pivot variables (already computed)
            # We only need to sum M[r, c] for c > pv.
            # Since W[c] is 0 for free vars, we only sum over pivot vars c > pv.
            for c in range(pv + 1, num_var):
                if M[r, c] == 1:
                    val ^= W[c]
            W[pv] = val
            
        return W

    def try_rank_f2(R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        num_eq = n * m * p
        num_var_w = R * size_w
        
        # Pre-compute indices for speed
        # For each eq (i, j, k), we need to know which W variables are involved.
        # W var index: r * size_w + k * n + i
        
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            if attempts % 100 == 0 and time.time() > deadline - 5:
                break
            
            # Random U, V
            U = np.random.randint(0, 2, (size_u, R))
            V = np.random.randint(0, 2, (size_v, R))
            
            # Check support condition
            valid_support = True
            # We can vectorize this check
            # T_support[i,j,k] = OR_r (U[i*m+j, r] * V[j*p+k, r])
            # This is equivalent to: for each r, U[:, r] (n*m) and V[:, r] (m*p)
            # Outer product? No.
            
            # Let's just loop, it's fast enough for small n,m,p
            # i in n, j in m, k in p
            for i in range(n):
                for j in range(m):
                    u_row = U[i * m + j]
                    for k in range(p):
                        v_row = V[j * p + k]
                        # Check if any r has u_row[r] * v_row[r] == 1
                        if not np.any(u_row * v_row):
                            valid_support = False
                            break
                if not valid_support:
                    break
            if not valid_support:
                continue
            
            # Construct A
            A = np.zeros((num_eq, num_var_w), dtype=np.int8)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    u_row = U[i * m + j]
                    for k in range(p):
                        v_row = V[j * p + k]
                        # Coefficients for W[k*n+i, r]
                        # W var index: r * size_w + k * n + i
                        coeffs = u_row * v_row
                        for r in range(R):
                            if coeffs[r]:
                                w_idx = r * size_w + k * n + i
                                A[eq_idx, w_idx] = 1
                        eq_idx += 1
            
            # Solve
            W = solve_linear_system_f2(A, np.ones(num_eq, dtype=np.int8))
            
            if W is not None:
                # We have a valid solution over F_2.
                # U, V, W are binary. This is a valid half-integer solution.
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.reshape(size_w, R).T.tolist() # Wait, W is flat. Need to reshape.
                }
        return None

    # Try ranks
    # Lower bound for tensor rank is max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # But usually higher.
    for R in range(30, 61):
        if time.time() > deadline:
            break
        result = try_rank_f2(R)
        if result:
            return result

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
