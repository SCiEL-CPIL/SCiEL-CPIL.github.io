"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    Uses a computational search approach: fix a sparse pattern for U and V,
    then solve for W linearly. Try multiple patterns to minimize rank.
    """
    import numpy as np
    from itertools import product
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # The correct condition:
    # C[i,k] = sum_r W[k*n+i, r] * X_r * Y_r
    # where X_r = sum_{a,b} U[a*m+b, r] * A[a,b]
    #       Y_r = sum_{c,d} V[c*p+d, r] * B[c,d]
    #
    # Coefficient of A[a,b]*B[c,d] in C[i,k] is:
    #   sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r]
    # This must equal 1 if a==i, b==c, d==k, else 0.
    #
    # Key insight: if we fix U and V, then for each (i,k) pair, the equation
    #   sum_r W[k*n+i, r] * (U[:,r] * V[:,r])_{(a,b,c,d)} = target
    # is linear in W. Specifically, define M_r = outer(U[:,r], V[:,r]) reshaped.
    # Then for each (i,k): sum_r W[k*n+i, r] * M_r[(i,j,j,k) part] = 1 for j=0..m-1
    # and = 0 for all other (a,b,c,d).
    
    # Strategy: Try rank from 12 upward. For each rank, try random/sparse patterns
    # for U and V, then solve the linear system for W.
    
    def check_solution(U, V, W):
        """Verify the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # For each (i,k), check that coefficient of A[a,b]*B[c,d] is correct
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                coeff = sum(W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] for r in range(rank))
                                target = 1.0 if (a == i and b == c and d == k) else 0.0
                                if abs(coeff - target) > 1e-9:
                                    return False
        return True
    
    def try_rank(rank, seed=0):
        """Try to find a valid decomposition of given rank."""
        np.random.seed(seed)
        for attempt in range(50):
            # Create sparse U and V patterns
            # U: (nm, rank), V: (mp, rank)
            # Each column of U should have a few non-zeros
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            
            for r in range(rank):
                # Pick 1-3 non-zero entries in U column r
                num_u = 1 + (attempt % 2)  # 1 or 2
                u_rows = np.random.choice(nm, size=min(num_u, nm), replace=False)
                U[u_rows, r] = 1.0
                
                # Pick 1-3 non-zero entries in V column r
                num_v = 1 + ((attempt + 1) % 2)  # 1 or 2
                v_rows = np.random.choice(mp, size=min(num_v, mp), replace=False)
                V[v_rows, r] = 1.0
            
            # Now solve for W: for each (i,k), we need
            # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = target(a,b,c,d; i,k)
            # This is a linear system in W.
            # Number of unknowns: rank * pn (W has pn*rank entries)
            # Number of equations: for each (i,k,a,b,c,d): pn * nm * mp = 8*6*12 = 576
            # But many are redundant. Let's set up the system properly.
            
            # For each (i,k), the condition is:
            # For all (a,b,c,d): sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = [a==i, b==c, d==k]
            
            # Let's build the linear system. Unknowns: W[k*n+i, r] for all (k,i,r).
            # Total unknowns: pn * rank = 8 * rank
            
            # Equations: for each (i,k,a,b,c,d) where the target is non-zero or we need zero.
            # Actually, we need ALL 576 equations to be satisfied.
            
            # Let's build the matrix. Each equation is indexed by (i,k,a,b,c,d).
            # Variable is indexed by (k,i,r) i.e. (k*n+i, r).
            
            num_vars = pn * rank
            num_eqs = n * p * nm * mp  # 8 * 6 * 12 = 576
            
            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for a in range(n):
                        for b in range(m):
                            for c in range(m):
                                for d in range(p):
                                    target = 1.0 if (a == i and b == c and d == k) else 0.0
                                    b_vec[eq_idx] = target
                                    # Coefficient of W[k*n+i, r] is U[a*m+b, r] * V[c*p+d, r]
                                    for r in range(rank):
                                        var_idx = (k * n + i) * rank + r
                                        A_mat[eq_idx, var_idx] = U[a*m+b, r] * V[c*p+d, r]
                                    eq_idx += 1
            
            # Solve the linear system
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if solution is valid
            W_sol = W_flat.reshape(pn, rank)
            
            # Verify
            valid = True
            for i in range(n):
                for k in range(p):
                    for a in range(n):
                        for b in range(m):
                            for c in range(m):
                                for d in range(p):
                                    coeff = sum(W_sol[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] for r in range(rank))
                                    target = 1.0 if (a == i and b == c and d == k) else 0.0
                                    if abs(coeff - target) > 1e-8:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format with integer/half-integer entries
                U_list = [[int(round(U[a, r] * 2)) / 2 for r in range(rank)] for a in range(nm)]
                V_list = [[int(round(V[a, r] * 2)) / 2 for r in range(rank)] for a in range(mp)]
                W_list = [[int(round(W_sol[a, r] * 2)) / 2 for r in range(rank)] for a in range(pn)]
                
                # Final verification with exact arithmetic
                U_arr = np.array(U_list, dtype=float)
                V_arr = np.array(V_list, dtype=float)
                W_arr = np.array(W_list, dtype=float)
                
                for i in range(n):
                    for k in range(p):
                        for a in range(n):
                            for b in range(m):
                                for c in range(m):
                                    for d in range(p):
                                        coeff = sum(W_arr[k*n+i, r] * U_arr[a*m+b, r] * V_arr[c*p+d, r] for r in range(rank))
                                        target = 1.0 if (a == i and b == c and d == k) else 0.0
                                        if abs(coeff - target) > 1e-9:
                                            return None
                
                return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 12 upward
    for rank in range(12, 25):
        for seed in range(20):
            result = try_rank(rank, seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
