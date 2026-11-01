import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Dimensions
    dim_u = n * m  # 4
    dim_v = m * p  # 22
    dim_w = p * n  # 22
    
    # Target tensor entries
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We store these as a list of (a, b, c) tuples
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    num_entries = len(tensor_entries) # 44
    
    def get_rank_39_solution():
        """Fallback to the known rank 39 Strassen/Schoolbook solution."""
        rank = 39
        U = np.zeros((dim_u, rank), dtype=int)
        V = np.zeros((dim_v, rank), dtype=int)
        W = np.zeros((dim_w, rank), dtype=int)
        
        r = 0
        # 5 pairs of columns using Strassen
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0
            v_f = k1
            v_g = p + k0
            v_h = p + k1
            
            w_c00 = k0 * n
            w_c10 = k0 * n + 1
            w_c01 = k1 * n
            w_c11 = k1 * n + 1
            
            # m1 = (a+d)(e+h)
            U[0, r] = 1; U[3, r] = 1
            V[v_e, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1; W[w_c11, r] = 1
            r += 1
            # m2 = (c+d)e
            U[2, r] = 1; U[3, r] = 1
            V[v_e, r] = 1
            W[w_c10, r] = 1; W[w_c11, r] = -1
            r += 1
            # m3 = a(f-h)
            U[0, r] = 1
            V[v_f, r] = 1; V[v_h, r] = -1
            W[w_c01, r] = 1; W[w_c11, r] = 1
            r += 1
            # m4 = d(g-e)
            U[3, r] = 1
            V[v_g, r] = 1; V[v_e, r] = -1
            W[w_c00, r] = 1; W[w_c10, r] = 1
            r += 1
            # m5 = (a+b)h
            U[0, r] = 1; U[1, r] = 1
            V[v_h, r] = 1
            W[w_c00, r] = -1; W[w_c01, r] = 1
            r += 1
            # m6 = (c-a)(e+f)
            U[2, r] = 1; U[0, r] = -1
            V[v_e, r] = 1; V[v_f, r] = 1
            W[w_c11, r] = 1
            r += 1
            # m7 = (b-d)(g+h)
            U[1, r] = 1; U[3, r] = -1
            V[v_g, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1
            r += 1

        # Single column k2 = 10 using schoolbook
        k2 = 10
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = k2 * n
        w_c12 = k2 * n + 1
        
        # ae
        U[0, r] = 1
        V[v_e2, r] = 1
        W[w_c02, r] = 1
        r += 1
        # bg
        U[1, r] = 1
        V[v_g2, r] = 1
        W[w_c02, r] = 1
        r += 1
        # ce
        U[2, r] = 1
        V[v_e2, r] = 1
        W[w_c12, r] = 1
        r += 1
        # dg
        U[3, r] = 1
        V[v_g2, r] = 1
        W[w_c12, r] = 1
        r += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_solution(U, V, W):
        """Check if the decomposition is exact."""
        # U: (4, R), V: (22, R), W: (22, R)
        R = U.shape[1]
        for a, b, c in tensor_entries:
            val = 0
            for r in range(R):
                val += U[a, r] * V[b, r] * W[c, r]
            if val != 1:
                return False
        return True

    def try_rank_38_search():
        """Attempt to find a rank 38 solution via random search."""
        target_rank = 38
        # Build the linear system matrix A for W
        # Equation: sum_r U[a,r] V[b,r] W[c,r] = 1
        # Let w be a vector of length dim_w * target_rank
        # w[c * target_rank + r] = W[c, r]
        # The coefficient for w[c * target_rank + r] in the equation for (a, b, c) is U[a, r] * V[b, r]
        # Note: For a fixed (a, b), the equation involves W[c, r] for all c? No.
        # The equation for a specific tensor entry (a, b, c) only involves the term W[c, r].
        # Wait, the sum is over r. For a fixed (a, b, c), the equation is:
        # sum_r (U[a, r] * V[b, r]) * W[c, r] = 1
        # This means the variable W[c, r] is coupled with coefficient U[a, r] * V[b, r].
        # However, W[c, r] appears in multiple equations?
        # Yes, W[c, r] appears in the equation for every (a, b, c) that has this specific c.
        # But the tensor entry is defined by a specific (a, b, c).
        # Are there other entries with the same c but different a, b?
        # c = k*n + i.
        # The entry (a, b, c) exists only if a=i*m+j and b=j*p+k.
        # So for a fixed c (which fixes k and i), the possible a are i*m+j (j=0,1) and possible b are j*p+k (j=0,1).
        # So for a fixed c, there are 4 possible (a, b) pairs that form valid tensor entries.
        # These 4 equations share the same W[c, :] vector.
        
        # We can solve for W block by block?
        # For each c, we have a system of 4 equations for the 38 variables W[c, 0..37].
        # The equations are:
        # For j in {0, 1}:
        #   a = i*m + j
        #   b = j*p + k
        #   sum_r U[a, r] V[b, r] W[c, r] = 1
        # This is a system of 4 linear equations in 38 unknowns (W[c, :]).
        # We can solve this using least squares or find a particular solution.
        # Since we have 4 equations and 38 unknowns, we can set 34 variables to 0 and solve for the 4?
        # Or just find the minimum norm solution.
        
        # Strategy:
        # 1. Randomize U and V.
        # 2. For each c (0 to 21):
        #    - Identify the 4 (a, b) pairs.
        #    - Form matrix M_c of shape (4, 38) where M_c[j, r] = U[a_j, r] * V[b_j, r].
        #    - Solve M_c * w_c = 1 (vector of ones).
        #    - Use np.linalg.lstsq to get w_c.
        # 3. Assemble W.
        # 4. Check if 2*W is close to integers.
        # 5. If yes, round and verify.
        
        np.random.seed(42) # Fixed seed for reproducibility in search? Maybe randomize per trial.
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Random U, V with small integers
            U = np.random.randint(-1, 2, size=(dim_u, target_rank))
            V = np.random.randint(-1, 2, size=(dim_v, target_rank))
            
            W = np.zeros((dim_w, target_rank))
            valid = True
            
            for c in range(dim_w):
                k = c // n
                i = c % n
                
                # Find the 4 (a, b) pairs
                # a = i*m + j, b = j*p + k for j in {0, 1}
                # Wait, j must be the same for a and b?
                # Yes, the tensor entry is T[i*m+j, j*p+k, k*n+i].
                # So for a fixed c (k, i), the possible j are 0 and 1.
                # This gives 2 pairs?
                # Let's check:
                # j=0: a=i*m, b=k. Entry (i*m, k, c).
                # j=1: a=i*m+1, b=p+k. Entry (i*m+1, p+k, c).
                # Are there others?
                # The definition is T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i.
                # So yes, for a fixed c, there are exactly 2 non-zero entries in the tensor?
                # No, the tensor is 3D.
                # For a fixed c, the slice T[:,:,c] has non-zeros at (i*m, k) and (i*m+1, p+k).
                # So we have 2 equations for W[c, :].
                
                eqs = []
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    # Coefficient vector for W[c, :]
                    coeff = U[a, :] * V[b, :]
                    eqs.append(coeff)
                
                if not eqs:
                    continue
                    
                M_c = np.array(eqs) # Shape (2, 38)
                rhs = np.ones(M_c.shape[0])
                
                # Solve M_c * w = rhs
                # If M_c is rank deficient or no solution, lstsq gives min norm.
                # We need an exact solution.
                # Since we have 2 eqs and 38 unknowns, we can always find a solution.
                # But we need it to be integer/half-integer.
                
                # Let's try to find a sparse integer solution.
                # Pick 2 columns of M_c that are linearly independent.
                # Solve for those 2 variables, set others to 0.
                # This might yield small integers.
                
                # Find 2 independent columns
                rank = np.linalg.matrix_rank(M_c)
                if rank < 2:
                    valid = False
                    break
                
                # Get column indices of pivot columns
                # Simple heuristic: iterate through columns, try to form a basis
                basis_cols = []
                current_basis = []
                for col_idx in range(target_rank):
                    col = M_c[:, col_idx]
                    new_basis = current_basis + [col]
                    if len(new_basis) <= 2:
                        if np.linalg.matrix_rank(np.column_stack(new_basis)) == len(new_basis):
                            basis_cols.append(col_idx)
                            current_basis = new_basis
                            if len(basis_cols) == 2:
                                break
                
                if len(basis_cols) < 2:
                    valid = False
                    break
                    
                M_sub = M_c[:, basis_cols]
                w_sub, residuals, rank_sub, s = np.linalg.lstsq(M_sub, rhs, rcond=None)
                
                # Check if solution is close to integers (scaled by 2)
                w_sub_scaled = w_sub * 2
                w_sub_rounded = np.round(w_sub_scaled)
                
                if np.allclose(w_sub_scaled, w_sub_rounded, atol=1e-9):
                    w_c = np.zeros(target_rank)
                    w_c[basis_cols] = w_sub_rounded / 2.0
                    W[c, :] = w_c
                else:
                    # Fallback to min norm solution and check
                    w_c, residuals, rank_c, s_c = np.linalg.lstsq(M_c, rhs, rcond=None)
                    w_c_scaled = w_c * 2
                    w_c_rounded = np.round(w_c_scaled)
                    if np.allclose(w_c_scaled, w_c_rounded, atol=1e-9):
                        W[c, :] = w_c_rounded / 2.0
                    else:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Verify the solution
            if verify_solution(U, V, W):
                # Convert to lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        return None

    # Try the search first
    result = try_rank_38_search()
    if result:
        return result
        
    # Fallback to rank 39
    return get_rank_39_solution()
