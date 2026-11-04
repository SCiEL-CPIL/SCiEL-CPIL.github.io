import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the rank-42 Strassen decomposition
    rank_full = 42
    U_full = np.zeros((nm, rank_full), dtype=float)
    V_full = np.zeros((mp, rank_full), dtype=float)
    W_full = np.zeros((pn, rank_full), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset, U, V, W):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        U[a10, offset + 1] += 1.0; U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0; W[w11, offset + 1] -= 1.0
        
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0; V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0; W[w11, offset + 2] += 1.0
        
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0; V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0; W[w10, offset + 3] += 1.0
        
        U[a00, offset + 4] += 1.0; U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0; W[w01, offset + 4] += 1.0
        
        U[a10, offset + 5] += 1.0; U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0; V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        U[a01, offset + 6] += 1.0; U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0; V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U_full, V_full, W_full)
                offset += 7

    def verify(U_mat, V_mat, W_mat, rank, tol=1e-8):
        """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    # Check if rank-42 works (should always work)
    if not verify(U_full, V_full, W_full, rank_full):
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank))
        best_V = np.zeros((mp, best_rank))
        best_W = np.zeros((pn, best_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    best_rank = rank_full
    best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()

    # Try to reduce rank by removing pairs of factors and solving the resulting system.
    # Strategy: For each pair of factor indices to remove, fix the remaining W columns,
    # then set up a linear system to solve for U and V adjustments.
    
    # Actually, a more practical approach: use the fact that the tensor equation
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] is multilinear.
    # If we fix W, it becomes bilinear in U and V.
    # If we fix both W and V, it becomes linear in U.
    
    # Here's a concrete strategy:
    # 1. Try removing 2 factors from the 42-factor Strassen solution.
    # 2. For each removal pattern, fix the remaining W values.
    # 3. Set up the linear system for U (treating V as unknowns coupled with U).
    
    # A simpler heuristic: try removing one factor from two different blocks.
    # The 6 blocks correspond to the 6 (i_start, j_start, k_start) combinations.
    # Block b has factors at indices [7*b, 7*b+1, ..., 7*b+6].
    
    # For each pair of distinct blocks (b1, b2) and each factor index within each,
    # try removing those 2 factors and check if a valid completion exists.
    
    # To check existence, we can try to solve the linear system.
    # With W fixed (remaining 40 columns), the equation is:
    # sum_{r in S} U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # This is still bilinear. But if we also fix V (remaining 40 columns from Strassen),
    # it becomes linear in U.
    
    # Let's try: fix both V and W to their Strassen values (with removed columns zeroed),
    # then solve for U linearly.
    
    # For a given set S of kept factor indices:
    # For each (a,b,c): sum_{r in S} U[a,r] * V_strassen[b,r] * W_strassen[c,r] = T[a,b,c]
    # This is a linear system in U[a,r] for r in S.
    # For each a, we have a system: sum_{r in S} U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c] for all (b,c).
    # But b and c are not independent: b = j*p+k, c = k*n+i, and a = i*m+j.
    # For fixed a = i*m+j, we need: for all k, sum_{r in S} U[a,r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (b,c) combinations, the sum should be 0.
    
    # This is getting complex. Let me try a different, more direct approach.
    
    # Direct approach: For each pair of factors to remove, set up the full bilinear system
    # and try to solve it using a simple iterative method or check if the Strassen solution
    # restricted to the remaining factors can be perturbed to work.
    
    # Given time constraints, let's try a small number of specific patterns and verify.
    
    found_better = False
    for trial in range(200):  # Limit trials
        if time.time() > deadline - 5:
            break
        
        # Pick two distinct blocks
        b1 = trial % 6
        b2 = (trial // 6) % 6
        if b1 == b2:
            b2 = (b2 + 1) % 6
        # Pick factors within each block
        f1 = (trial // 36) % 7
        f2 = (trial // 252) % 7
        
        remove_indices = sorted([b1 * 7 + f1, b2 * 7 + f2])
        if remove_indices[0] == remove_indices[1]:
            continue
            
        keep_indices = [t for t in range(42) if t not in remove_indices]
        r = len(keep_indices)
        
        if r >= best_rank:
            continue
        
        # Create candidate matrices with removed columns zeroed
        U_trial = U_full[:, keep_indices].copy()
        V_trial = V_full[:, keep_indices].copy()
        W_trial = W_full[:, keep_indices].copy()
        
        # Verify this truncated solution
        if verify(U_trial, V_trial, W_trial, r):
            best_rank = r
            best_U, best_V, best_W = U_trial, V_trial, W_trial
            found_better = True
            break

    # If no simple removal worked, try a linear solve approach for a few patterns.
    # For each pattern, fix W to Strassen values (kept columns), fix V to Strassen values,
    # and solve for U.
    if not found_better:
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            b1 = trial % 6
            b2 = (trial // 6) % 6
            if b1 == b2:
                b2 = (b2 + 1) % 6
            f1 = (trial // 36) % 7
            f2 = (trial // 252) % 7
            
            remove_indices = sorted([b1 * 7 + f1, b2 * 7 + f2])
            if remove_indices[0] == remove_indices[1]:
                continue
                
            keep_indices = [t for t in range(42) if t not in remove_indices]
            r = len(keep_indices)
            
            if r >= best_rank:
                continue
            
            V_fixed = V_full[:, keep_indices].copy()
            W_fixed = W_full[:, keep_indices].copy()
            
            # For each row a of U, solve the linear system
            U_sol = np.zeros((nm, r))
            valid = True
            for a in range(nm):
                i = a // m
                j = a % m
                # Build the linear system: for each (b, c), sum_{t in S} U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
                # Only (b,c) pairs that are consistent with a matter.
                # For fixed a = i*m+j, b must be of the form j*p+k for some k, and c must be k*n+i.
                # So for each k in range(p), b = j*p+k, c = k*n+i, and T[a,b,c] = 1.
                # For other (b,c) pairs, T[a,b,c] = 0.
                
                # Build the system matrix A_sys where A_sys[(b,c), t] = V[b,t] * W[c,t] for t in keep_indices
                # And the RHS vector rhs[(b,c)] = T[a,b,c]
                
                # We only need to consider (b,c) pairs. There are mp * pn = 24 * 12 = 288 such pairs.
                # The system is: A_sys @ U[a, :] = rhs
                # where A_sys has shape (288, r) and rhs has shape (288,).
                
                A_sys = np.zeros((mp * pn, r))
                rhs = np.zeros(mp * pn)
                
                for b in range(mp):
                    for c in range(pn):
                        bc = b * pn + c
                        # Check if (a,b,c) is a valid tensor entry
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        i_c = c % n
                        k_c = c // n
                        
                        if i_a == i_c and j_a == j_b and k_b == k_c:
                            rhs[bc] = 1.0
                        else:
                            rhs[bc] = 0.0
                        
                        for t_idx, t in enumerate(keep_indices):
                            A_sys[bc, t_idx] = V_fixed[b, t_idx] * W_fixed[c, t_idx]
                
                # Solve the least squares problem
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
                    U_sol[a, :] = sol
                except:
                    valid = False
                    break
            
            if valid:
                if verify(U_sol, V_fixed, W_fixed, r):
                    best_rank = r
                    best_U, best_V, best_W = U_sol, V_fixed, W_fixed
                    found_better = True
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
