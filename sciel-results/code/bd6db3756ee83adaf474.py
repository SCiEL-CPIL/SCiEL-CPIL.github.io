"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 4, 6

    def build_rank66():
        """Build the rank-66 decomposition."""
        U = np.zeros((n * m, 66))
        V = np.zeros((m * p, 66))
        W = np.zeros((p * n, 66))
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00, col] += 1; U[A11, col] += 1
            V[B00, col] += 1; V[B11, col] += 1
            W[C00, col] += 1; W[C11, col] += 1
            col += 1

            U[A10, col] += 1; U[A11, col] += 1
            V[B00, col] += 1
            W[C10, col] += 1; W[C11, col] -= 1
            col += 1

            U[A00, col] += 1
            V[B01, col] += 1; V[B11, col] -= 1
            W[C01, col] += 1; W[C11, col] += 1
            col += 1

            U[A11, col] += 1
            V[B10, col] += 1; V[B00, col] -= 1
            W[C00, col] += 1; W[C10, col] += 1
            col += 1

            U[A00, col] += 1; U[A01, col] += 1
            V[B11, col] += 1
            W[C00, col] -= 1; W[C01, col] += 1
            col += 1

            U[A11, col] += 1; U[A01, col] -= 1
            V[B00, col] += 1; V[B01, col] += 1
            W[C11, col] += 1
            col += 1

            U[A01, col] += 1; U[A11, col] -= 1
            V[B10, col] += 1; V[B11, col] += 1
            W[C00, col] += 1
            col += 1

        def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
            col += 1
            U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
            col += 1
            U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
            col += 1
            U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
            col += 1

        col_groups = [(0, 1), (2, 3), (4, 5)]
        for t0, t1 in col_groups:
            add_strassen(0, 0, 0, t0, 0, t0)
            add_strassen(0, 2, 2, t0, 0, t0)
            add_direct_1x2_2x2(2, 0, 0, t0, 2, t0)
            add_direct_1x2_2x2(2, 2, 2, t0, 2, t0)

        return U, V, W

    def verify(U_mat, V_mat, W_mat, r):
        """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        return max_err

    def try_eliminate_column(U, V, W, r, col_idx):
        """Try to express column col_idx as a linear combination of other columns.
        If successful, return the modified (U,V,W) with rank r-1."""
        # The tensor equation: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
        # We want: U[:,col_idx] ⊗ V[:,col_idx] ⊗ W[:,col_idx] = sum_{c != col_idx} alpha_c * U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # This is a nonlinear constraint. Instead, try: can we absorb column col_idx into others
        # by adding a scaled copy of some other column's contribution?
        # 
        # Alternative: check if removing column col_idx and adjusting others still works.
        # This is hard in general. Let's try a different approach:
        # 
        # Try to find if there exist coefficients such that we can merge two columns.
        # For now, try a simple heuristic: check if two columns have proportional U, V, W vectors.
        
        # Check for duplicate columns
        for c2 in range(r):
            if c2 == col_idx:
                continue
            # Check if U[:,col_idx] is proportional to U[:,c2]
            u1 = U[:, col_idx]
            u2 = U[:, c2]
            v1 = V[:, col_idx]
            v2 = V[:, c2]
            w1 = W[:, col_idx]
            w2 = W[:, c2]
            
            # Check proportionality
            if np.allclose(u1, 0) and np.allclose(u2, 0):
                continue
            if np.allclose(u1, 0) or np.allclose(u2, 0):
                continue
            
            # Find ratio from U
            nonzero_u1 = u1 != 0
            nonzero_u2 = u2 != 0
            common = nonzero_u1 & nonzero_u2
            if np.sum(common) < 2:
                continue
            
            ratios = u1[common] / u2[common]
            if not np.allclose(ratios, ratios[0], atol=1e-10):
                continue
            
            ratio_u = ratios[0]
            
            # Check V proportionality with same ratio
            nonzero_v1 = v1 != 0
            nonzero_v2 = v2 != 0
            common_v = nonzero_v1 & nonzero_v2
            if np.sum(common_v) < 1:
                continue
            ratios_v = v1[common_v] / v2[common_v]
            if not np.allclose(ratios_v, ratio_u, atol=1e-10):
                continue
            
            # Check W proportionality with same ratio
            nonzero_w1 = w1 != 0
            nonzero_w2 = w2 != 0
            common_w = nonzero_w1 & nonzero_w2
            if np.sum(common_w) < 1:
                continue
            ratios_w = w1[common_w] / w2[common_w]
            if not np.allclose(ratios_w, ratio_u, atol=1e-10):
                continue
            
            # All three are proportional with same ratio -> can merge
            # New column = u2*(1+ratio_u), v2*(1+ratio_u), w2*(1+ratio_u)
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            U_new[:, c2] = u2 * (1 + ratio_u)
            V_new[:, c2] = v2 * (1 + ratio_u)
            W_new[:, c2] = w2 * (1 + ratio_u)
            # Remove column col_idx
            if col_idx < c2:
                U_new = np.delete(U_new, col_idx, axis=1)
                V_new = np.delete(V_new, col_idx, axis=1)
                W_new = np.delete(W_new, col_idx, axis=1)
            else:
                U_new = np.delete(U_new, col_idx, axis=1)
                V_new = np.delete(V_new, col_idx, axis=1)
                W_new = np.delete(W_new, col_idx, axis=1)
            
            return U_new, V_new, W_new, True
        
        return None, None, None, False

    def try_merge_pair(U, V, W, r):
        """Try to find any pair of columns that can be merged."""
        for c1 in range(r):
            for c2 in range(c1 + 1, r):
                u1 = U[:, c1]
                u2 = U[:, c2]
                v1 = V[:, c1]
                v2 = V[:, c2]
                w1 = W[:, c1]
                w2 = W[:, c2]
                
                if np.allclose(u1, 0) or np.allclose(u2, 0):
                    continue
                if np.allclose(v1, 0) or np.allclose(v2, 0):
                    continue
                if np.allclose(w1, 0) or np.allclose(w2, 0):
                    continue
                
                # Find ratio from U
                nonzero_u1 = u1 != 0
                nonzero_u2 = u2 != 0
                common = nonzero_u1 & nonzero_u2
                if np.sum(common) < 2:
                    continue
                
                ratios = u1[common] / u2[common]
                if not np.allclose(ratios, ratios[0], atol=1e-10):
                    continue
                
                ratio = ratios[0]
                
                # Check V
                nonzero_v1 = v1 != 0
                nonzero_v2 = v2 != 0
                common_v = nonzero_v1 & nonzero_v2
                if np.sum(common_v) < 1:
                    continue
                ratios_v = v1[common_v] / v2[common_v]
                if not np.allclose(ratios_v, ratio, atol=1e-10):
                    continue
                
                # Check W
                nonzero_w1 = w1 != 0
                nonzero_w2 = w2 != 0
                common_w = nonzero_w1 & nonzero_w2
                if np.sum(common_w) < 1:
                    continue
                ratios_w = w1[common_w] / w2[common_w]
                if not np.allclose(ratios_w, ratio, atol=1e-10):
                    continue
                
                # Merge: new_col = col2 * (1 + ratio)
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                U_new[:, c2] = u2 * (1 + ratio)
                V_new[:, c2] = v2 * (1 + ratio)
                W_new[:, c2] = w2 * (1 + ratio)
                
                # Remove column c1
                U_new = np.delete(U_new, c1, axis=1)
                V_new = np.delete(V_new, c1, axis=1)
                W_new = np.delete(W_new, c1, axis=1)
                
                return U_new, V_new, W_new, True
        
        return None, None, None, False

    # Build initial rank-66 decomposition
    U, V, W = build_rank66()
    rank = 66

    # Try to reduce rank by merging proportional columns
    for attempt in range(20):
        if time.time() > deadline:
            break
        U_new, V_new, W_new, success = try_merge_pair(U, V, W, rank)
        if not success:
            break
        # Verify the new decomposition
        err = verify(U_new, V_new, W_new, rank - 1)
        if err < 1e-10:
            U, V, W = U_new, V_new, W_new
            rank -= 1
        else:
            break

    # Verify final decomposition
    err = verify(U, V, W, rank)
    if err > 1e-10:
        # Fallback to schoolbook rank 72
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        rank = rank_school
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Convert to lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
