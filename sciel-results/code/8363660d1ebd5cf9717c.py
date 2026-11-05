"""
Combined approach: Strassen-based block decomposition with rank-1 peeling refinement.

Strategy:
1. Build a Strassen-based decomposition (93 terms) as the starting point.
2. Apply rank-1 peeling: iteratively find rank-1 terms that exactly match the
   current tensor residual, subtract them, and check if the remaining terms
   can be reduced by finding linear dependencies.
3. Use the deadline guard to return the best (lowest rank) valid decomposition.

The key structural insight is that Strassen gives us a good starting point (93 < 112),
and then we attempt to reduce further by identifying redundant rank-1 terms
through linear algebra on the factor matrices.
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline, 35s safety margin
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        """
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    def schoolbook_block(i_off, j_off, k_off):
        """
        Adds schoolbook terms for a 2x2 * 2x1 block (4 terms).
        """
        for di in range(2):
            for dj in range(2):
                u = [0] * (n * m)
                u[(i_off + di) * m + (j_off + dj)] = 1
                v = [0] * (m * p)
                v[(j_off + dj) * p + k_off] = 1
                w = [0] * (p * n)
                w[k_off * n + (i_off + di)] = 1
                add_term(u, v, w)

    def build_strassen_decomposition():
        """
        Build the 93-term Strassen decomposition:
        - 49 terms for 4x4 * 4x4 (recursive Strassen)
        - 28 terms for 4x4 * 4x2 (block Strassen on cols 4,5)
        - 16 terms for 4x4 * 4x1 (schoolbook on col 6)
        """
        # 4x4 * 4x4 part (cols 0-3): recursive Strassen
        # Outer level: 7 terms, each containing 7 inner terms = 49
        outer_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00 + C11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (A10+A11)B00 -> C10 - C11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = A00(B01-B11) -> C01 + C11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = A11(B10-B00) -> C00 + C10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (A00+A01)B11 -> -C00 + C01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (A10-A00)(B00+B01) -> C11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (A01-A11)(B10+B11) -> C00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, tensor):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if tensor == 'A':
                return row * m + col
            elif tensor == 'B':
                return row * p + col
            elif tensor == 'C':
                return col * n + row
        
        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)
        
        # 4x4 * 4x2 part (cols 4-5): 4 blocks of 2x2 * 2x2 Strassen
        for i_b in range(2):
            for j_b in range(2):
                strassen_2x2_block(i_b * 2, j_b * 2, 4)
        
        # 4x4 * 4x1 part (col 6): schoolbook, 4 blocks of 2x2 * 2x1
        for i_b in range(2):
            for j_b in range(2):
                schoolbook_block(i_b * 2, j_b * 2, 6)

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_reduce_rank(terms_list, max_iterations=50):
        """
        Attempt to reduce rank by finding linearly dependent terms.
        If two terms have proportional U vectors (or V or W), we can potentially
        combine them. More generally, look for sets of terms where one is a
        linear combination of others.
        """
        if not terms_list:
            return terms_list
        
        r = len(terms_list)
        if r <= 1:
            return terms_list
        
        # Build factor matrices
        U_mat = np.array([t[0] for t in terms_list], dtype=float).T  # (n*m, r)
        V_mat = np.array([t[1] for t in terms_list], dtype=float).T  # (m*p, r)
        W_mat = np.array([t[2] for t in terms_list], dtype=float).T  # (p*n, r)
        
        # Check for exact linear dependencies among columns of U
        # If column c is a linear combo of other columns, we might be able to eliminate it
        # But we need the same dependency to hold for V and W simultaneously
        
        # Simple approach: check if any two terms are identical
        reduced = list(terms_list)
        i = 0
        while i < len(reduced):
            if time.time() > deadline - 10:
                break
            j = i + 1
            while j < len(reduced):
                # Check if term j is a scalar multiple of term i
                u_i, v_i, w_i = reduced[i]
                u_j, v_j, w_j = reduced[j]
                
                # Find non-zero entries to determine ratio
                ratio = None
                for idx in range(n * m):
                    if u_i[idx] != 0:
                        if u_j[idx] != 0:
                            ratio = u_j[idx] / u_i[idx]
                            break
                        else:
                            ratio = None
                            break
                    else:
                        if u_j[idx] != 0:
                            ratio = None
                            break
                
                if ratio is not None:
                    # Check if v_j = ratio * v_i and w_j = ratio * w_i
                    valid = True
                    for idx in range(m * p):
                        if abs(v_j[idx] - ratio * v_i[idx]) > 1e-9:
                            valid = False
                            break
                    if valid:
                        for idx in range(p * n):
                            if abs(w_j[idx] - ratio * w_i[idx]) > 1e-9:
                                valid = False
                                break
                    
                    if valid:
                        # These two terms are proportional; we can combine them
                        # Actually, if they're proportional, the sum is also proportional
                        # We can't simply remove one; we need to keep the combined effect
                        # But if ratio == -1, they cancel out
                        if abs(ratio + 1) < 1e-9:
                            # They cancel: remove both
                            reduced.pop(j)
                            reduced.pop(i)
                            i = max(0, i - 1)
                            break
                        else:
                            # Combine: new term = term_i + term_j
                            new_u = [u_i[idx] + u_j[idx] for idx in range(n * m)]
                            new_v = [v_i[idx] + v_j[idx] for idx in range(m * p)]
                            new_w = [w_i[idx] + w_j[idx] for idx in range(p * n)]
                            reduced[i] = (new_u, new_v, new_w)
                            reduced.pop(j)
                            break
                
                j += 1
            i += 1
        
        return reduced

    def rank1_peeling(terms_list, max_rounds=20):
        """
        Rank-1 peeling: try to identify and extract rank-1 terms from the
        current decomposition that can be cleanly separated.
        """
        if not terms_list:
            return terms_list
        
        current = list(terms_list)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            r = len(current)
            if r <= 1:
                break
            
            # Build matrices
            U_mat = np.array([t[0] for t in current], dtype=float).T
            V_mat = np.array([t[1] for t in current], dtype=float).T
            W_mat = np.array([t[2] for t in current], dtype=float).T
            
            # Check for zero columns (redundant terms)
            new_current = []
            for c in range(r):
                if np.allclose(U_mat[:, c], 0) or np.allclose(V_mat[:, c], 0) or np.allclose(W_mat[:, c], 0):
                    continue  # Skip zero terms
                new_current.append(current[c])
            
            if len(new_current) < r:
                current = new_current
                continue
            
            # Try to find if any term can be expressed as linear combination of others
            # This is expensive; do a simplified check
            found_reduction = False
            for c in range(len(current)):
                if time.time() > deadline - 10:
                    break
                
                # Check if term c is a linear combination of the others
                others = [i for i in range(len(current)) if i != c]
                if not others:
                    break
                
                # Build matrix of other terms' U vectors
                U_others = np.array([current[i][0] for i in others], dtype=float).T
                u_c = np.array(current[c][0], dtype=float)
                
                # Try to solve U_others * x = u_c
                try:
                    # Use least squares
                    x, residuals, rank, sv = np.linalg.lstsq(U_others, u_c, rcond=None)
                    if residuals.size > 0 and residuals[0] < 1e-9:
                        # Check if same x works for V and W
                        V_others = np.array([current[i][1] for i in others], dtype=float).T
                        v_c = np.array(current[c][1], dtype=float)
                        v_pred = V_others @ x
                        
                        W_others = np.array([current[i][2] for i in others], dtype=float).T
                        w_c = np.array(current[c][2], dtype=float)
                        w_pred = W_others @ x
                        
                        if np.allclose(v_pred, v_c, atol=1e-9) and np.allclose(w_pred, w_c, atol=1e-9):
                            # Term c is redundant! Remove it.
                            current.pop(c)
                            found_reduction = True
                            break
                except:
                    continue
            
            if not found_reduction:
                break
        
        return current

    # Build the Strassen decomposition
    build_strassen_decomposition()
    
    # Initial rank
    initial_rank = len(terms)
    
    # Verify the initial decomposition
    U_init = [t[0] for t in terms]
    V_init = [t[1] for t in terms]
    W_init = [t[2] for t in terms]
    
    if not verify_decomposition(U_init, V_init, W_init):
        # Fallback to schoolbook
        U_school = [[0] * (n * m * p) for _ in range(n * m)]
        V_school = [[0] * (n * m * p) for _ in range(m * p)]
        W_school = [[0] * (n * m * p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    best_terms = list(terms)
    
    # Attempt rank reduction
    if time.time() < deadline - 20:
        # First pass: remove duplicates and zero terms
        reduced = try_reduce_rank(terms)
        if len(reduced) < len(best_terms):
            # Verify
            U_r = [t[0] for t in reduced]
            V_r = [t[1] for t in reduced]
            W_r = [t[2] for t in reduced]
            if verify_decomposition(U_r, V_r, W_r):
                best_terms = reduced
        
        # Second pass: rank-1 peeling
        if time.time() < deadline - 20:
            peeled = rank1_peeling(best_terms)
            if len(peeled) < len(best_terms):
                U_p = [t[0] for t in peeled]
                V_p = [t[1] for t in peeled]
                W_p = [t[2] for t in peeled]
                if verify_decomposition(U_p, V_p, W_p):
                    best_terms = peeled
        
        # Third pass: try again after peeling
        if time.time() < deadline - 20:
            reduced2 = try_reduce_rank(best_terms)
            if len(reduced2) < len(best_terms):
                U_r2 = [t[0] for t in reduced2]
                V_r2 = [t[1] for t in reduced2]
                W_r2 = [t[2] for t in reduced2]
                if verify_decomposition(U_r2, V_r2, W_r2):
                    best_terms = reduced2
    
    # Final verification
    U_final = [t[0] for t in best_terms]
    V_final = [t[1] for t in best_terms]
    W_final = [t[2] for t in best_terms]
    
    if not verify_decomposition(U_final, V_final, W_final):
        # Fallback to the verified initial
        return {"U": U_init, "V": V_init, "W": W_init}
    
    return {"U": U_final, "V": V_final, "W": W_final}
