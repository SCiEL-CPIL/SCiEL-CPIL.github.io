import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    terms = []  # list of (u, v, w) triples

    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: {index: value}
        v_dict: {index: value}
        w_dict: {index: value}
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_dict.items():
            u[idx] = val
        for idx, val in v_dict.items():
            v[idx] = val
        for idx, val in w_dict.items():
            w[idx] = val
        terms.append((u, v, w))

    # --- Recursive Strassen for 4x4 * 4x4 ---
    # We generate 49 terms.
    # The 7 outer Strassen products M1..M7.
    # Each Mi is a 2x2 * 2x2 multiplication.
    # We decompose each Mi using the 7-term Strassen for 2x2 * 2x2.
    
    # Define the 7 outer Strassen products.
    # Each product is defined by:
    #   A_side: list of (i, j, coeff)
    #   B_side: list of (j, k, coeff)
    #   C_side: list of (k, i, coeff)
    # Wait, the C_side for the *product* Mi is not directly the C_side for the final C.
    # The product Mi is a 2x2 matrix. Let's call it P_i.
    # P_i is formed by multiplying two 2x2 matrices X_i and Y_i.
    # X_i is formed from A blocks, Y_i from B blocks.
    # The elements of P_i are then added/subtracted to form C.
    
    # Actually, it's simpler to think of the 49 terms as rank-1 tensors in the 4x4x4 space.
    # For each of the 7 outer steps, we have a "virtual" 2x2x2 multiplication.
    # The virtual A matrix elements are linear combos of A elements.
    # The virtual B matrix elements are linear combos of B elements.
    # The virtual C matrix elements are linear combos of C elements (with signs).
    
    # Let's define the 7 outer Strassen configurations.
    # Config: (A_positions, B_positions, C_positions)
    # A_positions: list of 4 (i, j, coeff) for the 2x2 virtual A
    # B_positions: list of 4 (j, k, coeff) for the 2x2 virtual B
    # C_positions: list of 4 (k, i, coeff) for the 2x2 virtual C
    
    # M1 = (A11 + A22) * (B11 + B22)
    # Virtual A:
    #   (0,0): A[0,0] + A[2,2] -> (0,0,1), (2,2,1)
    #   (0,1): A[0,1] + A[2,3] -> (0,1,1), (2,3,1)
    #   (1,0): A[1,0] + A[3,2] -> (1,0,1), (3,2,1)
    #   (1,1): A[1,1] + A[3,3] -> (1,1,1), (3,3,1)
    # Virtual B:
    #   (0,0): B[0,0] + B[2,2] -> (0,0,1), (2,2,1)
    #   (0,1): B[0,1] + B[2,3] -> (0,1,1), (2,3,1)
    #   (1,0): B[1,0] + B[3,2] -> (1,0,1), (3,2,1)
    #   (1,1): B[1,1] + B[3,3] -> (1,1,1), (3,3,1)
    # Virtual C:
    #   (0,0): C[0,0] -> (0,0,1)
    #   (0,1): C[0,1] -> (0,1,1)
    #   (1,0): C[1,0] -> (1,0,1)
    #   (1,1): C[1,1] -> (1,1,1)
    # Note: The C positions here are the *local* positions in the 2x2 result of the virtual multiplication.
    # The virtual multiplication result P1 is a 2x2 matrix.
    # P1[0,0] contributes to C[0,0] with +1
    # P1[0,1] contributes to C[0,1] with +1
    # P1[1,0] contributes to C[1,0] with +1
    # P1[1,1] contributes to C[1,1] with +1
    # Wait, in Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # So for M1, the contribution to C is:
    # C[0,0] += P1[0,0]
    # C[0,1] += P1[0,1]
    # C[1,0] += P1[1,0]
    # C[1,1] += P1[1,1]
    # This means the "Virtual C" for M1 has all +1 coefficients at the corresponding global positions.
    # Global positions for P1[0,0] is C[0,0]. P1[0,1] is C[0,1]. Etc.
    
    # Let's create a generic function to add the 7 inner Strassen terms for a given virtual 2x2x2 config.
    
    def add_inner_strassen(a_map, b_map, c_map):
        """
        a_map: dict mapping local (r,c) to list of (global_i, global_j, coeff)
        b_map: dict mapping local (r,c) to list of (global_j, global_k, coeff)
        c_map: dict mapping local (r,c) to list of (global_k, global_i, coeff)
        """
        # The 7 Strassen terms for 2x2*2x2:
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        
        # We need to generate 7 rank-1 terms.
        # Each term corresponds to one of the inner M1..M7.
        
        # Helper to build u, v, w dicts for a specific inner term
        def get_u(a_indices):
            u_dict = {}
            for (li, lj, coeff_a) in a_indices:
                for (gi, gj, coeff_g) in a_map[(li, lj)]:
                    idx = gi * m + gj
                    val = coeff_a * coeff_g
                    u_dict[idx] = u_dict.get(idx, 0) + val
            return u_dict

        def get_v(b_indices):
            v_dict = {}
            for (lj, lk, coeff_b) in b_indices:
                for (gj, gk, coeff_g) in b_map[(lj, lk)]:
                    idx = gj * p + gk
                    val = coeff_b * coeff_g
                    v_dict[idx] = v_dict.get(idx, 0) + val
            return v_dict

        def get_w(c_indices):
            w_dict = {}
            for (lk, li, coeff_c) in c_indices:
                for (gk, gi, coeff_g) in c_map[(lk, li)]:
                    idx = gk * n + gi
                    val = coeff_c * coeff_g
                    w_dict[idx] = w_dict.get(idx, 0) + val
            return w_dict

        # Inner M1: A(0,0)+A(1,1) * B(0,0)+B(1,1)
        # A indices: (0,0,1), (1,1,1)
        # B indices: (0,0,1), (1,1,1)
        # C indices: (0,0,1), (1,1,1)
        u = get_u([(0,0,1), (1,1,1)])
        v = get_v([(0,0,1), (1,1,1)])
        w = get_w([(0,0,1), (1,1,1)])
        add_term(u, v, w)

        # Inner M2: A(1,0)+A(1,1) * B(0,0)
        # A indices: (1,0,1), (1,1,1)
        # B indices: (0,0,1)
        # C indices: (1,0,1), (1,1,-1)
        u = get_u([(1,0,1), (1,1,1)])
        v = get_v([(0,0,1)])
        w = get_w([(1,0,1), (1,1,-1)])
        add_term(u, v, w)

        # Inner M3: A(0,0) * B(0,1)-B(1,1)
        # A indices: (0,0,1)
        # B indices: (0,1,1), (1,1,-1)
        # C indices: (0,1,1), (1,1,1)
        u = get_u([(0,0,1)])
        v = get_v([(0,1,1), (1,1,-1)])
        w = get_w([(0,1,1), (1,1,1)])
        add_term(u, v, w)

        # Inner M4: A(1,1) * B(1,0)-B(0,0)
        # A indices: (1,1,1)
        # B indices: (1,0,1), (0,0,-1)
        # C indices: (0,0,1), (1,0,1)
        u = get_u([(1,1,1)])
        v = get_v([(1,0,1), (0,0,-1)])
        w = get_w([(0,0,1), (1,0,1)])
        add_term(u, v, w)

        # Inner M5: A(0,0)+A(0,1) * B(1,1)
        # A indices: (0,0,1), (0,1,1)
        # B indices: (1,1,1)
        # C indices: (0,1,1), (0,0,-1)
        u = get_u([(0,0,1), (0,1,1)])
        v = get_v([(1,1,1)])
        w = get_w([(0,1,1), (0,0,-1)])
        add_term(u, v, w)

        # Inner M6: A(1,0)-A(0,0) * B(0,0)+B(0,1)
        # A indices: (1,0,1), (0,0,-1)
        # B indices: (0,0,1), (0,1,1)
        # C indices: (1,1,1)
        u = get_u([(1,0,1), (0,0,-1)])
        v = get_v([(0,0,1), (0,1,1)])
        w = get_w([(1,1,1)])
        add_term(u, v, w)

        # Inner M7: A(0,1)-A(1,1) * B(1,0)+B(1,1)
        # A indices: (0,1,1), (1,1,-1)
        # B indices: (1,0,1), (1,1,1)
        # C indices: (0,0,1)
        u = get_u([(0,1,1), (1,1,-1)])
        v = get_v([(1,0,1), (1,1,1)])
        w = get_w([(0,0,1)])
        add_term(u, v, w)

    # Define the 7 outer Strassen configs
    # a_map: local (r,c) -> [(global_i, global_j, coeff)]
    # b_map: local (r,c) -> [(global_j, global_k, coeff)]
    # c_map: local (r,c) -> [(global_k, global_i, coeff)]
    
    # M1
    a_map = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)]
    }
    b_map = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)]
    }
    c_map = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # M2 = (A21 + A22) * B11
    a_map = {
        (0,0): [(2,0,1)], # A21[0,0]
        (0,1): [(2,1,1)], # A21[0,1]
        (1,0): [(3,0,1), (2,0,0)], # Wait, A21 is rows 2-3, cols 0-1. A22 is rows 2-3, cols 2-3.
        # A21: (2,0), (2,1), (3,0), (3,1)
        # A22: (2,2), (2,3), (3,2), (3,3)
        # Virtual A for M2:
        # A' = A21 + A22
        # A'[0,0] = A21[0,0] + A22[0,0] = A[2,0] + A[2,2]
        # A'[0,1] = A21[0,1] + A22[0,1] = A[2,1] + A[2,3]
        # A'[1,0] = A21[1,0] + A22[1,0] = A[3,0] + A[3,2]
        # A'[1,1] = A21[1,1] + A22[1,1] = A[3,1] + A[3,3]
        (0,0): [(2,0,1), (2,2,1)],
        (0,1): [(2,1,1), (2,3,1)],
        (1,0): [(3,0,1), (3,2,1)],
        (1,1): [(3,1,1), (3,3,1)]
    }
    b_map = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)]
    }
    # C contribution for M2: C21 += M2, C22 -= M2
    # C21 is rows 2-3, cols 0-1. C22 is rows 2-3, cols 2-3.
    # P2[0,0] -> C[2,0] (+1)
    # P2[0,1] -> C[2,1] (+1)
    # P2[1,0] -> C[3,0] (+1)
    # P2[1,1] -> C[3,1] (+1)
    # And for C22:
    # P2[0,0] -> C[2,2] (-1)
    # P2[0,1] -> C[2,3] (-1)
    # P2[1,0] -> C[3,2] (-1)
    # P2[1,1] -> C[3,3] (-1)
    c_map = {
        (0,0): [(0,2,1), (2,2,-1)], # k=0, i=2 -> C[2,0]; k=2, i=2 -> C[2,2]
        (0,1): [(1,2,1), (3,2,-1)], # k=1, i=2 -> C[2,1]; k=3, i=2 -> C[2,3]
        (1,0): [(0,3,1), (2,3,-1)], # k=0, i=3 -> C[3,0]; k=2, i=3 -> C[3,2]
        (1,1): [(1,3,1), (3,3,-1)]  # k=1, i=3 -> C[3,1]; k=3, i=3 -> C[3,3]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # M3 = A11 * (B12 - B22)
    a_map = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)]
    }
    b_map = {
        (0,0): [(0,2,1)], # B12[0,0] = B[0,2]
        (0,1): [(0,3,1)], # B12[0,1] = B[0,3]
        (1,0): [(1,2,-1)], # -B22[0,0] = -B[2,2]
        (1,1): [(1,3,-1)]  # -B22[0,1] = -B[2,3]
    }
    # C contribution for M3: C12 += M3, C22 += M3
    # C12: rows 0-1, cols 2-3. C22: rows 2-3, cols 2-3.
    c_map = {
        (0,0): [(2,0,1)], # k=2, i=0 -> C[0,2]
        (0,1): [(3,0,1)], # k=3, i=0 -> C[0,3]
        (1,0): [(2,1,1)], # k=2, i=1 -> C[1,2]
        (1,1): [(3,1,1)]  # k=3, i=1 -> C[1,3]
    }
    # Wait, C22 contribution?
    # C22[0,0] = C[2,2]. P3[0,0] is local (0,0).
    # So c_map[0,0] should also include (2,2,1).
    c_map = {
        (0,0): [(2,0,1), (2,2,1)],
        (0,1): [(3,0,1), (3,2,1)],
        (1,0): [(2,1,1), (2,3,1)],
        (1,1): [(3,1,1), (3,3,1)]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # M4 = A22 * (B21 - B11)
    a_map = {
        (0,0): [(2,2,1)],
        (0,1): [(2,3,1)],
        (1,0): [(3,2,1)],
        (1,1): [(3,3,1)]
    }
    b_map = {
        (0,0): [(2,0,1)], # B21[0,0] = B[2,0]
        (0,1): [(2,1,1)], # B21[0,1] = B[2,1]
        (1,0): [(0,0,-1)], # -B11[0,0] = -B[0,0]
        (1,1): [(0,1,-1)]  # -B11[0,1] = -B[0,1]
    }
    # C contribution for M4: C11 += M4, C21 += M4
    c_map = {
        (0,0): [(0,0,1), (0,2,1)], # C[0,0] and C[2,0]
        (0,1): [(1,0,1), (1,2,1)], # C[0,1] and C[2,1]
        (1,0): [(0,1,1), (0,3,1)], # C[1,0] and C[3,0]
        (1,1): [(1,1,1), (1,3,1)]  # C[1,1] and C[3,1]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # M5 = (A11 + A12) * B22
    a_map = {
        (0,0): [(0,0,1), (0,2,1)],
        (0,1): [(0,1,1), (0,3,1)],
        (1,0): [(1,0,1), (1,2,1)],
        (1,1): [(1,1,1), (1,3,1)]
    }
    b_map = {
        (0,0): [(2,2,1)],
        (0,1): [(2,3,1)],
        (1,0): [(3,2,1)],
        (1,1): [(3,3,1)]
    }
    # C contribution for M5: C12 += M5, C11 -= M5
    c_map = {
        (0,0): [(2,0,1), (0,0,-1)], # C[0,2] and -C[0,0]
        (0,1): [(3,0,1), (1,0,-1)], # C[0,3] and -C[0,1]
        (1,0): [(2,1,1), (0,1,-1)], # C[1,2] and -C[1,0]
        (1,1): [(3,1,1), (1,1,-1)]  # C[1,3] and -C[1,1]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # M6 = (A21 - A11) * (B11 + B12)
    a_map = {
        (0,0): [(2,0,1), (0,0,-1)],
        (0,1): [(2,1,1), (0,1,-1)],
        (1,0): [(3,0,1), (1,0,-1)],
        (1,1): [(3,1,1), (1,1,-1)]
    }
    b_map = {
        (0,0): [(0,0,1), (0,2,1)],
        (0,1): [(0,1,1), (0,3,1)],
        (1,0): [(1,0,1), (1,2,1)],
        (1,1): [(1,1,1), (1,3,1)]
    }
    # C contribution for M6: C22 += M6
    c_map = {
        (0,0): [(2,2,1)],
        (0,1): [(3,2,1)],
        (1,0): [(2,3,1)],
        (1,1): [(3,3,1)]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # M7 = (A12 - A22) * (B21 + B22)
    a_map = {
        (0,0): [(0,2,1), (2,2,-1)],
        (0,1): [(0,3,1), (2,3,-1)],
        (1,0): [(1,2,1), (3,2,-1)],
        (1,1): [(1,3,1), (3,3,-1)]
    }
    b_map = {
        (0,0): [(2,0,1), (2,2,1)],
        (0,1): [(2,1,1), (2,3,1)],
        (1,0): [(3,0,1), (3,2,1)],
        (1,1): [(3,1,1), (3,3,1)]
    }
    # C contribution for M7: C11 += M7
    c_map = {
        (0,0): [(0,0,1)],
        (0,1): [(1,0,1)],
        (1,0): [(0,1,1)],
        (1,1): [(1,1,1)]
    }
    add_inner_strassen(a_map, b_map, c_map)

    # --- Schoolbook for 4x4 * 4x3 (k=4,5,6) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
