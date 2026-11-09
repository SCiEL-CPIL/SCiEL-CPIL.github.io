"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Strategy: Decompose into four 2x2x2 Strassen blocks.
    A is 2x4, B is 4x4, C is 2x4.
    Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3).
    Split B into B1 (2x2, rows 0,1) and B2 (2x2, rows 2,3).
    C = A1*B1 + A2*B2.
    Each 2x2*2x2 multiplication uses Strassen's algorithm (rank 7).
    Total rank = 4 * 7 = 28.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    dim_a = n * m  # 8
    dim_b = m * p  # 16
    dim_c = p * n  # 8
    
    # Target triples for verification
    target_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target_triples.add((a_idx, b_idx, c_idx))

    # Strassen 2x2x2 decomposition
    # Inputs: X (2x2), Y (2x2)
    # Outputs: Z = X * Y (2x2)
    # Indices:
    # X: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Y: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Z: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # Strassen formulas:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X11 - X10) * (Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    # We need to map this to the global indices.
    # Global A indices: i*m + j. 
    # For block 1 (j=0,1; k=0,1):
    #   A1 is A[:, 0:2]. Global A indices:
    #     X00 -> A[0,0] -> 0
    #     X01 -> A[0,1] -> 1
    #     X10 -> A[1,0] -> 4
    #     X11 -> A[1,1] -> 5
    #   B1 is B[0:2, 0:2]. Global B indices:
    #     Y00 -> B[0,0] -> 0
    #     Y01 -> B[0,1] -> 1
    #     Y10 -> B[1,0] -> 4
    #     Y11 -> B[1,1] -> 5
    #   Z1 is C[:, 0:2]. Global C indices (k*n + i):
    #     Z00 -> C[0,0] -> 0
    #     Z01 -> C[0,1] -> 1
    #     Z10 -> C[1,0] -> 2
    #     Z11 -> C[1,1] -> 3

    # For block 2 (j=0,1; k=2,3):
    #   A1 is same A[:, 0:2]. Global A indices: 0, 1, 4, 5
    #   B2 is B[0:2, 2:4]. Global B indices:
    #     Y00 -> B[0,2] -> 2
    #     Y01 -> B[0,3] -> 3
    #     Y10 -> B[1,2] -> 6
    #     Y11 -> B[1,3] -> 7
    #   Z2 is C[:, 2:4]. Global C indices:
    #     Z00 -> C[2,0] -> 4
    #     Z01 -> C[2,1] -> 5
    #     Z10 -> C[3,0] -> 6
    #     Z11 -> C[3,1] -> 7

    # For block 3 (j=2,3; k=0,1):
    #   A2 is A[:, 2:4]. Global A indices:
    #     X00 -> A[0,2] -> 2
    #     X01 -> A[0,3] -> 3
    #     X10 -> A[1,2] -> 6
    #     X11 -> A[1,3] -> 7
    #   B1 is B[0:2, 0:2]. Global B indices: 0, 1, 4, 5
    #   Z3 is C[:, 0:2]. Global C indices: 0, 1, 2, 3

    # For block 4 (j=2,3; k=2,3):
    #   A2 is A[:, 2:4]. Global A indices: 2, 3, 6, 7
    #   B2 is B[0:2, 2:4]. Global B indices: 2, 3, 6, 7
    #   Z4 is C[:, 2:4]. Global C indices: 4, 5, 6, 7

    # Define Strassen terms for a generic 2x2x2 block
    # Returns list of 7 tuples: (coeff_A, coeff_B, coeff_C) for each M
    # Actually, we need to construct U, V, W columns.
    # Each Strassen term M_k corresponds to one rank-1 term in the decomposition.
    # U[A_idx] = coeff_A, V[B_idx] = coeff_B, W[C_idx] = coeff_C (but W is summed)
    # Wait, the tensor is sum_t U[a,t] V[b,t] W[c,t].
    # For a specific Strassen term M, it contributes to Z.
    # Z = X * Y.
    # The term M1 = (X00+X11)(Y00+Y11) contributes to Z00, Z11.
    # Specifically, Z00 gets +1 * M1, Z11 gets +1 * M1.
    # So for the rank-1 term corresponding to M1:
    # U has non-zeros at X00, X11 with value 1.
    # V has non-zeros at Y00, Y11 with value 1.
    # W has non-zeros at Z00, Z11 with value 1.
    
    # Let's define the 7 Strassen terms for a block.
    # Each term is a set of (local_A_indices, local_B_indices, local_C_indices) with coefficients.
    # Since we are building U, V, W as matrices, each column t corresponds to one term.
    # For a term, U[a,t] is the coefficient for A element a.
    
    def get_strassen_terms(a_map, b_map, c_map):
        """
        a_map: dict {local_idx: global_idx} for A (X)
        b_map: dict {local_idx: global_idx} for B (Y)
        c_map: dict {local_idx: global_idx} for C (Z)
        Returns list of 7 dicts: {global_A: coeff, global_B: coeff, global_C: coeff}
        """
        terms = []
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        # Contributes to Z00 (+1), Z11 (+1)
        t = {}
        t[a_map[0]] = 1; t[a_map[3]] = 1
        t[b_map[0]] = 1; t[b_map[3]] = 1
        t[c_map[0]] = 1; t[c_map[3]] = 1
        terms.append(t)
        
        # M2 = (X10 + X11) * Y00
        # Contributes to Z10 (+1), Z11 (-1)
        t = {}
        t[a_map[2]] = 1; t[a_map[3]] = 1
        t[b_map[0]] = 1
        t[c_map[2]] = 1; t[c_map[3]] = -1
        terms.append(t)
        
        # M3 = X00 * (Y01 - Y11)
        # Contributes to Z01 (+1), Z11 (+1)
        t = {}
        t[a_map[0]] = 1
        t[b_map[1]] = 1; t[b_map[3]] = -1
        t[c_map[1]] = 1; t[c_map[3]] = 1
        terms.append(t)
        
        # M4 = X11 * (Y10 - Y00)
        # Contributes to Z00 (+1), Z10 (+1)
        t = {}
        t[a_map[3]] = 1
        t[b_map[2]] = 1; t[b_map[0]] = -1
        t[c_map[0]] = 1; t[c_map[2]] = 1
        terms.append(t)
        
        # M5 = (X00 + X01) * Y11
        # Contributes to Z00 (-1), Z01 (+1)
        t = {}
        t[a_map[0]] = 1; t[a_map[1]] = 1
        t[b_map[3]] = 1
        t[c_map[0]] = -1; t[c_map[1]] = 1
        terms.append(t)
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        # Contributes to Z11 (+1)
        t = {}
        t[a_map[2]] = 1; t[a_map[0]] = -1
        t[b_map[0]] = 1; t[b_map[1]] = 1
        t[c_map[3]] = 1
        terms.append(t)
        
        # M7 = (X11 - X10) * (Y10 + Y11)
        # Contributes to Z00 (+1)
        t = {}
        t[a_map[3]] = 1; t[a_map[2]] = -1
        t[b_map[2]] = 1; t[b_map[3]] = 1
        t[c_map[0]] = 1
        terms.append(t)
        
        return terms

    # Define maps for the 4 blocks
    # Block 1: A1 (cols 0,1), B1 (rows 0,1), C1 (cols 0,1)
    a_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    b_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    c_map_1 = {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Block 2: A1 (cols 0,1), B2 (rows 2,3), C2 (cols 2,3)
    a_map_2 = {0: 0, 1: 1, 2: 4, 3: 5}
    b_map_2 = {0: 2, 1: 3, 2: 6, 3: 7}
    c_map_2 = {0: 4, 1: 5, 2: 6, 3: 7}
    
    # Block 3: A2 (cols 2,3), B1 (rows 0,1), C1 (cols 0,1)
    a_map_3 = {0: 2, 1: 3, 2: 6, 3: 7}
    b_map_3 = {0: 0, 1: 1, 2: 4, 3: 5}
    c_map_3 = {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Block 4: A2 (cols 2,3), B2 (rows 2,3), C2 (cols 2,3)
    a_map_4 = {0: 2, 1: 3, 2: 6, 3: 7}
    b_map_4 = {0: 2, 1: 3, 2: 6, 3: 7}
    c_map_4 = {0: 4, 1: 5, 2: 6, 3: 7}

    all_terms = []
    all_terms.extend(get_strassen_terms(a_map_1, b_map_1, c_map_1))
    all_terms.extend(get_strassen_terms(a_map_2, b_map_2, c_map_2))
    all_terms.extend(get_strassen_terms(a_map_3, b_map_3, c_map_3))
    all_terms.extend(get_strassen_terms(a_map_4, b_map_4, c_map_4))
    
    rank = len(all_terms)
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(dim_a)]
    V = [[0] * rank for _ in range(dim_b)]
    W = [[0] * rank for _ in range(dim_c)]
    
    for t_idx, term in enumerate(all_terms):
        for idx, coeff in term.items():
            if idx < dim_a:
                U[idx][t_idx] = coeff
            elif idx < dim_a + dim_b:
                # This logic is wrong, indices are separate namespaces.
                pass
            
            # We need to distinguish A, B, C indices.
            # The term dict mixes them. Let's separate them during generation.
            
    # Let's redo the term generation to return separate dicts for U, V, W
    def get_strassen_terms_separated(a_map, b_map, c_map):
        terms = []
        
        # M1
        u = {a_map[0]: 1, a_map[3]: 1}
        v = {b_map[0]: 1, b_map[3]: 1}
        w = {c_map[0]: 1, c_map[3]: 1}
        terms.append((u, v, w))
        
        # M2
        u = {a_map[2]: 1, a_map[3]: 1}
        v = {b_map[0]: 1}
        w = {c_map[2]: 1, c_map[3]: -1}
        terms.append((u, v, w))
        
        # M3
        u = {a_map[0]: 1}
        v = {b_map[1]: 1, b_map[3]: -1}
        w = {c_map[1]: 1, c_map[3]: 1}
        terms.append((u, v, w))
        
        # M4
        u = {a_map[3]: 1}
        v = {b_map[2]: 1, b_map[0]: -1}
        w = {c_map[0]: 1, c_map[2]: 1}
        terms.append((u, v, w))
        
        # M5
        u = {a_map[0]: 1, a_map[1]: 1}
        v = {b_map[3]: 1}
        w = {c_map[0]: -1, c_map[1]: 1}
        terms.append((u, v, w))
        
        # M6
        u = {a_map[2]: 1, a_map[0]: -1}
        v = {b_map[0]: 1, b_map[1]: 1}
        w = {c_map[3]: 1}
        terms.append((u, v, w))
        
        # M7
        u = {a_map[3]: 1, a_map[2]: -1}
        v = {b_map[2]: 1, b_map[3]: 1}
        w = {c_map[0]: 1}
        terms.append((u, v, w))
        
        return terms

    all_terms = []
    all_terms.extend(get_strassen_terms_separated(a_map_1, b_map_1, c_map_1))
    all_terms.extend(get_strassen_terms_separated(a_map_2, b_map_2, c_map_2))
    all_terms.extend(get_strassen_terms_separated(a_map_3, b_map_3, c_map_3))
    all_terms.extend(get_strassen_terms_separated(a_map_4, b_map_4, c_map_4))
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(dim_a)]
    V = [[0] * rank for _ in range(dim_b)]
    W = [[0] * rank for _ in range(dim_c)]
    
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for a_idx, coeff in u_dict.items():
            U[a_idx][t_idx] = coeff
        for b_idx, coeff in v_dict.items():
            V[b_idx][t_idx] = coeff
        for c_idx, coeff in w_dict.items():
            W[c_idx][t_idx] = coeff
            
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check all target triples
    for a_idx in range(dim_a):
        for b_idx in range(dim_b):
            for c_idx in range(dim_c):
                val = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx])
                if (a_idx, b_idx, c_idx) in target_triples:
                    if val != 1:
                        return {"U": [[0]*32 for _ in range(dim_a)], "V": [[0]*32 for _ in range(dim_b)], "W": [[0]*32 for _ in range(dim_c)]}
                else:
                    if val != 0:
                        return {"U": [[0]*32 for _ in range(dim_a)], "V": [[0]*32 for _ in range(dim_b)], "W": [[0]*32 for _ in range(dim_c)]}
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
