import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Target tensor dimensions
    dim_a = n * m  # 16
    dim_b = m * p  # 20
    dim_c = p * n  # 20
    
    U = []
    V = []
    W = []
    
    # --- Part 1: 4x4 x 4x4 core (cols 0-3 of C) ---
    # Rank 49 using Strassen
    
    # Strassen 7 products for 4x4
    # A, B, C are 4x4.
    # Indices for A: i*4+j
    # Indices for B: j*5+k (we use k=0..3)
    # Indices for C: k*4+i (we use k=0..3)
    
    # Local 2x2 Strassen decomposition vectors (7 terms)
    # u, v, w are length 4 vectors for the 2x2 blocks
    strassen_local = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])),
        (np.array([1, 0, -1, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 1, 0]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        (np.array([0, 1, -1, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0])),
    ]
    
    # Define the 7 Strassen products for 4x4
    # Each product is defined by:
    # A_map: 4x4 matrix mapping global A indices to local 2x2 A' indices?
    # Actually, it's easier to define which global indices contribute to each local index with what coefficient.
    
    # Local indices: 0,1,2,3 correspond to 00, 01, 10, 11 of the 2x2 block.
    
    # Product 1: M1 = (A00+A11)(B00+B11)
    # A' = A00+A11 at 00,11? No.
    # A' is a 2x2 matrix.
    # A'00 = A00 + A11
    # A'01 = 0
    # A'10 = 0
    # A'11 = 0
    # Wait, Strassen multiplies two 2x2 matrices.
    # The "A" matrix for M1 is [[A00+A11, 0], [0, 0]]? No.
    # M1 = (A00+A11)(B00+B11). This is a product of two SCALARS? No, A00 is a 2x2 matrix.
    # So A' is the 2x2 matrix [[A00+A11, 0], [0, 0]]? No.
    # Strassen computes M1 = (A00+A11)(B00+B11).
    # Here A00, A11 are 2x2 matrices.
    # So A' is the 2x2 matrix formed by summing the blocks?
    # No, A00+A11 is a 2x2 matrix. Let's call it S_A.
    # B00+B11 is a 2x2 matrix. Let's call it S_B.
    # M1 = S_A * S_B.
    # So the "local A" is S_A, and "local B" is S_B.
    # S_A is a 2x2 matrix. Its entries are sums of entries of A00 and A11.
    # S_A[0,0] = A00[0,0] + A11[0,0]
    # S_A[0,1] = A00[0,1] + A11[0,1]
    # etc.
    
    # So for each Strassen product, we define:
    # A_coeffs: list of (global_idx, coeff) for each local entry (0,1,2,3)
    # B_coeffs: list of (global_idx, coeff) for each local entry (0,1,2,3)
    # C_coeffs: list of (global_idx, coeff) for each local entry (0,1,2,3)
    
    # Helper to get global index
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    # Product 1: M1 = (A00+A11)(B00+B11)
    # Local A:
    # 00: A00[0,0]+A11[0,0], A00[0,1]+A11[0,1], A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]
    # This means for local entry 0 (top-left of S_A), we have coeffs 1 for a_idx(0,0) and a_idx(2,0).
    # Wait, local entry 0 corresponds to S_A[0,0].
    # S_A[0,0] = A[0,0] + A[2,0].
    # So A_coeffs[0] = [(a_idx(0,0), 1), (a_idx(2,0), 1)]
    
    # Let's build the maps for the 7 products.
    
    products = []
    
    # 1. (A00+A11)(B00+B11)
    # A: 00->00,11; 01->01,11; 10->10,11; 11->11,11
    # B: 00->00,11; 01->01,11; 10->10,11; 11->11,11
    # C: 00->00,11; 01->01,11; 10->10,11; 11->11,11
    # Note: C contribution is to C00 and C11 blocks?
    # M1 contributes to C00 and C11.
    # C00 = M1 - M2 + M4 - M5
    # C11 = M1 - M3 - M4 + M6
    # So for M1, the C part is C00 + C11?
    # Yes, the local C for M1 is C00+C11.
    # Local C entry 00 corresponds to C00[0,0] + C11[0,0].
    
    def make_map(block1, block2, sign1=1, sign2=1):
        # block1, block2 are tuples (i0, j0)
        # Returns coeffs for local 0,1,2,3
        # Local 0: (i0, j0) of block1 + (i0, j0) of block2
        # This is for A or B.
        pass
        
    # It's easier to just hardcode the 7 products' index maps.
    
    # Product 1: A=(A00+A11), B=(B00+B11), C=(C00+C11)
    # A maps:
    # 0: a(0,0)+a(2,0)
    # 1: a(0,1)+a(2,1)
    # 2: a(1,0)+a(3,0)
    # 3: a(1,1)+a(3,1)
    # B maps:
    # 0: b(0,0)+b(2,0)
    # 1: b(0,1)+b(2,1)
    # 2: b(1,0)+b(3,0)
    # 3: b(1,1)+b(3,1)
    # C maps:
    # 0: c(0,0)+c(2,0)
    # 1: c(1,0)+c(3,0)
    # 2: c(0,1)+c(2,1)
    # 3: c(1,1)+c(3,1)
    
    # Product 2: A=(A10+A11), B=B00, C=C10
    # A maps:
    # 0: a(2,0)+a(0,0)? No. A10 is rows 2-3, cols 0-1. A11 is rows 2-3, cols 2-3.
    # A' = A10+A11.
    # A'[0,0] = A[2,0]+A[2,2]
    # A'[0,1] = A[2,1]+A[2,3]
    # A'[1,0] = A[3,0]+A[3,2]
    # A'[1,1] = A[3,1]+A[3,3]
    # B maps: B00.
    # 0: b(0,0)
    # 1: b(0,1)
    # 2: b(1,0)
    # 3: b(1,1)
    # C maps: C10.
    # 0: c(0,2)
    # 1: c(1,2)
    # 2: c(0,3)
    # 3: c(1,3)
    
    # We'll implement a helper to generate the global vectors for each of the 7 local terms.
    
    def add_strassen_product(a_map, b_map, c_map):
        """
        a_map, b_map, c_map: lists of 4 lists.
        a_map[l] is a list of (global_idx, coeff) for local index l.
        """
        for r in range(7):
            u_local, v_local, w_local = strassen_local[r]
            
            u_global = np.zeros(dim_a)
            v_global = np.zeros(dim_b)
            w_global = np.zeros(dim_c)
            
            for l in range(4):
                coeff_u = u_local[l]
                coeff_v = v_local[l]
                coeff_w = w_local[l]
                
                for idx, c in a_map[l]:
                    u_global[idx] += coeff_u * c
                for idx, c in b_map[l]:
                    v_global[idx] += coeff_v * c
                for idx, c in c_map[l]:
                    w_global[idx] += coeff_w * c
            
            U.append(u_global)
            V.append(v_global)
            W.append(w_global)
            
    # Define maps for 7 products
    # 1. (A00+A11)(B00+B11) -> C00+C11
    a1 = [
        [(a_idx(0,0), 1), (a_idx(2,0), 1)],
        [(a_idx(0,1), 1), (a_idx(2,1), 1)],
        [(a_idx(1,0), 1), (a_idx(3,0), 1)],
        [(a_idx(1,1), 1), (a_idx(3,1), 1)]
    ]
    b1 = [
        [(b_idx(0,0), 1), (b_idx(2,0), 1)],
        [(b_idx(0,1), 1), (b_idx(2,1), 1)],
        [(b_idx(1,0), 1), (b_idx(3,0), 1)],
        [(b_idx(1,1), 1), (b_idx(3,1), 1)]
    ]
    c1 = [
        [(c_idx(0,0), 1), (c_idx(2,0), 1)],
        [(c_idx(1,0), 1), (c_idx(3,0), 1)],
        [(c_idx(0,1), 1), (c_idx(2,1), 1)],
        [(c_idx(1,1), 1), (c_idx(3,1), 1)]
    ]
    add_strassen_product(a1, b1, c1)
    
    # 2. (A10+A11)B00 -> C10
    a2 = [
        [(a_idx(2,0), 1), (a_idx(2,2), 1)],
        [(a_idx(2,1), 1), (a_idx(2,3), 1)],
        [(a_idx(3,0), 1), (a_idx(3,2), 1)],
        [(a_idx(3,1), 1), (a_idx(3,3), 1)]
    ]
    b2 = [
        [(b_idx(0,0), 1)],
        [(b_idx(0,1), 1)],
        [(b_idx(1,0), 1)],
        [(b_idx(1,1), 1)]
    ]
    c2 = [
        [(c_idx(0,2), 1)],
        [(c_idx(1,2), 1)],
        [(c_idx(0,3), 1)],
        [(c_idx(1,3), 1)]
    ]
    add_strassen_product(a2, b2, c2)
    
    # 3. A00(B10+B11) -> C00
    a3 = [
        [(a_idx(0,0), 1)],
        [(a_idx(0,1), 1)],
        [(a_idx(1,0), 1)],
        [(a_idx(1,1), 1)]
    ]
    b3 = [
        [(b_idx(2,0), 1), (b_idx(2,2), 1)],
        [(b_idx(2,1), 1), (b_idx(2,3), 1)],
        [(b_idx(3,0), 1), (b_idx(3,2), 1)],
        [(b_idx(3,1), 1), (b_idx(3,3), 1)]
    ]
    c3 = [
        [(c_idx(0,0), 1)],
        [(c_idx(1,0), 1)],
        [(c_idx(0,1), 1)],
        [(c_idx(1,1), 1)]
    ]
    add_strassen_product(a3, b3, c3)
    
    # 4. (A00-A10)B01 -> C01
    a4 = [
        [(a_idx(0,0), 1), (a_idx(2,0), -1)],
        [(a_idx(0,1), 1), (a_idx(2,1), -1)],
        [(a_idx(1,0), 1), (a_idx(3,0), -1)],
        [(a_idx(1,1), 1), (a_idx(3,1), -1)]
    ]
    b4 = [
        [(b_idx(0,2), 1)],
        [(b_idx(0,3), 1)],
        [(b_idx(1,2), 1)],
        [(b_idx(1,3), 1)]
    ]
    c4 = [
        [(c_idx(2,0), 1)],
        [(c_idx(3,0), 1)],
        [(c_idx(2,1), 1)],
        [(c_idx(3,1), 1)]
    ]
    add_strassen_product(a4, b4, c4)
    
    # 5. (A01+A10)B00 -> C10
    a5 = [
        [(a_idx(0,2), 1), (a_idx(2,0), 1)],
        [(a_idx(0,3), 1), (a_idx(2,1), 1)],
        [(a_idx(1,2), 1), (a_idx(3,0), 1)],
        [(a_idx(1,3), 1), (a_idx(3,1), 1)]
    ]
    b5 = b2
    c5 = c2
    add_strassen_product(a5, b5, c5)
    
    # 6. (A01-A10)B11 -> C11
    a6 = [
        [(a_idx(0,2), 1), (a_idx(2,0), -1)],
        [(a_idx(0,3), 1), (a_idx(2,1), -1)],
        [(a_idx(1,2), 1), (a_idx(3,0), -1)],
        [(a_idx(1,3), 1), (a_idx(3,1), -1)]
    ]
    b6 = [
        [(b_idx(2,2), 1)],
        [(b_idx(2,3), 1)],
        [(b_idx(3,2), 1)],
        [(b_idx(3,3), 1)]
    ]
    c6 = [
        [(c_idx(2,2), 1)],
        [(c_idx(3,2), 1)],
        [(c_idx(2,3), 1)],
        [(c_idx(3,3), 1)]
    ]
    add_strassen_product(a6, b6, c6)
    
    # 7. (A01+A11)B10 -> C01
    a7 = [
        [(a_idx(0,2), 1), (a_idx(2,2), 1)],
        [(a_idx(0,3), 1), (a_idx(2,3), 1)],
        [(a_idx(1,2), 1), (a_idx(3,2), 1)],
        [(a_idx(1,3), 1), (a_idx(3,3), 1)]
    ]
    b7 = [
        [(b_idx(2,0), 1)],
        [(b_idx(2,1), 1)],
        [(b_idx(3,0), 1)],
        [(b_idx(3,1), 1)]
    ]
    c7 = [
        [(c_idx(2,0), 1)],
        [(c_idx(3,0), 1)],
        [(c_idx(2,1), 1)],
        [(c_idx(3,1), 1)]
    ]
    add_strassen_product(a7, b7, c7)
    
    # --- Part 2: 4x4 x 4x1 boundary (col 4 of C) ---
    # Rank 16 schoolbook
    for i in range(n):
        for j in range(m):
            u_vec = np.zeros(dim_a)
            v_vec = np.zeros(dim_b)
            w_vec = np.zeros(dim_c)
            
            a = i * m + j
            b = j * p + 4
            c = 4 * n + i
            
            u_vec[a] = 1
            v_vec[b] = 1
            w_vec[c] = 1
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
    return {
        "U": np.array(U).T.tolist(),
        "V": np.array(V).T.tolist(),
        "W": np.array(W).T.tolist()
    }
