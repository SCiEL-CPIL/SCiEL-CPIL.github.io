"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # We will build a decomposition using:
    # 1. Split the middle dimension m=3 into m1=1 and m2=2.
    # 2. For m1=1 part: A is 2x1, B is 1x8. C is 2x8.
    #    C[i,k] = A[i,0] * B[0,k].
    #    This requires 2*8 = 16 terms? No, it's a rank 1 tensor in (i,k) for each product?
    #    Actually, for fixed j=0, C[i,k] = sum_j A[i,j]B[j,k] = A[i,0]B[0,k].
    #    This is an outer product of vector A[:,0] and vector B[0,:].
    #    The tensor T_{i,k} = A_i B_k. This can be decomposed into rank 1?
    #    Wait, the tensor is T_{i,j,k}.
    #    For j=0, T_{i,0,k} = A_{i,0} B_{0,k}.
    #    This is a rank 1 tensor in indices (i,k).
    #    So we need 1 term? No, the "term" in matrix multiplication context corresponds to a scalar product.
    #    The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,k} ... no, = C_{i,k}.
    #    For the m1=1 block:
    #    We need to compute C_{i,k} = A_{i,0} B_{0,k} for i in {0,1}, k in {0..7}.
    #    This is 16 distinct products.
    #    Can we do better?
    #    If we view this as multiplying a 2x1 matrix by a 1x8 matrix, the rank is 2*1*8 = 16?
    #    Yes, because each output entry C_{i,k} depends on a unique pair (A_{i,0}, B_{0,k}).
    #    There is no sharing between different i or different k.
    #    So rank 16 is required for this part.
    
    # 3. For m2=2 part: A is 2x2, B is 2x8. C is 2x8.
    #    Split B into 4 blocks of 2x2.
    #    Each block multiplication is 2x2 by 2x2.
    #    We can use Strassen's algorithm for 2x2x2, which has rank 7.
    #    Total rank for this part: 4 blocks * 7 terms/block = 28 terms.
    
    # Total rank = 16 + 28 = 44.
    # Current schoolbook rank is 2*3*8 = 48.
    # So 44 < 48. This is an improvement.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 44
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 44
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 44
    
    # Helper to index U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # --- Part 1: m1=1 (j=0) ---
    # A is 2x1 (indices i=0,1; j=0)
    # B is 1x8 (indices j=0; k=0..7)
    # C is 2x8 (indices i=0,1; k=0..7)
    # C[i,k] = A[i,0] * B[0,k]
    # Each (i,k) pair is a separate term.
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 0), r] = 1
            V[v_idx(0, k), r] = 1
            W[w_idx(k, i), r] = 1
            r += 1
    
    # --- Part 2: m2=2 (j=1,2) ---
    # A is 2x2 (indices i=0,1; j=1,2)
    # B is 2x8 (indices j=1,2; k=0..7)
    # We split B into 4 blocks of columns: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For each block b, B_block is 2x2.
    # We use Strassen for each 2x2x2 multiplication.
    
    # Strassen coefficients for 2x2x2:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map local indices (a,b,c,d,e,f,g,h) to global indices.
    # Local A:
    # a = A[0,1] (global i=0, j=1)
    # b = A[0,2] (global i=0, j=2)
    # c = A[1,1] (global i=1, j=1)
    # d = A[1,2] (global i=1, j=2)
    
    # Local B for block b (covering k0, k1):
    # e = B[1, k0] (global j=1, k=k0)
    # f = B[1, k1] (global j=1, k=k1)
    # g = B[2, k0] (global j=2, k=k0)
    # h = B[2, k1] (global j=2, k=k1)
    
    # Local C for block b (covering i=0,1 and k0, k1):
    # C00 = C[0, k0] (global i=0, k=k0)
    # C01 = C[0, k1] (global i=0, k=k1)
    # C10 = C[1, k0] (global i=1, k=k0)
    # C11 = C[1, k1] (global i=1, k=k1)
    
    # Strassen terms definitions (U, V, W contributions for each M)
    # Each M is a product of a linear combo of A and a linear combo of B.
    # U vector: coefficients for A entries (a,b,c,d)
    # V vector: coefficients for B entries (e,f,g,h)
    # W vector: coefficients for C entries (C00, C01, C10, C11)
    
    # M1 = (a+d)(e+h)
    # U: a=1, d=1
    # V: e=1, h=1
    # W: C00=1, C11=1
    
    # M2 = (c+d)e
    # U: c=1, d=1
    # V: e=1
    # W: C10=1, C11=-1
    
    # M3 = a(f-h)
    # U: a=1
    # V: f=1, h=-1
    # W: C01=1, C11=1
    
    # M4 = d(g-e)
    # U: d=1
    # V: g=1, e=-1
    # W: C00=1, C10=1
    
    # M5 = (a+b)h
    # U: a=1, b=1
    # V: h=1
    # W: C00=-1, C01=1
    
    # M6 = (c-a)(e+f)
    # U: c=1, a=-1
    # V: e=1, f=1
    # W: C11=1
    
    # M7 = (b-d)(g+h)
    # U: b=1, d=-1
    # V: g=1, h=1
    # W: C00=1
    
    # Let's define these as lists of (local_index, coeff)
    # Local A indices: 0:a, 1:b, 2:c, 3:d
    # Local B indices: 0:e, 1:f, 2:g, 3:h
    # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict {local_a_idx: coeff}
        # V_coeffs: dict {local_b_idx: coeff}
        # W_coeffs: dict {local_c_idx: coeff}
        
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Map local indices to global indices for A, B, C
    # A local 0 (a) -> global i=0, j=1
    # A local 1 (b) -> global i=0, j=2
    # A local 2 (c) -> global i=1, j=1
    # A local 3 (d) -> global i=1, j=2
    a_local_to_global = {
        0: (0, 1),
        1: (0, 2),
        2: (1, 1),
        3: (1, 2)
    }
    
    # B local 0 (e) -> global j=1, k=k0
    # B local 1 (f) -> global j=1, k=k1
    # B local 2 (g) -> global j=2, k=k0
    # B local 3 (h) -> global j=2, k=k1
    # These depend on the block b (k0, k1)
    
    # C local 0 (C00) -> global i=0, k=k0
    # C local 1 (C01) -> global i=0, k=k1
    # C local 2 (C10) -> global i=1, k=k0
    # C local 3 (C11) -> global i=1, k=k1
    # These depend on the block b (k0, k1)
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b_local_to_global = {
            0: (1, k0),  # e
            1: (1, k1),  # f
            2: (2, k0),  # g
            3: (2, k1)   # h
        }
        
        c_local_to_global = {
            0: (0, k0),  # C00
            1: (0, k1),  # C01
            2: (1, k0),  # C10
            3: (1, k1)   # C11
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_a, coeff in u_coeffs.items():
                gi, gj = a_local_to_global[loc_a]
                U[u_idx(gi, gj), r] += coeff
            
            for loc_b, coeff in v_coeffs.items():
                gj, gk = b_local_to_global[loc_b]
                V[v_idx(gj, gk), r] += coeff
            
            for loc_c, coeff in w_coeffs.items():
                gi, gk = c_local_to_global[loc_c]
                W[w_idx(gk, gi), r] += coeff
            
            r += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
