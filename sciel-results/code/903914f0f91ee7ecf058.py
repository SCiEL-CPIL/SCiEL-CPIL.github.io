import numpy as np
from collections import defaultdict

def construct():
    n, m, p = 3, 4, 7
    
    U_dict = defaultdict(float)
    V_dict = defaultdict(float)
    W_dict = defaultdict(float)
    
    rank = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal rank
        U_dict[(u_idx, rank)] += coeff
        V_dict[(v_idx, rank)] += coeff
        W_dict[(w_idx, rank)] += coeff
        rank += 1
    
    # Helper to add trivial terms for a block
    def add_trivial(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term(u_idx, v_idx, w_idx, 1.0)
    
    # Helper to add Strassen terms for a 2x2x2 block
    # Indices: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    def add_strassen(i0, j0, k0):
        # The 7 products for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        
        # Term 1: M1
        # U: (A00+A11) -> u00=1, u11=1
        # V: (B00+B11) -> v00=1, v11=1
        # W: C00 += 1, C11 += 1
        add_term(i0*m + j0, k0*p + j0, i0*n + k0, 1.0) # This is not right, U index is (i,j)
        
        # Let's be precise with indices.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # U contributions: (i0, j0) coeff 1, (i0+1, j0+1) coeff 1
        # V contributions: (j0, k0) coeff 1, (j0+1, k0+1) coeff 1
        # W contributions:
        # C[i0, k0] += 1
        # C[i0+1, k0+1] += 1
        
        # We need to add a single term to the decomposition.
        # The term is a rank-1 tensor.
        # U vector has non-zeros at (i0, j0) and (i0+1, j0+1).
        # V vector has non-zeros at (j0, k0) and (j0+1, k0+1).
        # W vector has non-zeros at (k0, i0) and (k0+1, i0+1).
        
        # My add_term function only adds to ONE index in U, V, W.
        # I need to modify the approach or the helper.
        
        pass

    # The previous helper `add_term` was too simplistic for Strassen which involves sums in U and V.
    # Let's rewrite the logic to directly populate the dictionaries for the whole block.
    
    U_dict.clear()
    V_dict.clear()
    W_dict.clear()
    rank = 0
    
    def add_rank1_term(u_indices, v_indices, w_indices):
        """
        u_indices: list of (idx, coeff)
        v_indices: list of (idx, coeff)
        w_indices: list of (idx, coeff)
        """
        nonlocal rank
        for idx, coeff in u_indices:
            U_dict[(idx, rank)] += coeff
        for idx, coeff in v_indices:
            V_dict[(idx, rank)] += coeff
        for idx, coeff in w_indices:
            W_dict[(idx, rank)] += coeff
        rank += 1

    def add_trivial_block(i0, i1, j0, j1, k0, k1):
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_rank1_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

    def add_strassen_block(i0, j0, k0):
        # i0, i0+1 ; j0, j0+1 ; k0, k0+1
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [(i0*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)]
        v1 = [(j0*p+k0, 1.0), ((j0+1)*p+(k0+1), 1.0)]
        w1 = [(k0*n+i0, 1.0), ((k0+1)*n+(i0+1), 1.0)]
        add_rank1_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00
        u2 = [((i0+1)*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)]
        v2 = [(j0*p+k0, 1.0)]
        w2 = [((k0)*n+(i0+1), 1.0)] # C10 += M2
        add_rank1_term(u2, v2, w2)
        
        # M3 = A00(B10-B11)
        u3 = [(i0*m+j0, 1.0)]
        v3 = [((j0+1)*p+k0, 1.0), ((j0+1)*p+(k0+1), -1.0)]
        w3 = [((k0)*n+i0, 1.0)] # C00? No.
        # C01 = M3 + M5
        # C[i0, k0+1] += M3
        w3 = [((k0+1)*n+i0, 1.0)]
        add_rank1_term(u3, v3, w3)
        
        # M4 = A11(B01-B00)
        u4 = [((i0+1)*m+(j0+1), 1.0)]
        v4 = [(j0*p+(k0+1), 1.0), (j0*p+k0, -1.0)]
        # C10 += M4, C00 -= M4
        # This term contributes to multiple W entries?
        # Wait, a rank-1 term has a single W vector.
        # M4 is a scalar. It contributes to C10 with +1 and C00 with -1.
        # So W vector has two non-zeros.
        w4 = [((k0)*n+(i0+1), 1.0), (k0*n+i0, -1.0)]
        add_rank1_term(u4, v4, w4)
        
        # M5 = (A00+A01)B11
        u5 = [(i0*m+j0, 1.0), (i0*m+(j0+1), 1.0)]
        v5 = [((j0+1)*p+(k0+1), 1.0)]
        # C00 += M5, C01 += M5
        w5 = [(k0*n+i0, 1.0), ((k0+1)*n+i0, 1.0)]
        add_rank1_term(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B10)
        u6 = [((i0+1)*m+j0, 1.0), (i0*m+j0, -1.0)]
        v6 = [(j0*p+k0, 1.0), ((j0+1)*p+k0, 1.0)]
        # C11 += M6
        w6 = [((k0+1)*n+(i0+1), 1.0)]
        add_rank1_term(u6, v6, w6)
        
        # M7 = (A01-A11)(B01+B11)
        u7 = [(i0*m+(j0+1), 1.0), ((i0+1)*m+(j0+1), -1.0)]
        v7 = [(j0*p+(k0+1), 1.0), ((j0+1)*p+(k0+1), 1.0)]
        # C00 -= M7, C11 += M7
        w7 = [(k0*n+i0, -1.0), ((k0+1)*n+(i0+1), 1.0)]
        add_rank1_term(u7, v7, w7)

    # Partitioning strategy:
    # We have n=3, m=4, p=7.
    # We want to use Strassen on 2x2x2 blocks where possible.
    
    # Let's divide the indices into blocks.
    # i: 0, 1, 2
    # j: 0, 1, 2, 3
    # k: 0, 1, 2, 3, 4, 5, 6
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (7)
    add_strassen_block(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1} -> Strassen (7)
    add_strassen_block(0, 2, 0)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen (7)
    add_strassen_block(0, 0, 2)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3} -> Strassen (7)
    add_strassen_block(0, 2, 2)
    
    # Remaining k: 4, 5, 6.
    # i in {0,1}, j in {0,1}, k in {4,5,6} -> Trivial (2*2*3 = 12)
    add_trivial_block(0, 2, 0, 2, 4, 7)
    
    # i in {0,1}, j in {2,3}, k in {4,5,6} -> Trivial (2*2*3 = 12)
    add_trivial_block(0, 2, 2, 4, 4, 7)
    
    # Remaining i: 2.
    # i=2, j in {0,1}, k in {0,1} -> Trivial (1*2*2 = 4)
    add_trivial_block(2, 3, 0, 2, 0, 2)
    
    # i=2, j in {2,3}, k in {0,1} -> Trivial (1*2*2 = 4)
    add_trivial_block(2, 3, 2, 4, 0, 2)
    
    # i=2, j in {0,1}, k in {2,3} -> Trivial (1*2*2 = 4)
    add_trivial_block(2, 3, 0, 2, 2, 4)
    
    # i=2, j in {2,3}, k in {2,3} -> Trivial (1*2*2 = 4)
    add_trivial_block(2, 3, 2, 4, 2, 4)
    
    # i=2, j in {0,1}, k in {4,5,6} -> Trivial (1*2*3 = 6)
    add_trivial_block(2, 3, 0, 2, 4, 7)
    
    # i=2, j in {2,3}, k in {4,5,6} -> Trivial (1*2*3 = 6)
    add_trivial_block(2, 3, 2, 4, 4, 7)
    
    # Total Rank:
    # 4 Strassen blocks * 7 = 28
    # Trivial blocks:
    # 12 + 12 + 4 + 4 + 4 + 4 + 6 + 6 = 52
    # Total = 28 + 52 = 80.
    
    # This is an improvement over 84.
    
    # Convert dictionaries to arrays
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for (idx, r), val in U_dict.items():
        U[idx, r] = val
    for (idx, r), val in V_dict.items():
        V[idx, r] = val
    for (idx, r), val in W_dict.items():
        W[idx, r] = val
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
