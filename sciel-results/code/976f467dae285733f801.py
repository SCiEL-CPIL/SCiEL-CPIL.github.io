import time
import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    dim_A = n * m  # 4
    dim_B = m * p  # 8
    dim_C = p * n  # 8
    
    # Initialize matrices
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    # Helper to add a Strassen block
    def add_strassen_block(rank_offset, b_indices, c_indices):
        # b_indices: [e, f, g, h] mapping to global B indices
        # c_indices: [c00, c01, c10, c11] mapping to global C indices
        e, f, g, h = b_indices
        c00, c01, c10, c11 = c_indices
        
        # Term 1: (a+d)(e+h)
        U[0, rank_offset] += 1; U[3, rank_offset] += 1
        V[e, rank_offset] += 1; V[h, rank_offset] += 1
        W[c00, rank_offset] += 1
        W[c11, rank_offset] += 1
        
        # Term 2: (c+d)e
        U[2, rank_offset] += 1; U[3, rank_offset] += 1
        V[e, rank_offset] += 1
        W[c00, rank_offset] -= 1
        W[c10, rank_offset] += 1
        
        # Term 3: a(f-h)
        U[0, rank_offset] += 1
        V[f, rank_offset] += 1; V[h, rank_offset] -= 1
        W[c01, rank_offset] += 1
        W[c11, rank_offset] -= 1
        
        # Term 4: d(g-e)
        U[3, rank_offset] += 1
        V[g, rank_offset] += 1; V[e, rank_offset] -= 1
        W[c00, rank_offset] += 1
        W[c10, rank_offset] += 1
        
        # Term 5: (a+b)h
        U[0, rank_offset] += 1; U[1, rank_offset] += 1
        V[h, rank_offset] += 1
        W[c01, rank_offset] += 1
        W[c11, rank_offset] += 1
        
        # Term 6: (c-a)(e+f)
        U[2, rank_offset] += 1; U[0, rank_offset] -= 1
        V[e, rank_offset] += 1; V[f, rank_offset] += 1
        W[c10, rank_offset] += 1
        
        # Term 7: (b-d)(g+h)
        U[1, rank_offset] += 1; U[3, rank_offset] -= 1
        V[g, rank_offset] += 1; V[h, rank_offset] += 1
        W[c00, rank_offset] -= 1
        W[c11, rank_offset] += 1

    # Block 1: B cols 0,1 -> C cols 0,1
    # B indices: 0(0,0), 1(0,1), 4(1,0), 5(1,1)
    # C indices: 0(0,0), 1(0,1), 4(1,0), 5(1,1)
    add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5])
    
    # Block 2: B cols 2,3 -> C cols 2,3
    # B indices: 2(0,2), 3(0,3), 6(1,2), 7(1,3)
    # C indices: 2(0,2), 3(0,3), 6(1,2), 7(1,3)
    add_strassen_block(7, [2, 3, 6, 7], [2, 3, 6, 7])

    # Convert to lists of lists as required by the interface
    # The interface expects {'U': ..., 'V': ..., 'W': ...}
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # Note: The problem statement says W[k*n+i, r].
    # My C index mapping was k = i_n * p + i_p.
    # The problem statement says "output factor follows the official trace convention W[k*n+i,r]".
    # Let's check the indices in the prompt's seed code.
    # Seed code: k = i_n * p + k_p.
    # So C index is row * p + col.
    # My mapping: c00 is (0,0) -> 0*4+0=0. c01 is (0,1) -> 0*4+1=1. c10 is (1,0) -> 1*4+0=4. c11 is (1,1) -> 1*4+1=5.
    # This matches the seed code's convention.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
