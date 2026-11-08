import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 50. We aim for 46 using Strassen-like decomposition.
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # We will build the decomposition by summing contributions from 9 blocks.
    # Blocks are defined by splitting m=5 into 2, 2, 1 and p=5 into 2, 2, 1.
    # n=2 is not split.
    
    # Global sizes
    U = np.zeros((n * m, 46), dtype=float)
    V = np.zeros((m * p, 46), dtype=float)
    W = np.zeros((p * n, 46), dtype=float)
    
    rank_offset = 0
    
    # Helper to add a local decomposition
    # local_i, local_j, local_k ranges
    # global_i_offset, global_j_offset, global_k_offset
    # local_U, local_V, local_W are lists of tuples (idx, val)
    
    def add_block(gi_off, gj_off, gk_off, local_n, local_m, local_p, local_U, local_V, local_W):
        nonlocal rank_offset
        for r in range(len(local_U)):
            for (li, lj, val_u) in local_U[r]:
                gi = gi_off + li
                gj = gj_off + lj
                U[gi * m + gj, rank_offset + r] = val_u
            
            for (lj, lk, val_v) in local_V[r]:
                gj = gj_off + lj
                gk = gk_off + lk
                V[gj * p + gk, rank_offset + r] = val_v
            
            for (lk, li, val_w) in local_W[r]:
                gk = gk_off + lk
                gi = gi_off + li
                W[gk * n + gi, rank_offset + r] = val_w
        
        rank_offset += len(local_U)

    # 1. M_2,2,2 (Strassen)
    # A: 2x2, B: 2x2, C: 2x2
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    U_222 = []
    V_222 = []
    W_222 = []
    
    # M1: (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    U_222.append([(0,0,1), (1,1,1)])
    V_222.append([(0,0,1), (1,1,1)])
    W_222.append([(0,0,1), (1,1,1)]) # k=0,i=0 and k=1,i=1
    
    # M2: (A10+A11)B00
    # C10 += M2, C11 -= M2
    U_222.append([(1,0,1), (1,1,1)])
    V_222.append([(0,0,1)])
    W_222.append([(0,1,1), (1,1,-1)]) # k=0,i=1 and k=1,i=1
    
    # M3: A00(B10-B11)
    # C01 += M3, C11 += M3
    U_222.append([(0,0,1)])
    V_222.append([(1,0,1), (1,1,-1)])
    W_222.append([(0,0,1), (1,1,1)]) # k=0,i=0 and k=1,i=1
    
    # M4: A11(B01-B00)
    # C00 += M4, C10 += M4
    U_222.append([(1,1,1)])
    V_222.append([(0,1,1), (0,0,-1)])
    W_222.append([(1,0,1), (0,1,1)]) # k=1,i=0 and k=0,i=1
    
    # M5: (A00+A01)B11
    # C00 -= M5, C01 += M5
    U_222.append([(0,0,1), (0,1,1)])
    V_222.append([(1,1,1)])
    W_222.append([(0,0,-1), (0,0,1)]) # k=0,i=0 (for C00? No C01 is k=1? No C01 is i=0, k=1)
    # C00 is i=0, k=0. C01 is i=0, k=1.
    # M5 contributes to C00 with -1 and C01 with +1.
    # W index for C00: k=0, i=0 -> 0.
    # W index for C01: k=1, i=0 -> 2.
    W_222[-1] = [(0,0,-1), (1,0,1)]
    
    # M6: (A10-A00)(B00+B01)
    # C11 += M6
    U_222.append([(1,0,1), (0,0,-1)])
    V_222.append([(0,0,1), (0,1,1)])
    W_222.append([(1,1,1)]) # k=1, i=1
    
    # M7: (A01-A11)(B10+B11)
    # C00 += M7
    U_222.append([(0,1,1), (1,1,-1)])
    V_222.append([(1,0,1), (1,1,1)])
    W_222.append([(0,0,1)]) # k=0, i=0
    
    # 2. M_2,2,1
    # A: 2x2, B: 2x1, C: 2x1
    # Rank 4
    U_221 = []
    V_221 = []
    W_221 = []
    
    # M1: A00 B00 -> C00
    U_221.append([(0,0,1)])
    V_221.append([(0,0,1)])
    W_221.append([(0,0,1)])
    
    # M2: A01 B10 -> C00
    U_221.append([(0,1,1)])
    V_221.append([(1,0,1)])
    W_221.append([(0,0,1)])
    
    # M3: A10 B00 -> C10
    U_221.append([(1,0,1)])
    V_221.append([(0,0,1)])
    W_221.append([(0,1,1)])
    
    # M4: A11 B10 -> C10
    U_221.append([(1,1,1)])
    V_221.append([(1,0,1)])
    W_221.append([(0,1,1)])
    
    # 3. M_2,1,2
    # A: 2x1, B: 1x2, C: 2x2
    # Rank 4
    U_212 = []
    V_212 = []
    W_212 = []
    
    # M1: A0 B0 -> C00
    U_212.append([(0,0,1)])
    V_212.append([(0,0,1)])
    W_212.append([(0,0,1)])
    
    # M2: A0 B1 -> C01
    U_212.append([(0,0,1)])
    V_212.append([(0,1,1)])
    W_212.append([(1,0,1)])
    
    # M3: A1 B0 -> C10
    U_212.append([(1,0,1)])
    V_212.append([(0,0,1)])
    W_212.append([(0,1,1)])
    
    # M4: A1 B1 -> C11
    U_212.append([(1,0,1)])
    V_212.append([(0,1,1)])
    W_212.append([(1,1,1)])
    
    # 4. M_2,1,1
    # A: 2x1, B: 1x1, C: 2x1
    # Rank 2
    U_211 = []
    V_211 = []
    W_211 = []
    
    # M1: A0 B0 -> C00
    U_211.append([(0,0,1)])
    V_211.append([(0,0,1)])
    W_211.append([(0,0,1)])
    
    # M2: A1 B0 -> C10
    U_211.append([(1,0,1)])
    V_211.append([(0,0,1)])
    W_211.append([(0,1,1)])

    # Block 1,1: A(2x2) B(2x2) -> C(2x2)
    # gi_off=0, gj_off=0, gk_off=0
    add_block(0, 0, 0, 2, 2, 2, U_222, V_222, W_222)
    
    # Block 1,2: A(2x2) B(2x2) -> C(2x2)
    # gi_off=0, gj_off=0, gk_off=2
    add_block(0, 0, 2, 2, 2, 2, U_222, V_222, W_222)
    
    # Block 1,3: A(2x2) B(2x1) -> C(2x1)
    # gi_off=0, gj_off=0, gk_off=4
    add_block(0, 0, 4, 2, 2, 1, U_221, V_221, W_221)
    
    # Block 2,1: A(2x2) B(2x2) -> C(2x2)
    # gi_off=0, gj_off=2, gk_off=0
    add_block(0, 2, 0, 2, 2, 2, U_222, V_222, W_222)
    
    # Block 2,2: A(2x2) B(2x2) -> C(2x2)
    # gi_off=0, gj_off=2, gk_off=2
    add_block(0, 2, 2, 2, 2, 2, U_222, V_222, W_222)
    
    # Block 2,3: A(2x2) B(2x1) -> C(2x1)
    # gi_off=0, gj_off=2, gk_off=4
    add_block(0, 2, 4, 2, 2, 1, U_221, V_221, W_221)
    
    # Block 3,1: A(2x1) B(1x2) -> C(2x2)
    # gi_off=0, gj_off=4, gk_off=0
    add_block(0, 4, 0, 2, 1, 2, U_212, V_212, W_212)
    
    # Block 3,2: A(2x1) B(1x2) -> C(2x2)
    # gi_off=0, gj_off=4, gk_off=2
    add_block(0, 4, 2, 2, 1, 2, U_212, V_212, W_212)
    
    # Block 3,3: A(2x1) B(1x1) -> C(2x1)
    # gi_off=0, gj_off=4, gk_off=4
    add_block(0, 4, 4, 2, 1, 1, U_211, V_211, W_211)
    
    # Convert to lists of lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
