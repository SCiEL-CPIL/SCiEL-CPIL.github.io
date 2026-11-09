import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U: i*2+j
    V: j*2+k
    W: k*2+i
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # c00, c11
    
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # c10, c11
    
    # M3: a00(b10-b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # c01, c11
    
    # M4: a11(b01-b00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # c00, c10
    
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # c00, c01
    
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # c11
    
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # c00
    
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition (Rank 49).
    U: 16x49, V: 16x49, W: 16x49
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for rb in range(7):
        for rl in range(7):
            r = rb * 7 + rl
            # Kronecker product logic
            # U4[i*4+j, r] = U2[ib*2+jb, rb] * U2[il*2+jl, rl]
            # i = ib*2 + il, j = jb*2 + jl
            
            # Iterate over block indices
            for ib in range(2):
                for il in range(2):
                    i = ib * 2 + il
                    for jb in range(2):
                        for jl in range(2):
                            j = jb * 2 + jl
                            
                            idx_u4 = i * 4 + j
                            idx_u2_b = ib * 2 + jb
                            idx_u2_l = il * 2 + jl
                            
                            U[idx_u4, r] = U2[idx_u2_b, rb] * U2[idx_u2_l, rl]
                            
                            # V4[j*4+k, r] = V2[jb*2+kb, rb] * V2[jl*2+kl, rl]
                            # j is same as above. k = kb*2 + kl
                            for kb in range(2):
                                for kl in range(2):
                                    k = kb * 2 + kl
                                    idx_v4 = j * 4 + k
                                    idx_v2_b = jb * 2 + kb
                                    idx_v2_l = jl * 2 + kl
                                    
                                    V[idx_v4, r] = V2[idx_v2_b, rb] * V2[idx_v2_l, rl]
                                    
                                    # W4[k*4+i, r] = W2[kb*2+ib, rb] * W2[kl*2+il, rl]
                                    # k is same. i is same.
                                    idx_w4 = k * 4 + i
                                    idx_w2_b = kb * 2 + ib
                                    idx_w2_l = kl * 2 + il
                                    
                                    W[idx_w4, r] = W2[idx_w2_b, rb] * W2[idx_w2_l, rl]
                                    
    return U, V, W

def _schoolbook_4x4x3():
    """
    Returns U, V, W for 4x4x3 Schoolbook decomposition (Rank 48).
    U: 16x48, V: 12x48, W: 12x48
    """
    rank = 48
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((12, rank), dtype=int)
    
    for t in range(rank):
        i = t // 12
        rem = t % 12
        j = rem // 3
        k = rem % 3
        
        U[i * 4 + j, t] = 1
        V[j * 3 + k, t] = 1
        W[k * 4 + i, t] = 1
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 7
    
    U1, V1, W1 = _strassen_4x4x4()
    U2, V2, W2 = _schoolbook_4x4x3()
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    
    # U: 16 x total_rank
    U = np.hstack([U1, U2])
    
    # V: 28 x total_rank
    V = np.zeros((28, total_rank))
    
    # Map V1 rows to V rows
    # V1 index: j*4 + k (k in 0..3)
    # V global index: j*7 + k_glob
    for r in range(16):
        j = r // 4
        k = r % 4
        g = j * 7 + k
        V[g, 0:rank1] = V1[r]
        
    # Map V2 rows to V rows
    # V2 index: j*3 + k_loc (k_loc in 0..2)
    # V global index: j*7 + k_glob (k_glob in 4..6)
    for r in range(12):
        j = r // 3
        k_loc = r % 3
        k_glob = k_loc + 4
        g = j * 7 + k_glob
        V[g, rank1:rank1+rank2] = V2[r]
        
    # W: 28 x total_rank
    W = np.zeros((28, total_rank))
    
    # Map W1 rows to W rows
    # W1 index: k*4 + i (k in 0..3)
    # W global index: k_glob*4 + i
    for r in range(16):
        k = r // 4
        i = r % 4
        g = k * 4 + i
        W[g, 0:rank1] = W1[r]
        
    # Map W2 rows to W rows
    # W2 index: k_loc*4 + i (k_loc in 0..2)
    # W global index: k_glob*4 + i (k_glob in 4..6)
    for r in range(12):
        k_loc = r // 4
        i = r % 4
        k_glob = k_loc + 4
        g = k_glob * 4 + i
        W[g, rank1:rank1+rank2] = W2[r]
        
    return {"U": U, "V": V, "W": W}
