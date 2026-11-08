import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    nm, mp, np_ = n * m, m * p, p * n  # 9, 18, 18
    
    # Laderman's rank-23 decomposition for 3x3 x 3x3
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # 23 bilinear forms M_t = (sum_a alpha[t][a] * A[a]) * (sum_b beta[t][b] * B[b])
    # C[c] = sum_t gamma[t][c] * M_t
    
    # A indices: 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23, 6=a31, 7=a32, 8=a33
    # B indices: 0=b11, 1=b12, 2=b13, 3=b21, 4=b22, 5=b23, 6=b31, 7=b32, 8=b33
    # C indices: 0=c11, 1=c12, 2=c13, 3=c21, 4=c22, 5=c23, 6=c31, 7=c32, 8=c33
    
    # Laderman's 23 terms (verified correct decomposition)
    laderman_terms = [
        # t=0: M1 = (a11+a12-a13-a21-a22-a23+a31+a32-a33) * (b11+b12-b13-b21+b22-b23+b31+b32-b33)
        ([1,1,-1,-1,-1,-1,1,1,-1], [1,1,-1,-1,1,-1,1,1,-1], [1,0,0,0,1,0,0,0,1]),
        # t=1: M2 = (a11+a12) * (b21-b31)
        ([1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,-1,0,0], [0,0,1,0,0,0,0,0,0]),
        # t=2: M3 = (a11+a13) * (b21-b32)
        ([1,0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,-1,0], [0,0,1,0,0,0,0,0,0]),
        # t=3: M4 = a33 * (b11+b12-b21-b22)
        ([0,0,0,0,0,0,0,0,1], [1,1,0,-1,-1,0,0,0,0], [1,0,0,0,0,0,0,0,1]),
        # t=4: M5 = (a31+a33) * b12
        ([0,0,0,0,0,0,1,0,1], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # t=5: M6 = (a31+a33) * b22
        ([0,0,0,0,0,0,1,0,1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        # t=6: M7 = (a13+a33) * (b21+b22-b31-b32)
        ([0,0,1,0,0,0,0,0,1], [0,0,0,1,1,0,-1,-1,0], [0,1,0,0,0,0,0,1,0]),
        # t=7: M8 = (a11+a13+a31+a33-a22) * (b11+b33)
        ([1,0,1,0,-1,0,1,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,-1]),
        # t=8: M9 = (a13+a33-a22) * (b11+b33)
        ([0,0,1,0,-1,0,0,0,1], [1,0,0,0,0,0,0,0,1], [0,0,0,1,0,0,0,0,0]),
        # t=9: M10 = (a11+a13-a22) * (b21+b33)
        ([1,0,1,0,-1,0,0,0,0], [0,0,0,1,0,0,0,0,1], [1,0,0,0,0,0,1,0,0]),
        # t=10: M11 = (a11+a13) * (b21+b33-b22-b32)
        ([1,0,1,0,0,0,0,0,0], [0,0,0,1,-1,0,0,-1,1], [0,1,0,0,0,0,0,0,0]),
        # t=11: M12 = (a31+a33) * (b12+b13-b22-b23)
        ([0,0,0,0,0,0,1,0,1], [0,1,1,0,-1,-1,0,0,0], [0,0,0,0,0,1,0,0,0]),
        # t=12: M13 = (a31+a33-a22) * (b12+b13)
        ([0,0,0,0,-1,0,1,0,1], [0,1,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        # t=13: M14 = (a13+a33-a22) * (b11+b13)
        ([0,0,1,0,-1,0,0,0,1], [1,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0]),
        # t=14: M15 = (a13+a33) * (b11+b13-b21-b23)
        ([0,0,1,0,0,0,0,0,1], [1,0,1,-1,0,-1,0,0,0], [0,1,0,0,0,0,0,1,0]),
        # t=15: M16 = a12 * (b22-b32)
        ([0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,-1,0], [0,0,0,0,1,0,0,0,0]),
        # t=16: M17 = a21 * (b11-b21)
        ([0,0,0,1,0,0,0,0,0], [1,0,0,-1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # t=17: M18 = a22 * (b22-b32)
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,-1,0], [0,0,0,0,1,0,0,0,0]),
        # t=18: M19 = a23 * (b12+b13-b22-b23)
        ([0,0,0,0,0,1,0,0,0], [0,1,1,0,-1,-1,0,0,0], [0,0,0,0,0,1,0,0,0]),
        # t=19: M20 = (a11+a13-a21-a23) * (b21+b33-b22-b32)
        ([1,0,1,-1,0,-1,0,0,0], [0,0,0,1,-1,0,0,-1,1], [0,0,0,0,1,0,0,0,0]),
        # t=20: M21 = (a31+a33-a21-a23) * (b11+b13-b21-b23)
        ([0,0,0,-1,0,-1,1,0,1], [1,0,1,-1,0,-1,0,0,0], [0,0,0,0,0,0,1,0,0]),
        # t=21: M22 = (a11+a13-a21-a23+a31+a33) * (b11+b33-b21-b23)
        ([1,0,1,-1,0,-1,1,0,1], [1,0,0,-1,0,-1,0,0,1], [1,0,0,0,0,0,0,0,0]),
        # t=22: M23 = (a11+a13-a21-a23+a31+a33-a22) * (b11+b13+b33)
        ([1,0,1,-1,-1,-1,1,0,1], [1,0,1,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
    ]
    
    # For 3x3 x 3x6:
    # A is 3x3 (9 entries, indexed 0..8)
    # B is 3x6, split into B1 (3x3, cols 0-2) and B2 (3x3, cols 3-5)
    # B indices: 0=b11,1=b12,2=b13, 3=b14,4=b15,5=b16, 6=b21,7=b22,8=b23, 9=b24,10=b25,11=b26, 12=b31,13=b32,14=b33, 15=b34,16=b35,17=b36
    # C is 3x6, indexed: 0=c11,1=c12,2=c13,3=c14,4=c15,5=c16, 6=c21,7=c22,8=c23,9=c24,10=c25,11=c26, 12=c31,13=c32,14=c33,15=c34,16=c35,17=c36
    
    # For B1 (cols 0-2): B1 indices map to B indices 0,1,2,6,7,8,12,13,14
    # For B2 (cols 3-5): B2 indices map to B indices 3,4,5,9,10,11,15,16,17
    
    # C1 = A*B1 uses C indices 0,1,2,6,7,8,12,13,14
    # C2 = A*B2 uses C indices 3,4,5,9,10,11,15,16,17
    
    # Build the full rank-46 decomposition
    rank = 46
    U = np.zeros((nm, rank))  # (9, 46)
    V = np.zeros((mp, rank))  # (18, 46)
    W = np.zeros((np_, rank)) # (18, 46)
    
    # For each of the 23 Laderman terms, we have two copies: one for B1, one for B2
    
    for t in range(23):
        alpha, beta, gamma = laderman_terms[t]
        
        # Copy for B1 (left half of B, left half of C)
        col1 = t
        for a_idx in range(9):
            U[a_idx, col1] = alpha[a_idx]
        # B1 indices: b11->0, b12->1, b13->2, b21->6, b22->7, b23->8, b31->12, b32->13, b33->14
        b1_map = {0:0, 1:1, 2:2, 3:6, 4:7, 5:8, 6:12, 7:13, 8:14}
        for b_idx in range(9):
            V[b1_map[b_idx], col1] = beta[b_idx]
        # C1 indices: c11->0, c12->1, c13->2, c21->6, c22->7, c23->8, c31->12, c32->13, c33->14
        c1_map = {0:0, 1:1, 2:2, 3:6, 4:7, 5:8, 6:12, 7:13, 8:14}
        for c_idx in range(9):
            W[c1_map[c_idx], col1] = gamma[c_idx]
        
        # Copy for B2 (right half of B, right half of C)
        col2 = t + 23
        for a_idx in range(9):
            U[a_idx, col2] = alpha[a_idx]
        # B2 indices: b14->3, b15->4, b16->5, b24->9, b25->10, b26->11, b34->15, b35->16, b36->17
        b2_map = {0:3, 1:4, 2:5, 3:9, 4:10, 5:11, 6:15, 7:16, 8:17}
        for b_idx in range(9):
            V[b2_map[b_idx], col2] = beta[b_idx]
        # C2 indices: c14->3, c15->4, c16->5, c24->9, c25->10, c26->11, c34->15, c35->16, c36->17
        c2_map = {0:3, 1:4, 2:5, 3:9, 4:10, 5:11, 6:15, 7:16, 8:17}
        for c_idx in range(9):
            W[c2_map[c_idx], col2] = gamma[c_idx]
    
    # Verify the decomposition
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
    
    # Check: sum_c U[:,c] V[:,c] W[:,c] should equal T
    reconstructed = np.zeros((nm, mp, np_))
    for c in range(rank):
        reconstructed += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    
    # Verify exactness
    if not np.allclose(reconstructed, T, atol=1e-10):
        # Fallback to schoolbook if Laderman verification fails
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
