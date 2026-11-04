"""Kronecker product decomposition for 4x4x7 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Kronecker product: 4x4x7 = (2x2x7) ⊗ (2x2x1)
    T₁: 2x2x7 (n₁=2,m₁=2,p₁=7) - schoolbook rank 14
    T₂: 2x2x1 (n₂=2,m₂=2,p₂=1) - schoolbook rank 4
    Total rank: 14 * 4 = 56
    """
    # T₁: 2x2x7 schoolbook
    n1, m1, p1 = 2, 2, 7
    rank1 = n1 * m1 * p1  # 14
    U1 = [[0]*rank1 for _ in range(n1*m1)]  # 4 x 14
    V1 = [[0]*rank1 for _ in range(m1*p1)]  # 14 x 14
    W1 = [[0]*rank1 for _ in range(p1*n1)]  # 14 x 14
    
    c = 0
    for i in range(n1):
        for k in range(m1):
            for j in range(p1):
                U1[i*m1 + k][c] = 1
                V1[k*p1 + j][c] = 1
                W1[j*n1 + i][c] = 1
                c += 1
    
    # T₂: 2x2x1 schoolbook
    n2, m2, p2 = 2, 2, 1
    rank2 = n2 * m2 * p2  # 4
    U2 = [[0]*rank2 for _ in range(n2*m2)]  # 4 x 4
    V2 = [[0]*rank2 for _ in range(m2*p2)]  # 2 x 4
    W2 = [[0]*rank2 for _ in range(p2*n2)]  # 2 x 4
    
    c = 0
    for i in range(n2):
        for k in range(m2):
            for j in range(p2):
                U2[i*m2 + k][c] = 1
                V2[k*p2 + j][c] = 1
                W2[j*n2 + i][c] = 1
                c += 1
    
    # Combined tensor: n=4, m=4, p=7
    n, m, p = 4, 4, 7
    rank = rank1 * rank2  # 56
    U = [[0]*rank for _ in range(n*m)]  # 16 x 56
    V = [[0]*rank for _ in range(m*p)]  # 28 x 56
    W = [[0]*rank for _ in range(p*n)]  # 28 x 56
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            
            # U: index (i1*n2+i2)*(m1*m2) + (k1*m2+k2)
            for i1 in range(n1):
                for k1 in range(m1):
                    v1 = U1[i1*m1 + k1][c1]
                    if v1 == 0: continue
                    for i2 in range(n2):
                        for k2 in range(m2):
                            v2 = U2[i2*m2 + k2][c2]
                            if v2 == 0: continue
                            idx = (i1*n2 + i2)*(m1*m2) + (k1*m2 + k2)
                            U[idx][c] = v1 * v2
            
            # V: index (k1*m2+k2)*(p1*p2) + (j1*p2+j2)
            for k1 in range(m1):
                for j1 in range(p1):
                    v1 = V1[k1*p1 + j1][c1]
                    if v1 == 0: continue
                    for k2 in range(m2):
                        for j2 in range(p2):
                            v2 = V2[k2*p2 + j2][c2]
                            if v2 == 0: continue
                            idx = (k1*m2 + k2)*(p1*p2) + (j1*p2 + j2)
                            V[idx][c] = v1 * v2
            
            # W: index (j1*p2+j2)*(n1*n2) + (i1*n2+i2)
            for j1 in range(p1):
                for i1 in range(n1):
                    v1 = W1[j1*n1 + i1][c1]
                    if v1 == 0: continue
                    for j2 in range(p2):
                        for i2 in range(n2):
                            v2 = W2[j2*n2 + i2][c2]
                            if v2 == 0: continue
                            idx = (j1*p2 + j2)*(n1*n2) + (i1*n2 + i2)
                            W[idx][c] = v1 * v2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
