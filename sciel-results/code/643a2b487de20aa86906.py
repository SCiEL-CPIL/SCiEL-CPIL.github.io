import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition for n x m * m x p."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    recon = np.zeros((p * n, n * m, m * p))
    for r in range(rank):
        recon += np.einsum('i,j,k->ijk', W[:, r], U[:, r], V[:, r])
    return np.allclose(T, recon, atol=tol)

def _rank14_2x3x3():
    """
    Return a rank-14 decomposition for 2x3 * 3x3.
    U: (6, 14), V: (9, 14), W: (6, 14)
    A index: i*3+j, B index: j*3+k, C/W index: k*2+i
    """
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # W index: k*2+i
    
    U = np.zeros((6, 14))
    V = np.zeros((9, 14))
    W = np.zeros((6, 14))
    
    # P1: (a0+a3)(b0+b3+b6)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1; V[6,0]=1
    # a0*b0->C00, a0*b3->C01, a0*b6->C02, a3*b0->C10, a3*b3->C11, a3*b6->C12
    W[0,0]=1; W[2,0]=1; W[4,0]=1
    W[1,0]=1; W[3,0]=1; W[5,0]=1
    
    # P2: (a0-a3)(b0-b3-b6)
    U[0,1]=1; U[3,1]=-1
    V[0,1]=1; V[3,1]=-1; V[6,1]=-1
    # a0*b0->C00, -a0*b3->C01, -a0*b6->C02, -a3*b0->C10, a3*b3->C11, a3*b6->C12
    W[0,1]=1; W[2,1]=-1; W[4,1]=-1
    W[1,1]=-1; W[3,1]=1; W[5,1]=1
    
    # P3: (a1+a4)(b1+b4+b7)
    U[1,2]=1; U[4,2]=1
    V[1,2]=1; V[4,2]=1; V[7,2]=1
    # a1*b1->C01, a1*b4->C02, a1*b7->C02 (wait, b7 is B[2,1], so k=1)
    # a1 is A[0,1], b1 is B[0,1] -> C[0,1] (W[2])
    # a1*b4: A[0,1]*B[1,1] -> C[0,1] (W[2])
    # a1*b7: A[0,1]*B[2,1] -> C[0,1] (W[2])
    # a4*b1: A[1,1]*B[0,1] -> C[1,1] (W[3])
    # a4*b4: A[1,1]*B[1,1] -> C[1,1] (W[3])
    # a4*b7: A[1,1]*B[2,1] -> C[1,1] (W[3])
    # So all contribute to C[0,1] or C[1,1]
    W[2,2]=3; W[3,2]=3
    
    # P4: (a1-a4)(b1-b4-b7)
    U[1,3]=1; U[4,3]=-1
    V[1,3]=1; V[4,3]=-1; V[7,3]=-1
    # a1*b1->C01, -a1*b4->C01, -a1*b7->C01, -a4*b1->C11, a4*b4->C11, a4*b7->C11
    W[2,3]=3; W[3,3]=3
    
    # P5: (a2+a5)(b2+b5+b8)
    U[2,4]=1; U[5,4]=1
    V[2,4]=1; V[5,4]=1; V[8,4]=1
    # a2*b2: A[0,2]*B[0,2] -> C[0,2] (W[4])
    # a2*b5: A[0,2]*B[1,2] -> C[0,2] (W[4])
    # a2*b8: A[0,2]*B[2,2] -> C[0,2] (W[4])
    # a5*b2: A[1,2]*B[0,2] -> C[1,2] (W[5])
    # a5*b5: A[1,2]*B[1,2] -> C[1,2] (W[5])
    # a5*b8: A[1,2]*B[2,2] -> C[1,2] (W[5])
    W[4,4]=3; W[5,4]=3
    
    # P6: (a2-a5)(b2-b5-b8)
    U[2,5]=1; U[5,5]=-1
    V[2,5]=1; V[5,5]=-1; V[8,5]=-1
    W[4,5]=3; W[5,5]=3
    
    # Now I have 6 products. Let's check what's covered:
    # C00: P1(1*a0*b0) + P2(1*a0*b0) = 2*a0*b0. Need a0*b0 + a1*b3 + a2*b6.
    # This is not working. The sum of P1 and P2 gives 2*a0*b0, 0*a0*b3, 0*a0*b6, etc.
    # So this approach is flawed.
    
    return None

def _find_rank14_2x3x3():
    """
    Find a rank-14 decomposition for 2x3 * 3x3.
    Returns (U, V, W) or None.
    """
    return None

def _construct_rank42():
    """
    Construct a rank-42 decomposition for 2x3 * 3x9.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 2, 3, 9
    
    # Schoolbook rank-54
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
