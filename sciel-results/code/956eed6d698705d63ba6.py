import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-44 matrix multiplication decomposition for 2x3 * 3x8.
    Uses Strassen's algorithm for the 2x2 * 2x8 part (4 blocks of 2x2*2x2)
    and schoolbook for the 2x1 * 1x8 part.
    Total rank: 4 * 7 + 2 * 8 = 28 + 16 = 44.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 8
    rank = 44
    
    # U: (n*m, rank) = (6, 44)
    # V: (m*p, rank) = (24, 44)
    # W: (p*n, rank) = (16, 44)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # A indices:
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    # B is split into 4 blocks of width 2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    for blk in range(4):
        bo = 2 * blk
        
        # V indices for B[0:2, bo:bo+2]
        # B[0, bo]   -> V[bo]
        # B[0, bo+1] -> V[bo+1]
        # B[1, bo]   -> V[8 + bo]
        # B[1, bo+1] -> V[8 + bo + 1]
        
        ve = bo
        vf = bo + 1
        vg = 8 + bo
        vh = 8 + bo + 1
        
        # W indices for C[0:2, bo:bo+2]
        # C[i, k] -> W[k*n + i] = W[2*k + i]
        # C[0, bo]   -> W[2*bo]
        # C[0, bo+1] -> W[2*(bo+1)] = W[2*bo+2]
        # C[1, bo]   -> W[2*bo+1]
        # C[1, bo+1] -> W[2*(bo+1)+1] = W[2*bo+3]
        
        w00 = 2 * bo
        w01 = 2 * bo + 2
        w10 = 2 * bo + 1
        w11 = 2 * bo + 3
        
        # Strassen P1 = (a+d)(e+h)
        # a=U[0], d=U[4], e=V[ve], h=V[vh]
        # Contributes to C[0,0] (w00) and C[1,1] (w11)
        U[0][col] = 1
        U[4][col] = 1
        V[ve][col] = 1
        V[vh][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # Strassen P2 = (c+d)e
        # c=U[3], d=U[4], e=V[ve]
        # Contributes to C[1,0] (w10) and C[1,1] (w11) with -1
        U[3][col] = 1
        U[4][col] = 1
        V[ve][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # Strassen P3 = a(f-h)
        # a=U[0], f=V[vf], h=V[vh]
        # Contributes to C[0,1] (w01) and C[1,1] (w11)
        U[0][col] = 1
        V[vf][col] = 1
        V[vh][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # Strassen P4 = d(g-e)
        # d=U[4], g=V[vg], e=V[ve]
        # Contributes to C[0,0] (w00) and C[1,0] (w10)
        U[4][col] = 1
        V[vg][col] = 1
        V[ve][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # Strassen P5 = (a+b)h
        # a=U[0], b=U[1], h=V[vh]
        # Contributes to C[0,0] (w00) with -1 and C[0,1] (w01)
        U[0][col] = 1
        U[1][col] = 1
        V[vh][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # Strassen P6 = (c-a)(e+f)
        # c=U[3], a=U[0], e=V[ve], f=V[vf]
        # Contributes to C[1,1] (w11)
        U[3][col] = 1
        U[0][col] = -1
        V[ve][col] = 1
        V[vf][col] = 1
        W[w11][col] = 1
        col += 1
        
        # Strassen P7 = (b-d)(g+h)
        # b=U[1], d=U[4], g=V[vg], h=V[vh]
        # Contributes to C[0,0] (w00)
        U[1][col] = 1
        U[4][col] = -1
        V[vg][col] = 1
        V[vh][col] = 1
        W[w00][col] = 1
        col += 1
        
    # Part 2: A[0:2, 2] * B[2, 0:8]
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B[2,k] -> V[16+k]
    # C[i,k] -> W[2k+i]
    
    for k in range(8):
        for i in range(2):
            u_idx = i * 3 + 2
            v_idx = 16 + k
            w_idx = 2 * k + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    if col != rank:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank_fallback = n * m * p
        U = [[0] * rank_fallback for _ in range(n * m)]
        V = [[0] * rank_fallback for _ in range(m * p)]
        W = [[0] * rank_fallback for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
