import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U: (8, rank)
    # V: (20, rank)
    # W: (10, rank)
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y, z, w_map):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        # M1 = (y00+y11)*(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        k0, i0 = w_map[0,0]
        k3, i3 = w_map[1,1]
        W[w_idx(k0,i0)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        # M2 = (y10+y11)*z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        k2, i2 = w_map[1,0]
        W[w_idx(k2,i2)][col] += 1
        W[w_idx(k3,i3)][col] -= 1
        col += 1
        
        # M3 = y00*(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        k1, i1 = w_map[0,1]
        W[w_idx(k1,i1)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        # M4 = y11*(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w_idx(k0,i0)][col] += 1
        W[w_idx(k2,i2)][col] += 1
        col += 1
        
        # M5 = (y00+y01)*z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] -= 1
        W[w_idx(k1,i1)][col] += 1
        col += 1
        
        # M6 = (y10-y00)*(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        # M7 = (y01-y11)*(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] += 1
        col += 1

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    y1 = [(0,0),(0,1),(1,0),(1,1)]
    z1 = [(0,0),(0,1),(1,0),(1,1)]
    w1 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y1, z1, w1)
    
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    y2 = [(0,2),(0,3),(1,2),(1,3)]
    z2 = [(2,0),(2,1),(3,0),(3,1)]
    w2 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y2, z2, w2)
    
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    y3 = [(0,0),(0,1),(1,0),(1,1)]
    z3 = [(0,2),(0,3),(1,2),(1,3)]
    w3 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y3, z3, w3)
    
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    y4 = [(0,2),(0,3),(1,2),(1,3)]
    z4 = [(2,2),(2,3),(3,2),(3,3)]
    w4 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y4, z4, w4)
    
    # Block 5: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    # Direct multiplication: 4 terms
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    return {"U": U, "V": V, "W": W}
