import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 2, 4, 5
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_block(y, z, w_map):
        nonlocal col
        if time.time() > deadline:
            return
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        k0, i0 = w_map[0,0]
        k3, i3 = w_map[1,1]
        W[w_idx(k0,i0)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        k2, i2 = w_map[1,0]
        W[w_idx(k2,i2)][col] += 1
        W[w_idx(k3,i3)][col] -= 1
        col += 1
        
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        k1, i1 = w_map[0,1]
        W[w_idx(k1,i1)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w_idx(k0,i0)][col] += 1
        W[w_idx(k2,i2)][col] += 1
        col += 1
        
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] -= 1
        W[w_idx(k1,i1)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] += 1
        col += 1

    y1 = [(0,0),(0,1),(1,0),(1,1)]
    z1 = [(0,0),(0,1),(1,0),(1,1)]
    w1 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y1, z1, w1)
    
    y2 = [(0,2),(0,3),(1,2),(1,3)]
    z2 = [(2,0),(2,1),(3,0),(3,1)]
    w2 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y2, z2, w2)
    
    y3 = [(0,0),(0,1),(1,0),(1,1)]
    z3 = [(0,2),(0,3),(1,2),(1,3)]
    w3 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y3, z3, w3)
    
    y4 = [(0,2),(0,3),(1,2),(1,3)]
    z4 = [(2,2),(2,3),(3,2),(3,3)]
    w4 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y4, z4, w4)
    
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
