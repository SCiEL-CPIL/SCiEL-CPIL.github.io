import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for t in range(rank):
                    s += U[a][t] * V[b][t] * W[c][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W factor matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    
    Strassen's algorithm decomposes 2x2x2 multiplication into 7 rank-1 terms.
    The 7 products are:
      m1 = (a11+a22)(b11+b22)
      m2 = (a21+a22)b11
      m3 = a11(b12-b22)
      m4 = a22(b21-b11)
      m5 = (a11+a12)b22
      m6 = (a21-a11)(b11+b12)
      m7 = (a12-a22)(b21+b22)
    
    Result:
      c11 = m1 - m4 + m5 + m7
      c12 = m3 + m5
      c21 = m2 + m4
      c22 = m1 - m3 + m2 - m6
    """
    # U: 4 rows (a11, a12, a21, a22) x 7 cols
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a11
        [0, 0, 0, 0, 1, 0, 1],    # a12
        [0, 1, 0, 0, 0, 1, 0],    # a21
        [1, 1, 0, 1, 0, 0, -1],   # a22
    ]
    
    # V: 4 rows (b11, b12, b21, b22) x 7 cols
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b11
        [1, 0, 1, 0, 0, 1, 0],    # b12
        [0, 0, 0, 1, 0, 0, 1],    # b21
        [1, 0, -1, 0, 1, 0, 1],   # b22
    ]
    
    # W: 4 rows (c11, c12, c21, c22) x 7 cols
    W = [
        [1, 0, 0, -1, 1, 0, 1],   # c11
        [0, 0, 1, 0, 1, 0, 0],    # c12
        [0, 1, 0, 1, 0, 0, 0],    # c21
        [1, 0, -1, 0, 0, -1, 0],  # c22
    ]
    
    return U, V, W

def _build_kronecker_decomposition():
    """Build a decomposition using Kronecker product structure.
    
    Split m=3 into m1=1 and m2=2, and p=8 into p1=4 and p2=4.
    Use Strassen for the 2x2x2 block (i, j in {1,2}, k in {4,5,6,7} -> treat as 2x2x2 with p=2)
    and schoolbook for the 1x2x4 block (i=0, j in {0,1}, k in {0,1,2,3}).
    
    Actually, let's use a cleaner approach:
    - Split the 2x3x8 tensor into two parts:
      Part 1: 2x1x4 (i=0,1; j=0; k=0,1,2,3) - rank 8 schoolbook
      Part 2: 2x2x4 (i=0,1; j=1,2; k=0,1,2,3) - can use Strassen-like
    
    For 2x2x4: split k into two groups of 2, giving two 2x2x2 blocks.
    Each 2x2x2 uses Strassen (rank 7), so total rank = 8 + 7 + 7 = 22? No, that's not right.
    
    Let me reconsider. The Kronecker approach:
    2x3x8 = (2x2x2) ⊗ (1x3x4) doesn't quite work directly.
    
    Better approach: 
    Split m=3 as 1+2, and p=8 as 4+4.
    Block (0,0): 2x1x4 -> rank 8 (schoolbook)
    Block (0,1): 2x2x4 -> rank 16 schoolbook, or use Strassen on 2x2x2 blocks
    Block (1,0): 2x1x4 -> rank 8 (schoolbook)  
    Block (1,1): 2x2x4 -> rank 16 schoolbook
    
    Total schoolbook: 8+16+8+16 = 48
    
    Using Strassen for the 2x2x4 blocks:
    Each 2x2x4 can be seen as two 2x2x2 blocks (split k into 2+2).
    Each 2x2x2 uses Strassen rank 7.
    So each 2x2x4 uses rank 14.
    Total: 8 + 14 + 8 + 14 = 44.
    """
    n, m, p = 2, 3, 8
    
    U = [[0] * 44 for _ in range(n * m)]  # 6 rows
    V = [[0] * 44 for _ in range(m * p)]  # 24 rows
    W = [[0] * 44 for _ in range(p * n)]  # 16 rows
    
    col = 0
    
    # Strassen factors for 2x2x2
    sU, sV, sW = _strassen_2x2x2()
    
    # Block (i_block=0, j_block=0): i=0,1; j=0; k=0,1,2,3 -> 2x1x4 schoolbook
    # U indices: i*m+j = i*3+0 = 0,1
    # V indices: j*p+k = 0*8+k = k for k=0..3
    # W indices: k*n+i = k*2+i
    for i in range(2):
        for k in range(4):
            u_idx = i * 3 + 0
            v_idx = 0 * 8 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    # col is now 8
    
    # Block (i_block=0, j_block=1): i=0,1; j=1,2; k=0,1,2,3 -> 2x2x4
    # Split k into k1=0,1 and k2=2,3
    # For k1=0,1: 2x2x2 Strassen
    # U indices: i*3+j where j=1,2 -> 3,4,6,7
    # V indices: j*8+k where j=1,2 and k=0,1 -> 8,9,16,17
    # W indices: k*2+i where k=0,1 and i=0,1 -> 0,1,2,3
    for t in range(7):
        for i_local in range(2):
            for j_local in range(2):
                u_idx = i_local * 3 + (1 + j_local)  # j=1 or 2
                U[u_idx][col] = sU[i_local][t]
        
        for j_local in range(2):
            for k_local in range(2):
                v_idx = (1 + j_local) * 8 + k_local  # k=0 or 1
                V[v_idx][col] = sV[j_local][t]
        
        for k_local in range(2):
            for i_local in range(2):
                w_idx = k_local * 2 + i_local
                W[w_idx][col] = sW[i_local][t]
        
        col += 1
    # col is now 8 + 7 = 15
    
    # Block (i_block=1, j_block=0): i=0,1; j=0; k=4,5,6,7 -> 2x1x4 schoolbook
    for i in range(2):
        for k in range(4):
            u_idx = i * 3 + 0
            v_idx = 0 * 8 + (4 + k)
            w_idx = (4 + k) * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    # col is now 15 + 8 = 23
    
    # Block (i_block=1, j_block=1): i=0,1; j=1,2; k=4,5,6,7 -> 2x2x4
    # Split k into k1=4,5 and k2=6,7
    # For k1=4,5: 2x2x2 Strassen
    for t in range(7):
        for i_local in range(2):
            for j_local in range(2):
                u_idx = i_local * 3 + (1 + j_local)
                U[u_idx][col] = sU[i_local][t]
        
        for j_local in range(2):
            for k_local in range(2):
                v_idx = (1 + j_local) * 8 + (4 + k_local)
                V[v_idx][col] = sV[j_local][t]
        
        for k_local in range(2):
            for i_local in range(2):
                w_idx = (4 + k_local) * 2 + i_local
                W[w_idx][col] = sW[i_local][t]
        
        col += 1
    # col is now 23 + 7 = 30
    
    # Wait, that's only 30. Let me recount.
    # Block 1: 2x1x4 = 8 terms
    # Block 2: 2x2x4 with Strassen on 2x2x2 = 7 terms (but this only covers k=0,1)
    # I need to handle k=2,3 separately for block 2.
    
    # Let me redo this more carefully.
    
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    
    sU, sV, sW = _strassen_2x2x2()
    
    # Block (j=0, k=0..3): 2x1x4 schoolbook, 8 terms
    for i in range(2):
        for k in range(4):
            u_idx = i * 3 + 0
            v_idx = 0 * 8 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    # col = 8
    
    # Block (j=1,2, k=0,1): 2x2x2 Strassen, 7 terms
    for t in range(7):
        for i_local in range(2):
            for j_local in range(2):
                u_idx = i_local * 3 + (1 + j_local)
                U[u_idx][col] = sU[i_local][t]
        
        for j_local in range(2):
            for k_local in range(2):
                v_idx = (1 + j_local) * 8 + k_local
                V[v_idx][col] = sV[j_local][t]
        
        for k_local in range(2):
            for i_local in range(2):
                w_idx = k_local * 2 + i_local
                W[w_idx][col] = sW[i_local][t]
        
        col += 1
    # col = 15
    
    # Block (j=1,2, k=2,3): 2x2x2 Strassen, 7 terms
    for t in range(7):
        for i_local in range(2):
            for j_local in range(2):
                u_idx = i_local * 3 + (1 + j_local)
                U[u_idx][col] = sU[i_local][t]
        
        for j_local in range(2):
            for k_local in range(2):
                v_idx = (1 + j_local) * 8 + (2 + k_local)
                V[v_idx][col] = sV[j_local][t]
        
        for k_local in range(2):
            for i_local in range(2):
                w_idx = (2 + k_local) * 2 + i_local
                W[w_idx][col] = sW[i_local][t]
        
        col += 1
    # col = 22
    
    # Block (j=0, k=4..7): 2x1x4 schoolbook, 8 terms
    for i in range(2):
        for k in range(4):
            u_idx = i * 3 + 0
            v_idx = 0 * 8 + (4 + k)
            w_idx = (4 + k) * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    # col = 30
    
    # Block (j=1,2, k=4,5): 2x2x2 Strassen, 7 terms
    for t in range(7):
        for i_local in range(2):
            for j_local in range(2):
                u_idx = i_local * 3 + (1 + j_local)
                U[u_idx][col] = sU[i_local][t]
        
        for j_local in range(2):
            for k_local in range(2):
                v_idx = (1 + j_local) * 8 + (4 + k_local)
                V[v_idx][col] = sV[j_local][t]
        
        for k_local in range(2):
            for i_local in range(2):
                w_idx = (4 + k_local) * 2 + i_local
                W[w_idx][col] = sW[i_local][t]
        
        col += 1
    # col = 37
    
    # Block (j=1,2, k=6,7): 2x2x2 Strassen, 7 terms
    for t in range(7):
        for i_local in range(2):
            for j_local in range(2):
                u_idx = i_local * 3 + (1 + j_local)
                U[u_idx][col] = sU[i_local][t]
        
        for j_local in range(2):
            for k_local in range(2):
                v_idx = (1 + j_local) * 8 + (6 + k_local)
                V[v_idx][col] = sV[j_local][t]
        
        for k_local in range(2):
            for i_local in range(2):
                w_idx = (6 + k_local) * 2 + i_local
                W[w_idx][col] = sW[i_local][t]
        
        col += 1
    # col = 44
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    # Start with schoolbook (rank 48) as the guaranteed fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    # Try the Kronecker/Strassen decomposition (rank 44)
    try:
        candidate = _build_kronecker_decomposition()
        if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            best = candidate
            best_rank = 44
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
