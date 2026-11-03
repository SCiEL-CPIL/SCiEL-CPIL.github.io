"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen for 2x2x2 blocks
    # A is 2x4, B is 4x6
    # Split A into A1 (2x2), A2 (2x2)
    # Split B into B1 (2x6), B2 (2x6)
    # C = A1 B1 + A2 B2
    # B1 is 2x6, split into B11, B12, B13 (2x2 each)
    # A1 B1 = A1 B11 + A1 B12 + A1 B13
    # Each A1 Bj is 2x2x2x2 multiplication, rank 7
    # Total rank 3*7 + 3*7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen factors for 2x2x2
    # U_idx: i*2+j
    # V_idx: j*2+k
    # W_idx: k*2+i
    # M1: U(0,0)=1, U(1,1)=1; V(0,0)=1, V(1,1)=1; W(0,0)=1, W(1,1)=1
    # M2: U(1,0)=1, U(1,1)=1; V(0,0)=1; W(1,0)=1, W(1,1)=1? No.
    # Let's derive M2: (X10+X11)Y00
    # X part: X10, X11. U[1*2+0]=1, U[1*2+1]=1.
    # Y part: Y00. V[0*2+0]=1.
    # Z part: Z10, Z11. W[0*2+1]=1, W[1*2+1]=1.
    
    strassen_U = [
        [1, 0, 0, 0, 1, 0, 0], # M1: X00, X11
        [0, 1, 0, 0, 0, 1, 0], # M2: X10, X11
        [1, 0, 0, 0, 0, 0, 0], # M3: X00
        [0, 0, 0, 1, 0, 0, 0], # M4: X11
        [1, 1, 0, 0, 0, 0, 0], # M5: X00, X01
        [0, 1, -1, 0, 0, 0, 0],# M6: X10, X00
        [0, 0, 0, 1, 0, 0, 0], # M7: X01, X11? No.
    ]
    # M7: (X01-X11)(Y10+Y11)
    # X part: X01, X11. U[0*2+1]=1, U[1*2+1]=-1.
    
    # Let's just use a known rank 7 decomposition table.
    # Indices: U(i,j), V(j,k), W(k,i)
    # r=0: M1
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    # r=1: M2
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (1,0)=1, (1,1)=1
    # r=2: M3
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (0,1)=1
    # r=3: M4
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (1,0)=1
    # r=4: M5
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (0,0)=-1, (0,1)=1
    # r=5: M6
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (1,1)=1
    # r=6: M7
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1

    # We have 6 such blocks (3 for A1, 3 for A2)
    # Block 0: A1 B11
    # Block 1: A1 B12
    # Block 2: A1 B13
    # Block 3: A2 B21
    # Block 4: A2 B22
    # Block 5: A2 B23
    
    # Mapping local indices (i,j,k) in 2x2x2 to global indices
    # A1 is rows 0,1 of A, cols 0,1.
    # A2 is rows 0,1 of A, cols 2,3.
    # B1 is rows 0,1 of B, cols 0..5.
    # B2 is rows 2,3 of B, cols 0..5.
    
    # For A1 B11:
    # i in 0,1 -> global i
    # j in 0,1 -> global j
    # k in 0,1 -> global k
    # U global idx: i*4 + j
    # V global idx: j*6 + k
    # W global idx: k*2 + i
    
    # For A1 B12:
    # k in 2,3 -> global k = k_local + 2
    
    # For A1 B13:
    # k in 4,5 -> global k = k_local + 4
    
    # For A2 B21:
    # j in 2,3 -> global j = j_local + 2
    
    # For A2 B22:
    # j in 2,3, k in 2,3
    
    # For A2 B23:
    # j in 2,3, k in 4,5

    blocks = [
        (0, 0), # A1 B11
        (0, 2), # A1 B12
        (0, 4), # A1 B13
        (2, 0), # A2 B21
        (2, 2), # A2 B22
        (2, 4), # A2 B23
    ]
    
    # Strassen factors defined locally
    S_U = [
        [1, 0, 0, 1], # M1: (0,0), (1,1)
        [0, 0, 1, 1], # M2: (1,0), (1,1)
        [1, 0, 0, 0], # M3: (0,0)
        [0, 0, 0, 1], # M4: (1,1)
        [1, 1, 0, 0], # M5: (0,0), (0,1)
        [0, 0, 1, -1],# M6: (1,0), (0,0)
        [0, 1, 0, -1],# M7: (0,1), (1,1)
    ]
    S_V = [
        [1, 0, 0, 1], # M1: (0,0), (1,1)
        [1, 0, 0, 0], # M2: (0,0)
        [0, 1, 0, -1],# M3: (0,1), (1,1)
        [0, -1, 1, 0],# M4: (1,0), (0,0)
        [0, 0, 0, 1], # M5: (1,1)
        [1, 1, 0, 0], # M6: (0,0), (0,1)
        [0, 0, 1, 1], # M7: (1,0), (1,1)
    ]
    S_W = [
        [1, 0, 0, 1], # M1: (0,0), (1,1)
        [0, 1, 0, 1], # M2: (1,0), (1,1)
        [0, 1, 0, 0], # M3: (0,1)
        [0, 1, 0, 0], # M4: (1,0)
        [-1, 1, 0, 0],# M5: (0,0), (0,1)
        [0, 0, 0, 1], # M6: (1,1)
        [1, 0, 0, 1], # M7: (0,0), (1,1)
    ]
    
    # Note: S_W indices are k*2+i.
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 2
    # (1,1) -> 3
    
    # Fill U, V, W
    for block_idx, (j_offset, k_offset) in enumerate(blocks):
        for r_local in range(7):
            r_global = block_idx * 7 + r_local
            
            # U
            for i in range(2):
                for j_local in range(2):
                    val = S_U[r_local][i * 2 + j_local]
                    if val != 0:
                        j_global = j_local + j_offset
                        U[i * 4 + j_global][r_global] = val
                        
            # V
            for j_local in range(2):
                for k_local in range(2):
                    val = S_V[r_local][j_local * 2 + k_local]
                    if val != 0:
                        j_global = j_local + j_offset
                        k_global = k_local + k_offset
                        V[j_global * 6 + k_global][r_global] = val
                        
            # W
            for k_local in range(2):
                for i in range(2):
                    val = S_W[r_local][k_local * 2 + i]
                    if val != 0:
                        k_global = k_local + k_offset
                        W[k_global * 2 + i][r_global] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
