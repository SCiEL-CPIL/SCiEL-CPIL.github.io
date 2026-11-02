"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 38 decomposition:
    # Split m=4 into two 2s.
    # Split p=5 into 2 and 3.
    # For each 2x2x2 block, use Strassen (rank 7).
    # For each 2x2x3 block, use Schoolbook (rank 12).
    # Total: 2 * (7 + 12) = 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]  # 8 x 38
    V = [[0] * rank for _ in range(m * p)]  # 20 x 38
    W = [[0] * rank for _ in range(p * n)]  # 10 x 38
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h) -> C11+=, C12+=, C21+=, C22+=
    # M2 = (c+d)(g+h) -> C21+=, C22+=
    # M3 = a(f-h) -> C12+=, C22-=
    # M4 = c(e-g) -> C11-=, C21+=
    # M5 = (a+b)g -> C11+=, C12+=
    # M6 = (c+d)f -> C21+=, C22+=
    # M7 = (b-d)(h-g) -> C12-=, C21-=
    
    # Indices for 2x2 A: 0:a, 1:b, 2:c, 3:d
    # Indices for 2x2 B: 0:e, 1:f, 2:g, 3:h
    # Indices for 2x2 C: 0:C00, 1:C01, 2:C10, 3:C11
    
    def add_strassen(u_base, v_base, w_base):
        nonlocal col
        # u_base, v_base, w_base are offsets into the global arrays
        # A indices: u_base+0, u_base+1, u_base+2, u_base+3
        # B indices: v_base+0, v_base+1, v_base+2, v_base+3
        # C indices: w_base+0, w_base+1, w_base+2, w_base+3
        
        # M1
        U[u_base+0][col] += 1; U[u_base+3][col] += 1
        V[v_base+0][col] += 1; V[v_base+3][col] += 1
        W[w_base+0][col] += 1; W[w_base+1][col] += 1
        W[w_base+2][col] += 1; W[w_base+3][col] += 1
        col += 1
        
        # M2
        U[u_base+2][col] += 1; U[u_base+3][col] += 1
        V[v_base+2][col] += 1; V[v_base+3][col] += 1
        W[w_base+2][col] += 1; W[w_base+3][col] += 1
        col += 1
        
        # M3
        U[u_base+0][col] += 1
        V[v_base+1][col] += 1; V[v_base+3][col] -= 1
        W[w_base+1][col] += 1; W[w_base+3][col] -= 1
        col += 1
        
        # M4
        U[u_base+2][col] += 1
        V[v_base+0][col] += 1; V[v_base+2][col] -= 1
        W[w_base+0][col] -= 1; W[w_base+2][col] += 1
        col += 1
        
        # M5
        U[u_base+0][col] += 1; U[u_base+1][col] += 1
        V[v_base+2][col] += 1
        W[w_base+0][col] += 1; W[w_base+1][col] += 1
        col += 1
        
        # M6
        U[u_base+2][col] += 1; U[u_base+3][col] += 1
        V[v_base+1][col] += 1
        W[w_base+2][col] += 1; W[w_base+3][col] += 1
        col += 1
        
        # M7
        U[u_base+1][col] += 1; U[u_base+3][col] -= 1
        V[v_base+3][col] += 1; V[v_base+2][col] -= 1
        W[w_base+1][col] -= 1; W[w_base+2][col] -= 1
        col += 1

    def add_schoolbook_2x2x3(u_base, v_base, w_base):
        nonlocal col
        # A is 2x2 (indices 0..3)
        # B is 2x3 (indices 0..5: e,f,g,h,i,j)
        # C is 2x3 (indices 0..5: C00,C01,C02,C10,C11,C12)
        
        # C00 = a*e + b*h
        U[u_base+0][col] = 1; V[v_base+0][col] = 1; W[w_base+0][col] = 1; col += 1
        U[u_base+1][col] = 1; V[v_base+3][col] = 1; W[w_base+0][col] = 1; col += 1
        
        # C01 = a*f + b*i
        U[u_base+0][col] = 1; V[v_base+1][col] = 1; W[w_base+1][col] = 1; col += 1
        U[u_base+1][col] = 1; V[v_base+4][col] = 1; W[w_base+1][col] = 1; col += 1
        
        # C02 = a*g + b*j
        U[u_base+0][col] = 1; V[v_base+2][col] = 1; W[w_base+2][col] = 1; col += 1
        U[u_base+1][col] = 1; V[v_base+5][col] = 1; W[w_base+2][col] = 1; col += 1
        
        # C10 = c*e + d*h
        U[u_base+2][col] = 1; V[v_base+0][col] = 1; W[w_base+3][col] = 1; col += 1
        U[u_base+3][col] = 1; V[v_base+3][col] = 1; W[w_base+3][col] = 1; col += 1
        
        # C11 = c*f + d*i
        U[u_base+2][col] = 1; V[v_base+1][col] = 1; W[w_base+4][col] = 1; col += 1
        U[u_base+3][col] = 1; V[v_base+4][col] = 1; W[w_base+4][col] = 1; col += 1
        
        # C12 = c*g + d*j
        U[u_base+2][col] = 1; V[v_base+2][col] = 1; W[w_base+5][col] = 1; col += 1
        U[u_base+3][col] = 1; V[v_base+5][col] = 1; W[w_base+5][col] = 1; col += 1

    # Block 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, all cols)
    # A1 indices in U: 0..3 (i*m+j)
    # B1 indices in V: 0..9 (j*p+k) -> j=0,1. k=0..4.
    # C1 indices in W: 0..9 (k*n+i) -> k=0..4, i=0,1.
    
    # Split B1 into B1L (cols 0-1) and B1R (cols 2-4)
    # B1L: V indices 0..3 (j=0,1; k=0,1)
    # B1R: V indices 4..9 (j=0,1; k=2,3,4)
    
    # C1L: W indices for k=0,1.
    # W index = k*n + i.
    # k=0: i=0->0, i=1->1.
    # k=1: i=0->2, i=1->3.
    # So C1L W indices: 0,1,2,3.
    
    # C1R: W indices for k=2,3,4.
    # k=2: i=0->4, i=1->5.
    # k=3: i=0->6, i=1->7.
    # k=4: i=0->8, i=1->9.
    # So C1R W indices: 4,5,6,7,8,9.
    
    add_strassen(0, 0, 0) # A1 * B1L
    add_schoolbook_2x2x3(0, 4, 4) # A1 * B1R
    
    # Block 2: A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, all cols)
    # A2 indices in U: 4..7 (i*m+j, j=2,3)
    # i=0: j=2->2, j=3->3. (Wait, i*m+j. m=4. i=0, j=2 -> 2. i=0, j=3 -> 3. i=1, j=2 -> 6. i=1, j=3 -> 7.)
    # So A2 U indices: 2,3,6,7.
    # But my add_strassen expects contiguous indices 0..3.
    # I need to map them.
    
    # Let's create a helper to map indices.
    # For A2:
    # a -> U[2], b -> U[3], c -> U[6], d -> U[7].
    
    # B2 indices in V: j=2,3.
    # j=2: k=0..4 -> V[10..14].
    # j=3: k=0..4 -> V[15..19].
    # B2L (k=0,1):
    # j=2, k=0 -> V[10]. j=2, k=1 -> V[11].
    # j=3, k=0 -> V[15]. j=3, k=1 -> V[16].
    # So B2L V indices: 10, 11, 15, 16.
    
    # B2R (k=2,3,4):
    # j=2, k=2 -> V[12]. j=2, k=3 -> V[13]. j=2, k=4 -> V[14].
    # j=3, k=2 -> V[17]. j=3, k=3 -> V[18]. j=3, k=4 -> V[19].
    # So B2R V indices: 12, 13, 14, 17, 18, 19.
    
    # C2 indices in W: same as C1.
    # C2L W indices: 0,1,2,3.
    # C2R W indices: 4,5,6,7,8,9.
    
    # We need a version of add_strassen that takes index lists.
    
    def add_strassen_mapped(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [a,b,c,d]
        # v_idx: [e,f,g,h]
        # w_idx: [C00,C01,C10,C11]
        
        # M1
        U[u_idx[0]][col] += 1; U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1; V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1; W[w_idx[1]][col] += 1
        W[w_idx[2]][col] += 1; W[w_idx[3]][col] += 1
        col += 1
        
        # M2
        U[u_idx[2]][col] += 1; U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1; V[v_idx[3]][col] += 1
        W[w_idx[2]][col] += 1; W[w_idx[3]][col] += 1
        col += 1
        
        # M3
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1; V[v_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1; W[w_idx[3]][col] -= 1
        col += 1
        
        # M4
        U[u_idx[2]][col] += 1
        V[v_idx[0]][col] += 1; V[v_idx[2]][col] -= 1
        W[w_idx[0]][col] -= 1; W[w_idx[2]][col] += 1
        col += 1
        
        # M5
        U[u_idx[0]][col] += 1; U[u_idx[1]][col] += 1
        V[v_idx[2]][col] += 1
        W[w_idx[0]][col] += 1; W[w_idx[1]][col] += 1
        col += 1
        
        # M6
        U[u_idx[2]][col] += 1; U[u_idx[3]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[2]][col] += 1; W[w_idx[3]][col] += 1
        col += 1
        
        # M7
        U[u_idx[1]][col] += 1; U[u_idx[3]][col] -= 1
        V[v_idx[3]][col] += 1; V[v_idx[2]][col] -= 1
        W[w_idx[1]][col] -= 1; W[w_idx[2]][col] -= 1
        col += 1

    def add_schoolbook_2x2x3_mapped(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [a,b,c,d]
        # v_idx: [e,f,g,h,i,j]
        # w_idx: [C00,C01,C02,C10,C11,C12]
        
        U[u_idx[0]][col] = 1; V[v_idx[0]][col] = 1; W[w_idx[0]][col] = 1; col += 1
        U[u_idx[1]][col] = 1; V[v_idx[3]][col] = 1; W[w_idx[0]][col] = 1; col += 1
        
        U[u_idx[0]][col] = 1; V[v_idx[1]][col] = 1; W[w_idx[1]][col] = 1; col += 1
        U[u_idx[1]][col] = 1; V[v_idx[4]][col] = 1; W[w_idx[1]][col] = 1; col += 1
        
        U[u_idx[0]][col] = 1; V[v_idx[2]][col] = 1; W[w_idx[2]][col] = 1; col += 1
        U[u_idx[1]][col] = 1; V[v_idx[5]][col] = 1; W[w_idx[2]][col] = 1; col += 1
        
        U[u_idx[2]][col] = 1; V[v_idx[0]][col] = 1; W[w_idx[3]][col] = 1; col += 1
        U[u_idx[3]][col] = 1; V[v_idx[3]][col] = 1; W[w_idx[3]][col] = 1; col += 1
        
        U[u_idx[2]][col] = 1; V[v_idx[1]][col] = 1; W[w_idx[4]][col] = 1; col += 1
        U[u_idx[3]][col] = 1; V[v_idx[4]][col] = 1; W[w_idx[4]][col] = 1; col += 1
        
        U[u_idx[2]][col] = 1; V[v_idx[2]][col] = 1; W[w_idx[5]][col] = 1; col += 1
        U[u_idx[3]][col] = 1; V[v_idx[5]][col] = 1; W[w_idx[5]][col] = 1; col += 1

    # A1 indices: [0, 1, 4, 5] (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # Wait, i*m+j.
    # i=0, j=0 -> 0.
    # i=0, j=1 -> 1.
    # i=1, j=0 -> 4.
    # i=1, j=1 -> 5.
    u_idx_A1 = [0, 1, 4, 5]
    
    # B1L indices:
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # j=1, k=0 -> 5.
    # j=1, k=1 -> 6.
    v_idx_B1L = [0, 1, 5, 6]
    
    # C1L indices:
    # k=0, i=0 -> 0.
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    w_idx_C1L = [0, 1, 2, 3]
    
    add_strassen_mapped(u_idx_A1, v_idx_B1L, w_idx_C1L)
    
    # B1R indices:
    # j=0, k=2 -> 2.
    # j=0, k=3 -> 3.
    # j=0, k=4 -> 4.
    # j=1, k=2 -> 7.
    # j=1, k=3 -> 8.
    # j=1, k=4 -> 9.
    v_idx_B1R = [2, 3, 4, 7, 8, 9]
    
    # C1R indices:
    # k=2, i=0 -> 4.
    # k=2, i=1 -> 5.
    # k=3, i=0 -> 6.
    # k=3, i=1 -> 7.
    # k=4, i=0 -> 8.
    # k=4, i=1 -> 9.
    w_idx_C1R = [4, 5, 6, 7, 8, 9]
    
    add_schoolbook_2x2x3_mapped(u_idx_A1, v_idx_B1R, w_idx_C1R)
    
    # A2 indices:
    # i=0, j=2 -> 2.
    # i=0, j=3 -> 3.
    # i=1, j=2 -> 6.
    # i=1, j=3 -> 7.
    u_idx_A2 = [2, 3, 6, 7]
    
    # B2L indices:
    # j=2, k=0 -> 10.
    # j=2, k=1 -> 11.
    # j=3, k=0 -> 15.
    # j=3, k=1 -> 16.
    v_idx_B2L = [10, 11, 15, 16]
    
    # C2L indices: same as C1L.
    w_idx_C2L = [0, 1, 2, 3]
    
    add_strassen_mapped(u_idx_A2, v_idx_B2L, w_idx_C2L)
    
    # B2R indices:
    # j=2, k=2 -> 12.
    # j=2, k=3 -> 13.
    # j=2, k=4 -> 14.
    # j=3, k=2 -> 17.
    # j=3, k=3 -> 18.
    # j=3, k=4 -> 19.
    v_idx_B2R = [12, 13, 14, 17, 18, 19]
    
    # C2R indices: same as C1R.
    w_idx_C2R = [4, 5, 6, 7, 8, 9]
    
    add_schoolbook_2x2x3_mapped(u_idx_A2, v_idx_B2R, w_idx_C2R)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
