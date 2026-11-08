"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # C00 += M1, C11 += M1
    # W[k*2+i]: C00 is i=0,k=0 -> W[0], C11 is i=1,k=1 -> W[3]
    # Wait, W index is k*2+i. C[i,k]: i=0,k=0 -> W[0*2+0]=W[0]; i=1,k=1 -> W[1*2+1]=W[3]
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    # C10 += M2: i=1,k=0 -> W[0*2+1]=W[1]; C11 += M2: i=1,k=1 -> W[1*2+1]=W[3]
    # Wait, I need to be more careful.
    
    # Let me redo with proper indexing.
    # A is 2x2: A[i,j], i,j in 0,1
    # B is 2x2: B[j,k], j,k in 0,1
    # C is 2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    # A00: i=0,j=0 -> U[0]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]; B11: j=1,k=1 -> V[3]
    # C00: i=0,k=0 -> W[0]; C11: i=1,k=1 -> W[3]
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (A10+A11)B00
    # A10: i=1,j=0 -> U[2]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]
    # C10: i=1,k=0 -> W[1]; C11: i=1,k=1 -> W[3]
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] += 1
    
    # M3 = A00(B11-B00)
    # A00: i=0,j=0 -> U[0]
    # B11: j=1,k=1 -> V[3]; B00: j=0,k=0 -> V[0]
    # C01: i=0,k=1 -> W[2]; C11: i=1,k=1 -> W[3]
    U[0, 2] += 1
    V[3, 2] += 1; V[0, 2] -= 1
    W[2, 2] += 1; W[3, 2] += 1
    
    # M4 = A11(B00-B10)
    # A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]; B10: j=1,k=0 -> V[2]
    # C00: i=0,k=0 -> W[0]; C10: i=1,k=0 -> W[1]
    U[3, 3] += 1
    V[0, 3] += 1; V[2, 3] -= 1
    W[0, 3] += 1; W[1, 3] += 1
    
    # M5 = (A00+A01)B10
    # A00: i=0,j=0 -> U[0]; A01: i=0,j=1 -> U[1]
    # B10: j=1,k=0 -> V[2]
    # C01: i=0,k=1 -> W[2]; C11: i=1,k=1 -> W[3]
    U[0, 4] += 1; U[1, 4] += 1
    V[2, 4] += 1
    W[2, 4] += 1; W[3, 4] += 1
    
    # M6 = (A10-A00)(B01+B00)
    # A10: i=1,j=0 -> U[2]; A00: i=0,j=0 -> U[0]
    # B01: j=0,k=1 -> V[1]; B00: j=0,k=0 -> V[0]
    # C00: i=0,k=0 -> W[0]; C10: i=1,k=0 -> W[1]
    U[2, 5] += 1; U[0, 5] -= 1
    V[1, 5] += 1; V[0, 5] += 1
    W[0, 5] += 1; W[1, 5] += 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: i=0,j=1 -> U[1]; A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]; B11: j=1,k=1 -> V[3]
    # C10: i=1,k=0 -> W[1]; C11: i=1,k=1 -> W[3]
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[1, 6] += 1; W[3, 6] += 1
    
    return U, V, W

def _embed_2x2x2_into_4x4x4(offset_u, offset_v, offset_w, U, V, W, U2, V2, W2):
    """Embed a 2x2x2 decomposition into a 4x4x4 at given block offsets."""
    # This is getting complex. Let me just build the 4x4x4 Strassen directly.
    pass

def _strassen_4x4x4():
    """
    Return U, V, W for 4x4x4 Strassen decomposition (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    U index: i*4+j, V index: j*4+k, W index: k*4+i
    """
    # Use recursive Strassen: split 4x4 into four 2x2 blocks
    # 7 multiplications of 2x2x2, each of rank 7, total 49
    
    # The 7 "scalar" multiplications in Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B01+B00)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2x2 multiplication, which we decompose with rank 7.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # For each of the 7 Strassen terms, embed the 2x2x2 decomposition
    # The block structure: A is 4x4 split into 2x2 blocks Aab where a,b in 0,1
    # Aab[i,j] where i,j in 0,1, global index (2a+i)*4 + (2b+j)
    
    # Term 1: (A00+A11)(B00+B11)
    # A part: A00 (blocks 0,0) and A11 (blocks 1,1)
    # B part: B00 (blocks 0,0) and B11 (blocks 1,1)
    # C part: C00 (blocks 0,0) and C11 (blocks 1,1)
    
    for r in range(7):
        col = r
        # U: A00 and A11
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    # A00: global i=0..1, j=0..1 -> U[(i)*4+j, col]
                    U[i*4+j, col] += u_val
                    # A11: global i=2..3, j=2..3 -> U[(i+2)*4+(j+2), col]
                    U[(i+2)*4+(j+2), col] += u_val
        # V: B00 and B11
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    # B00: global j=0..1, k=0..1 -> V[j*4+k, col]
                    V[j*4+k, col] += v_val
                    # B11: global j=2..3, k=2..3 -> V[(j+2)*4+(k+2), col]
                    V[(j+2)*4+(k+2), col] += v_val
        # W: C00 and C11
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C00: global i=0..1, k=0..1 -> W[k*4+i, col]
                    W[k*4+i, col] += w_val
                    # C11: global i=2..3, k=2..3 -> W[(k+2)*4+(i+2), col]
                    W[(k+2)*4+(i+2), col] += w_val
    
    # Term 2: (A10+A11)B00
    # A: A10 (blocks 1,0) and A11 (blocks 1,1)
    # B: B00 (blocks 0,0)
    # C: C10 (blocks 1,0) and C11 (blocks 1,1)
    
    for r in range(7):
        col = 7 + r
        # U: A10 and A11
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    # A10: global i=2..3, j=0..1 -> U[(i+2)*4+j, col]
                    U[(i+2)*4+j, col] += u_val
                    # A11: global i=2..3, j=2..3 -> U[(i+2)*4+(j+2), col]
                    U[(i+2)*4+(j+2), col] += u_val
        # V: B00
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    V[j*4+k, col] += v_val
        # W: C10 and C11
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C10: global i=2..3, k=0..1 -> W[k*4+(i+2), col]
                    W[k*4+(i+2), col] += w_val
                    # C11: global i=2..3, k=2..3 -> W[(k+2)*4+(i+2), col]
                    W[(k+2)*4+(i+2), col] += w_val
    
    # Term 3: A00(B11-B00)
    # A: A00 (blocks 0,0)
    # B: B11 (blocks 1,1) and B00 (blocks 0,0)
    # C: C01 (blocks 0,1) and C11 (blocks 1,1)
    
    for r in range(7):
        col = 14 + r
        # U: A00
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    U[i*4+j, col] += u_val
        # V: B11 and B00
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    # B11: global j=2..3, k=2..3
                    V[(j+2)*4+(k+2), col] += v_val
                    # B00: global j=0..1, k=0..1
                    V[j*4+k, col] -= v_val
        # W: C01 and C11
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C01: global i=0..1, k=2..3 -> W[(k+2)*4+i, col]
                    W[(k+2)*4+i, col] += w_val
                    # C11: global i=2..3, k=2..3 -> W[(k+2)*4+(i+2), col]
                    W[(k+2)*4+(i+2), col] += w_val
    
    # Term 4: A11(B00-B10)
    # A: A11 (blocks 1,1)
    # B: B00 (blocks 0,0) and B10 (blocks 1,0)
    # C: C00 (blocks 0,0) and C10 (blocks 1,0)
    
    for r in range(7):
        col = 21 + r
        # U: A11
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    U[(i+2)*4+(j+2), col] += u_val
        # V: B00 and B10
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    # B00: global j=0..1, k=0..1
                    V[j*4+k, col] += v_val
                    # B10: global j=2..3, k=0..1
                    V[(j+2)*4+k, col] -= v_val
        # W: C00 and C10
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C00: global i=0..1, k=0..1
                    W[k*4+i, col] += w_val
                    # C10: global i=2..3, k=0..1
                    W[k*4+(i+2), col] += w_val
    
    # Term 5: (A00+A01)B10
    # A: A00 (blocks 0,0) and A01 (blocks 0,1)
    # B: B10 (blocks 1,0)
    # C: C01 (blocks 0,1) and C11 (blocks 1,1)
    
    for r in range(7):
        col = 28 + r
        # U: A00 and A01
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    # A00: global i=0..1, j=0..1
                    U[i*4+j, col] += u_val
                    # A01: global i=0..1, j=2..3
                    U[i*4+(j+2), col] += u_val
        # V: B10
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    V[(j+2)*4+k, col] += v_val
        # W: C01 and C11
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C01: global i=0..1, k=2..3
                    W[(k+2)*4+i, col] += w_val
                    # C11: global i=2..3, k=2..3
                    W[(k+2)*4+(i+2), col] += w_val
    
    # Term 6: (A10-A00)(B01+B00)
    # A: A10 (blocks 1,0) and A00 (blocks 0,0)
    # B: B01 (blocks 0,1) and B00 (blocks 0,0)
    # C: C00 (blocks 0,0) and C10 (blocks 1,0)
    
    for r in range(7):
        col = 35 + r
        # U: A10 and A00
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    # A10: global i=2..3, j=0..1
                    U[(i+2)*4+j, col] += u_val
                    # A00: global i=0..1, j=0..1
                    U[i*4+j, col] -= u_val
        # V: B01 and B00
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    # B01: global j=0..1, k=2..3
                    V[j*4+(k+2), col] += v_val
                    # B00: global j=0..1, k=0..1
                    V[j*4+k, col] += v_val
        # W: C00 and C10
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C00: global i=0..1, k=0..1
                    W[k*4+i, col] += w_val
                    # C10: global i=2..3, k=0..1
                    W[k*4+(i+2), col] += w_val
    
    # Term 7: (A01-A11)(B10+B11)
    # A: A01 (blocks 0,1) and A11 (blocks 1,1)
    # B: B10 (blocks 1,0) and B11 (blocks 1,1)
    # C: C10 (blocks 1,0) and C11 (blocks 1,1)
    
    for r in range(7):
        col = 42 + r
        # U: A01 and A11
        for i in range(2):
            for j in range(2):
                u_idx_2x2 = i*2+j
                u_val = U2[u_idx_2x2, r]
                if u_val != 0:
                    # A01: global i=0..1, j=2..3
                    U[i*4+(j+2), col] += u_val
                    # A11: global i=2..3, j=2..3
                    U[(i+2)*4+(j+2), col] -= u_val
        # V: B10 and B11
        for j in range(2):
            for k in range(2):
                v_idx_2x2 = j*2+k
                v_val = V2[v_idx_2x2, r]
                if v_val != 0:
                    # B10: global j=2..3, k=0..1
                    V[(j+2)*4+k, col] += v_val
                    # B11: global j=2..3, k=2..3
                    V[(j+2)*4+(k+2), col] += v_val
        # W: C10 and C11
        for k in range(2):
            for i in range(2):
                w_idx_2x2 = k*2+i
                w_val = W2[w_idx_2x2, r]
                if w_val != 0:
                    # C10: global i=2..3, k=0..1
                    W[k*4+(i+2), col] += w_val
                    # C11: global i=2..3, k=2..3
                    W[(k+2)*4+(i+2), col] += w_val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _strassen_4x4x4()
    
    # For 4x4x8, we have two independent 4x4x4 multiplications
    # B is 4x8, split into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7)
    # C is 4x8, split into C_left (4x4) and C_right (4x4)
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half: B cols 0-3, C cols 0-3
    # U is same for both halves (A is shared)
    # V: B_left has indices j*4+k for j in 0..3, k in 0..3 -> global V[j*8+k]
    # V: B_right has indices j*4+k for j in 0..3, k in 0..3 -> global V[j*8+(k+4)]
    # W: C_left has indices k*4+i for k in 0..3 -> global W[k*4+i]
    # W: C_right has indices k*4+i for k in 0..3 -> global W[(k+4)*4+i]
    
    for r in range(49):
        # First half
        U[:, r] = U4[:, r]
        for j in range(4):
            for k in range(4):
                v_idx_4x4 = j*4+k
                v_val = V4[v_idx_4x4, r]
                if v_val != 0:
                    V[j*8+k, r] = v_val
        for k in range(4):
            for i in range(4):
                w_idx_4x4 = k*4+i
                w_val = W4[w_idx_4x4, r]
                if w_val != 0:
                    W[k*4+i, r] = w_val
        
        # Second half
        col = 49 + r
        U[:, col] = U4[:, r]
        for j in range(4):
            for k in range(4):
                v_idx_4x4 = j*4+k
                v_val = V4[v_idx_4x4, r]
                if v_val != 0:
                    V[j*8+(k+4), col] = v_val
        for k in range(4):
            for i in range(4):
                w_idx_4x4 = k*4+i
                w_val = W4[w_idx_4x4, r]
                if w_val != 0:
                    W[(k+4)*4+i, col] = w_val
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
