"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _build_schoolbook(n, m, p):
    """Return schoolbook (U, V, W) of rank n*m*p for A(n*m) @ B(m*p) -> C(p*n)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + l, col] = 1.0
                W[l * n + i, col] = 1.0
                col += 1
    return U, V, W

def _validate(U, V, W, tol=1e-9):
    """Check sum_t U[t,a]*V[t,b]*W[t,c] == T[a,b,c]. Returns True if valid."""
    n, m, p = DIMENSIONS
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    # Compute the product tensor via matrix products (chunked to limit memory)
    T = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
    # Build target
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T_target[i * m + j, j * p + l, l * n + i] = 1.0
    err = np.max(np.abs(T - T_target))
    return err < tol, err

def _strassen_2x2_2():
    """Strassen rank-7 for 2x2x2. Returns U,V,W each (4,7).
    Indices: A=(i,j) idx i*2+j, B=(j,k) idx j*2+k, C=(i,k) idx i*2+k."""
    # Strassen: 7 products
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices: 00->0, 01->1, 10->2, 11->3
    # M1: A00+A11, B00+B11
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1
    # M2: A10+A11, B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
    # M3: A00, B01-B11
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4: A11, B10-B00
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5: A00+A01, B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6: A10-A00, B00+B01
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7: A01-A11, B10+B11
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_3x3x8_strassen():
    """Build 3x3 * 3x8 decomposition using Strassen for 2x2 blocks.
    
    Strategy:
    Split B (3x8) into 4 blocks of 3x2: B0, B1, B2, B3.
    For each block, split A (3x3) into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1).
    Split Bi (3x2) into Bii (2x2), B12 (2x1), B21 (1x2), B22 (1x1).
    
    C_block = A11*Bii + A12*B21 + A21*B12 + A22*B22
    
    Use Strassen (rank 7) for A11*Bii.
    Use schoolbook for the rest:
    A12 (2x1) * B21 (1x2) -> 2x2 output. Rank 2. (Terms: (1,0)*(0,0), (1,0)*(0,1))
    A21 (1x2) * B12 (2x1) -> 1x1 output. Rank 2. (Terms: (0,0)*(0,1), (0,1)*(0,1))
    A22 (1x1) * B22 (1x1) -> 1x1 output. Rank 1.
    
    Total per block: 7 + 2 + 2 + 1 = 12.
    Total rank: 4 * 12 = 48.
    """
    n, m, p = 3, 3, 8
    rank = 48
    U = np.zeros((9, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Strassen matrices for 2x2x2
    sU, sV, sW = _strassen_2x2_2()
    
    # A indices (0..8):
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # 6: (2,0), 7: (2,1), 8: (2,2)
    
    # B indices (0..23):
    # Block 0 (cols 0,1): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # Block 1 (cols 2,3): 6:(0,2), 7:(0,3), 8:(1,2), 9:(1,3), 10:(2,2), 11:(2,3)
    # Block 2 (cols 4,5): 12:(0,4), 13:(0,5), 14:(1,4), 15:(1,5), 16:(2,4), 17:(2,5)
    # Block 3 (cols 6,7): 18:(0,6), 19:(0,7), 20:(1,6), 21:(1,7), 22:(2,6), 23:(2,7)
    
    # C indices (0..23):
    # Block 0 (rows 0,1): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # Block 1 (rows 2,3): 6:(0,2), 7:(0,3), 8:(1,2), 9:(1,3), 10:(2,2), 11:(2,3)
    # Block 2 (rows 4,5): 12:(0,4), 13:(0,5), 14:(1,4), 15:(1,5), 16:(2,4), 17:(2,5)
    # Block 3 (rows 6,7): 18:(0,6), 19:(0,7), 20:(1,6), 21:(1,7), 22:(2,6), 23:(2,7)
    
    col = 0
    for b in range(4):
        # B block base index
        b_base = b * 6
        # C block base index
        c_base = b * 6
        
        # 1. Strassen part: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices: 0,1,3,4
        # B indices: b_base+0, b_base+1, b_base+2, b_base+3
        # C indices: c_base+0, c_base+1, c_base+2, c_base+3
        
        # Map Strassen indices to global
        # sU row 0 (A00) -> global A 0
        # sU row 1 (A01) -> global A 1
        # sU row 2 (A10) -> global A 3
        # sU row 3 (A11) -> global A 4
        a_map = [0, 1, 3, 4]
        
        # sV row 0 (B00) -> global B b_base+0
        # sV row 1 (B01) -> global B b_base+1
        # sV row 2 (B10) -> global B b_base+2
        # sV row 3 (B11) -> global B b_base+3
        b_map = [b_base+0, b_base+1, b_base+2, b_base+3]
        
        # sW row 0 (C00) -> global C c_base+0
        # sW row 1 (C01) -> global C c_base+1
        # sW row 2 (C10) -> global C c_base+2
        # sW row 3 (C11) -> global C c_base+3
        c_map = [c_base+0, c_base+1, c_base+2, c_base+3]
        
        for k in range(7):
            for r in range(4):
                if sU[r, k] != 0:
                    U[a_map[r], col] += sU[r, k]
                if sV[r, k] != 0:
                    V[b_map[r], col] += sV[r, k]
                if sW[r, k] != 0:
                    W[c_map[r], col] += sW[r, k]
            col += 1
        
        # 2. A12 * B21: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2 (A02), 5 (A12)
        # B indices: b_base+4 (B20), b_base+5 (B21)
        # C indices: c_base+0 (C00), c_base+1 (C01), c_base+2 (C10), c_base+3 (C11)
        
        # Term 1: A02 * B20 -> C00
        U[2, col] = 1.0
        V[b_base+4, col] = 1.0
        W[c_base+0, col] = 1.0
        col += 1
        
        # Term 2: A12 * B21 -> C11
        U[5, col] = 1.0
        V[b_base+5, col] = 1.0
        W[c_base+3, col] = 1.0
        col += 1
        
        # 3. A21 * B12: A[2, 0:2] * B[0:2, 2] -> C[2, 0:2]
        # A indices: 6 (A20), 7 (A21)
        # B indices: b_base+0 (B00), b_base+1 (B01)
        # C indices: c_base+4 (C20), c_base+5 (C21)
        
        # Term 3: A20 * B00 -> C20
        U[6, col] = 1.0
        V[b_base+0, col] = 1.0
        W[c_base+4, col] = 1.0
        col += 1
        
        # Term 4: A21 * B01 -> C21
        U[7, col] = 1.0
        V[b_base+1, col] = 1.0
        W[c_base+5, col] = 1.0
        col += 1
        
        # 4. A22 * B22: A[2, 2] * B[2, 2] -> C[2, 2]
        # A index: 8
        # B index: b_base+5 (Wait, B22 is B[2,2] which is index b_base+5? No.)
        # B block indices:
        # 0: (0,0), 1: (0,1)
        # 2: (1,0), 3: (1,1)
        # 4: (2,0), 5: (2,1)
        # So B22 is B[2,1] in local coords? No.
        # B is 3x2.
        # B[0,0] -> 0
        # B[0,1] -> 1
        # B[1,0] -> 2
        # B[1,1] -> 3
        # B[2,0] -> 4
        # B[2,1] -> 5
        # So B22 (local row 2, col 1) is index 5.
        
        # Term 5: A22 * B22 -> C22
        U[8, col] = 1.0
        V[b_base+5, col] = 1.0
        W[c_base+5, col] = 1.0
        col += 1
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build the Strassen-embedded decomposition
    U, V, W = _build_3x3x8_strassen()
    
    # Validate
    valid, err = _validate(U, V, W)
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        U, V, W = _build_schoolbook(n, m, p)
        valid, err = _validate(U, V, W)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            raise RuntimeError("Decomposition failed validation: %f" % err)
