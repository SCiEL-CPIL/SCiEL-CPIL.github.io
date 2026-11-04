import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _laderman_4x4():
    """
    Laderman's rank-37 decomposition for 4x4 x 4x4 -> 4x4.
    Returns U(16,37), V(16,37), W(16,37) with W indexed by (j*4+i).
    """
    U = np.zeros((16, 37))
    V = np.zeros((16, 37))
    W = np.zeros((16, 37))
    
    def add(r, u_terms, v_terms, w_terms):
        for idx, c in u_terms:
            U[idx, r] += c
        for idx, c in v_terms:
            V[idx, r] += c
        for idx, c in w_terms:
            W[idx, r] += c
    
    # Using 0-indexed: A[i,k], B[k,j], C[i,j], W[j*4+i]
    # Laderman 37 terms (verified formulation):
    
    # 7 Strassen-like terms for the top-left 2x2 block structure
    # M1=(A00+A11)(B00+B11)
    add(0, [(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(4,1)])
    # M2=(A20+A21)B00  -> note: A20 means row2,col0 -> idx=2*4+0=8
    add(1, [(8,1),(9,1)], [(0,1)], [(1,1),(4,-1)])
    # M3=A00(B01-B11)
    add(2, [(0,1)], [(1,1),(5,-1)], [(0,1),(4,1)])  # C01: W[1*4+0]=W[4], C11: W[1*4+1]=W[5]... 
    
    # This is getting error-prone. Let me use a different verified source.
    # I'll construct it programmatically from known correct bilinear forms.
    
    U[:] = 0; V[:] = 0; W[:] = 0
    
    # Verified Laderman 37-term decomposition
    # Each term: (list of (A_idx, coeff)), (list of (B_idx, coeff)), (list of (C_idx, coeff))
    # A_idx = i*4+k, B_idx = k*4+j, C_idx = j*4+i
    
    terms = [
        # Term 0: (A00+A11)(B00+B11) -> C00+C11
        ([(0,1),(5,1)], [(0,1),(5,1)], [(0,1),(5,1)]),
        # Term 1: (A20+A21)B00 -> C20-C21  (A20=idx8, A21=idx9)
        ([(8,1),(9,1)], [(0,1)], [(8,1),(9,-1)]),
        # Term 2: A00(B01-B11) -> C01+C11  (B01=idx1, B11=idx5)
        ([(0,1)], [(1,1),(5,-1)], [(1,1),(5,1)]),
        # Term 3: A11(B10-B00) -> C10+C00  (B10=idx4, B00=idx0)
        ([(5,1)], [(4,1),(0,-1)], [(4,1),(0,1)]),
        # Term 4: (A00+A01)B11 -> C00-C10  (A00=0, A01=1, B11=5)
        ([(0,1),(1,1)], [(5,1)], [(0,1),(4,-1)]),
        # Term 5: (A20-A00)(B00+B01) -> C21  (A20=8, A00=0, B00=0, B01=1)
        ([(8,1),(0,-1)], [(0,1),(1,1)], [(9,1)]),
        # Term 6: (A01-A11)(B10+B11) -> C10  (A01=1, A11=5, B10=4, B11=5)
        ([(1,1),(5,-1)], [(4,1),(5,1)], [(4,1)]),
    ]
    
    # Terms 7-16: handle interactions with indices 2,3
    # M7=(A02+A20)(B02+B20) -> C02+C20
    terms.append(([(2,1),(8,1)], [(2,1),(8,1)], [(2,1),(8,1)]))
    # M8=(A03+A30)(B03+B30) -> C03+C30
    terms.append(([(3,1),(12,1)], [(3,1),(12,1)], [(3,1),(12,1)]))
    # M9=(A12+A21)(B12+B21) -> C12+C21
    terms.append(([(6,1),(9,1)], [(6,1),(9,1)], [(6,1),(9,1)]))
    # M10=(A13+A31)(B13+B31) -> C13+C31
    terms.append(([(7,1),(13,1)], [(7,1),(13,1)], [(7,1),(13,1)]))
    # M11=(A23+A32)(B23+B32) -> C23+C32
    terms.append(([(11,1),(14,1)], [(11,1),(14,1)], [(11,1),(14,1)]))
    
    # M12=(A02+A20)(B03+B30) -> C03+C20
    terms.append(([(2,1),(8,1)], [(3,1),(12,1)], [(3,1),(8,1)]))
    # M13=(A03+A30)(B02+B20) -> C02+C30
    terms.append(([(3,1),(12,1)], [(2,1),(8,1)], [(2,1),(12,1)]))
    # M14=(A12+A21)(B13+B31) -> C13+C21
    terms.append(([(6,1),(9,1)], [(7,1),(13,1)], [(7,1),(9,1)]))
    # M15=(A13+A31)(B12+B21) -> C12+C31
    terms.append(([(7,1),(13,1)], [(6,1),(9,1)], [(6,1),(13,1)]))
    # M16=(A23+A32)(B23+B32) -> C23+C32 (duplicate of M11? No)
    # Need distinct terms. Let me use:
    # M16=(A02+A30)(B03+B20) -> C03+C20? This is getting messy.
    
    # I'm not confident in the Laderman construction. Let me use a safer approach.
    # Use Strassen rank 49 for 4x4x4, giving total rank 49+49+16=114.
    # But try to reduce: for the last column (4x1), can I do better than 16?
    # No, 4x4 x 4x1 -> 4x1 requires rank 16 (each output needs 4 mults).
    
    # Alternative: group columns differently. 4x4 x 4x9 = 4x4 x (4x3 + 4x3 + 4x3)
    # For 4x4 x 4x3: rank at most 4*4*3=48, or use Strassen-like?
    # Actually, for rectangular, we can still use block algorithms.
    
    # Let me just use the verified Strassen approach and ensure correctness.
    return None  # Signal to use Strassen

def _strassen_2x2():
    """Strassen 2x2x2, rank 7. U(4,7),V(4,7),W(4,7). W[j*2+i]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1=(A00+A11)(B00+B11) -> C00+C11
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A10+A11)B00 -> C10-C11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3=A00(B01-B11) -> C01+C11
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4=A11(B10-B00) -> C10+C00
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5=(A00+A01)B11 -> C00-C10
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6=(A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _strassen_4x4x4():
    """Strassen 4x4x4 via 2-level, rank 49. U(16,49),V(16,49),W(16,49)."""
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Corrected Strassen 2x2x2 terms (verified):
    # M1=(A00+A11)(B00+B11) -> C00+C11
    # M2=(A10+A11)B00 -> C10-C11
    # M3=A00(B01-B11) -> C01+C11
    # M4=A11(B10-B00) -> C10+C00
    # M5=(A00+A01)B11 -> C00-C10
    # M6=(A10-A00)(B00+B01) -> C11
    # M7=(A01-A11)(B10+B11) -> C00
    
    big = [
        # (A_block_terms, B_block_terms, C_block_terms)
        # Each term: (block_row, block_col, coeff)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(1,0,-1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for bi, (A_b, B_b, C_b) in enumerate(big):
        for si in range(7):
            r = bi*7 + si
            for (bi_i, bi_k, ca) in A_b:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*bi_i + i2; k = 2*bi_k + k2
                        U[i*4+k, r] += ca * U2[i2*2+k2, si]
            for (bk_i, bk_j, cb) in B_b:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2*bk_i + k2; j = 2*bk_j + j2
                        V[k*4+j, r] += cb * V2[k2*2+j2, si]
            for (cj_i, cj_j, cc) in C_b:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2*cj_i + j2; i = 2*cj_j + i2
                        W[j*4+i, r] += cc * W2[j2*2+i2, si]
    return U, V, W

def _build_4x4x9():
    """
    4x4 x 4x9 -> 4x9.
    Exploit rank-sharing: A is the same for all column blocks.
    Use a single U(16, 49) for Strassen on A, and separate V/W for each column block.
    For cols 0-3: V uses Strassen, W uses Strassen W4.
    For cols 4-7: V uses Strassen, W uses Strassen W4 shifted.
    For col 8: direct rank 16.
    
    Total rank = 49 + 49 + 16 = 114, but U is shared.
    Actually in the tensor decomposition, rank is the number of columns in U,V,W.
    Since U must have the same number of columns as V and W, we can't truly share.
    But we CAN set U to be the same 49 columns repeated, and let V/W differ.
    That still gives rank 114.
    
    Better approach: Use the fact that for the last column (4x1), we need rank 16.
    For the first 8 columns, we can use a 4x4 x 4x8 decomposition.
    Split 4x8 into two 4x4 blocks: rank 49+49=98 for those, plus 16 for last col = 114.
    
    Alternative: Can we do 4x4 x 4x8 with rank < 98?
    4x4 x 4x8 = two independent 4x4 x 4x4, so rank >= 2*37 = 74 (if Laderman works).
    But without verified Laderman, we stick with Strassen: 98 + 16 = 114.
    
    Wait - the key insight from inspiration 3: U is identical across blocks.
    In the tensor formulation, if U[:, 0:49] = U[:, 49:98], that's fine - 
    it just means those rank-1 terms happen to have the same A-factor.
    The rank is still 114. There's no way to reduce below 114 with this approach.
    
    Let me verify correctness first, then see if there's a better decomposition.
    """
    U4, V4, W4 = _strassen_4x4x4()
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Block 1: cols 0-3
    U[:, 0:49] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W4[j*4+i, :]
    
    # Block 2: cols 4-7
    U[:, 49:98] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W4[j*4+i, :]
    
    # Block 3: col 8 (direct, rank 16)
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1
            V[k*9+8, r] = 1
            W[8*4+i, r] = 1
    
    return U, V, W, rank

def construct():
    start = time.time()
    deadline = start + 140
    
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_4x4x9()
    
    # Verify with multiple random tests
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 9)
        C_exp = A @ B
        
        Ur = U.reshape(4, 4, rank)
        Vr = V.reshape(4, 9, rank)
        Wr = W.reshape(9, 4, rank)
        
        a = np.einsum('ikr,ik->ir', Ur, A)
        b = np.einsum('kjr,kj->jr', Vr, B)
        C = np.einsum('jir,ir,jr->ij', Wr, a, b)
        
        if not np.allclose(C, C_exp, atol=1e-10):
            # Fallback: direct rank 144
            rank = 144
            U = np.zeros((16, rank))
            V = np.zeros((36, rank))
            W = np.zeros((36, rank))
            idx = 0
            for i in range(4):
                for j in range(9):
                    for k in range(4):
                        U[i*4+k, idx] = 1
                        V[k*9+j, idx] = 1
                        W[j*4+i, idx] = 1
                        idx += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
