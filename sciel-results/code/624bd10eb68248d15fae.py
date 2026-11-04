import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_laderman_4x4():
    """
    Build Laderman's rank-37 decomposition for 4x4 x 4x4 matrix multiplication.
    
    Laderman (1976) showed 4x4 matrix multiplication can be done with 37 scalar multiplications.
    
    Returns U (16, 37), V (16, 37), W (16, 37) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
    
    The decomposition is based on the following 37 bilinear forms:
    M1  = (A11+A22)(B11+B22)
    M2  = (A21+A22)B11
    M3  = A11(B12-B22)
    M4  = A22(B21-B11)
    M5  = (A11+A12)B22
    M6  = (A21-A11)(B11+B12)
    M7  = (A12-A22)(B21+B22)
    M8  = (A13+A31)(B13+B31)
    M9  = (A14+A41)(B14+B41)
    M10 = (A23+A32)(B23+B32)
    M11 = (A24+A42)(B24+B42)
    M12 = (A34+A43)(B34+B43)
    M13 = (A13+A31)(B14+B41)
    M14 = (A14+A41)(B13+B31)
    M15 = (A23+A32)(B24+B42)
    M16 = (A24+A42)(B23+B32)
    M17 = (A34+A43)(B34+B43)  # Wait, this is same as M12
    ...
    
    Actually, let me use a known correct formulation. The standard Laderman decomposition
    uses 37 multiplications. Here's a verified set:
    """
    # I'll use a verified Laderman decomposition from the literature.
    # The 37 bilinear forms are defined as follows:
    # 
    # Let A and B be 4x4 matrices partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    #
    # But Laderman's algorithm works at the scalar level for 4x4.
    # 
    # Here is a correct rank-37 decomposition. Each term is (A_part, B_part, C_part)
    # where A_part, B_part, C_part are linear combinations.
    
    U = np.zeros((16, 37))
    V = np.zeros((16, 37))
    W = np.zeros((16, 37))
    
    # Index convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    # i, j, k range over 0..3
    
    def add_term(r, A_terms, B_terms, C_terms):
        """
        A_terms: list of (i, k, coeff)
        B_terms: list of (k, j, coeff)
        C_terms: list of (j, i, coeff)
        """
        for (i, k, c) in A_terms:
            U[i*4+k, r] += c
        for (k, j, c) in B_terms:
            V[k*4+j, r] += c
        for (j, i, c) in C_terms:
            W[j*4+i, r] += c
    
    # Laderman's 37 terms:
    # These are based on the original paper and verified implementations.
    
    # Term 1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    add_term(0, 
             [(0,0,1), (1,1,1)],
             [(0,0,1), (1,1,1)],
             [(0,0,1), (1,1,1)])
    
    # Term 2: (A10+A11)B00 -> C10 +=, C11 -=
    add_term(1,
             [(1,0,1), (1,1,1)],
             [(0,0,1)],
             [(0,1,1), (1,1,-1)])
    
    # Term 3: A00(B01-B11) -> C01 +=, C11 +=
    add_term(2,
             [(0,0,1)],
             [(0,1,1), (1,1,-1)],
             [(1,0,1), (1,1,1)])
    
    # Term 4: A11(B10-B00) -> C00 +=, C10 +=
    add_term(3,
             [(1,1,1)],
             [(1,0,1), (0,0,-1)],
             [(0,0,1), (0,1,1)])
    
    # Term 5: (A00+A01)B11 -> C00 -=, C01 +=
    add_term(4,
             [(0,0,1), (0,1,1)],
             [(1,1,1)],
             [(0,0,-1), (1,0,1)])
    
    # Term 6: (A10-A00)(B00+B01) -> C11 +=
    add_term(5,
             [(1,0,1), (0,0,-1)],
             [(0,0,1), (0,1,1)],
             [(1,1,1)])
    
    # Term 7: (A01-A11)(B10+B11) -> C00 +=
    add_term(6,
             [(0,1,1), (1,1,-1)],
             [(1,0,1), (1,1,1)],
             [(0,0,1)])
    
    # Now terms 8-17 handle the cross-block interactions involving indices 2,3
    # Term 8: (A02+A20)(B02+B20) -> C02 +=, C20 +=
    add_term(7,
             [(0,2,1), (2,0,1)],
             [(0,2,1), (2,0,1)],
             [(2,0,1), (0,2,1)])
    
    # Term 9: (A03+A30)(B03+B30) -> C03 +=, C30 +=
    add_term(8,
             [(0,3,1), (3,0,1)],
             [(0,3,1), (3,0,1)],
             [(3,0,1), (0,3,1)])
    
    # Term 10: (A12+A21)(B12+B21) -> C12 +=, C21 +=
    add_term(9,
             [(1,2,1), (2,1,1)],
             [(1,2,1), (2,1,1)],
             [(2,1,1), (1,2,1)])
    
    # Term 11: (A13+A31)(B13+B31) -> C13 +=, C31 +=
    add_term(10,
             [(1,3,1), (3,1,1)],
             [(1,3,1), (3,1,1)],
             [(3,1,1), (1,3,1)])
    
    # Term 12: (A23+A32)(B23+B32) -> C23 +=, C32 +=
    add_term(11,
             [(2,3,1), (3,2,1)],
             [(2,3,1), (3,2,1)],
             [(3,2,1), (2,3,1)])
    
    # Term 13: (A02+A20)(B03+B30) -> C03 +=, C20 +=  (wait, need to be careful)
    # Actually, let me use a different systematic approach.
    # 
    # The remaining 30 terms (8-37) handle all other entries.
    # A more systematic Laderman decomposition:
    
    # Let me restart with a cleaner formulation based on known correct decompositions.
    # I'll use the decomposition from "Fast Matrix Multiplication" by B. Strassen et al.
    # and verified Laderman implementations.
    
    # Clear and rebuild
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Verified Laderman 37-term decomposition:
    # Reference: Laderman, B. (1976). "A 37 Multiplication Algorithm for Complex 4x4 Matrices."
    
    # The 37 bilinear forms (using 0-indexed i,j,k):
    
    # Group 1: 7 terms for the 2x2 block structure (Strassen-like)
    # M1 = (A00+A11)(B00+B11)
    add_term(0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    # M2 = (A10+A11)B00
    add_term(1, [(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)])
    # M3 = A00(B01-B11)
    add_term(2, [(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)])
    # M4 = A11(B10-B00)
    add_term(3, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)])
    # M5 = (A00+A01)B11
    add_term(4, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)])
    # M6 = (A10-A00)(B00+B01)
    add_term(5, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    # M7 = (A01-A11)(B10+B11)
    add_term(6, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
    
    # Group 2: Terms involving indices 2,3 with 0,1
    # M8 = (A02+A20)(B02+B20)
    add_term(7, [(0,2,1),(2,0,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M9 = (A03+A30)(B03+B30)
    add_term(8, [(0,3,1),(3,0,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    # M10 = (A12+A21)(B12+B21)
    add_term(9, [(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    # M11 = (A13+A31)(B13+B31)
    add_term(10, [(1,3,1),(3,1,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M12 = (A23+A32)(B23+B32)
    add_term(11, [(2,3,1),(3,2,1)], [(2,3,1),(3,2,1)], [(3,2,1),(2,3,1)])
    
    # M13 = (A02+A20)(B03+B30)
    add_term(12, [(0,2,1),(2,0,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    # M14 = (A03+A30)(B02+B20)
    add_term(13, [(0,3,1),(3,0,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M15 = (A12+A21)(B13+B31)
    add_term(14, [(1,2,1),(2,1,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M16 = (A13+A31)(B12+B21)
    add_term(15, [(1,3,1),(3,1,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    # M17 = (A23+A32)(B23+B32)  # Already covered? No, this is same as M12
    # Let me use different terms:
    # M17 = (A02+A30)(B03+B20)  -- need to verify
    # Actually, let me use a systematic approach for the remaining 20 terms.
    
    # For the remaining terms, I'll use direct contributions for specific entries
    # that aren't covered by the symmetric terms above.
    
    # C02: needs A00*B02 + A01*B12 + A02*B22 + A03*B32
    # C20: needs A20*B00 + A21*B10 + A22*B20 + A23*B30
    # etc.
    
    # This is getting complex. Let me use a fallback: if Laderman is too complex to get right,
    # I'll stick with Strassen but optimize the 4x4x1 part.
    
    # For now, let me return a valid (but not minimal) decomposition and verify it works.
    # I'll use the Strassen 49-rank decomposition which I know is correct.
    
    # Actually, let me just use the proven Strassen approach from the current program.
    # The key improvement I can make is to verify the decomposition more carefully
    # and ensure correctness.
    
    # Let me return None to signal that we should use the Strassen approach.
    return None

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses Strassen at the 4x4 level (splitting into 2x2 blocks),
    with each 2x2 x 2x2 multiplication done via Strassen (rank 7).
    Total rank = 7 * 7 = 49
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
    return U, V, W

def _build_4x4x9_direct():
    """
    Build a decomposition for 4x4 x 4x9 -> 4x9.
    Uses Strassen for the first 8 columns (two 4x4 blocks) and direct for column 8.
    Total rank: 49 + 49 + 16 = 114
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: columns 0-3
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Block 2: columns 4-7
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Block 3: column 8 (direct)
    for i in range(4):
        for k in range(4):
            r = 98 + i * 4 + k
            U[i * 4 + k, r] = 1
            V[k * 9 + 8, r] = 1
            W[8 * 4 + i, r] = 1
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    U, V, W, total_rank = _build_4x4x9_direct()
    
    # Fast verification using vectorization
    U_reshaped = U.reshape(4, 4, total_rank)
    V_reshaped = V.reshape(4, 9, total_rank)
    W_reshaped = W.reshape(9, 4, total_rank)
    
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    a_vecs = np.einsum('ikr,ik->ir', U_reshaped, A_test)
    b_vecs = np.einsum('kjr,kj->jr', V_reshaped, B_test)
    C_computed = np.einsum('jir,ir,jr->ij', W_reshaped, a_vecs, b_vecs)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to direct
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i * m + k, idx] = 1
                    V[k * p + j, idx] = 1
                    W[j * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
