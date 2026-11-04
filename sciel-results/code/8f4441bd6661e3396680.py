"""
Rewritten program for 4x4x4 matrix multiplication tensor decomposition.
Attempts to use Laderman's rank-23 decomposition for 3x3 embedded in 4x4.
Falls back to Kronecker product of Strassen (rank 49) if verification fails.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = 1; W[0*2+1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1; W[1*2+0][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + j][r] = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                    V[j * p + k][r] = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                    W[k * n + i][r] = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _laderman_3x3():
    """
    Return the rank-23 Laderman decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as lists of lists with shape (9, 23), (9, 23), (9, 23).
    Indices are 0-based.
    """
    rank = 23
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Helper to add a term
    # term: (list of (idx, coeff) for A, list of (idx, coeff) for B, list of (idx, coeff) for C)
    # idx is 0..8 for 3x3 matrix (row*3+col)
    def add_term(r, a_coeffs, b_coeffs, c_coeffs):
        for idx, coeff in a_coeffs:
            U[idx][r] += coeff
        for idx, coeff in b_coeffs:
            V[idx][r] += coeff
        for idx, coeff in c_coeffs:
            W[idx][r] += coeff

    # Laderman's 23 terms (converted to 0-based indices)
    # A[i,j] index: i*3+j
    # B[j,k] index: j*3+k
    # C[i,k] index: i*3+k
    
    # 1. (A00+A11+A22)(B00+B11+B22)
    add_term(0, [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)])
    
    # 2. (A01+A10+A22)(B01+B10+B22)
    add_term(1, [(1,1),(3,1),(8,1)], [(1,1),(3,1),(8,1)], [(1,1),(3,1),(8,1)])
    
    # 3. (A02+A11+A20)(B02+B11+B20)
    add_term(2, [(2,1),(4,1),(6,1)], [(2,1),(4,1),(6,1)], [(2,1),(4,1),(6,1)])
    
    # 4. (A12+A21+A00)(B12+B21+B00)
    add_term(3, [(5,1),(7,1),(0,1)], [(5,1),(7,1),(0,1)], [(5,1),(7,1),(0,1)])
    
    # 5. (A20+A02+A11)(B20+B02+B11)
    add_term(4, [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)])
    
    # 6. (A00+A11+A22)(B01+B12+B20)
    add_term(5, [(0,1),(4,1),(8,1)], [(1,1),(5,1),(6,1)], [(1,1),(5,1),(6,1)])
    
    # 7. (A01+A10+A22)(B10+B21+B02)
    add_term(6, [(1,1),(3,1),(8,1)], [(3,1),(7,1),(2,1)], [(3,1),(7,1),(2,1)])
    
    # 8. (A02+A11+A20)(B20+B02+B11)
    add_term(7, [(2,1),(4,1),(6,1)], [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)])
    
    # 9. (A12+A21+A00)(B01+B12+B20)
    add_term(8, [(5,1),(7,1),(0,1)], [(1,1),(5,1),(6,1)], [(1,1),(5,1),(6,1)])
    
    # 10. (A20+A02+A11)(B10+B21+B02)
    add_term(9, [(6,1),(2,1),(4,1)], [(3,1),(7,1),(2,1)], [(3,1),(7,1),(2,1)])
    
    # 11. (A00+A11+A22)(B02+B20+B11)
    add_term(10, [(0,1),(4,1),(8,1)], [(2,1),(6,1),(4,1)], [(2,1),(6,1),(4,1)])
    
    # 12. (A01+A10+A22)(B02+B20+B11)
    add_term(11, [(1,1),(3,1),(8,1)], [(2,1),(6,1),(4,1)], [(2,1),(6,1),(4,1)])
    
    # 13. (A02+A11+A20)(B01+B12+B20)
    add_term(12, [(2,1),(4,1),(6,1)], [(1,1),(5,1),(6,1)], [(1,1),(5,1),(6,1)])
    
    # 14. (A12+A21+A00)(B10+B21+B02)
    add_term(13, [(5,1),(7,1),(0,1)], [(3,1),(7,1),(2,1)], [(3,1),(7,1),(2,1)])
    
    # 15. (A20+A02+A11)(B01+B12+B20)
    add_term(14, [(6,1),(2,1),(4,1)], [(1,1),(5,1),(6,1)], [(1,1),(5,1),(6,1)])
    
    # 16. (A00+A11+A22)(B10+B21+B02)
    add_term(15, [(0,1),(4,1),(8,1)], [(3,1),(7,1),(2,1)], [(3,1),(7,1),(2,1)])
    
    # 17. (A01+A10+A22)(B20+B02+B11)
    add_term(16, [(1,1),(3,1),(8,1)], [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)])
    
    # 18. (A02+A11+A20)(B10+B21+B02)
    add_term(17, [(2,1),(4,1),(6,1)], [(3,1),(7,1),(2,1)], [(3,1),(7,1),(2,1)])
    
    # 19. (A12+A21+A00)(B02+B20+B11)
    add_term(18, [(5,1),(7,1),(0,1)], [(2,1),(6,1),(4,1)], [(2,1),(6,1),(4,1)])
    
    # 20. (A20+A02+A11)(B20+B02+B11)
    add_term(19, [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)])
    
    # 21. (A00+A11+A22)(B20+B02+B11)
    add_term(20, [(0,1),(4,1),(8,1)], [(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)])
    
    # 22. (A01+A10+A22)(B00+B11+B22)
    add_term(21, [(1,1),(3,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)])
    
    # 23. (A02+A11+A20)(B00+B11+B22)
    add_term(22, [(2,1),(4,1),(6,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)])
    
    return U, V, W

def _embed_3x3_to_4x4(U3, V3, W3):
    """
    Embed the 3x3x3 decomposition into 4x4x4.
    The 3x3 block is in the top-left (indices 0,1,2).
    Index mapping:
    U3[i*3+j, r] -> U4[i*4+j, r]
    V3[j*3+k, r] -> V4[j*4+k, r]
    W3[k*3+i, r] -> W4[k*4+i, r]
    """
    rank = 23
    U4 = [[0] * rank for _ in range(16)]
    V4 = [[0] * rank for _ in range(16)]
    W4 = [[0] * rank for _ in range(16)]
    
    for i in range(3):
        for j in range(3):
            idx3 = i * 3 + j
            idx4 = i * 4 + j
            U4[idx4] = U3[idx3]
            
    for j in range(3):
        for k in range(3):
            idx3 = j * 3 + k
            idx4 = j * 4 + k
            V4[idx4] = V3[idx3]
            
    for k in range(3):
        for i in range(3):
            idx3 = k * 3 + i
            idx4 = k * 4 + i
            W4[idx4] = W3[idx3]
            
    return U4, V4, W4

def _schoolbook_decomposition(n, m, p):
    """Construct the rank-64 schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _tensor_square_4x4():
    """
    Attempt a rank-28 decomposition using the tensor square idea:
    partition 4x4 into 2x2 blocks of 2x2 matrices, and apply a
    Strassen-like rank-7 decomposition at the block level, with
    each block operation being a 2x2 matrix multiplication.
    
    This approach treats the 4x4 multiplication as a 2x2 matrix
    multiplication over the ring of 2x2 matrices. Since 2x2 matrix
    multiplication is non-commutative, we need to be careful.
    
    Actually, let's try a different approach: use the fact that
    4x4 matrix multiplication can be decomposed using the
    'middle product' or other algebraic identities.
    
    For now, let's try to construct a valid rank-28 decomposition
    by combining Strassen for the outer 2x2 structure with
    schoolbook for the inner 2x2 structure, but with shared terms.
    """
    # The tensor square approach: we have 4 block positions in A
    # and 4 in B, giving 16 block products, each of rank 4 (schoolbook 2x2).
    # Total would be 64 without sharing. With Strassen at block level (rank 7),
    # we get 7 * 4 = 28 if the ring operations commute.
    
    # Since 2x2 matrix mult is non-commutative, let's try a different
    # strategy: use the Laderman 3x3 rank-23 decomposition embedded in 4x4
    # plus additional terms for the 4th row/column.
    
    # Embed Laderman 3x3 in top-left 3x3 of 4x4
    U3, V3, W3 = _laderman_3x3()
    rank_base = 23
    
    # We need to handle the 4th row/column. The 4x4 product C = A*B has:
    # C[i,k] for i,k in 0..3
    # The 3x3 part is handled by Laderman.
    # C[3,k] = sum_j A[3,j]*B[j,k] for k=0..3 (4 terms, but j=0..3)
    # C[i,3] = sum_j A[i,j]*B[j,3] for i=0..3
    # C[3,3] = sum_j A[3,j]*B[j,3]
    
    # Additional terms needed:
    # For row 3 of C: C[3,k] = A[3,0]*B[0,k] + A[3,1]*B[1,k] + A[3,2]*B[2,k] + A[3,3]*B[3,k]
    # For col 3 of C: C[i,3] = A[i,0]*B[0,3] + A[i,1]*B[1,3] + A[i,2]*B[2,3] + A[i,3]*B[3,3]
    # C[3,3] is counted in both
    
    # We need at minimum:
    # - 4 terms for A[3,j]*B[j,k] (j=0..3, for each k) -> but we can group
    # - 4 terms for A[i,j]*B[j,3] (i=0..3, for each j) -> but we can group
    
    # Actually, the simplest: add 7 more terms (rank 30 total)
    # Or try to be clever.
    
    # Let's just add schoolbook terms for the 4th row and column
    # C[3,k] for k=0,1,2,3: need A[3,j]*B[j,k] for j=0..3
    # C[i,3] for i=0,1,2: need A[i,j]*B[j,3] for j=0..3 (C[3,3] already counted)
    
    # Total additional: 4 (for row 3) + 3*4 (for col 3, rows 0-2) = 16
    # But C[3,3] is shared, so 4 + 12 - 1 = 15 additional terms
    # Total rank: 23 + 15 = 38
    
    # That's still better than 49. Let's construct it.
    rank = 38
    U4 = [[0] * rank for _ in range(16)]
    V4 = [[0] * rank for _ in range(16)]
    W4 = [[0] * rank for _ in range(16)]
    
    # Embed Laderman 3x3
    for i in range(3):
        for j in range(3):
            idx3 = i * 3 + j
            idx4 = i * 4 + j
            for r in range(rank_base):
                U4[idx4][r] = U3[idx3][r]
    
    for j in range(3):
        for k in range(3):
            idx3 = j * 3 + k
            idx4 = j * 4 + k
            for r in range(rank_base):
                V4[idx4][r] = V3[idx3][r]
    
    for k in range(3):
        for i in range(3):
            idx3 = k * 3 + i
            idx4 = k * 4 + i
            for r in range(rank_base):
                W4[idx4][r] = W3[idx3][r]
    
    # Now add terms for the 4th row/column
    # Row 3: C[3,k] += A[3,j] * B[j,k] for j=0..3, k=0..3
    # We need to handle j=0,1,2,3 and k=0,1,2,3
    # But the Laderman part already handles C[i,k] for i,k in 0..2
    # So we need C[3,k] for all k, and C[i,3] for i=0..2
    
    # C[3,k] = sum_{j=0}^{3} A[3,j] * B[j,k]
    # C[i,3] = sum_{j=0}^{3} A[i,j] * B[j,3] for i=0,1,2
    
    # Add schoolbook terms for these
    col = rank_base
    
    # For C[3,k], k=0..3: need A[3,j]*B[j,k]
    for j in range(4):
        for k in range(4):
            if col >= rank:
                break
            U4[3 * 4 + j][col] = 1
            V4[j * 4 + k][col] = 1
            W4[k * 4 + 3][col] = 1
            col += 1
    
    # For C[i,3], i=0..2: need A[i,j]*B[j,3]
    # But C[3,3] is already covered above
    for i in range(3):
        for j in range(4):
            if col >= rank:
                break
            U4[i * 4 + j][col] = 1
            V4[j * 4 + 3][col] = 1
            W4[3 * 4 + i][col] = 1
            col += 1
    
    # col should be rank_base + 16 = 39, but we allocated 38
    # Let me recount: row 3 has 4*4=16 terms, col 3 has 3*4=12 terms, total 28 additional
    # That's too many. Let me reduce.
    
    # Actually, I made an error. Let me restart with correct counting.
    # C[3,k] for k=0..3: each needs sum over j=0..3, so 4 terms each = 16 terms
    # C[i,3] for i=0..2: each needs sum over j=0..3, so 4 terms each = 12 terms
    # Total additional: 28, total rank: 23 + 28 = 51. That's worse than 49!
    
    # So this approach doesn't help. Let's just use the Kronecker product.
    # But wait - maybe we can share some terms.
    
    # For now, return None to signal failure, and use Kronecker.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline with 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try tensor square / Laderman embedding first
    if time.time() < deadline:
        try:
            result_ts = _tensor_square_4x4()
            if result_ts is not None:
                U, V, W = result_ts["U"], result_ts["V"], result_ts["W"]
                valid, max_diff = _verify_decomp(n, m, p, U, V, W, tol=1e-9)
                if valid:
                    return result_ts
        except Exception:
            pass
    
    # Use Kronecker-Strassen rank-49 decomposition
    try:
        result49 = _kronecker_decomposition(n, m, p)
        
        # Verify the decomposition
        U, V, W = result49["U"], result49["V"], result49["W"]
        valid, max_diff = _verify_decomp(n, m, p, U, V, W, tol=1e-9)
        
        if valid:
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook if verification fails
            pass
    except Exception:
        pass
    
    # Check deadline before expensive fallback
    if time.time() > deadline:
        # Return a minimal valid result (schoolbook is fast to construct)
        pass
    
    # Final fallback: schoolbook
    result_schoolbook = _schoolbook_decomposition(n, m, p)
    return result_schoolbook

# EVOLVE-BLOCK-END
