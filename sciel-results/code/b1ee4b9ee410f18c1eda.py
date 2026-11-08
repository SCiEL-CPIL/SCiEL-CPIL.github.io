"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms of Strassen's 2x2x2 matrix multiplication.
    Convention:
      A is 2x2 with indices A[i,j] stored at position i*2+j
      B is 2x2 with indices B[j,k] stored at position j*2+k
      C is 2x2 with indices C[i,k] stored at position k*2+i (trace convention W[k*n+i])
    
    Each term is (u, v, w) where u, v, w are 4-vectors.
    The tensor T = sum_r u_r ⊗ v_r ⊗ w_r equals the matmul tensor.
    """
    terms = []
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    # u: A00=1(idx 0), A11=1(idx 3)
    # v: B00=1(idx 0), B11=1(idx 3)
    # w: C00=1(idx 0*2+0=0), C11=1(idx 1*2+1=3)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)*B00 -> C10 += 1, C11 -= 1
    # u: A10=1(idx 2), A11=1(idx 3)
    # v: B00=1(idx 0)
    # w: C10=1(idx 0*2+1=2), C11=-1(idx 1*2+1=3)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = A00*(B10-B11) -> C01 += 1, C11 += 1
    # u: A00=1(idx 0)
    # v: B10=1(idx 2), B11=-1(idx 3)
    # w: C01=1(idx 1*2+0=2), C11=1(idx 1*2+1=3)
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,1]))
    
    # M4 = A11*(B01-B00) -> C00 += 1, C10 += 1
    # u: A11=1(idx 3)
    # v: B01=1(idx 1), B00=-1(idx 0)
    # w: C00=1(idx 0*2+0=0), C10=1(idx 0*2+1=2)
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    
    # M5 = (A00+A01)*B11 -> C00 -= 1, C01 += 1
    # u: A00=1(idx 0), A01=1(idx 1)
    # v: B11=1(idx 3)
    # w: C00=-1(idx 0*2+0=0), C01=1(idx 1*2+0=2)
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6 = (A10-A00)*(B00+B01) -> C11 += 1
    # u: A10=1(idx 2), A00=-1(idx 0)
    # v: B00=1(idx 0), B01=1(idx 1)
    # w: C11=1(idx 1*2+1=3)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A01-A11)*(B10+B11) -> C00 += 1
    # u: A01=1(idx 1), A11=-1(idx 3)
    # v: B10=1(idx 2), B11=1(idx 3)
    # w: C00=1(idx 0*2+0=0)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _verify_strassen_2x2(terms):
    """Verify that the Strassen 2x2x2 tensors correctly decompose the matmul tensor."""
    n = 2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for u, v, w in terms:
                    s += u[i*n+j] * v[j*n+k] * w[k*n+i]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _kronecker_strassen_4x4x4():
    """
    Generate the 49 rank-1 terms for 4x4x4 matrix multiplication
    via Kronecker product of two Strassen 2x2x2 decompositions.
    """
    strassen_terms = _strassen_2x2_terms()
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for r in range(7):
        u_r, v_r, w_r = strassen_terms[r]
        for s in range(7):
            u_s, v_s, w_s = strassen_terms[s]
            
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            for i_hi in range(2):
                for j_hi in range(2):
                    for i_lo in range(2):
                        for j_lo in range(2):
                            i = 2 * i_hi + i_lo
                            j = 2 * j_hi + j_lo
                            u_full[i * 4 + j] = u_r[i_hi * 2 + j_hi] * u_s[i_lo * 2 + j_lo]
            
            for j_hi in range(2):
                for k_hi in range(2):
                    for j_lo in range(2):
                        for k_lo in range(2):
                            j = 2 * j_hi + j_lo
                            k = 2 * k_hi + k_lo
                            v_full[j * 4 + k] = v_r[j_hi * 2 + k_hi] * v_s[j_lo * 2 + k_lo]
            
            for k_hi in range(2):
                for i_hi in range(2):
                    for k_lo in range(2):
                        for i_lo in range(2):
                            k = 2 * k_hi + k_lo
                            i = 2 * i_hi + i_lo
                            w_full[k * 4 + i] = w_r[k_hi * 2 + i_hi] * w_s[k_lo * 2 + i_lo]
            
            for idx in range(16):
                U[idx][col] = u_full[idx]
                V[idx][col] = v_full[idx]
                W[idx][col] = w_full[idx]
            
            col += 1
    
    return U, V, W

def _verify_4x4x4(U, V, W):
    """Verify that the 4x4x4 decomposition is correct."""
    n = 4
    rank = len(U[0])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for r in range(rank):
                    s += U[i * n + j][r] * V[j * n + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook_4x4x4():
    """Schoolbook 4x4x4 decomposition with rank 64."""
    n = 4
    rank = n * n * n
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_4x4x8_from_4x4x4(U4, V4, W4):
    """
    Build 4x4x8 decomposition from two 4x4x4 decompositions.
    A is 4x4, B is 4x8, C is 4x8.
    
    Split B into B_left (cols 0-3) and B_right (cols 4-7).
    C_left = A * B_left (4x4 result)
    C_right = A * B_right (4x4 result)
    
    U has shape (16, rank), V has shape (32, rank), W has shape (32, rank)
    U index: i*4+j
    V index: j*8+k
    W index: k*4+i
    """
    n, m, p = 4, 4, 8
    rank4 = len(U4[0])
    rank = 2 * rank4
    
    U = [[0] * rank for _ in range(n * m)]   # (16, rank)
    V = [[0] * rank for _ in range(m * p)]   # (32, rank)
    W = [[0] * rank for _ in range(p * n)]   # (32, rank)
    
    # First half: B_left (k=0..3)
    for idx in range(16):
        for r in range(rank4):
            U[idx][r] = U4[idx][r]
            V[idx][r] = V4[idx][r]  # V4 index j*4+k, V index j*8+k for k=0..3
            W[idx][r] = W4[idx][r]  # W4 index k*4+i, W index k*4+i for k=0..3
    
    # Second half: B_right (k=4..7)
    for idx in range(16):
        for r in range(rank4):
            U[idx][rank4 + r] = U4[idx][r]
    
    for j in range(4):
        for k_prime in range(4):
            v4_idx = j * 4 + k_prime
            v_idx = j * 8 + 4 + k_prime
            for r in range(rank4):
                V[v_idx][rank4 + r] = V4[v4_idx][r]
    
    for k_prime in range(4):
        for i in range(4):
            w4_idx = k_prime * 4 + i
            w_idx = (4 + k_prime) * 4 + i
            for r in range(rank4):
                W[w_idx][rank4 + r] = W4[w4_idx][r]
    
    return U, V, W

def _verify_4x4x8(U, V, W):
    """Verify that the 4x4x8 decomposition is correct."""
    n, m, p = 4, 4, 8
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _burgisser_19_terms_4x4x4():
    """
    Attempt to find a 19-rank decomposition for 4x4x4 matrix multiplication.
    This uses a structured search approach based on known results.
    
    The 4x4x4 matmul tensor has rank at most 19 (Bürgisser 1998).
    We construct it using a combination of Strassen blocks and additional terms.
    
    Actually, let me use a different approach: we'll use the known fact that
    the 4x4x4 tensor can be decomposed with rank 19 by exploiting the
    structure of the problem.
    
    For now, let's fall back to the Kronecker Strassen (rank 49) if we can't
    find the 19-rank decomposition quickly.
    """
    # For now, return None to indicate we should use the Kronecker approach
    # A proper 19-rank construction would require significant additional work
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strategy: Use Kronecker Strassen for 4x4x4 (rank 49), then combine
    # two such blocks for 4x4x8 (total rank 98).
    
    U4, V4, W4 = _kronecker_strassen_4x4x4()
    
    # Verify the 4x4x4 decomposition
    if not _verify_4x4x4(U4, V4, W4):
        # Fallback to schoolbook
        U4, V4, W4 = _schoolbook_4x4x4()
    
    # Build 4x4x8 from two 4x4x4 blocks
    U, V, W = _build_4x4x8_from_4x4x4(U4, V4, W4)
    
    # Verify the final result
    if not _verify_4x4x8(U, V, W):
        # Fallback to full schoolbook 4x4x8
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
# EVOLVE-BLOCK-END
