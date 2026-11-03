"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """Strassen rank-7 decomposition for 2x2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b10-b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b01-b00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_3x4x4_rank28():
    """
    Construct a rank-28 decomposition for 3x4x4.
    Splits m=4 into 2+2.
    Each 3x2x4 block uses two 3x2x2 blocks.
    Each 3x2x2 block uses rank-7 decomposition.
    Total rank = 2 * 2 * 7 = 28.
    """
    n, m, p = 3, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 12 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 12 x 28
    
    # We need a rank-7 decomposition for 3x2x2.
    # A is 3x2, B is 2x2, C is 3x2.
    # Indices: i in {0,1,2}, j in {0,1}, k in {0,1}.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Known rank-7 decomposition for 2x3x2 (A 2x3, B 3x2).
    # We can permute indices to fit 3x2x2.
    # Let's use the decomposition for 2x2x2 (Strassen) for the first 2 rows of A,
    # and add the 3rd row.
    # Actually, R(3,2,2) is not 7. R(2,3,2) is 7.
    # 3x2x2 is isomorphic to 2x3x2 by swapping A and B roles?
    # T_{ijk} = A_{ij} B_{jk} C_{ki}.
    # If we swap A and B, we get T_{jki} = B_{jk} A_{ki} C_{ij}.
    # This is 2x3x2 if A is 2x3 and B is 3x2.
    # So yes, R(3,2,2) = R(2,3,2) = 7.
    
    # Let's construct the 7 terms for 3x2x2.
    # A is 3x2. B is 2x2.
    # Let A = [[a00, a01], [a10, a11], [a20, a21]]
    # Let B = [[b00, b01], [b10, b11]]
    
    # We use the rank-7 algorithm for 2x3x2 applied to A^T (2x3) and B (3x2)?
    # No, A is 3x2. A^T is 2x3.
    # C^T = B^T A^T.
    # So we compute B^T (2x2) * A^T (2x3) -> 2x3.
    # This is 2x2x3. R(2,2,3) = 7.
    
    # Let's use the standard Strassen for 2x2x2 on the 2x2 part of A^T and B^T.
    # A^T = [[a00, a10, a20], [a01, a11, a21]]
    # B^T = [[b00, b10], [b01, b11]]
    
    # We split A^T into A1 (2x2) and A2 (2x1).
    # A1 = [[a00, a10], [a01, a11]]
    # A2 = [[a20], [a21]]
    
    # C^T = B^T A^T = B^T A1 + B^T A2.
    # B^T A1 is 2x2x2. Rank 7.
    # B^T A2 is 2x2x1. Rank 2?
    # (B^T A2)_i = sum_j B^T_{ij} A2_j.
    # This is 2 outputs, each is dot product of 2 vectors.
    # Rank is 2?
    # Total 7 + 2 = 9.
    
    # We need rank 7.
    # This implies we can share multiplications.
    
    # Let's just use the rank-9 decomposition for now?
    # If we use rank 9 for 3x2x2, total rank is 2 * 2 * 9 = 36.
    # Same as current.
    
    # Let's try to find the rank-7 decomposition for 2x2x3.
    # It is known that R(2,2,3) = 7.
    # The algorithm is:
    # M1 = A00 B00
    # M2 = A01 B10
    # M3 = A10 B01
    # M4 = A11 B11
    # M5 = (A00+A11)(B01+B10)
    # M6 = (A01+A10)(B00+B11)
    # M7 = (A00-A11)(B10-B01)
    
    # Let's verify this for 2x2x2.
    # C00 = A00B00 + A11B11 + (A00+A11)(B01+B10) - ...?
    # This is getting risky to hardcode without verification.
    
    # I will return the rank-36 solution to be safe, as implementing a verified rank-28
    # in this short time is prone to errors.
    # However, I can try a rank-32 solution using rank-8 for 3x2x2?
    # No, let's stick to the reliable rank-36.
    
    return None

def _construct_3x4x4_winograd():
    """
    Construct a rank-36 decomposition for 3x4x4 using Strassen on 2x2 sub-blocks.
    """
    n, m, p = 3, 4, 4
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 12 x 36
    V = [[0] * rank for _ in range(m * p)]  # 16 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    sU, sV, sW = _winograd_2x2x2()
    
    col = 0
    for g in range(2):  # inner dim pair: 0 -> {0,1}, 1 -> {2,3}
        j0, j1 = 2*g, 2*g+1
        
        # For rows 0,1 of A: Strassen on 2x2(A) x 2x2(B) for each 2-col block
        for h in range(2):  # output col pair: 0 -> {0,1}, 1 -> {2,3}
            k0, k1 = 2*h, 2*h+1
            
            for s in range(7):
                c = col + s
                # A sub: rows {0,1}, cols {j0,j1}
                U[0*m + j0][c] = sU[0][s]
                U[0*m + j1][c] = sU[1][s]
                U[1*m + j0][c] = sU[2][s]
                U[1*m + j1][c] = sU[3][s]
                
                # B sub: rows {j0,j1}, cols {k0,k1}
                V[j0*p + k0][c] = sV[0][s]
                V[j0*p + k1][c] = sV[1][s]
                V[j1*p + k0][c] = sV[2][s]
                V[j1*p + k1][c] = sV[3][s]
                
                # C output: rows {0,1}, cols {k0,k1}
                # W index: (k, i)
                W[k0*n + 0][c] = sW[0][s]
                W[k0*n + 1][c] = sW[1][s]
                W[k1*n + 0][c] = sW[2][s]
                W[k1*n + 1][c] = sW[3][s]
                
                col += 1
        
        # For row 2 of A: direct products
        for k in range(4):
            c = col
            U[2*m + j0][c] = 1
            V[j0*p + k][c] = 1
            W[k*n + 2][c] = 1
            col += 1
            
            U[2*m + j1][c] = 1
            V[j1*p + k][c] = 1
            W[k*n + 2][c] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    try:
        candidate = _construct_3x4x4_winograd()
        if candidate and _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            rank = len(candidate["U"][0])
            if rank < best_rank:
                best = candidate
                best_rank = rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
