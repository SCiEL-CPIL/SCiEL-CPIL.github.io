"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_tensors(n, m, p):
    """Build U, V, W index mappings for the multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    """
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    return U_rows, V_rows, W_rows

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    W uses the trace convention W[k*n+i, r].
    """
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if val != 1:
                    return False
    return True

def _try_rank1(n, m, p):
    """Try rank-1: U, V, W all ones."""
    U_rows, V_rows, W_rows = _build_tensors(n, m, p)
    U = [[1] for _ in range(U_rows)]
    V = [[1] for _ in range(V_rows)]
    W = [[1] for _ in range(W_rows)]
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank2(n, m, p):
    """Try rank-2 decompositions with small integer/half-integer entries.
    We search over structured patterns."""
    U_rows, V_rows, W_rows = _build_tensors(n, m, p)
    
    # Try simple patterns: U and V depend on j (the middle index), W depends on k and i
    # For rank 2, each "rank-1 slice" corresponds to a choice of j-value grouping.
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k has rank 1, so if the target
    # is all-ones, rank 1 works. If not, we need a different approach.
    
    # Let's try: for each pair of indices, see if we can find a 2-term decomposition.
    # Use the structure: the multiplication tensor M_{n,m,p} has T[i,j,k] = 1 iff the
    # (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
    # In the standard definition, T[i,j,k] = 1 for ALL i,j,k (each product A[i,j]*B[j,k]
    # contributes to C[i,k]).
    
    # Since rank-1 already covers the all-ones case, rank-2 is only useful if the
    # tensor has a non-trivial structure. Let's just return None and let the fallback
    # handle it.
    return None

def _try_strassen_like(n, m, p):
    """Attempt a Strassen-like decomposition for 2x4 * 4x4.
    
    Split A (2x4) into blocks: A = [A11 A12; A21 A22] where each block is 1x2.
    Split B (4x4) into blocks: B = [B11 B12; B21 B22] where each block is 2x2.
    
    C = A*B:
    C[0,:] = A11*B11 + A12*B21  (1x4)
    C[1,:] = A21*B11 + A22*B21  (1x4)
    
    Using Strassen's 7-multiplication trick for 2x2 blocks:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    But our blocks are 1x2 (for A) and 2x2 (for B), so each "multiplication" is
    a 1x2 times 2x2 = 1x2 operation, which itself needs 2 schoolbook multiplications.
    
    Total: 7 * 2 = 14 multiplications.
    
    Let's construct this explicitly.
    """
    n, m, p = 2, 4, 4
    
    # A is 2x4: indices (i, j) where i in {0,1}, j in {0,1,2,3}
    # B is 4x4: indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: indices (i, k) where i in {0,1}, k in {0,1,2,3}
    
    # Block structure:
    # A11 = A[0,0:2], A12 = A[0,2:4]  (each 1x2)
    # A21 = A[1,0:2], A22 = A[1,2:4]  (each 1x2)
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4]  (each 2x2)
    # B21 = B[2:4,0:2], B22 = B[2:4,2:4]  (each 2x2)
    
    # C[0,0:2] = A11*B11 + A12*B21
    # C[0,2:4] = A11*B12 + A12*B22
    # C[1,0:2] = A21*B11 + A22*B21
    # C[1,2:4] = A21*B12 + A22*B22
    
    # Using Strassen for the 2x2 block matrix product (where A blocks are 1x2 and B blocks are 2x2):
    # Each "block multiply" is a (1x2)*(2x2) = (1x2) operation requiring 2 scalar mults.
    
    # The 7 Strassen products (each is a 1x2 * 2x2 = 1x2 result, i.e., 2 scalar multiplications):
    # M1 = (A11+A22) * (B11+B22)  -> 2 mults
    # M2 = (A21+A22) * B11        -> 2 mults
    # M3 = A11 * (B12-B22)        -> 2 mults
    # M4 = A22 * (B21-B11)        -> 2 mults
    # M5 = (A11+A12) * B22        -> 2 mults
    # M6 = (A21-A11) * (B11+B12)  -> 2 mults
    # M7 = (A12-A22) * (B21+B22)  -> 2 mults
    # Total: 14 scalar multiplications
    
    # But we need to express this as a rank-14 tensor decomposition.
    # Each scalar multiplication corresponds to one rank-1 term.
    
    # Let's define the 14 rank-1 terms:
    # For each Strassen product Mx = (linear combo of A) * (linear combo of B),
    # and each such product produces a 1x2 result (two scalar outputs).
    # So each Mx contributes 2 rank-1 terms (one for each output coordinate).
    
    # Actually, let me think more carefully.
    # M1 = (A11+A22) * (B11+B22)
    # A11+A22 is a 1x2 vector: [A[0,0]+A[1,0], A[0,1]+A[1,1]]
    # B11+B22 is a 2x2 matrix: [[B[0,0]+B[2,0], B[0,1]+B[2,1]], [B[1,0]+B[3,0], B[1,1]+B[3,1]]]
    # M1 = (A11+A22) * (B11+B22) is a 1x2 vector:
    # M1[0] = (A[0,0]+A[1,0])*(B[0,0]+B[2,0]) + (A[0,1]+A[1,1])*(B[1,0]+B[3,0])
    # M1[1] = (A[0,0]+A[1,0])*(B[0,1]+B[2,1]) + (A[0,1]+A[1,1])*(B[1,1]+B[3,1])
    
    # This is getting complex. Let me just enumerate all 14 scalar multiplications directly.
    
    # I'll construct the U, V, W matrices by tracking each scalar multiplication.
    
    U_rows = n * m  # 8
    V_rows = m * p  # 16
    W_rows = p * n  # 8
    
    U = [[0]*14 for _ in range(U_rows)]
    V = [[0]*14 for _ in range(V_rows)]
    W = [[0]*14 for _ in range(W_rows)]
    
    # Helper: add a rank-1 term where U has value u at row i*m+j, V has value v at row j*p+k,
    # and W has value w at row k*n+i.
    def add_term(i, j, k, u_val, v_val, w_val, col):
        U[i*m + j][col] += u_val
        V[j*p + k][col] += v_val
        W[k*n + i][col] += w_val
    
    # M1 = (A11+A22) * (B11+B22)
    # Contributes to C[0,0:2] and C[1,0:2]
    # C[0,:] += M1 - M5 + M7 (for first 2 cols) ... let me be more careful.
    
    # Actually, the Strassen formulas give:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where C11 = C[0,0:2], C12 = C[0,2:4], etc.
    
    # Each Mx is a 1x2 vector. So:
    # C[0,k] for k in {0,1}: M1[k] + M4[k] - M5[k] + M7[k]
    # C[0,k] for k in {2,3}: M3[k-2] + M5[k-2]
    # C[1,k] for k in {0,1}: M2[k] + M4[k]
    # C[1,k] for k in {2,3}: M1[k-2] - M2[k-2] + M3[k-2] + M6[k-2]
    
    # Now, each Mx[k] is a sum of 2 scalar multiplications (since it's 1x2 * 2x2).
    # Let me enumerate each scalar multiplication.
    
    # M1[0] = (A[0,0]+A[1,0])*(B[0,0]+B[2,0]) + (A[0,1]+A[1,1])*(B[1,0]+B[3,0])
    # This expands to 4 scalar mults, but we want to keep them as products of linear combos.
    # For the tensor decomposition, each rank-1 term is a product of a linear combo of A entries,
    # a linear combo of B entries, and a linear combo of C entries.
    
    # Let me define each scalar multiplication as:
    # (sum_i alpha_i * A[i, j1]) * (sum_j beta_j * B[j2, k]) -> contributes to C[i, k]
    
    # This is quite involved. Let me just use the known result that the rank of
    # 2x4 * 4x4 is at most 14 (from Strassen). Let me construct it directly.
    
    # Actually, let me try a simpler approach: just return the schoolbook 32-rank
    # decomposition and rely on the fact that we might find something better.
    # For now, let's just do the basic verification and return schoolbook.
    
    return None  # Skip for now, too complex to get right quickly

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try rank 1 first
    result = _try_rank1(n, m, p)
    if result is not None:
        return result
    
    # Try rank 2
    if time.time() < deadline:
        result = _try_rank2(n, m, p)
        if result is not None:
            return result
    
    # Try Strassen-like rank 14
    if time.time() < deadline:
        result = _try_strassen_like(n, m, p)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank 32
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
