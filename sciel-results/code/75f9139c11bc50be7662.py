"""Construct a rank-30 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(n, m, p, U, V, W):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)"""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(Ua[i*m+j, rr] * Va[j*p+k, rr] * Wa[k*n+i, rr] for rr in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank34_strassen():
    """Build rank-34 decomposition for 3x3 * 3x4 using Strassen on 2x2 blocks.
    
    Partition: A (3x3) split as rows 2+1, cols 2+1
               B (3x4) split as rows 2+1, cols 2+2
    
    C[0:2,0:2] = A11*B11 + A12*B21
    C[0:2,2:4] = A11*B12 + A12*B22
    C[2:3,0:2] = A21*B11 + A22*B21
    C[2:3,2:4] = A21*B12 + A22*B22
    
    Strassen for 2x2*2x2: 7 mults each for A11*B11 and A11*B12
    Schoolbook for remaining: 4+4+4+4+2+2 = 20
    Total: 7+7+20 = 34
    """
    n, m, p = 3, 3, 4
    terms = []  # list of (u_indices, v_indices, w_indices, coefficient)
    
    # Index mapping:
    # A[i,j]: i in 0..2, j in 0..2
    # B[j,k]: j in 0..2, k in 0..3
    # C[i,k]: i in 0..2, k in 0..3
    # U index: i*m+j = i*3+j
    # V index: j*p+k = j*4+k
    # W index: k*n+i = k*3+i
    
    # Strassen for A11*B11 (2x2 * 2x2 -> 2x2)
    # A11 = [[A00,A01],[A10,A11]], B11 = [[B00,B01],[B10,B11]] (B indices: B[j,k] with j=0,1 and k=0,1)
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    # C11[0,0] = M1+M4-M5+M7
    # C11[0,1] = M3+M5
    # C11[1,0] = M2+M4
    # C11[1,1] = M1-M2+M3+M6
    
    # Each Strassen term is a product of (sum of A entries) * (sum of B entries)
    # contributing to (sum of C entries)
    
    # M1 = (A00+A11)*(B00+B11) -> contributes to C[0,0] and C[1,1]
    # A part: A[0,0] and A[1,1]
    # B part: B[0,0] and B[1,0] (since B[j,k], j is row of B = col of A)
    # Wait, B[j,k] where j indexes rows of B (= cols of A), k indexes cols of B (= cols of C)
    # A11 block: A[i,j] for i in {0,1}, j in {0,1}
    # B11 block: B[j,k] for j in {0,1}, k in {0,1}
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    # This is: (A00*B00 + A00*B10 + A11*B00 + A11*B10)
    # C[0,0] gets: A00*B00 + A11*B10 (from A11*B11 part)
    # C[1,1] gets: A00*B00 + A11*B10 ... wait no.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B10) where B00=B[0,0], B10=B[1,0]
    # M2 = (A10+A11)*B00 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A00*(B01-B11) = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A11*(B10-B00) = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A00+A10)*B11 = (A[0,0]+A[1,0])*B[1,1]
    # M6 = (A11-A10)*(B00+B10) = (A[1,1]-A[1,0])*(B[0,0]+B[1,0])
    # M7 = (A01-A11)*(B11+B01) = (A[0,1]-A[1,1])*(B[1,1]+B[0,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For tensor decomposition, each "multiplication" M is a rank-1 term:
    # M = (sum of A coefficients) * (sum of B coefficients)
    # contributing with coefficient to each C entry.
    
    # Let me represent each term as: for each (i,j) in A with coeff ca,
    # for each (j',k) in B with coeff cb (where j'=j for valid terms),
    # the contribution to C[i,k] is ca*cb*coeff_c
    
    # Actually, for the tensor decomposition, each rank-1 term is:
    # U[i*3+j, r] * V[j*4+k, r] * W[k*3+i, r]
    # summed over r equals delta for the correct (i,j,k).
    
    # A Strassen term like M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,0])
    # expands to: A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[1,1]*B[0,0] + A[1,1]*B[1,0]
    # Each of these is a schoolbook term! So M1 as a single "multiplication" in the
    # algorithm means we compute the product of two linear combinations, which in
    # tensor decomposition terms is NOT a single rank-1 term.
    
    # I need to think about this differently. In tensor decomposition, each rank-1
    # term corresponds to one scalar multiplication. Strassen reduces the number of
    # scalar multiplications by using linear combinations.
    
    # For the tensor decomposition, a Strassen multiplication like:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    # is represented as a single rank-1 term where:
    # U has 1 at positions corresponding to A[0,0] and A[1,1]
    # V has 1 at positions corresponding to B[0,0] and B[1,0]
    # W has the appropriate coefficients for the C entries this contributes to.
    
    # So for M1 contributing to C[0,0] with coeff +1 and C[1,1] with coeff +1:
    # U[0*3+0, r] = 1  (A[0,0])
    # U[1*3+1, r] = 1  (A[1,1])
    # V[0*4+0, r] = 1  (B[0,0])
    # V[1*4+0, r] = 1  (B[1,0])
    # W[0*3+0, r] = 1  (C[0,0] gets +1)
    # W[1*3+1, r] = 1  (C[1,1] gets +1)
    
    # Let me build all terms this way.
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        """u_coeffs: list of (idx, coeff), v_coeffs: list of (idx, coeff), w_coeffs: list of (idx, coeff)"""
        term = {'U': [], 'V': [], 'W': []}
        for idx, c in u_coeffs:
            term['U'].append((idx, c))
        for idx, c in v_coeffs:
            term['V'].append((idx, c))
        for idx, c in w_coeffs:
            term['W'].append((idx, c))
        terms.append(term)
    
    # Helper to create index lists
    def A_idx(i, j): return i * 3 + j  # U index
    def B_idx(j, k): return j * 4 + k  # V index
    def C_idx(k, i): return k * 3 + i  # W index
    
    # --- Strassen for A11*B11 (rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B) ---
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    # Contributes: C[0,0] += M1, C[1,1] += M1
    add_strassen_term(
        [(A_idx(0,0), 1), (A_idx(1,1), 1)],
        [(B_idx(0,0), 1), (B_idx(1,0), 1)],
        [(C_idx(0,0), 1), (C_idx(1,1), 1)]
    )
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # Contributes: C[1,0] += M2, C[1,1] -= M2
    add_strassen_term(
        [(A_idx(1,0), 1), (A_idx(1,1), 1)],
        [(B_idx(0,0), 1)],
        [(C_idx(0,1), 1), (C_idx(1,1), -1)]
    )
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # Contributes: C[0,1] += M3, C[1,1] += M3
    add_strassen_term(
        [(A_idx(0,0), 1)],
        [(B_idx(0,1), 1), (B_idx(1,1), -1)],
        [(C_idx(1,0), 1), (C_idx(1,1), 1)]
    )
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # Contributes: C[0,0] += M4, C[1,0] += M4
    add_strassen_term(
        [(A_idx(1,1), 1)],
        [(B_idx(1,0), 1), (B_idx(0,0), -1)],
        [(C_idx(0,0), 1), (C_idx(0,1), 1)]
    )
    
    # M5 = (A[0,0]+A[1,0]) * B[1,1]
    # Contributes: C[0,0] -= M5, C[0,1] += M5
    add_strassen_term(
        [(A_idx(0,0), 1), (A_idx(1,0), 1)],
        [(B_idx(1,1), 1)],
        [(C_idx(0,0), -1), (C_idx(1,0), 1)]
    )
    
    # M6 = (A[1,1]-A[1,0]) * (B[0,0]+B[1,0])
    # Contributes: C[1,1] += M6
    add_strassen_term(
        [(A_idx(1,1), 1), (A_idx(1,0), -1)],
        [(B_idx(0,0), 1), (B_idx(1,0), 1)],
        [(C_idx(1,1), 1)]
    )
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,1]+B[0,1])
    # Contributes: C[0,0] += M7
    add_strassen_term(
        [(A_idx(0,1), 1), (A_idx(1,1), -1)],
        [(B_idx(1,1), 1), (B_idx(0,1), 1)],
        [(C_idx(0,0), 1)]
    )
    
    # --- Strassen for A11*B12 (rows 0-1, cols 0-1 of A; rows 0-1, cols 2-3 of B) ---
    # Same structure but B columns are 2,3
    # M1' = (A[0,0]+A[1,1]) * (B[0,2]+B[1,2])
    # Contributes: C[0,2] += M1', C[1,3] += M1'
    add_strassen_term(
        [(A_idx(0,0), 1), (A_idx(1,1), 1)],
        [(B_idx(0,2), 1), (B_idx(1,2), 1)],
        [(C_idx(2,0), 1), (C_idx(3,1), 1)]
    )
    
    # M2' = (A[1,0]+A[1,1]) * B[0,2]
    # Contributes: C[1,2] += M2', C[1,3] -= M2'
    add_strassen_term(
        [(A_idx(1,0), 1), (A_idx(1,1), 1)],
        [(B_idx(0,2), 1)],
        [(C_idx(2,1), 1), (C_idx(3,1), -1)]
    )
    
    # M3' = A[0,0] * (B[0,3]-B[1,3])
    # Contributes: C[0,3] += M3', C[1,3] += M3'
    add_strassen_term(
        [(A_idx(0,0), 1)],
        [(B_idx(0,3), 1), (B_idx(1,3), -1)],
        [(C_idx(3,0), 1), (C_idx(3,1), 1)]
    )
    
    # M4' = A[1,1] * (B[1,2]-B[0,2])
    # Contributes: C[0,2] += M4', C[1,2] += M4'
    add_strassen_term(
        [(A_idx(1,1), 1)],
        [(B_idx(1,2), 1), (B_idx(0,2), -1)],
        [(C_idx(2,0), 1), (C_idx(2,1), 1)]
    )
    
    # M5' = (A[0,0]+A[1,0]) * B[1,3]
    # Contributes: C[0,2] -= M5', C[0,3] += M5'
    add_strassen_term(
        [(A_idx(0,0), 1), (A_idx(1,0), 1)],
        [(B_idx(1,3), 1)],
        [(C_idx(2,0), -1), (C_idx(3,0), 1)]
    )
    
    # M6' = (A[1,1]-A[1,0]) * (B[0,2]+B[1,2])
    # Contributes: C[1,3] += M6'
    add_strassen_term(
        [(A_idx(1,1), 1), (A_idx(1,0), -1)],
        [(B_idx(0,2), 1), (B_idx(1,2), 1)],
        [(C_idx(3,1), 1)]
    )
    
    # M7' = (A[0,1]-A[1,1]) * (B[1,3]+B[0,3])
    # Contributes: C[0,2] += M7'
    add_strassen_term(
        [(A_idx(0,1), 1), (A_idx(1,1), -1)],
        [(B_idx(1,3), 1), (B_idx(0,3), 1)],
        [(C_idx(2,0), 1)]
    )
    
    # --- Schoolbook for A12*B21 (A rows 0-1, col 2; B row 2, cols 0-1) ---
    # C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
    add_strassen_term([(A_idx(0,2), 1)], [(B_idx(2,0), 1)], [(C_idx(0,0), 1)])
    add_strassen_term([(A_idx(0,2), 1)], [(B_idx(2,1), 1)], [(C_idx(1,0), 1)])
    add_strassen_term([(A_idx(1,2), 1)], [(B_idx(2,0), 1)], [(C_idx(0,1), 1)])
    add_strassen_term([(A_idx(1,2), 1)], [(B_idx(2,1), 1)], [(C_idx(1,1), 1)])
    
    # --- Schoolbook for A12*B22 (A rows 0-1, col 2; B row 2, cols 2-3) ---
    # C[0,2] += A[0,2]*B[2,2], C[0,3] += A[0,2]*B[2,3]
    # C[1,2] += A[1,2]*B[2,2], C[1,3] += A[1,2]*B[2,3]
    add_strassen_term([(A_idx(0,2), 1)], [(B_idx(2,2), 1)], [(C_idx(2,0), 1)])
    add_strassen_term([(A_idx(0,2), 1)], [(B_idx(2,3), 1)], [(C_idx(3,0), 1)])
    add_strassen_term([(A_idx(1,2), 1)], [(B_idx(2,2), 1)], [(C_idx(2,1), 1)])
    add_strassen_term([(A_idx(1,2), 1)], [(B_idx(2,3), 1)], [(C_idx(3,1), 1)])
    
    # --- Schoolbook for A21*B11 (A row 2, cols 0-1; B rows 0-1, cols 0-1) ---
    # C[2,0] += A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] += A[2,0]*B[0,1] + A[2,1]*B[1,1]
    add_strassen_term([(A_idx(2,0), 1)], [(B_idx(0,0), 1)], [(C_idx(0,2), 1)])
    add_strassen_term([(A_idx(2,1), 1)], [(B_idx(1,0), 1)], [(C_idx(0,2), 1)])
    add_strassen_term([(A_idx(2,0), 1)], [(B_idx(0,1), 1)], [(C_idx(1,2), 1)])
    add_strassen_term([(A_idx(2,1), 1)], [(B_idx(1,1), 1)], [(C_idx(1,2), 1)])
    
    # --- Schoolbook for A21*B12 (A row 2, cols 0-1; B rows 0-1, cols 2-3) ---
    # C[2,2] += A[2,0]*B[0,2] + A[2,1]*B[1,2]
    # C[2,3] += A[2,0]*B[0,3] + A[2,1]*B[1,3]
    add_strassen_term([(A_idx(2,0), 1)], [(B_idx(0,2), 1)], [(C_idx(2,2), 1)])
    add_strassen_term([(A_idx(2,1), 1)], [(B_idx(1,2), 1)], [(C_idx(2,2), 1)])
    add_strassen_term([(A_idx(2,0), 1)], [(B_idx(0,3), 1)], [(C_idx(3,2), 1)])
    add_strassen_term([(A_idx(2,1), 1)], [(B_idx(1,3), 1)], [(C_idx(3,2), 1)])
    
    # --- Schoolbook for A22*B21 (A[2,2]; B row 2, cols 0-1) ---
    # C[2,0] += A[2,2]*B[2,0], C[2,1] += A[2,2]*B[2,1]
    add_strassen_term([(A_idx(2,2), 1)], [(B_idx(2,0), 1)], [(C_idx(0,2), 1)])
    add_strassen_term([(A_idx(2,2), 1)], [(B_idx(2,1), 1)], [(C_idx(1,2), 1)])
    
    # --- Schoolbook for A22*B22 (A[2,2]; B row 2, cols 2-3) ---
    # C[2,2] += A[2,2]*B[2,2], C[2,3] += A[2,2]*B[2,3]
    add_strassen_term([(A_idx(2,2), 1)], [(B_idx(2,2), 1)], [(C_idx(2,2), 1)])
    add_strassen_term([(A_idx(2,2), 1)], [(B_idx(2,3), 1)], [(C_idx(3,2), 1)])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r_idx, term in enumerate(terms):
        for idx, coeff in term['U']:
            U[idx][r_idx] = coeff
        for idx, coeff in term['V']:
            V[idx][r_idx] = coeff
        for idx, coeff in term['W']:
            W[idx][r_idx] = coeff
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Build schoolbook rank-36 decomposition."""
    n, m, p = 3, 3, 4
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

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 3, 3, 4
    
    # Try rank-34 Strassen-based decomposition
    try:
        result = _build_rank34_strassen()
        if _verify(n, m, p, result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        return _build_schoolbook()
    
    # Fallback: schoolbook rank 36
    return _build_schoolbook()
# EVOLVE-BLOCK-END
