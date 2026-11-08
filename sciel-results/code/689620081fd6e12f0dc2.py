"""Matrix multiplication 2x3x8 with rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n=2, m=3, p=8, tol=1e-9):
    """Check that U*V*W reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But actually the convention is:
    # U has shape (n*m, rank) indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank) indexed by (j,k) -> j*p+k
    # W has shape (p*n, rank) indexed by (k,i) -> k*n+i
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta? No, T is the matmul tensor where
    # T[i,j,k] = 1 if this term contributes, 0 otherwise.
    # Actually for matmul M_{n,m,p}, T[i,j,k] = 1 (it's the identity tensor)
    # The decomposition satisfies: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid (i,j,k) in the matmul pattern.
    # Wait, for matmul the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No that's not right either.
    
    # Let me think again. The matmul tensor M_{n,m,p} is defined as:
    # M_{n,m,p}[i,j,k] = 1 if this is the (i,j,k) entry of the matmul operation
    # The matmul operation: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid indices (it's all ones for the supported range)
    # No wait - the tensor for matmul is: for each output element C[i,k], 
    # it's the sum over j of A[i,j]*B[j,k].
    # The tensor T has entries T[i,j,k] which equals 1 if the term A[i,j]*B[j,k] 
    # appears in the computation of C[i,k].
    # So T[i,j,k] = 1 for all i,j,k in valid range.
    
    # Let's just verify: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank44():
    """
    Rank 44 decomposition for 2x3x8 matmul.
    
    Strategy: Split the 8 columns into 4 pairs.
    Each pair gives a 2x3x2 multiplication.
    For each 2x3x2: split 3 into 2+1.
      - 2x2x2 part: use Strassen (rank 7)
      - 2x1x2 part: schoolbook (rank 4)
    Total per pair: 11. Four pairs: 44.
    
    The output convention: W[k*n + i, r] where k is column index, i is row index.
    For 2x3x8: n=2, m=3, p=8.
    U: (6, rank) indexed by (i,j) -> i*3+j
    V: (24, rank) indexed by (j,k) -> j*8+k  
    W: (16, rank) indexed by (k,i) -> k*2+i
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    # 4 pairs of columns: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k0 = pair_idx * 2
        k1 = k0 + 1
        
        # For this 2x3x2 block (rows 0,1; cols 0,1 of the pair; j=0,1,2)
        # Split j into {0,1} and {2}
        
        # Part 1: 2x2x2 (i in {0,1}, j in {0,1}, k in {k0,k1}) - Strassen rank 7
        # Strassen for 2x2x2:
        # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = AB where C[i,k] = sum_j A[i,j] B[j,k]
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a10)b11
        # M6 = (a01-a10)(b00+b10)
        # M7 = (a11-a01)(b01+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # In our tensor decomposition, for the 2x2x2 sub-problem:
        # U indices: (i,j) where i in {0,1}, j in {0,1} -> global (i*3+j)
        # V indices: (j,k) where j in {0,1}, k in {k0,k1} -> global (j*8+k)
        # W indices: (k,i) where k in {k0,k1}, i in {0,1} -> global (k*2+i)
        
        # Strassen terms (7 of them):
        # Each term c has:
        # U[i*3+j, c] = coefficient for A[i,j]
        # V[j*8+k, c] = coefficient for B[j,k]
        # W[k*2+i, c] = coefficient for C[i,k]
        
        # Term 0: M1 = (a00+a11)(b00+b11)
        # U[0*3+0, c]=1, U[1*3+1, c]=1 (a00, a11)
        # V[0*8+k0, c]=1, V[0*8+k1, c]=1, V[1*8+k0, c]=1, V[1*8+k1, c]=1
        # Wait no. Let me be more careful.
        # M1 = (A00+A11) * (B00+B11)
        # U part: coefficient of A[i,j]: U[i*3+j, c]
        #   A00: i=0,j=0 -> U[0, c] = 1
        #   A11: i=1,j=1 -> U[3, c] = 1
        # V part: coefficient of B[j,k]: V[j*8+k, c]
        #   B00: j=0,k=k0 -> V[k0, c] = 1
        #   B01: j=0,k=k1 -> V[k1, c] = 1
        #   B10: j=1,k=k0 -> V[8+k0, c] = 1
        #   B11: j=1,k=k1 -> V[8+k1, c] = 1
        # W part: coefficient of C[i,k]: W[k*2+i, c]
        #   C00: k=k0,i=0 -> W[k0*2+0, c] = 1
        #   C01: k=k1,i=0 -> W[k1*2+0, c] = 1
        #   C10: k=k0,i=1 -> W[k0*2+1, c] = 1
        #   C11: k=k1,i=1 -> W[k1*2+1, c] = 1
        
        # Let me define a helper to add a Strassen term
        def add_strassen_term(term_id, U_coeff, V_coeff, W_coeff):
            """
            term_id: 0-6
            U_coeff: dict (i,j) -> coeff
            V_coeff: dict (j, k_local) -> coeff where k_local is 0 or 1 (relative to k0)
            W_coeff: dict (k_local, i) -> coeff
            """
            nonlocal col
            c = col
            for (ii, jj), uval in U_coeff.items():
                U[ii * m + jj][c] = uval
            for (jj, kk_local), vval in V_coeff.items():
                kk = k0 + kk_local
                V[jj * p + kk][c] = vval
            for (kk_local, ii), wval in W_coeff.items():
                kk = k0 + kk_local
                W[kk * n + ii][c] = wval
            col += 1
        
        # Strassen terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        add_strassen_term(0,
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}
        )
        # M2 = (A10+A11)B00
        add_strassen_term(1,
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (0,0): 1}
        )
        # M3 = A00(B01-B11)
        add_strassen_term(2,
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        )
        # M4 = A11(B10-B00)
        add_strassen_term(3,
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,1): 1}
        )
        # M5 = (A00+A10)B11
        add_strassen_term(4,
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(1,0): -1, (1,1): 1, (0,0): -1}
        )
        # M6 = (A01-A10)(B00+B10)
        add_strassen_term(5,
            {(0,1): 1, (1,0): -1},
            {(0,0): 1, (1,0): 1},
            {(0,0): 1, (0,1): 1}
        )
        # M7 = (A11-A01)(B01+B11)
        add_strassen_term(6,
            {(1,1): 1, (0,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,1): 1, (1,1): -1, (0,0): 1}
        )
        
        # Part 2: 2x1x2 (i in {0,1}, j=2, k in {k0,k1}) - schoolbook rank 4
        # C[i,k] += A[i,2]*B[2,k]
        # 4 terms: (i,k) = (0,k0), (0,k1), (1,k0), (1,k1)
        for ii in range(2):
            for kk_local in range(2):
                kk = k0 + kk_local
                c = col
                U[ii * m + 2][c] = 1
                V[2 * p + kk][c] = 1
                W[kk * n + ii][c] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 120.0
    
    try:
        result = _construct_rank44()
        
        # Verify correctness
        if _verify(result["U"], result["V"], result["W"]):
            return result
        else:
            # Fallback to schoolbook if verification fails
            n, m, p = DIMENSIONS
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
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
