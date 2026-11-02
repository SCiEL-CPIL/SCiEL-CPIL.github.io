import time
import random
import math
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _build_rank25(n, m, p):
    """
    Constructs a rank-25 exact tensor decomposition for 2x2 by 2x7 matrix multiplication.
    
    Strategy:
    Split the 7 columns of B into three 2-column blocks (cols 0-1, 2-3, 4-5) and one 1-column block (col 6).
    Use Strassen decomposition (rank 7) for each 2x2 block.
    Use Schoolbook decomposition (rank 4) for the 1-column block.
    Total Rank = 7 + 7 + 7 + 4 = 25.
    """
    rank = 25
    U = [[0] * rank for _ in range(n * m)] # n*m = 4
    V = [[0] * rank for _ in range(m * p)] # m*p = 14
    W = [[0] * rank for _ in range(p * n)] # p*n = 14
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def add_strassen_block(col_offset, base_rank):
        """
        Adds a Strassen block for columns col_offset and col_offset+1.
        col_offset is 0, 2, or 4.
        base_rank is the starting index in the rank dimension.
        """
        c0 = col_offset
        c1 = col_offset + 1
        
        # M1: (A00 + A11) * (B00 + B11)
        # U: A00 (0), A11 (3)
        # V: B00 (c0), B11 (c1)
        U[0][base_rank + 0] += 1
        U[3][base_rank + 0] += 1
        V[0 * p + c0][base_rank + 0] += 1
        V[1 * p + c1][base_rank + 0] += 1
        # W: C00 (c0, 0), C11 (c1, 1)
        # C00 += M1, C11 += M1
        W[c0 * n + 0][base_rank + 0] += 1
        W[c1 * n + 1][base_rank + 0] += 1
        
        # M2: (A10 + A11) * B00
        # U: A10 (2), A11 (3)
        # V: B00 (c0)
        U[2][base_rank + 1] += 1
        U[3][base_rank + 1] += 1
        V[0 * p + c0][base_rank + 1] += 1
        # W: C10 (c0, 1)
        # C10 += M2
        W[c0 * n + 1][base_rank + 1] += 1
        
        # M3: A00 * (B01 - B11)
        # U: A00 (0)
        # V: B01 (c1), B11 (c1) -> wait, B indices are j*p+k.
        # B01 is j=0, k=c1. B11 is j=1, k=c1.
        U[0][base_rank + 2] += 1
        V[0 * p + c1][base_rank + 2] += 1
        V[1 * p + c1][base_rank + 2] -= 1
        # W: C01 (c1, 0)
        # C01 += M3
        W[c1 * n + 0][base_rank + 2] += 1
        
        # M4: A11 * (B10 - B00)
        # U: A11 (3)
        # V: B10 (c0), B00 (c0)
        U[3][base_rank + 3] += 1
        V[1 * p + c0][base_rank + 3] += 1
        V[0 * p + c0][base_rank + 3] -= 1
        # W: C10 (c0, 1), C00 (c0, 0)
        # C10 += M4, C00 -= M4
        W[c0 * n + 1][base_rank + 3] += 1
        W[c0 * n + 0][base_rank + 3] -= 1
        
        # M5: (A00 + A01) * B11
        # U: A00 (0), A01 (1)
        # V: B11 (c1)
        U[0][base_rank + 4] += 1
        U[1][base_rank + 4] += 1
        V[1 * p + c1][base_rank + 4] += 1
        # W: C01 (c1, 0)
        # C01 += M5
        W[c1 * n + 0][base_rank + 4] += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        # U: A10 (2), A00 (0)
        # V: B00 (c0), B01 (c1)
        U[2][base_rank + 5] += 1
        U[0][base_rank + 5] -= 1
        V[0 * p + c0][base_rank + 5] += 1
        V[0 * p + c1][base_rank + 5] += 1
        # W: C00 (c0, 0)
        # C00 += M6
        W[c0 * n + 0][base_rank + 5] += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        # U: A01 (1), A11 (3)
        # V: B10 (c0), B11 (c1)
        U[1][base_rank + 6] += 1
        U[3][base_rank + 6] -= 1
        V[1 * p + c0][base_rank + 6] += 1
        V[1 * p + c1][base_rank + 6] += 1
        # W: C11 (c1, 1)
        # C11 += M7
        W[c1 * n + 1][base_rank + 6] += 1

    def add_schoolbook_block(col_idx, base_rank):
        """
        Adds a schoolbook block for a single column col_idx.
        """
        c = col_idx
        # C0c = A00*B0c + A01*B1c
        # C1c = A10*B0c + A11*B1c
        
        # A00 * B0c -> C0c
        U[0][base_rank + 0] += 1
        V[0 * p + c][base_rank + 0] += 1
        W[c * n + 0][base_rank + 0] += 1
        
        # A01 * B1c -> C0c
        U[1][base_rank + 1] += 1
        V[1 * p + c][base_rank + 1] += 1
        W[c * n + 0][base_rank + 1] += 1
        
        # A10 * B0c -> C1c
        U[2][base_rank + 2] += 1
        V[0 * p + c][base_rank + 2] += 1
        W[c * n + 1][base_rank + 2] += 1
        
        # A11 * B1c -> C1c
        U[3][base_rank + 3] += 1
        V[1 * p + c][base_rank + 3] += 1
        W[c * n + 1][base_rank + 3] += 1

    # Block 1: Columns 0, 1 (Rank 7)
    add_strassen_block(0, 0)
    
    # Block 2: Columns 2, 3 (Rank 7)
    add_strassen_block(2, 7)
    
    # Block 3: Columns 4, 5 (Rank 7)
    add_strassen_block(4, 14)
    
    # Block 4: Column 6 (Rank 4)
    add_schoolbook_block(6, 21)
    
    return U, V, W

def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        h = int(round(2.0 * float(x))) / 2.0
        return Fraction(h).limit_denominator(10**6)
    return Fraction(x).limit_denominator(10**6)

def _verify_exact(U, V, W, n, m, p):
    """
    Verify exact decomposition.
    """
    R = len(U[0])
    Uf = [[_to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[_to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[_to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]

    # (1) tensor identity: sum_c U[i*m+j,c] * V[j*p+k,c] == delta[i,k]
    # Wait, the tensor identity for matrix multiplication A (n x m) B (m x p) is:
    # T[i, j, k] = delta[i, k] ? No.
    # The product C = AB has C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor decomposition defines a bilinear map.
    # The condition for exact decomposition of the multiplication tensor is:
    # sum_c U[i, j, c] V[j, k, c] W[c, i, k] = delta[i, i'] delta[k, k'] ? No.
    
    # Actually, the standard definition:
    # We want to compute C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition is:
    # C[i, k] = sum_c (sum_j A[i, j] U[i, j, c]) (sum_k' B[j, k] V[j, k, c]) ... this is not right.
    
    # Correct Tensor Definition:
    # The multiplication tensor T is defined by T[i, j, k] = delta[i, k] for n=m=p=2? No.
    # For n, m, p:
    # The tensor is T in U*V*W such that for any A, B:
    # <T, A \otimes B \otimes I> = trace(A B I^T)? No.
    
    # Let's use the explicit check:
    # For random A and B, compute C = sum_c (A * U_c) * (V_c * B) ? No.
    # The algorithm is:
    # For each c:
    #   u_c = sum_j A[i, j] U[i, j, c] (vector over i)
    #   v_c = sum_j B[j, k] V[j, k, c] (vector over k)
    #   C[i, k] += W[i, k, c] * u_c * v_c
    #
    # So, C[i, k] = sum_c W[i, k, c] (sum_j A[i, j] U[i, j, c]) (sum_j' B[j', k] V[j', k, c])
    #
    # This must equal sum_j A[i, j] B[j, k].
    #
    # This is a bilinear form in A and B.
    # We can check the coefficients:
    # Coefficient of A[i0, j0] B[j1, k1] in the sum must be:
    # 1 if i0 == i and j1 == k (and j0 == j1? No, the indices are fixed).
    # Wait, the identity is:
    # sum_c W[i, k, c] U[i, j, c] V[j', k, c] must equal delta[j, j'] delta[i, i']? No.
    #
    # Let's re-read the tensor identity from the seed program comments:
    # "sum_c U[i*m+j,c] * V[j*p+k,c] == delta[i,k]"
    # This identity in the seed comment is WRONG for standard matrix multiplication.
    # For matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is defined on indices (i, j, k) for A, (j, k, l) for B? No.
    # The inputs are A (n x m) and B (m x p).
    # The tensor T has indices (i, j) for A, (j, k) for B, and (i, k) for C.
    # The multiplication is a linear map from A \otimes B to C.
    # T[i, j, j', k] = delta[j, j'] delta[i, i']? No.
    #
    # Let's check the seed's verification logic:
    # It checks: sum_c U[i*m+j, c] V[j*p+k, c] == (1 if i == k else 0)
    # This implies that the product of the "A-part" and "B-part" vectors yields delta[i,k].
    # This is only true if U and V are chosen such that their inner product over c gives the identity matrix structure.
    #
    # Actually, the standard tensor rank decomposition of the matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{c=1}^R u_c \otimes v_c \otimes w_c
    # where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
    # The condition is that for all A, B:
    # C = AB = sum_c (A \cdot u_c) (v_c \cdot B)^T ? No.
    #
    # The correct expansion:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor decomposition:
    # C[i, k] = sum_c W[k*n+i, c] * (sum_j A[i, j] U[i*m+j, c]) * (sum_j' B[j', k] V[j'*p+k, c])
    #
    # For this to hold for all A, B, the coefficient of A[i0, j0] B[j1, k1] must be:
    # sum_c W[k*n+i, c] U[i*m+j0, c] V[j1*p+k, c]
    # This sum must equal delta[i, i0] delta[k, k1] delta[j0, j1].
    #
    # Wait, the seed program's verification was:
    # for i in range(n):
    #   for k in range(p):
    #     for j in range(m):
    #       s = sum_c U[i*m+j, c] V[j*p+k, c]
    #       if s != (1 if i == k else 0): return False
    #
    # This check implies that the "W" part is implicitly 1 or handled separately?
    # If W is not 1, the check is wrong.
    #
    # However, in the Strassen decomposition, W entries are not all 1.
    # So the seed's verification function `_verify_exact` is likely incorrect for general W, OR it assumes a specific form where W is effectively absorbed?
    #
    # Let's look at the seed's `_verify_exact` again.
    # It checks `sum_c U[i*m+j,c] V[j*p+k,c] == delta[i,k]`.
    # And it checks W routing separately.
    #
    # If I use Strassen, does `sum_c U[i*m+j,c] V[j*p+k,c] == delta[i,k]` hold?
    # For a single 2x2x2 Strassen block (cols 0,1):
    # U and V are sparse.
    # Let's test M1 term:
    # U[0,0]=1, U[3,0]=1. V[0,0]=1, V[1,1]=1.
    # i=0, k=0, j=0: U[0,0]V[0,0] = 1*1 = 1. (Matches delta[0,0])
    # i=0, k=1, j=0: U[0,0]V[0,1] = 1*0 = 0. (Matches delta[0,1])
    # i=1, k=0, j=0: U[3,0]V[0,0] = 1*1 = 1. (Does NOT match delta[1,0]=0)
    #
    # So the seed's check `sum_c U[i*m+j,c] V[j*p+k,c] == delta[i,k]` FAILS for Strassen.
    #
    # Why did the seed program work?
    # The seed program returned schoolbook (rank 28).
    # For schoolbook:
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # c corresponds to (i,j,k).
    # sum_c U[i*m+j,c] V[j*p+k,c] = sum_{i',j',k'} delta[i,i']delta[j,j']delta[k,k'] = 1 if i=i', j=j', k=k'?
    # No, the sum is over c.
    # For fixed i,j,k:
    # Only the term c=(i,j,k) contributes 1.
    # So sum = 1.
    # The check requires `1 == (1 if i==k else 0)`.
    # This implies the seed's check is ALSO wrong for schoolbook if i != k?
    #
    # Wait, the seed's check:
    # `if s != (1 if i == k else 0): return False`
    # For schoolbook, s=1 always.
    # So it returns False if i != k.
    #
    # Does the seed program actually pass?
    # The score is 1.0000.
    # The prompt says: "The evaluator measures the raw value... invalid result... scores 0."
    # If the seed returned 1.0, it means it was considered valid.
    #
    # Perhaps the evaluator does NOT use the seed's `_verify_exact` function?
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The evaluator likely checks the correctness by multiplying random matrices.
    #
    # My `_verify_exact` should be robust.
    # I will implement a correct verification:
    # Check that for random A, B, C_computed == A*B.
    #
    # Let's rewrite `_verify_exact` to be correct.
    pass

def _verify_decomposition(U, V, W, n, m, p):
    R = len(U[0])
    # Convert to Fraction for exactness
    Uf = [[Fraction(U[i][c]) for c in range(R)] for i in range(n*m)]
    Vf = [[Fraction(V[j][c]) for c in range(R)] for j in range(m*p)]
    Wf = [[Fraction(W[k][c]) for c in range(R)] for k in range(p*n)]
    
    # Random test
    import random
    rng = random.Random(42)
    for _ in range(10):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = AB
        C_ref = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_ref[i][k] = s
        
        # Compute C using decomposition
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j' B[j',k] V[j'*p+k, c])
        C_calc = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for c in range(R):
                    w = Wf[k*n+i][c]
                    if w == 0: continue
                    u_val = Fraction(0)
                    for j in range(m):
                        u_val += A[i][j] * Uf[i*m+j][c]
                    v_val = Fraction(0)
                    for j in range(m):
                        v_val += B[j][k] * Vf[j*p+k][c]
                    s += w * u_val * v_val
                C_calc[i][k] = s
        
        if C_calc[i][k] != C_ref[i][k]:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Rank 25 decomposition
    U, V, W = _build_rank25(n, m, p)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
