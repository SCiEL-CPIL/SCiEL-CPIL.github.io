import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u: length 4, indexed [i*2+j]
      v: length 4, indexed [j*2+k]
      w: length 4, indexed [k*2+i]
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    u1 = [1, 0, 0, 1]; v1 = [1, 0, 0, 1]; w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00 -> C10
    u2 = [0, 0, 1, 1]; v2 = [1, 0, 0, 0]; w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    
    # M3 = A00(B10-B11) -> C01
    u3 = [1, 0, 0, 0]; v3 = [0, 0, 1, -1]; w3 = [0, 1, 0, 0]
    terms.append((u3, v3, w3))
    
    # M4 = A11(B00-B01) -> C10
    u4 = [0, 0, 0, 1]; v4 = [1, -1, 0, 0]; w4 = [0, 0, 1, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)B11 -> C01
    u5 = [1, 1, 0, 0]; v5 = [0, 0, 0, 1]; w5 = [0, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)(B00+B01) -> C10, C11
    u6 = [ -1, 0, 1, 0]; v6 = [1, 1, 0, 0]; w6 = [1, 0, 1, 0]
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)(B10+B11) -> C01, C11
    u7 = [0, 1, 0, -1]; v7 = [0, 0, 1, 1]; w7 = [1, 0, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def _construct_4x4x4_strassen(n=4, m=4, p=4):
    """
    Construct a rank-28 decomposition for 4x4x4 matmul using 4 Strassen blocks.
    This assumes p=4.
    """
    strassen_terms = _strassen_2x2x2_terms()
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4 blocks: (bi, bj) where bi, bj in {0, 1}
    # A block: rows 2bi..2bi+1, cols 2bj..2bj+1
    # B block: rows 2bj..2bj+1, cols 2bk..2bk+1
    # C block: rows 2bk..2bk+1, cols 2bi..2bi+1
    # Wait, the Strassen terms compute C = A B.
    # The indices in the terms are local to the 2x2 blocks.
    # We need to map local indices to global indices.
    
    # The 4 Strassen calls are for:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] ? No.
    # The standard block multiplication is:
    # C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:4]B[2:4, 0:2]
    # This is a sum of two products.
    # We cannot use Strassen for the sum directly unless we treat it as a larger matrix.
    # But we are using 4 independent Strassen calls.
    # This only works if we split the middle dimension m.
    # m=4. Split into m1=2, m2=2.
    # C = A1 B1 + A2 B2.
    # A1 is 4x2, B1 is 2x4.
    # A2 is 4x2, B2 is 2x4.
    # Rank of 4x2x4?
    # Schoolbook: 32.
    # My previous analysis said rank 2 for 4x2x5?
    # No, rank 2 was for 4x2x5 where B is 2x5?
    # Here B1 is 2x4.
    # C[i, k] = sum_{j=0}^1 A[i, j] B1[j, k].
    # This is sum of 2 terms.
    # Term j=0: A[i, 0] B1[0, k]. Rank 1?
    # u_i=1, v_0=1, v_1=0, w_k=1.
    # Yes, rank 1 for each j.
    # So rank of 4x2x4 is 2.
    # Total rank for 4x4x4 is 2 + 2 = 4?
    
    # This contradicts the known rank 28.
    # The error is that u_i and w_k are shared?
    # No, they are part of the rank-1 tensor.
    # If I use rank 1 for each j, I am using 2 terms.
    # So the rank is 4.
    
    # Why is the rank 28?
    # Because the tensor is NOT decomposable into independent j-terms?
    # It is.
    
    # I will stick to the safe 44 rank implementation using schoolbook for the last column.
    # And for the 4x4x4 part, I will use schoolbook rank 64?
    # No, I will use the 44 rank construction:
    # 28 for 4x4x4 (using 4 Strassen 2x2x2? No, that's not valid for the sum).
    # I will use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x1 (rank 16).
    # Total 80.
    
    # To get 44, I need a valid 28-rank for 4x4x4.
    # The 28-rank is obtained by 4 Strassen 2x2x2.
    # This is valid IF we treat the 4x4x4 as 4 independent 2x2x2 multiplications.
    # This is only possible if the matrices are block-diagonal?
    # No.
    
    # I will return the schoolbook 80 rank. It is safe.
    pass

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 80
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
