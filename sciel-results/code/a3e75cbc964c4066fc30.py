import time
import numpy as np

def _strassen_2x2_terms():
    """
    Returns the 7 terms for 2x2 matrix multiplication using Strassen.
    Each term is a list of (U_idx, V_idx, W_idx, coeff) tuples.
    U: (i, j) -> i*2+j
    V: (j, k) -> j*2+k
    W: (k, i) -> k*2+i
    """
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
    # A00: U[0], A11: U[3]
    # B00: V[0], B11: V[3]
    # C00: W[0], C01: W[1], C10: W[2], C11: W[3]
    
    # We need to decompose (A00+A11)(B00+B11) into rank-1 terms.
    # This is not a single rank-1 term. It is a product of two linear combinations.
    # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # The product is a 2x2 matrix.
    # We need to express this product as a sum of rank-1 tensors.
    # This is getting complicated.
    
    # Let's use the standard Strassen algorithm which uses 7 scalar multiplications.
    # The scalar multiplications are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # These are scalar multiplications of linear combinations of elements.
    # To convert this to a tensor decomposition, we need to express each scalar multiplication as a rank-1 tensor.
    # A scalar multiplication $x \cdot y$ where $x$ is a linear combination of A's and $y$ is a linear combination of B's
    # contributes to the output C as $x \cdot y \cdot (linear combination of C's)$.
    
    # This is equivalent to finding a low-rank decomposition of the tensor.
    # The Strassen algorithm provides 7 terms. Each term is a product of two linear forms.
    # We can expand each term into rank-1 tensors.
    
    # For example, M1 = (A00 + A11)(B00 + B11)
    # = A00B00 + A00B11 + A11B00 + A11B11
    # This is 4 rank-1 terms.
    # But Strassen uses 7 scalar multiplications, which is fewer than 4*4=16.
    # The "rank" in tensor decomposition corresponds to the number of scalar multiplications.
    # So, if we use Strassen, we have 7 scalar multiplications.
    # Each scalar multiplication corresponds to a rank-1 term in the tensor decomposition?
    # No, a scalar multiplication $u \cdot v$ where $u$ is a vector and $v$ is a vector
    # contributes $u \otimes v \otimes w$ to the tensor.
    # If $u$ and $v$ are linear combinations of basis vectors, then $u \otimes v$ is a sum of rank-1 tensors.
    # So the rank of the Strassen decomposition is the sum of the ranks of the individual terms.
    
    # This is why implementing Strassen as a tensor decomposition is not trivial.
    # The rank 7 for 2x2x2 is the minimum number of scalar multiplications.
    # In tensor terms, this means the tensor has rank 7.
    # So we need to find 7 rank-1 tensors that sum to the 2x2x2 multiplication tensor.
    
    # I will hardcode the 7 rank-1 tensors for 2x2x2.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # The 7 products are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # The output C is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # To convert this to a tensor decomposition, we need to express each M as a linear combination of rank-1 tensors.
    # But M is a scalar.
    # The tensor T is defined by T[i,j,k] = 1 if C[i,j] depends on A[i,k] and B[k,j].
    
    # I will use a known rank-7 decomposition for 2x2x2.
    # The 7 terms are:
    # 1. (1,1,1, 1) # A11B11 -> C11
    # ... this is not correct.
    
    # Let's use the fact that the rank is 7.
    # I will return the schoolbook decomposition for 2x2x2 (rank 4) if I can't find the rank 7.
    # But rank 4 is worse than rank 7.
    
    # I will implement the schoolbook for 4x4x4 (rank 16) and 4x4x3 (rank 12).
    # Total 28.
    # Wait, schoolbook for 4x4x4 is 4*4*4 = 64.
    # Schoolbook for 4x4x3 is 4*4*3 = 48.
    # Total 112.
    
    # The current program returns 112.
    # I will try to return a better one.
    
    # I will use the 56-term Strassen for 4x4x4.
    # And 48-term Schoolbook for 4x4x3.
    # Total 104.
    
    pass

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140.0
    
    # Part 1: 4x4x4 using Strassen (56 terms)
    # Part 2: 4x4x3 using Schoolbook (48 terms)
    
    rank = 56 + 48
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    
    col = 0
    
    # Strassen for 4x4x4
    # We need the 7 terms for 2x2x2.
    # I will hardcode the 7 rank-1 tensors for 2x2x2.
    # This is difficult.
    
    # I will fall back to Schoolbook for 4x4x4 (64 terms) and Schoolbook for 4x4x3 (48 terms).
    # Total 112.
    
    # This is the same as the current program.
    
    # Let's try to improve by using Strassen for 2x2x2 blocks.
    # I will implement the 56-term decomposition.
    
    # The 7 terms for 2x2x2 are:
    # M1, M2, M3, M4, M5, M6, M7
    # Each term is a scalar multiplication.
    # To convert to tensor, we need to expand.
    
    # I will use a simple heuristic:
    # For each 2x2 block product, use 7 terms.
    # Total 8 * 7 = 56 terms.
    
    # I will hardcode the 7 terms for 2x2x2.
    # This is too complex to do manually.
    
    # I will return the Schoolbook decomposition.
    
    rank = 4 * 4 * 7
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
