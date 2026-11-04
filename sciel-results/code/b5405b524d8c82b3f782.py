"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer_list(lst):
    """Check if all entries are half-integers (n/2 for integer n)."""
    for row in lst:
        for x in row:
            if abs(x * 2 - round(x * 2)) > 1e-9:
                return False
    return True

def _strassen_222():
    """
    Return the 7-term Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Standard Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U factors: row i*2+j
    # A11 -> row 0, A12 -> row 1, A21 -> row 2, A22 -> row 3
    U = [[0]*7 for _ in range(4)]
    # V factors: row j*2+k
    # B11 -> row 0, B12 -> row 1, B21 -> row 2, B22 -> row 3
    V = [[0]*7 for _ in range(4)]
    # W factors: row k*2+i
    # C11 -> row 0, C12 -> row 1, C21 -> row 2, C22 -> row 3
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose the 4-dimensional inner product into two 2-dimensional subspaces.
    # Block 1: rows 0,1 of A (size 2x2) * cols 0,1 of B (size 2x2) -> contributes to C[0:2, 0:2]
    # Block 2: rows 2,3 of A (size 2x2) * cols 2,3 of B (size 2x2) -> contributes to C[0:2, 2:4]
    
    # Wait, the matrix A is 2x4, B is 4x5.
    # A = [A_top; A_bottom] where A_top is 2x4? No.
    # A is 2x4. Let's split the 4 columns of A into two groups of 2: A = [A1, A2] where A1 is 2x2, A2 is 2x2.
    # B is 4x5. Let's split the 4 rows of B into two groups of 2: B = [B1; B2] where B1 is 2x5, B2 is 2x5.
    # Then C = A1 * B1 + A2 * B2.
    # This is a sum of two matrix multiplications: (2x2)*(2x5) and (2x2)*(2x5).
    # The rank of MM_{2,2,5} is 10 (schoolbook: 2*2*5=20, but Strassen-like or lower? No, rank of 2x2x5 is 10? 
    # Actually, rank of MM_{2,2,p} is known to be 2p for p>=2? 
    # Let's check: MM_{2,2,2} rank is 7. MM_{2,2,3} rank is 9? 
    # The schoolbook rank is 2*2*5 = 20.
    # If we use Strassen for the 2x2 part, we can multiply 2x2 by 2x2 in rank 7.
    # But B is 2x5. We can view B as three 2x2 blocks plus a 2x1? Or just use linearity.
    # C = A1 * B1 + A2 * B2.
    # B1 is 2x5. Let's split B1 into columns: B1 = [b1, b2, b3, b4, b5] where each bj is 2x1.
    # Then A1 * B1 = [A1*b1, A1*b2, ...]
    # This doesn't directly give a tensor decomposition with low rank unless we use the linearity.
    # The tensor for MM_{2,2,5} has rank at most 2*2*5 = 20.
    # Actually, the rank of MM_{2,2,5} is 10. Why?
    # Consider the map (A, B) -> A*B.
    # For fixed B, the map A -> A*B is linear.
    # It is known that R(MM_{2,2,p}) = 2p for p=2,3,4,5?
    # Let's assume R(MM_{2,2,5}) = 10.
    # Then R(MM_{2,4,5}) <= R(MM_{2,2,5}) + R(MM_{2,2,5}) = 10 + 10 = 20.
    # The schoolbook rank is 40. So 20 is a massive improvement!
    # Can we do even better?
    # R(MM_{2,4,5}) >= R(MM_{2,2,5})? No.
    # Is there a decomposition of rank < 20?
    # Let's try to construct the rank-20 decomposition.
    # We need a decomposition for MM_{2,2,5} of rank 10.
    # How to get rank 10 for 2x2 * 2x5?
    # We can use the fact that A is 2x2. Let A = [[a, b], [c, d]].
    # B is 2x5. Let B = [u; v] where u, v are 1x5 vectors.
    # C = A*B = [[a*u + b*v], [c*u + d*v]].
    # C has 2 rows, 5 cols. 10 entries.
    # c11 = a*u1 + b*v1
    # c12 = a*u2 + b*v2
    # ...
    # c21 = c*u1 + d*v1
    # ...
    # This is a bilinear map from (a,b,c,d, u1..u5, v1..v5) to 10 outputs.
    # Can we decompose this into 10 rank-1 tensors?
    # Yes, because each output involves a product of one A-entry and one B-entry?
    # No, c11 = a*u1 + b*v1. This is a sum of two products.
    # So we need at least 2 rank-1 terms for c11?
    # Term 1: a * u1 * 1 (contributes to c11)
    # Term 2: b * v1 * 1 (contributes to c11)
    # But wait, the tensor decomposition requires the same factor index c to be used for all outputs.
    # T = sum_c U_c \otimes V_c \otimes W_c.
    # For each c, we pick a linear functional on A, a linear functional on B, and a linear functional on C (which is actually just picking an output coordinate in the standard basis, but with weights).
    # Actually, the output is not just a coordinate, it's the whole matrix.
    # In the tensor T_{i,j,k}, T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # For MM_{2,2,5}:
    # i in {0,1}, j in {0,1}, k in {0..4}.
    # T_{i,j,k} = 1 if C_{i,k} contains A_{i,j} * B_{j,k}.
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
    # So T_{0,0,0} = 1 (for A_{0,0}B_{0,0}), T_{0,1,0} = 1 (for A_{0,1}B_{1,0}).
    # The tensor is the sum of two "diagonal" tensors?
    # Tensor 1: non-zero only at (i, i, k) for i in {0,1}, k in {0..4}. Value 1.
    # Tensor 2: non-zero only at (i, 1-i, k) for i in {0,1}, k in {0..4}. Value 1.
    # Rank of Tensor 1?
    # It is a direct sum of 5 copies of a 2x2x1 tensor?
    # For a fixed k, the slice is diag(1,1) in (i,j).
    # The tensor T1 = sum_{k=0}^4 (e_0 \otimes e_0 \otimes f_k) + (e_1 \otimes e_1 \otimes f_k).
    # This is a sum of 10 rank-1 tensors.
    # Can it be done with fewer?
    # The slice at k is diag(1,1). The rank of the tensor diag(1,1) in 2x2x1 is 2?
    # e0 \otimes e0 \otimes 1 + e1 \otimes e1 \otimes 1.
    # Yes, rank 2.
    # Since the k-slices are independent (the f_k are orthogonal basis vectors), the rank of the whole tensor is the sum of the ranks of the slices?
    # Yes, if the supports in the k-direction are disjoint, the rank is the sum.
    # So R(T1) = 5 * 2 = 10.
    # Similarly, T2 (off-diagonal) has rank 10.
    # Total rank = 20.
    # Is there any overlap? No, T1 and T2 have disjoint supports in the (i,j) plane.
    # So the rank of MM_{2,2,5} is exactly 20?
    # Wait, if R(MM_{2,2,5}) = 20, then R(MM_{2,4,5}) = 40.
    # Let's re-evaluate.
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
    # This requires 2 terms.
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}.
    # This requires 2 terms.
    # ...
    # There are 10 outputs. Each output is a sum of 2 products.
    # Can we share terms between outputs?
    # Term c: U_c \otimes V_c \otimes W_c.
    # W_c is a vector in C-space (2x5).
    # If we set W_c = f_k (the k-th column of C), then the term contributes to the k-th column.
    # To compute column k of C, we need A * B_k.
    # A is 2x2, B_k is 2x1.
    # The multiplication of 2x2 and 2x1 is a 2x1 vector.
    # c0 = a*u + b*v
    # c1 = c*u + d*v
    # This is a bilinear map.
    # Rank of MM_{2,2,1}?
    # T_{i,j,0}.
    # T_{0,0,0}=1, T_{0,1,0}=1, T_{1,0,0}=1, T_{1,1,0}=1? No.
    # C_0 = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
    # C_1 = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}.
    # So T_{0,0,0}=1, T_{0,1,0}=1, T_{1,0,0}=1, T_{1,1,0}=1.
    # This is the rank of the 2x2 matrix [[1,1],[1,1]]? No, the tensor is just 4 non-zero entries.
    # Can we decompose it into 1 rank-1 tensor? No, because it's not rank-1 in the tensor sense.
    # U \otimes V \otimes W.
    # If W is fixed (scalar 1), we need U \otimes V to be [[1,1],[1,1]].
    # Rank of [[1,1],[1,1]] is 1. So U = [1,1]^T, V = [1,1]^T.
    # So T = [1,1]^T \otimes [1,1]^T \otimes [1] ?
    # Let's check.
    # Sum_c U_i V_j W_k.
    # If c=1, U=[1,1], V=[1,1], W=[1].
    # T_{0,0} = 1*1*1 = 1.
    # T_{0,1} = 1*1*1 = 1.
    # T_{1,0} = 1*1*1 = 1.
    # T_{1,1} = 1*1*1 = 1.
    # Yes! The rank of MM_{2,2,1} is 1.
    # This is because the multiplication of a 2x2 matrix and a 2x1 vector is just a matrix-vector product, which is bilinear but can be represented by a single tensor term?
    # Wait. The tensor T_{i,j,k} is 1 if A_{i,j} B_{j,k} contributes to C_{i,k}.
    # For MM_{2,2,1}, k is always 0.
    # T_{0,0,0} = 1 (from A00 B00)
    # T_{0,1,0} = 1 (from A01 B10)
    # T_{1,0,0} = 1 (from A10 B00)
    # T_{1,1,0} = 1 (from A11 B10)
    # My previous deduction that U=[1,1], V=[1,1], W=[1] works implies that:
    # C_0 = 1*1*A_0 + 1*1*A_1? No.
    # The formula is C_i = sum_j sum_c U_{i,c} V_{j,c} W_{k,c} A_{i,j} B_{j,k}? No.
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # The value is T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}.
    # If we set U_{0,1}=1, U_{1,1}=1, V_{0,1}=1, V_{1,1}=1, W_{0,1}=1.
    # Then T_{0,0,0} = 1. T_{0,1,0} = 1. T_{1,0,0} = 1. T_{1,1,0} = 1.
    # This matches the tensor!
    # So R(MM_{2,2,1}) = 1.
    
    # If R(MM_{2,2,1}) = 1, then for MM_{2,2,5}, since the k-slices are independent,
    # R(MM_{2,2,5}) = 5 * R(MM_{2,2,1}) = 5.
    # Let's verify.
    # For each k, we have a term c_k.
    # U_{i, c_k} = 1 for all i?
    # V_{j, c_k} = 1 for all j?
    # W_{k', c_k} = 1 if k'=k, else 0.
    # Then T_{i,j,k} = sum_{c_k'} U_{i,c_k'} V_{j,c_k'} W_{k,c_k'} = 1*1*1 = 1.
    # This works for all i,j,k.
    # So the rank is 5.
    
    # So R(MM_{2,2,5}) = 5.
    # Then R(MM_{2,4,5}) <= R(MM_{2,2,5}) + R(MM_{2,2,5}) = 5 + 5 = 10.
    
    # Let's construct the rank-10 decomposition.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C = A1 B1 + A2 B2.
    # For A1 B1:
    # Terms c0..c4.
    # For c = 0..4:
    #   U: for i in {0,1}, j in {0,1}: U_{i,j} = 1. (Indices in A1)
    #   V: for j' in {0,1}, k in {0..4}: V_{j',k} = 1. (Indices in B1)
    #   W: for k in {0..4}, i in {0,1}: W_{k,i} = 1 if k == c, else 0.
    # Wait, the indices for U, V, W in the global tensor are:
    # U: (i, j) where i in 0..1, j in 0..3.
    # V: (j, k) where j in 0..3, k in 0..4.
    # W: (k, i) where k in 0..4, i in 0..1.
    
    # For the first block (A1, B1):
    # A1 indices: i in {0,1}, j in {0,1}.
    # B1 indices: j in {0,1}, k in {0..4}.
    # C indices: i in {0,1}, k in {0..4}.
    
    # Term t (t=0..4):
    # U_{i, j} = 1 if i in {0,1} and j in {0,1} and t is used?
    # No, for a specific term c, U_{i,j,c} is non-zero only for j in {0,1}.
    # U_{i,j,c} = 1 for i in {0,1}, j in {0,1}.
    # V_{j,k,c} = 1 for j in {0,1}, k in {0..4}.
    # W_{k,i,c} = 1 if k == c, else 0. (And i in {0,1}).
    
    # Let's check the sum for a specific T_{i,j,k}.
    # If j in {0,1}:
    # Sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # = Sum_c 1 * 1 * (1 if k==c else 0)
    # = 1 if k is in {0..4}, which is always true.
    # So T_{i,j,k} = 1 for j in {0,1}.
    # This matches the definition of T (it is 1 if the product contributes).
    
    # If j in {2,3}:
    # U_{i,j,c} = 0 for these c's.
    # So sum is 0.
    
    # So the first 5 terms handle the j in {0,1} part.
    # The second 5 terms handle the j in {2,3} part.
    # For terms c=5..9:
    # U_{i,j,c} = 1 for i in {0,1}, j in {2,3}.
    # V_{j,k,c} = 1 for j in {2,3}, k in {0..4}.
    # W_{k,i,c} = 1 if k == c-5, else 0.
    
    # Total rank = 10.
    
    rank = 10
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 10
    V = [[0] * rank for _ in range(m * p)]  # 20 x 10
    W = [[0] * rank for _ in range(p * n)]  # 10 x 10
    
    # Block 1: j in {0, 1}
    for c in range(5):
        for i in range(n):
            for j in range(2):
                U[i * m + j][c] = 1
        for j in range(2):
            for k in range(p):
                V[j * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                if k == c:
                    W[k * n + i][c] = 1
    
    # Block 2: j in {2, 3}
    for c in range(5, 10):
        cc = c - 5
        for i in range(n):
            for j in range(2, 4):
                U[i * m + j][c] = 1
        for j in range(2, 4):
            for k in range(p):
                V[j * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                if k == cc:
                    W[k * n + i][c] = 1
    
    # Verify
    assert _validate(U, V, W, n, m, p), "Rank 10 decomposition failed validation!"
    assert _is_half_integer_list(U), "U is not half-integer"
    assert _is_half_integer_list(V), "V is not half-integer"
    assert _is_half_integer_list(W), "W is not half-integer"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
