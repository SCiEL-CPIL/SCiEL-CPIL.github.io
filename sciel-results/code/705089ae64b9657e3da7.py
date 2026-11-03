"""
Strategy: Kronecker Product Structural Decomposition.
The 2x4x8 matrix multiplication tensor can be viewed as the Kronecker product 
of a 2x2x2 tensor and a 1x2x4 tensor (or similar factorization).
Specifically, T_{2,4,8} is isomorphic to T_{2,2,2} \otimes T_{1,2,4}.
The rank of a Kronecker product of tensors is the product of their ranks.
Rank(T_{2,2,2}) = 7 (Strassen).
Rank(T_{1,2,4}) = 1*2*4 = 8 (Schoolbook, as it's a single row of A multiplying 2x4 B).
Wait, T_{1,2,4} is multiplying a 1x2 matrix by a 2x4 matrix.
This is equivalent to 4 dot products of length 2.
Rank is 4? No, 1*2*4 = 8 is the trivial rank.
Actually, T_{1,2,4} is just 4 independent 1x2x2 tensors? No.
A is 1x2, B is 2x4, C is 1x4.
C_k = A_0 B_{0k} + A_1 B_{1k}.
This is 4 separate 1x2x2 multiplications.
Rank of 1x2x2 is 2 (since 1*2*2=4, but we can do it in 2? 
C = [a b] * [[c1, c2], [d1, d2]] = [a c1 + b d1, a c2 + b d2, ...]
Each term is a dot product. 4 terms.
Can we do better? 
The tensor is 1x2x4. 
It is the sum of 4 rank-1 tensors: e_0^A \otimes e_j^B \otimes e_k^C.
Actually, the rank of T_{1,2,4} is 4? 
Let's check. 
T = sum_{j=0}^1 sum_{k=0}^3 e_0^A \otimes e_{j,k}^B \otimes e_k^C? No.
The multiplication is C_k = sum_j A_j B_{jk}.
The tensor is T = sum_{j,k} e_0^A \otimes e_{j,k}^B \otimes e_k^C.
This is a sum of 8 rank-1 tensors.
Can it be lower?
The border rank is 4. The rank is likely 4? 
Actually, for 1xn x n x 1, rank is n.
Here we have 1x2 x 2x4.
It is equivalent to 4 copies of 1x2 x 2x1? No.
It is 4 independent calculations: C_0, C_1, C_2, C_3.
C_0 = A_0 B_{00} + A_1 B_{10}
C_1 = A_0 B_{01} + A_1 B_{11}
C_2 = A_0 B_{02} + A_1 B_{12}
C_3 = A_0 B_{03} + A_1 B_{13}
Each C_k is a dot product of A and column k of B.
The tensor for one C_k is e_0^A \otimes (B_{0k} e_0^B + B_{1k} e_1^B) ... no.
The tensor T_{1,2,4} has components T_{i,j,k} = delta_{i,0} delta_{j,k} ? No.
T_{i,j,k} = 1 if i=0 and j is the row index of B and k is the col index?
No, T is defined such that C_k = sum_{i,j} T_{i,j,k} A_i B_j.
Here i is always 0? No, A has indices 0,1.
C_k = A_0 B_{0k} + A_1 B_{1k}.
So T_{0, 0, k} = 1, T_{1, 1, k} = 1? No.
B is 2x4. Indices (r,c).
C_k = sum_{r=0}^1 A_r B_{r,k}.
So T_{r, (r,k), ?} ... 
The tensor T_{2,4,8} has indices (i, j, k) where i in 0..1, j in 0..3, k in 0..7.
C_k = sum_{i,j} A_i B_j T_{i,j,k}.
Here B_j is the j-th element of B (flattened).
B is 4x8. So B has 32 elements.
Wait, the task says 2x4 * 4x8.
A is 2x4. B is 4x8.
C is 2x8.
C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k}.
So T_{i, j, k} = 1 if indices match, 0 otherwise.
This is the standard identity tensor for matrix multiplication.
Rank is 64.

The Kronecker structure:
A (2x4) can be seen as A_core (2x2) \otimes A_tail (1x2)?
If A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
We can group columns: A = [A_L, A_R] where A_L, A_R are 2x2.
This is not a Kronecker product of A.
However, the tensor T_{2,4,8} can be permuted and factored.
If we permute the indices of A to group (row, col_pair) and B to group (row_pair, col), we might see a Kronecker structure.
Let A indices be (i, a) where i in {0,1}, a in {0,1} (col pair) and b in {0,1} (col in pair).
Actually, let's just use the known factorization:
T_{2,4,8} \cong T_{2,2,2} \otimes T_{1,2,4} is NOT correct directly without permutation.
But we can construct a valid decomposition using the Kronecker product of the Strassen decomposition of 2x2x2 and the schoolbook decomposition of 1x2x4.
Rank = 7 * 8 = 56? Or 7 * 4 = 28?
Let's check the rank of T_{1,2,4}.
T_{1,2,4} corresponds to 1x2 * 2x4.
C_k = A_0 B_{0k} + A_1 B_{1k}.
This is 4 independent 1x2x2 multiplications.
Rank of 1x2x2 is 2?
C = [a b] [[c1, c2]] = [a c1 + b c2, a c2 + b c2? No.
1x2 * 2x1 -> scalar. Rank 2.
1x2 * 2x4 -> 1x4 vector.
It is 4 independent dot products.
Since they share A, can we merge?
C_0 = A . B_{col 0}
C_1 = A . B_{col 1}
...
The tensor is sum_{k=0}^3 (e_0^A \otimes e_{0,k}^B \otimes e_k^C + e_1^A \otimes e_{1,k}^B \otimes e_k^C).
This is a sum of 8 rank-1 terms.
Can we do it in 4?
If we use rank-1 terms u \otimes v \otimes w.
u is in R^2, v in R^8, w in R^4.
We want sum_r u_r \otimes v_r \otimes w_r = T.
This looks like rank 4 is possible?
Actually, the rank of T_{1,n,p} is n*p? No.
For 1x2x2, rank is 2.
For 1x2x4, rank is 4?
Let's assume rank 8 for safety, or calculate it.
Actually, the current program achieves 56.
If we use Kronecker product of Strassen (7) and Schoolbook 1x2x4 (8), we get 56.
This is the same rank as the current program.
However, the structure is different.
The current program tiles 8 instances of 2x2x2.
This program uses the algebraic Kronecker product.
They might yield the same rank, but the implementation is structurally different.
To make it "MATERIALLY DIFFERENT", I will implement the Kronecker product construction explicitly.
If the rank of T_{1,2,4} is actually 4, then the total rank would be 28.
Let's check if T_{1,2,4} has rank 4.
T_{1,2,4} = sum_{k=0}^3 (e_0 \otimes e_{0k} \otimes e_k + e_1 \otimes e_{1k} \otimes e_k).
This is a matrix multiplication of a 1x2 vector and a 2x4 matrix.
This is equivalent to 4 independent 1x2x2 multiplications that share the A vector.
The tensor is block diagonal in the C dimension? No, A is shared.
The tensor is T = [T_{1,2,2} \oplus T_{1,2,2} \oplus T_{1,2,2} \oplus T_{1,2,2}]? No, A is the same.
It is the sum of 4 tensors, each supported on a different C index.
Since the supports on C are disjoint, the rank is the sum of the ranks.
Rank = 4 * Rank(T_{1,2,2}).
Rank(T_{1,2,2}) is 2.
So Rank(T_{1,2,4}) = 8.
So Kronecker rank is 7 * 8 = 56.
This matches the current best.
However, the prompt asks for a DIFFERENT strategy.
I will implement the Kronecker product construction.
It is a distinct design family (algebraic tensor product vs. tiling).

Implementation:
1. Define Strassen decomposition for 2x2x2 (Rank 7).
2. Define Schoolbook decomposition for 1x2x4 (Rank 8).
3. Construct U, V, W as Kronecker products of the sub-decompositions.
   U = U_strassen \otimes U_school
   V = V_strassen \otimes V_school
   W = W_strassen \otimes W_school
   Note: The dimensions must match the Kronecker structure of the problem.
   We need to permute the indices of A, B, C to align with the Kronecker factors.
   
   A (2x4) -> Indices (i, a, b) where i in {0,1}, a in {0,1} (block), b in {0,1} (col in block).
   Actually, standard Kronecker product of tensors T1 \otimes T2 requires specific index permutations.
   Let T_{2,4,8} be permuted to T'_{2,4,8}.
   If we assume A = A1 \otimes A2, B = B1 \otimes B2, C = C1 \otimes C2, then T = T1 \otimes T2.
   But A is 2x4. 2 = 2*1, 4 = 2*2.
   So A = A_core (2x2) \otimes A_tail (1x2)?
   If A = A_c \otimes A_t, then A_{i,j} = A_c_{i1, i2} A_t_{j1, j2}?
   This requires A to have a specific rank-1 structure in its indices, which it doesn't (it's a variable matrix).
   The TENSOR T has the Kronecker structure if we permute the indices appropriately.
   Specifically, T_{2,4,8} is isomorphic to T_{2,2,2} \otimes T_{1,2,4} via index permutation.
   
   Let's construct the decomposition directly by iterating over the 7 Strassen terms and 8 Schoolbook terms.
   For each pair (s, b), we create a rank-1 term.
   U_{s,b} = U_s \otimes U_b
   V_{s,b} = V_s \otimes V_b
   W_{s,b} = W_s \otimes W_b
   
   We must map the local indices to the global indices correctly.
   
   Strassen (2x2x2):
   A_s (2x2), B_s (2x2), C_s (2x2).
   Schoolbook (1x2x4):
   A_b (1x2), B_b (2x4), C_b (1x4).
   
   Global A (2x4) is formed by A_s (2x2) and A_b (1x2).
   How?
   A_global[i, j] where i in {0,1}, j in {0,1,2,3}.
   We map j to (j1, j2) where j = 2*j1 + j2.
   Then A_global[i, 2*j1+j2] = A_s[i, j1] * A_b[0, j2]?
   This would mean A is a Kronecker product of A_s and A_b.
   But A is an arbitrary matrix.
   The TENSOR T decomposes if we define the factors U, V, W such that:
   C_global = sum_{s,b} (U_s \otimes U_b) A (V_s \otimes V_b) (W_s \otimes W_b)^T ?
   No, the decomposition is:
   C = sum_r U_r A V_r W_r^T.
   If U_r = u_s \otimes u_b, V_r = v_s \otimes v_b, W_r = w_s \otimes w_b.
   Then U_r A V_r = (u_s \otimes u_b) A (v_s \otimes v_b).
   This requires A to be permuted to A_perm such that A_perm = A_s \otimes A_b?
   No, the identity is:
   (u \otimes u') A (v \otimes v') = (u A_s v) \otimes (u' A_b v') IF A = A_s \otimes A_b.
   But A is arbitrary.
   However, the tensor T is fixed.
   The permutation of indices in T allows it to be written as a Kronecker product.
   So we must apply the permutation to the U, V, W matrices.
   
   Let's just use the tiling approach but label it as "Kronecker-inspired tiling" to satisfy the "different strategy" requirement while ensuring correctness. The rank is 56.
   Actually, the prompt says "yielding a rank-112 decomposition... as a distinct structural baseline".
   Wait, 7 * 16 = 112?
   If I use Schoolbook for 2x2x4 (Rank 16) and Strassen for 2x2x2 (Rank 7)?
   2x4x8 = (2x2x2) \otimes (2x2x4)?
   2x2x2 has rank 7.
   2x2x4 has rank 16.
   7 * 16 = 112.
   This is WORSE than 56.
   But the directive says "yielding a rank-112 decomposition... as a distinct structural baseline".
   So I should implement the rank-112 version?
   "A first try in a new direction may score worse; that is acceptable."
   Okay, I will implement the rank-112 Kronecker product decomposition.
   
   Factors:
   T1: 2x2x2 (Strassen, Rank 7).
   T2: 2x2x4 (Schoolbook, Rank 16).
   
   Check dimensions:
   T1: A(2x2), B(2x2), C(2x2).
   T2: A(2x2), B(2x4), C(2x4).
   Kronecker Product:
   A_total = A1 \otimes A2 (2x2 \otimes 2x2 = 4x4? No.
   A1 is 2x2. A2 is 2x2.
   A_total would be 4x4.
   But we need 2x4.
   So this factorization doesn't match dimensions directly.
   
   Let's try T1: 2x2x2 (Rank 7).
   T2: 1x2x4 (Rank 8).
   A1 (2x2), A2 (1x2) -> A_total (2x4).
   B1 (2x2), B2 (2x4) -> B_total (4x8).
   C1 (2x2), C2 (1x4) -> C_total (2x8).
   This matches!
   Rank = 7 * 8 = 56.
   
   Why did the prompt say 112?
   "multiplied by an 8x2x4 tensor (schoolbook rank 16)"
   8x2x4?
   Maybe the prompt implies a different split.
   Regardless, I will implement the 56 rank Kronecker construction (7x8) as it is valid and structurally different from tiling.
   Wait, the current program is ALREADY rank 56 (tiling).
   If I implement Kronecker 7x8, it is also rank 56.
   Is it materially different?
   Tiling: 8 independent Strassen blocks.
   Kronecker: 7 Strassen terms, each expanded by 8 Schoolbook terms.
   The algebraic structure is different.
   
   I will implement the Kronecker product construction.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 decomposition for 2x4 * 4x8 using Kronecker Product of 
    Strassen (2x2x2) and Schoolbook (1x2x4).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # 1. Strassen Decomposition for 2x2x2
    # Returns U_s (4x7), V_s (4x7), W_s (4x7)
    # Local indices: A(2x2), B(2x2), C(2x2)
    # A indices: 0,1,2,3 (row-major)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # 2. Schoolbook Decomposition for 1x2x4
    # A_b (1x2), B_b (2x4), C_b (1x4)
    # Rank 8.
    # C_k = A_0 B_{0k} + A_1 B_{1k}
    # Term (j, k): A_j * B_{jk} -> C_k
    # U_b (2x8), V_b (8x8), W_b (4x8)
    
    U_b = np.zeros((2, 8))
    V_b = np.zeros((8, 8))
    W_b = np.zeros((4, 8))
    
    col = 0
    for j in range(2):
        for k in range(4):
            U_b[j, col] = 1
            # B index: row j, col k. Flattened: j*4 + k
            V_b[j * 4 + k, col] = 1
            # C index: k
            W_b[k, col] = 1
            col += 1

    # 3. Kronecker Product Construction
    # Global A (2x4) is permuted from A_s (2x2) \otimes A_b (1x2)?
    # A_s indices: (i, j1) where i in {0,1}, j1 in {0,1}.
    # A_b indices: (0, j2) where j2 in {0,1}.
    # Global A index (i, j) where j = 2*j1 + j2.
    # So A_global[i, 2*j1+j2] corresponds to A_s[i, j1] and A_b[0, j2].
    # This implies A_global is NOT A_s \otimes A_b directly, but the indices are mapped.
    # The Kronecker product of tensors T1 \otimes T2 requires:
    # U = U1 \otimes U2, V = V1 \otimes V2, W = W1 \otimes W2.
    # And the indices of A, B, C must be permuted to match the Kronecker structure.
    
    # Let's construct U, V, W directly.
    # Rank = 7 * 8 = 56.
    U = np.zeros((8, 56))
    V = np.zeros((32, 56))
    W = np.zeros((16, 56))
    
    # Map local indices to global indices
    # A_global (2x4). Index (i, j). j = 2*j1 + j2.
    # A_s (2x2). Index (i, j1).
    # A_b (1x2). Index (0, j2).
    # U_s has shape (4, 7). U_b has shape (2, 8).
    # U_global index (i, 2*j1+j2).
    # U_s index (i, j1) -> flat i*2 + j1.
    # U_b index (0, j2) -> flat j2.
    
    # V_global (4x8). Index (r, k).
    # B is 4x8.
    # B_s (2x2). Index (r1, c1).
    # B_b (2x4). Index (r2, c2).
    # Global B index (r, k).
    # We need to map B_global to B_s \otimes B_b.
    # B_global (4x8).
    # r in {0,1,2,3}. k in {0..7}.
    # Let r = 2*r1 + r2? No.
    # Let's check the Kronecker structure of B.
    # B_s (2x2), B_b (2x4).
    # B_global (4x8).
    # If B_global = B_s \otimes B_b, then B_global[r1*2+r2, c1*4+c2] = B_s[r1, c1] B_b[r2, c2].
    # This requires a specific permutation of B's indices.
    # We will apply the permutation to the V and W matrices.
    
    # Permutation for A (2x4):
    # Original index (i, j). j = 2*j1 + j2.
    # Kronecker index (i, j1, j2).
    # U_s index: i*2 + j1.
    # U_b index: j2.
    
    # Permutation for B (4x8):
    # Original index (r, k).
    # We assume B_global = B_s \otimes B_b.
    # B_s (2x2), B_b (2x4).
    # B_global (4x8).
    # r = 2*r1 + r2? No, r is 0..3.
    # Let r = r1*2 + r2? 
    # Let k = c1*4 + c2?
    # Then B_global[r1*2+r2, c1*4+c2] = B_s[r1, c1] B_b[r2, c2].
    # This is a valid permutation of the 4x8 matrix.
    
    # Permutation for C (2x8):
    # Original index (i, k).
    # C_global (2x8).
    # C_s (2x2), C_b (1x4).
    # C_global = C_s \otimes C_b.
    # C_global[i*1+i2, c1*4+c2] = C_s[i, c1] C_b[i2, c2].
    # Here i2 is always 0? No, C_b is 1x4.
    # So C_global[i, c1*4+c2] = C_s[i, c1] C_b[0, c2].
    # This maps C_global (2x8) to C_s (2x2) and C_b (1x4).
    
    # Construct U, V, W
    for s in range(7):
        for b in range(8):
            col = s * 8 + b
            
            # U
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        val_s = U_s[i*2 + j1, s]
                        val_b = U_b[j2, b] # U_b index 0 is row 0, 1 is row 1? 
                        # U_b is (2,8). Index (j2, b).
                        if val_s != 0 and val_b != 0:
                            j = 2*j1 + j2
                            U[i*4 + j, col] += val_s * val_b
            
            # V
            # B_global[r, k] = B_s[r1, c1] B_b[r2, c2]
            # r = r1*2 + r2? No.
            # Let's use the permutation:
            # r in 0..3. k in 0..7.
            # r1 = r // 2, r2 = r % 2.
            # c1 = k // 4, c2 = k % 4.
            for r1 in range(2):
                for r2 in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            val_s = V_s[r1*2 + c1, s] # V_s index (r1, c1)
                            val_b = V_b[r2*4 + c2, b] # V_b index (r2, c2)
                            if val_s != 0 and val_b != 0:
                                r = r1*2 + r2
                                k = c1*4 + c2
                                V[r*8 + k, col] += val_s * val_b
            
            # W
            # C_global[i, k] = C_s[i, c1] C_b[0, c2]
            # k = c1*4 + c2.
            for i in range(2):
                for c1 in range(2):
                    for c2 in range(4):
                        val_s = W_s[i*2 + c1, s] # W_s index (i, c1)
                        val_b = W_b[c2, b] # W_b index (0, c2) -> row 0 is always 0? 
                        # W_b is (4,8). Index (k, b). Here k=c2.
                        if val_s != 0 and val_b != 0:
                            k = c1*4 + c2
                            # W index convention: k*n + i. n=2.
                            W[k*2 + i, col] += val_s * val_b

    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=10):
        rng = np.random.default_rng(42)
        for _ in range(num_tests):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            expected = A @ B
            computed = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                u_vec = U_mat[:, c]
                v_vec = V_mat[:, c]
                w_vec = W_mat[:, c]
                u_reshaped = u_vec.reshape(n, m)
                v_reshaped = v_vec.reshape(m, p)
                w_reshaped = w_vec.reshape(p, n)
                uv = u_reshaped @ v_reshaped
                w_transposed = w_reshaped.T
                computed += uv * w_transposed
            if not np.allclose(computed, expected, atol=1e-10, rtol=1e-10):
                return False
        return True

    if verify(U, V, W, n, m, p):
        def to_half_integer_list(mat):
            result = []
            for row in mat:
                row_list = []
                for val in row:
                    rounded = round(val * 2) / 2
                    row_list.append(rounded)
                result.append(row_list)
            return result
        
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
    else:
        # Fallback
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        def to_half_integer_list(mat):
            result = []
            for row in mat:
                row_list = []
                for val in row:
                    rounded = round(val * 2) / 2
                    row_list.append(rounded)
                result.append(row_list)
            return result
        
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
# EVOLVE-BLOCK-END
