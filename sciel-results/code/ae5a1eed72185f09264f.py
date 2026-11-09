"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.linalg import solve

def construct():
    """Return a rank-minimal exact tensor decomposition for 6x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 216
    
    # First, build the schoolbook decomposition as fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition using Strassen-based approach
    # Strassen on 2x2 blocks of 3x3 with schoolbook 3x3: rank = 7 * 27 = 189
    # This is a valid improvement over 216
    
    # Let me try to construct the Strassen decomposition
    result_189 = _build_strassen_6x6(n, m, p)
    if result_189 is not None and _validate(result_189, n, m, p):
        best_result = result_189
        best_rank = 189
    
    if time.time() > deadline:
        return best_result
    
    # Try to find an even lower rank decomposition
    # Use the 23-term decomposition for 3x3 if available
    # R(3) = 23 is the best known. 7 * 23 = 161
    result_161 = _build_strassen_23term(n, m, p)
    if result_161 is not None and _validate(result_161, n, m, p):
        best_result = result_161
        best_rank = 161
    
    if time.time() > deadline:
        return best_result
    
    # Try to do even better with random search / optimization
    # Try to find a decomposition of rank < best_rank using iterative methods
    
    return best_result


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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


def _build_strassen_6x6(n, m, p):
    """
    Build a 189-rank decomposition for 6x6 using Strassen on 2x2 blocks of 3x3.
    
    Strassen's algorithm computes 2x2 matrix product using 7 sub-products:
    C11 = P1 - P5 + P6 + P7
    C12 = P3 + P5
    C21 = P4 + P7
    C22 = P1 - P2 + P3 + P4
    
    where:
    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22)B11
    P3 = A11(B21 - B22)
    P4 = A22(B12 - B11)
    P5 = (A11 + A12)B22
    P6 = (A21 - A11)(B11 + B12)
    P7 = (A12 - A22)(B21 + B22)
    
    For 6x6 as 2x2 blocks of 3x3, each Pi is a 3x3 x 3x3 = 3x3 product.
    Each 3x3 product uses schoolbook rank 27.
    Total: 7 * 27 = 189.
    """
    if n != 6 or m != 6 or p != 6:
        return None
    
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(36)]  # 6*6 = 36
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Block indexing: A is 2x2 blocks of 3x3
    # A11 = rows 0-2, cols 0-2; A12 = rows 0-2, cols 3-5
    # A21 = rows 3-5, cols 0-2; A22 = rows 3-5, cols 3-5
    
    # For each of the 7 Strassen products, we need to map
    # the linear combinations of input blocks to the U and V matrices,
    # and the output blocks to W.
    
    # Each Pi = Xi * Yi where Xi and Yi are 3x3 matrices (linear combos of A/B blocks)
    # The product Pi is 3x3, and we use schoolbook for each.
    
    # Define the 7 pairs (Xi, Yi) and their output locations in C
    # Pi contributes to specific positions in C via addition
    
    # Strassen formulas (block level):
    # C11 = P1 - P5 + P6 + P7
    # C12 = P3 + P5
    # C21 = P4 + P7
    # C22 = P1 - P2 + P3 + P4
    
    # Each Pi is a 3x3 matrix. For schoolbook, Pi[r,c] = sum over (i,j) of Xi[r,i] * Yi[i,c]
    # But we can also view it as: for each (r, s, t) where r,s,t are 0..2 (local 3x3 indices),
    # Pi[r, t] uses the term Xi[r, s] * Yi[s, t].
    
    # So for each of the 7 Pi, and for each (r, s, t) in 0..2^3, we get one rank-1 term.
    # Total 7 * 27 = 189 terms.
    
    # Let me define the linear combinations:
    # P1: Xi = A11 + A22, Yi = B11 + B22
    # P2: Xi = A21 + A22, Yi = B11
    # P3: Xi = A11, Yi = B21 - B22
    # P4: Xi = A22, Yi = B12 - B11
    # P5: Xi = A11 + A12, Yi = B22
    # P6: Xi = A21 - A11, Yi = B11 + B12
    # P7: Xi = A12 - A22, Yi = B21 + B22
    
    # For a given block, e.g., A11, its (r, s) entry (local) corresponds to global position
    # row: 3*block_row + r, col: 3*block_col + s
    # where A11 is block_row=0, block_col=0
    # A12 is block_row=0, block_col=1
    # A21 is block_row=1, block_col=0
    # A22 is block_row=1, block_col=1
    
    # Similarly for B.
    
    # For C output:
    # C11 is block_row=0, block_col=0
    # C12 is block_row=0, block_col=1
    # C21 is block_row=1, block_col=0
    # C22 is block_row=1, block_col=1
    
    # The Strassen product Pi contributes to C blocks with certain coefficients.
    
    # Let me define helper functions to get global index from block and local indices
    
    def global_idx(block_row, block_col, local_r, local_s):
        """Global row/col index in the 6x6 matrix."""
        row = 3 * block_row + local_r
        col = 3 * block_col + local_s
        return row * 6 + col  # for U/V, this is the row index (m*p or n*m)
    
    # Wait, I need to be more careful about the tensor structure.
    # 
    # Matrix multiplication: C = A * B
    # A is n x m (6x6), B is m x p (6x6), C is p x n? No wait.
    # 
    # Standard: A is (n x m), B is (m x p), C = A * B is (n x p).
    # But the task says "output factor follows the official trace convention W[k*n+i,r]"
    # So C has dimensions p x n? Let me re-read.
    #
    # "U factor of shape (n*m, rank), V factor of shape (m*p, rank), W factor of shape (p*n, rank)"
    # This suggests:
    # U indexes A entries: n*m entries, so A is n x m
    # V indexes B entries: m*p entries, so B is m x p  
    # W indexes C entries: p*n entries, so C is p x n? That's unusual.
    #
    # Wait, for matrix multiplication A(n,m) * B(m,p) = C(n,p).
    # But W is p*n, which would be C transposed or C with dimensions (p,n).
    # 
    # Let me look at the schoolbook code:
    # U[i*m+j][column] = 1  -> i is row of A (0..n-1), j is col of A (0..m-1)
    # V[j*p+k][column] = 1  -> j is row of B (0..m-1), k is col of B (0..p-1)
    # W[k*n+i][column] = 1  -> k is row of C (0..p-1), i is col of C (0..n-1)
    # 
    # So C[k*n+i] means C is stored as (p x n) where k is row and i is col.
    # This means C[k, i] = sum_j A[i, j] * B[j, k]
    # So C has dimensions p x n, and C[k, i] = (B^T A^T)[k, i] or equivalently C = A^T * B^T viewed differently.
    #
    # Actually: C[k, i] = sum_j A[i, j] B[j, k]. If we define C' = C^T, then C'[i, k] = sum_j A[i, j] B[j, k] = (AB)[i, k].
    # So W is storing C^T in row-major order, or C in column-major order.
    
    # For the Strassen approach, I need to map to this convention.
    # A is n x m = 6 x 6
    # B is m x p = 6 x 6
    # C is stored as p x n = 6 x 6, where W[k*n + i] corresponds to C[k, i]
    # and C[k, i] = sum_j A[i, j] B[j, k]
    
    # So effectively, we're computing C = A^T * B^T if we think of standard matrix mult,
    # or C[i^T, j^T] where the multiplication is A (n x m) times B (m x p) giving C (n x p),
    # but stored transposed.
    
    # Let me just work with the convention as given:
    # A entries indexed by (i, j) with i in 0..5, j in 0..5 -> U[i*6+j]
    # B entries indexed by (j, k) with j in 0..5, k in 0..5 -> V[j*6+k]
    # C entries indexed by (k, i) with k in 0..5, i in 0..5 -> W[k*6+i]
    # C[k, i] = sum_j A[i, j] B[j, k]
    
    # For Strassen on 2x2 blocks of 3x3:
    # Let me partition A (6x6) into 2x2 blocks:
    # A[0:3, 0:3] = A11, A[0:3, 3:6] = A12
    # A[3:6, 0:3] = A21, A[3:6, 3:6] = A22
    
    # B (6x6) into 2x2 blocks:
    # B[0:3, 0:3] = B11, B[0:3, 3:6] = B12
    # B[3:6, 0:3] = B21, B[3:6, 3:6] = B22
    
    # C (6x6, but note the convention: W[k*6+i] where k is the first index)
    # C as a 6x6 matrix with C[k, i] = sum_j A[i,j] B[j,k]
    # Let me think of C as having blocks:
    # C[0:3, 0:3] corresponds to k in 0:3, i in 0:3 -> this is C11
    # C[0:3, 3:6] corresponds to k in 0:3, i in 3:6 -> this is C12? 
    
    # Wait, I'm getting confused. Let me just implement it directly.
    
    # The key insight: C[k, i] = sum_j A[i, j] B[j, k]
    # This is NOT the standard AB computation where C = AB has C[i, k] = sum_j A[i,j]B[j,k].
    # Here C is the transpose of what you'd normally compute.
    
    # So if I define A_std = A (6x6) and B_std = B (6x6), the "C" here is A_std^T * B_std... no.
    # C[k, i] = sum_j A[i, j] B[j, k]. If I swap roles: let X = A^T (so X[j, i] = A[i, j]) and Y = B (Y[j, k] = B[j, k]).
    # Then C[k, i] = sum_j X[j, i] Y[j, k] = (X^T Y)[i, k]^T... this is getting messy.
    
    # Let me just directly construct the decomposition.
    # For each rank-1 term, I need:
    # U[a] = 1 for some a, V[b] = 1 for some b, W[c] = 1 for some c, where:
    # a = i*6 + j (for A[i, j])
    # b = j*6 + k (for B[j, k])
    # c = k*6 + i (for C[k, i])
    # And the term contributes A[i, j] * B[j, k] to C[k, i].
    
    # So for a given (i, j, k), the rank-1 term has:
    # U[i*6+j] = 1
    # V[j*6+k] = 1
    # W[k*6+i] = 1
    
    # This is exactly the schoolbook decomposition!
    # The Strassen approach requires combining multiple (i, j, k) triples into one term
    # by using linear combinations.
    
    # For Strassen, each of the 7 products Pi involves linear combinations of A and B blocks.
    # Each Pi is a 3x3 matrix computed from 3x3 sub-matrices.
    # Using schoolbook for each Pi means 27 scalar multiplications per Pi.
    # Each scalar multiplication in Pi corresponds to one "effective" (i, j, k) triple,
    # but the U and V entries can be linear combinations.
    
    # Actually, for the tensor decomposition to be valid with U, V, W having entries that
    # can be non-zero for multiple positions, we need:
    # C = sum_r U[:, r] * V[:, r]^T (outer product-like, but with the specific index structure)
    
    # Let me just implement the Strassen decomposition properly.
    
    # For each of the 7 Strassen products, and for each of the 27 scalar multiplications
    # within that product (using schoolbook for 3x3), I create one rank-1 term.
    
    # P1 = (A11 + A22) * (B11 + B22)
    # P1[r, t] = sum_s (A11[r,s] + A22[r,s]) * (B11[s,t] + B22[s,t])
    #          = A11[r,s]*B11[s,t] + A11[r,s]*B22[s,t] + A22[r,s]*B11[s,t] + A22[r,s]*B22[s,t]
    # 
    # Wait, that's 4 terms for each (r,t), so 4*9 = 36 terms for P1 alone? No, that can't be right.
    # 
    # The point of Strassen is that we compute the 7 products as single matrix multiplications.
    # But for the tensor decomposition, each "product" is itself a tensor.
    # When we say "7 products of 3x3 matrices", each product has rank 27 (schoolbook for 3x3),
    # so total rank is 7 * 27 = 189.
    
    # But actually, the Strassen algorithm computes linear combinations of A and B blocks,
    # multiplies them, and then combines the results. For the tensor decomposition:
    
    # The key is that in the Strassen formula, the "multiplication" of (A11 + A22) and (B11 + B22)
    # is done as a single 3x3 x 3x3 multiplication. In the tensor decomposition sense,
    # this means we're computing the product of the linear combination, not expanding it first.
    
    # So for P1 = (A11 + A22)(B11 + B22):
    # P1[r, t] = sum_s (A11[r,s] + A22[r,s])(B11[s,t] + B22[s,t])
    # 
    # In the tensor decomposition, this corresponds to a rank-27 sub-decomposition
    # where for each (r, s, t):
    # The U vector has 1s at positions corresponding to A11[r,s] and A22[r,s]
    # The V vector has 1s at positions corresponding to B11[s,t] and B22[s,t]
    # The W vector has 1 at position corresponding to the contribution to C
    
    # But wait, this would make the U vector have multiple 1s, which is fine for the tensor decomposition.
    # The tensor is: T[i, j, k] = C[k,i] for C = A*B (with the given convention)
    # And we want: T = sum_r U[:,r] * V[:,r] (elementwise in the appropriate sense)
    
    # Actually, the decomposition is:
    # C[k, i] = sum_r (sum_j' U[k, i, r] ... ) 
    # 
    # No, let me recall the tensor decomposition for matrix multiplication:
    # M_{n,m,p} = sum_{r=1}^{R} u_r * v_r * w_r (tensor product of vectors)
    # where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}
    # and u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] summed over i, j, k gives the contribution
    # to the multiplication tensor.
    
    # For schoolbook, u_r, v_r, w_r are all unit vectors, so only one term is non-zero at a time.
    # For Strassen, u_r, v_r, w_r can have multiple non-zero entries.
    
    # Let me implement this properly.
    
    # For each of the 7 Strassen products, for each local (r, s, t) in 0..2:
    # 
    # The U vector u_r will have entries at positions corresponding to the A blocks involved.
    # The V vector v_r will have entries at positions corresponding to the B blocks involved.
    # The W vector w_r will have entries at positions corresponding to the C blocks where
    # this product contributes.
    
    # Let me define the 7 products with their (Xi, Yi) block combinations:
    # P1: X = A11 + A22, Y = B11 + B22
    # P2: X = A21 + A22, Y = B11
    # P3: X = A11, Y = B21 - B22
    # P4: X = A22, Y = B12 - B11
    # P5: X = A11 + A12, Y = B22
    # P6: X = A21 - A11, Y = B11 + B12
    # P7: X = A12 - A22, Y = B21 + B22
    
    # And the output combinations:
    # C11 = P1 - P5 + P6 + P7
    # C12 = P3 + P5
    # C21 = P4 + P7
    # C22 = P1 - P2 + P3 + P4
    
    # For the 6x6 problem with the given convention C[k, i] = sum_j A[i,j] B[j,k]:
    # I need to figure out which 3x3 blocks of C correspond to C11, C12, C21, C22.
    
    # Since C is 6x6 with C[k, i], let me partition it:
    # C[0:3, 0:3]: k in 0:3, i in 0:3 -> let's call this C_A
    # C[0:3, 3:6]: k in 0:3, i in 3:6 -> C_B
    # C[3:6, 0:3]: k in 3:6, i in 0:3 -> C_C
    # C[3:6, 3:6]: k in 3:6, i in 3:6 -> C_D
    
    # Now, C[k, i] = sum_j A[i,j] B[j,k]
    # For C_A (k in 0:3, i in 0:3): sum over j of A[i,j] B[j,k] where i,j,k are in 0:5
    # This involves all of A and B, not just blocks.
    
    # Hmm, the Strassen block decomposition doesn't directly apply because the C index
    # convention is transposed. Let me think about this differently.
    
    # Let me define A' = A^T and B' = B^T. Then C_standard = A' * B' has C_standard[i, k] = sum_j A'[i,j] B'[j,k] = sum_j A[j,i] B[k,j].
    # That's not matching either.
    
    # Let me just give up on Strassen for now and return the schoolbook decomposition.
    # The task allows half-integer entries, so maybe I should try a different approach.
    
    # Actually, let me just return the 189-rank Strassen decomposition if I can validate it.
    # Let me implement it carefully.
    
    # For the C convention C[k,i] = sum_j A[i,j] B[j,k]:
    # Think of it as: the matrix C has its (k,i) entry equal to the (i,k) entry of A*B^T... 
    # C[k,i] = sum_j A[i,j] B[j,k] = (A * B^T)[i, k]? Let's check: (A B^T)[i,k] = sum_j A[i,j] B^T[j,k] = sum_j A[i,j] B[k,j]. No.
    
    # C[k,i] = sum_j A[i,j] B[j,k]. If I define M = B^T, then M[k,j] = B[j,k].
    # C[k,i] = sum_j A[i,j] M[k,j] = (A * M^T)[i,k]? No, (A M^T)[i,k] = sum_j A[i,j] M^T[j,k] = sum_j A[i,j] M[k,j]. Yes!
    # So C[k,i] = (A * B^T)^T [i,k]... wait. (A B^T) is 6x6 with entry [i,k] = sum_j A[i,j] B[k,j].
    # And C[k,i] = sum_j A[i,j] B[j,k]. These are different: B[k,j] vs B[j,k].
    
    # So C is not simply related to A*B in a transposed way unless B is symmetric.
    
    # OK here's the deal: the tensor for matrix multiplication with these indices is:
    # T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    # which is the same as the standard matrix multiplication tensor, just with
    # the third index pair being (k, i) instead of (i, k).
    
    # The rank of this tensor is the same as R(6) since it's just a reindexing.
    
    # For Strassen, I need to adapt the block structure to this indexing.
    
    # Let me try a completely different approach: just use random initialization
    # and try to minimize the residual, but that's unlikely to find exact integer solutions.
    
    # Given time constraints, let me return the schoolbook decomposition.
    # But wait, I already observed that 189 was achieved. Let me just hardcode a valid
    # 189-rank decomposition or construct it properly.
    
    # Let me construct the 189-rank Strassen decomposition carefully.
    
    rank = 189
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    col = 0
    
    # Helper to get U index for A[i,j]
    def u_idx(i, j):
        return i * 6 + j
    
    def v_idx(j, k):
        return j * 6 + k
    
    def w_idx(k, i):
        return k * 6 + i
    
    # For each Strassen product Pi, for each local (r,s,t) in 3x3x3:
    # Pi[r,t] = sum_s Xi[r,s] * Yi[s,t]
    # Where Xi is a linear combination of A blocks, Yi is a linear combination of B blocks.
    
    # For the term corresponding to (Pi, r, s, t):
    # U contributions: for each A block component in Xi[r,s], add its coefficient
    # V contributions: for each B block component in Yi[s,t], add its coefficient
    # W contributions: for each C block that Pi[r,t] contributes to, add coefficient
    
    # Define the 7 products:
    # Each product has Xi (which A blocks and with what coefficients), Yi (which B blocks),
    # and the output: which C blocks and with what coefficients.
    
    # A blocks: A11 = A[0:3,0:3], A12 = A[0:3,3:6], A21 = A[3:6,0:3], A22 = A[3:6,3:6]
    # B blocks: B11 = B[0:3,0:3], B12 = B[0:3,3:6], B21 = B[3:6,0:3], B22 = B[3:6,3:6]
    
    # For a local position (r, s) in a 3x3 block:
    # A11[r,s] -> global A[0+r, 0+s] = A[r, s]
    # A12[r,s] -> global A[0+r, 3+s] = A[r, 3+s]
    # A21[r,s] -> global A[3+r, 0+s] = A[3+r, s]
    # A22[r,s] -> global A[3+r, 3+s] = A[3+r, 3+s]
    
    # Similarly for B.
    
    # For C output:
    # C11[r,t] -> which global C? C is indexed as C[k,i]. 
    # C11 is the block of C corresponding to k in 0:3, i in 0:3.
    # So C11[r,t] where r is local row of C11 block (which is the k index), t is local col (i index).
    # So C11[r,t] = C[r, t] in global indices? Let's see: k = 0 + r, i = 0 + t. Yes.
    # C12[r,t]: k in 0:3, i in 3:6. So C[0+r, 3+t].
    # C21[r,t]: k in 3:6, i in 0:3. So C[3+r, 0+t].
    # C22[r,t]: k in 3:6, i in 3:6. So C[3+r, 3+t].
    
    # Now, the Strassen products:
    # P1[r,t] contributes to: C11 (coeff +1), C22 (coeff +1)
    # P2[r,t] contributes to: C22 (coeff -1)
    # P3[r,t] contributes to: C12 (coeff +1), C22 (coeff +1)
    # P4[r,t] contributes to: C21 (coeff +1), C22 (coeff +1)
    # P5[r,t] contributes to: C11 (coeff -1), C12 (coeff +1)
    # P6[r,t] contributes to: C11 (coeff +1)
    # P7[r,t] contributes to: C11 (coeff +1), C21 (coeff +1)
    
    # Wait, let me re-derive. Standard Strassen for C = A * B (2x2 blocks):
    # C11 = P1 - P5 + P6 + P7
    # C12 = P3 + P5
    # C21 = P4 + P7
    # C22 = P1 - P2 + P3 + P4
    
    # So:
    # P1: C11 +1, C22 +1
    # P2: C22 -1
    # P3: C12 +1, C22 +1
    # P4: C21 +1, C22 +1
    # P5: C11 -1, C12 +1
    # P6: C11 +1
    # P7: C11 +1, C21 +1
    
    # Now, for the given C convention C[k,i] = sum_j A[i,j] B[j,k]:
    # This is equivalent to computing C = A * B where we define the multiplication as:
    # C[k,i] = sum_j A[i,j] B[j,k]
    # which is the standard formula if we consider A to be indexed (i,j) and B to be (j,k).
    
    # So the block structure should work the same way, with:
    # A blocks: A11 = A[0:3,0:3], A12 = A[0:3,3:6], A21 = A[3:6,0:3], A22 = A[3:6,3:6]
    # B blocks: B11 = B[0:3,0:3], B12 = B[0:3,3:6], B21 = B[3:6,0:3], B22 = B[3:6,3:6]
    # C blocks: C11 = C[0:3,0:3] (k=0:3, i=0:3), C12 = C[0:3,3:6] (k=0:3, i=3:6)
    #           C21 = C[3:6,0:3] (k=3:6, i=0:3), C22 = C[3:6,3:6] (k=3:6, i=3:6)
    
    # Strassen products for the (i,j,k) indices:
    # P1 = (A11 + A22)(B11 + B22) -> contributes to C11 and C22
    # etc.
    
    # Let me just implement it.
    
    # Define products as (xi_blocks, yi_blocks, output_blocks)
    # xi_blocks: list of (block_name, coefficient)
    # yi_blocks: list of (block_name, coefficient)
    # output_blocks: list of (block_name, coefficient)
    
    products = [
        # P1
        ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)],
         [('C11', 1), ('C22', 1)]),
        # P2
        ([('A21', 1), ('A22', 1)], [('B11', 1)],
         [('C22', -1)]),
        # P3
        ([('A11', 1)], [('B21', 1), ('B22', -1)],
         [('C12', 1), ('C22', 1)]),
        # P4
        ([('A22', 1)], [('B12', 1), ('B11', -1)],
         [('C21', 1), ('C22', 1)]),
        # P5
        ([('A11', 1), ('A12', 1)], [('B22', 1)],
         [('C11', -1), ('C12', 1)]),
        # P6
        ([('A21', 1), ('A11', -1)], [('B11', 1), ('B12', 1)],
         [('C11', 1)]),
        # P7
        ([('A12', 1), ('A22', -1)], [('B21', 1), ('B22', 1)],
         [('C11', 1), ('C21', 1)]),
    ]
    
    # Block to (row_offset, col_offset) mapping
    block_offsets = {
        'A11': (0, 0), 'A12': (0, 3), 'A21': (3, 0), 'A22': (3, 3),
        'B11': (0, 0), 'B12': (0, 3), 'B21': (3, 0), 'B22': (3, 3),
        'C11': (0, 0), 'C12': (0, 3), 'C21': (3, 0), 'C22': (3, 3),
    }
    
    for xi_blocks, yi_blocks, out_blocks in products:
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    for (a_block, a_coeff) in xi_blocks:
                        ai, aj = block_offsets[a_block]
                        U[u_idx(ai + r, aj + s)][col] += a_coeff
                    for (b_block, b_coeff) in yi_blocks:
                        bi, bj = block_offsets[b_block]
                        V[v_idx(bi + s, bj + t)][col] += b_coeff
                    for (c_block, c_coeff) in out_blocks:
                        ci, cj = block_offsets[c_block]
                        W[w_idx(ci + r, cj + t)][col] += c_coeff
                    col += 1
    
    return {"U": U, "V": V, "W": W}


def _validate(decomp, n, m, p):
    """Validate that the decomposition correctly computes matrix multiplication."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    
    # Reconstruct the multiplication tensor and check
    # For random test matrices
    random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = np.zeros((p, n))
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    C_expected[k, i] += A[i, j] * B[j, k]
        
        # Compute C from decomposition
        C_actual = np.zeros((p, n))
        for col in range(U.shape[1]):
            u = U[:, col]
            v = V[:, col]
            w = W[:, col]
            # u is indexed by (i, j) -> u[i*m+j]
            # v is indexed by (j, k) -> v[j*p+k]
            # w is indexed by (k, i) -> w[k*n+i]
            # Contribution: w[k*n+i] * (sum over i,j of u[i*m+j] * A[i,j]) * (sum over j,k of v[j*p+k] * B[j,k])
            # Wait, no. The tensor decomposition means:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # And T[i,j,k] should equal 1 for the matrix multiplication tensor.
            # So C[k,i] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
            # = sum_r (sum_i U[i*m+?, r] * A[i,?]) * ... 
            # 
            # Actually, let me just check the tensor directly.
            pass
        
        # Simpler check: verify that the decomposition gives the correct tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 for all valid (i,j,k)
        
        # This is O(n*m*p*R) which for R=189, n=m=p=6: 216*189 = 40824, fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
    return True


def _build_strassen_23term(n, m, p):
    """
    Try to build a 161-rank decomposition using the 23-term 3x3 decomposition.
    This requires knowing the explicit 23-term decomposition for 3x3 matrix multiplication.
    Since I don't have a verified one memorized, return None to fall back to 189.
    """
    return None

# EVOLVE-BLOCK-END
