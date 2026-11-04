import numpy as np
from collections import defaultdict

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> 36 x R
    # V: (m*p, rank) -> 36 x R
    # W: (p*n, rank) -> 36 x R (index k*n+i)
    
    # Strassen blocks 3x3
    # A blocks: 0:11, 1:12, 2:21, 3:22
    # B blocks: 0:11, 1:12, 2:21, 3:22
    # C blocks: 0:11, 1:12, 2:21, 3:22
    
    # Coefficients for A linear combinations in each of 7 steps
    # S_i = sum c_ab A_ab
    a_coeffs = [
        {0: 1, 3: 1},    # M1: A11 + A22
        {2: 1, 3: 1},    # M2: A21 + A22
        {0: 1},          # M3: A11
        {3: 1},          # M4: A22
        {0: 1, 1: 1},    # M5: A11 + A12
        {2: 1, 0: -1},   # M6: A21 - A11
        {1: 1, 3: -1},   # M7: A12 - A22
    ]
    
    # Coefficients for B linear combinations
    b_coeffs = [
        {0: 1, 3: 1},    # M1: B11 + B22
        {0: 1},          # M2: B11
        {1: 1, 2: -1},   # M3: B12 - B22
        {2: 1, 0: -1},   # M4: B21 - B11
        {3: 1},          # M5: B22
        {0: 1, 1: 1},    # M6: B11 + B12
        {2: 1, 3: 1},    # M7: B21 + B22
    ]
    
    # Coefficients for C linear combinations
    # C_ab = sum d_ci M_i
    c_coeffs = [
        {0: 1, 3: 1},    # M1: C11 + C22
        {2: -1, 3: 1},   # M2: -C21 + C22
        {1: 1, 3: 1},    # M3: C12 + C22
        {0: 1, 2: 1},    # M4: C11 + C21
        {0: -1, 1: 1},   # M5: -C11 + C12
        {3: 1},          # M6: C22
        {0: 1},          # M7: C11
    ]
    
    # Offsets for 3x3 blocks
    offsets = {0: (0, 0), 1: (0, 3), 2: (3, 0), 3: (3, 3)}
    
    # We will build U, V, W as lists of lists or numpy arrays.
    # First, determine the rank.
    # Each of 7 steps has a 3x3 matrix multiply.
    # Schoolbook 3x3 has 9 terms.
    # Total rank = 7 * 9 = 63.
    
    rank = 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Iterate through each Strassen step
    for step in range(7):
        # For each step, we have a 3x3 product S @ T.
        # S[u, v] = sum_{a_block} a_coeffs[step][a_block] * A[a_block][u, v]
        # T[v, w] = sum_{b_block} b_coeffs[step][b_block] * B[b_block][v, w]
        # Product P[u, w] = sum_v S[u, v] T[v, w]
        
        # We decompose P[u, w] into 9 terms: sum_v S[u, v] T[v, w]
        # Each term is S[u, v] * T[v, w].
        # This is a product of two linear combinations.
        # S[u, v] = sum_{a_block} alpha_{a_block} A_{a_block}[u, v]
        # T[v, w] = sum_{b_block} beta_{b_block} B_{b_block}[v, w]
        # Term = (sum alpha A) (sum beta B) = sum_{a_block, b_block} alpha beta A_{a_block}[u, v] B_{b_block}[v, w]
        
        # This term contributes to C blocks according to c_coeffs[step].
        # C[c_block][u, w] += c_coeff * Term
        
        # So for each u, v, w in 0..2:
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # This is one scalar multiplication in the 3x3 product.
                    # It corresponds to the term S[u, v] * T[v, w].
                    # We assign a rank index to this.
                    r_idx = step * 9 + u * 3 + v * 3 + w # Wait, 9 terms per step.
                    # u, v, w are 0..2.
                    # Index in 3x3x3: u*9 + v*3 + w?
                    # Let's just use a counter.
                    # Actually, we can just use the index directly.
                    
                    # Wait, the rank is 63.
                    # Step 0: 0..8
                    # Step 1: 9..17
                    # ...
                    # Step 6: 54..62
                    r_idx = step * 9 + u * 3 + v * 3 + w
                    # This mapping is not quite right. u, v, w determine the term.
                    # The 9 terms are for (u, v, w) where v is the summation index?
                    # No, in schoolbook 3x3, for each output (u, w), we sum over v.
                    # So there are 3*3*3 = 27 scalar multiplications?
                    # No. Schoolbook 3x3:
                    # C[u, w] = sum_{v=0}^2 A[u, v] B[v, w].
                    # Total multiplications = 3*3*3 = 27?
                    # No, rank of 3x3 matrix multiplication is 9?
                    # No, schoolbook rank is $n^3 = 27$ for $3 \times 3$.
                    # Strassen rank for $2 \times 2$ is 7.
                    # For $3 \times 3$, if we use Strassen on $2 \times 2$ blocks, it's not integer.
                    # The standard schoolbook rank for $n \times n$ is $n^3$.
                    # So for $3 \times 3$, rank is 27.
                    # Total rank = 7 * 27 = 189.
                    
                    # Let's re-verify the rank of 3x3 schoolbook.
                    # C = A B.
                    # C[i, j] = sum_k A[i, k] B[k, j].
                    # Number of terms = n * n * n = 27.
                    # So rank is 27.
                    
                    # So total rank = 7 * 27 = 189.
                    
                    pass
    
    # Recalculate rank
    rank = 7 * 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for step in range(7):
        for u in range(3):
            for w in range(3):
                for v in range(3):
                    r_idx = step * 27 + u * 9 + w * 3 + v
                    # Term: S[u, v] * T[v, w]
                    # S[u, v] = sum_{a_block} a_coeff[a_block] * A[a_block][u, v]
                    # T[v, w] = sum_{b_block} b_coeff[b_block] * B[b_block][v, w]
                    # Product = sum_{a_block, b_block} a_coeff[a_block] b_coeff[b_block] A[a_block][u, v] B[b_block][v, w]
                    
                    # This product contributes to C[c_block][u, w] with coeff c_coeff[c_block]
                    # So total coeff for A[a_block][u, v] B[b_block][v, w] is:
                    # sum_{c_block} c_coeff[c_block] * a_coeff[a_block] * b_coeff[b_block]
                    
                    # Wait, the rank-1 term in the decomposition is:
                    # U_{I, J, r} V_{J, K, r} W_{K, I, r}
                    # Here, the term is A[a_block][u, v] B[b_block][v, w].
                    # Global I = offset_a_row + u
                    # Global J = offset_a_col + v
                    # Global J' = offset_b_row + v
                    # Global K = offset_b_col + w
                    
                    # For this to be a valid term in the tensor identity, we need J = J'.
                    # J = offset_a_col + v.
                    # J' = offset_b_row + v.
                    # offset_a_col is 0 or 3. offset_b_row is 0 or 3.
                    # So we need offset_a_col = offset_b_row.
                    # i.e., a_block_col == b_block_row.
                    
                    # If a_block_col != b_block_row, this term is "invalid" (contributes 0 to identity).
                    # But it is part of the Strassen algorithm.
                    # The Strassen algorithm sums to the correct result.
                    # The "invalid" terms must cancel out or be zero in the final sum?
                    # No, they are not zero.
                    # This implies that the simple mapping of Strassen terms to rank-1 tensors
                    # does NOT work if we require J=J' for each rank-1 term.
                    
                    # However, the tensor identity allows U_{ij, r} and V_{jk, r} to be non-zero
                    # for different j's in different r's.
                    # But for a FIXED r, U_{ij, r} and V_{jk, r} must share the same j?
                    # No. The sum is over r.
                    # For a fixed i, j, k, the value is sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
                    # If for a specific r, U_{ij, r} is non-zero (so A_{ij} is involved) and V_{jk, r} is non-zero (so B_{jk} is involved),
                    # then the product U_{ij, r} V_{jk, r} is non-zero.
                    # This contributes to the sum for (i, j, k).
                    # If the term in the algorithm was A_{i, j_1} B_{j_2, k} with j_1 != j_2,
                    # then U_{i, j_1, r} != 0 and V_{j_2, k, r} != 0.
                    # For the coordinate (i, j, k), we sum U_{i, j, r} V_{j, k, r}.
                    # If j != j_1, U_{i, j, r} = 0.
                    # If j != j_2, V_{j, k, r} = 0.
                    # So if j_1 != j_2, there is no j such that both are non-zero.
                    # So this term contributes 0 to all (i, j, k).
                    
                    # This means "invalid" terms (j_1 != j_2) are useless for the tensor identity.
                    # So we must only include terms where j_1 = j_2.
                    # In Strassen, do the valid terms sum to 1?
                    # This is the key question.
                    
                    # If not, then Strassen's algorithm cannot be directly converted to a tensor decomposition
                    # of rank 63 (or 189) using this simple method.
                    
                    # However, it is known that the tensor rank of matrix multiplication is the minimum
                    # number of multiplications in a bilinear algorithm.
                    # Strassen is a bilinear algorithm.
                    # So there MUST be a tensor decomposition of rank 63.
                    
                    # The catch: The linear combinations in Strassen are not just sums of A_{ij}.
                    # They are sums of BLOCKS.
                    # But a block is a set of A_{ij}.
                    # So it is a linear combination.
                    
                    # Perhaps the "invalid" terms cancel out with other "invalid" terms from other steps?
                    # No, they contribute 0 to the tensor identity.
                    
                    # I suspect the tensor decomposition of Strassen's algorithm involves
                    # a different basis for the tensor modes.
                    # But the task requires the standard basis.
                    
                    # Let's try the schoolbook decomposition for 6x6 first to ensure correctness.
                    # Rank 216.
                    
                    pass

    # Let's just implement Schoolbook 6x6 to get a valid score.
    # Then we can try to optimize.
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
