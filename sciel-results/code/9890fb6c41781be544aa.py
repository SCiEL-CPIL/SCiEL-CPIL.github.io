"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Use Strassen-like decomposition for the 3x3 core
    # Split: A is 2x3, B is 3x5
    # We can decompose into blocks:
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x5, B2 is 2x5
    # C = A1*B1 + A2*B2
    # For A2*B2 (2x2 * 2x2), use Strassen rank 7
    # For A1*B1 (2x1 * 1x5), use schoolbook rank 2
    
    # Total rank = 2 (for A1*B1) + 7 (for A2*B2) = 9? No, that's not right.
    # Let me think more carefully.
    
    # Actually, let's use a different block structure:
    # Split A (2x3) into [A_top | A_bottom] where A_top is 1x3, A_bottom is 1x3
    # Split B (3x5) into [B_top; B_mid; B_bot] where each is 1x5
    # This doesn't help directly.
    
    # Better approach: Use the fact that we can decompose 3x3 multiplication with rank 7 (Strassen)
    # and extend to 2x3 * 3x5
    
    # Strategy: Pad A to 3x3 (add a zero row), pad B to 3x5 (keep as is)
    # Then 3x3 * 3x5 can be decomposed using Strassen on the 3x3 part
    
    # Actually, the cleanest approach:
    # A (2x3) * B (3x5) = C (2x5)
    # Think of this as: for each row i of A (i=0,1) and each column k of B (k=0..4):
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use a rank-7 decomposition for the 3x3 matrix multiplication core,
    # then extend to handle the rectangular case.
    
    # Let's use a direct approach: decompose into 3 groups of p=5 columns
    # Group 1: j=0, Group 2: j=1, Group 3: j=2
    # For each group, we have A[:,j] (2x1) * B[j,:] (1x5) = outer product (2x5)
    # Each outer product has rank 1, so 3 groups give rank 3? No, that's not how tensor rank works.
    
    # Let me just implement a known low-rank decomposition.
    # For 2x3 * 3x5, the schoolbook rank is 2*3*5 = 30.
    # We can potentially do better by sharing factors.
    
    # Key insight: We can group the inner dimension j.
    # C = sum_j A[:,j] * B[j,:]
    # Each term A[:,j] * B[j,:] is a rank-1 2x5 matrix.
    # But in tensor decomposition, we need to express the multiplication tensor.
    
    # The multiplication tensor T has indices (i, j, k) where:
    # T[i, j, k] = 1 if we multiply A[i,j] by B[j,k]
    # Wait, no. The tensor for matrix multiplication M_{n,m,p} has indices (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p), and
    # T[i, j, k] contributes to C[i, k] via A[i, j] * B[j, k].
    
    # The standard tensor decomposition:
    # C[i,k] = sum_r U[i*m + something, r] * V[... , r] * W[..., r]
    
    # Actually, the standard form is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form with U, V, W of shape (n*m, r), (m*p, r), (p*n, r):
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]  
    # W[k*n + i, r] corresponds to C[i,k] (note: k*n + i, not i*n + k)
    
    # So the decomposition is:
    # C[i,k] = sum_r U[i*m + j_r, r] * V[j_r * p + k_r, r] * W[k_r * n + i_r, r]
    # where for each r, we have fixed indices (i_r, j_r, k_r) and
    # U[i*m + j, r] = 1 if (i,j) = (i_r, j_r) else 0
    # V[j*p + k, r] = 1 if (j,k) = (j_r, k_r) else 0
    # W[k*n + i, r] = 1 if (k,i) = (k_r, i_r) else 0
    
    # This is the schoolbook decomposition with rank n*m*p.
    
    # To reduce rank, we need to find a decomposition where each term r
    # can contribute to multiple (i,j,k) triples.
    
    # For a general low-rank decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,j'} * delta_{k,k'} ... 
    # No, that's not right either.
    
    # The correct condition is:
    # For all i, k: C[i,k] = sum_j A[i,j] * B[j,k]
    # This means the tensor T satisfies:
    # T[i, j, k] = 1 for all valid (i,j,k), and the multiplication is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    # In the U,V,W decomposition:
    # C[i,k] = sum_r U[i*m + j_i_r, r] * V[j_v_r * p + k_v_r, r] * W[k_w_r * n + i_w_r, r]
    # But this only works if for each r, U is nonzero at exactly one (i,j) position,
    # V at exactly one (j,k) position, and W at exactly one (k,i) position,
    # AND these are consistent (same j, same k, same i).
    
    # That's the schoolbook. To get lower rank, we need U, V, W to have
    # multiple nonzero entries per column, such that the sum still gives the correct result.
    
    # This is a non-trivial optimization problem. Let me try a heuristic approach.
    
    # For now, let's try to find a decomposition with rank < 30 using a search.
    # But that might be too slow. Let me try a structured approach.
    
    # Known result: rank(2x3*3x5) <= rank(3x3*3x3) + extra terms
    # Actually, 2x3*3x5 can be embedded in 3x3*3x5 by adding a zero row to A.
    # And 3x3*3x5 can be decomposed by splitting the 3x3 part.
    
    # Let me try: decompose 3x3 matrix multiplication with rank 7 (Strassen),
    # then extend to 3x3 * 3x5.
    
    # For 3x3 * 3x5:
    # Split B (3x5) into [B1; B2; B3] where each is 1x5.
    # Then C = A * B = sum_j A[:,j] * B[j,:]
    # This is 3 outer products, each of rank 1 in the output, but in tensor terms...
    
    # I think the cleanest approach is to just do a small search for a better decomposition.
    # Let me try rank 28, 27, etc.
    
    # Actually, let me just implement a simple improvement: use the fact that
    # some terms can be combined.
    
    # For a 2x3 * 3x5 multiplication, we have 30 schoolbook terms.
    # Can we combine any? 
    
    # Let me try a different strategy: use bilinear algorithms.
    # The Strassen algorithm for 2x2 gives rank 7 instead of 8.
    # For 2x3 * 3x5, we can potentially use a similar idea.
    
    # Split A (2x3) into [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split B (3x5) into [B1; B2] where B1 is 2x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 * 2x5. We can decompose this using Strassen on the 2x2 part.
    # A2*B2 is 2x1 * 1x5, which is just an outer product, rank 2 in tensor terms?
    # No, 2x1 * 1x5 has tensor rank 2*1*5 = 10 in schoolbook, but it's actually
    # a sum of 5 outer products of rank 1 vectors, so tensor rank might be lower.
    
    # This is getting complicated. Let me just try a greedy/local search approach.
    
    # For simplicity and reliability, let me return the schoolbook decomposition
    # but with a small optimization: check if we can reduce by 1 or 2 terms.
    
    # Actually, let me just implement the schoolbook and add a simple local search
    # to try to reduce the rank by replacing pairs of terms.
    
    # Given the time constraint, let me just return schoolbook rank 30 for now,
    # but structure the code so it's easy to improve.
    
    # Wait, the inspiration says we achieved rank 28. Let me try to replicate that.
    # The strategy was: "Exploits rectangular structure by decomposing into 2x2 Strassen blocks (rank 7) and schoolbook parts"
    
    # So: Split the 3 inner dimension into 2+1.
    # A (2x3) = [A_2x2 | A_2x1]
    # B (3x5) = [B_2x5; B_1x5]
    # C = A_2x2 * B_2x5 + A_2x1 * B_1x5
    
    # For A_2x2 * B_2x5: This is like multiplying a 2x2 matrix by a 2x5 matrix.
    # We can think of B_2x5 as [B_col1 | B_col2 | ... | B_col5] where each B_colj is 2x1.
    # Then A_2x2 * B_colj is a 2x1 vector.
    # Using Strassen on the 2x2 * 2x1 multiplication: rank 7 instead of 8? 
    # Actually, 2x2 * 2x1 has schoolbook rank 2*2*1 = 4. Strassen for 2x2*2x2 is rank 7,
    # but for 2x2*2x1, the schoolbook rank is already 4, and I'm not sure Strassen helps.
    
    # Let me reconsider. The Strassen decomposition for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B11)  # Wait, this is wrong
    # The standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2 * 2x5, we can apply Strassen to each column of B independently,
    # but that would give 7*5 = 35 terms, which is worse than schoolbook 2*2*5 = 20.
    
    # Alternatively, we can treat B_2x5 as a single matrix and find a rank-7 decomposition
    # for 2x2 * 2x5. But I don't know if such a decomposition exists.
    
    # Let me try a different approach: just do a local search to improve the schoolbook.
    
    # For now, return schoolbook to ensure correctness, then try to improve.
    
    rank_schoolbook = n * m * p
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Try to reduce rank by combining terms
    # For each pair of terms, check if they can be merged
    # This is a heuristic approach
    
    # Check time
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
    
    # Try local search: for each pair of columns, see if we can replace them
    # with a single column that still satisfies the decomposition
    # This is complex, so let's just return schoolbook for now.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
