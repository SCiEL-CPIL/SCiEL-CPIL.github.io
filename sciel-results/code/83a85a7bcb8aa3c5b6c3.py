"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # The schoolbook rank is n*m*p = 3*3*5 = 45.
    # We attempt to find a lower rank decomposition.
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 23 (Landsberg, etc.)
    # - 3x3x4: rank 20? (Not sure, but let's check)
    # - 3x3x5: ?
    
    # Let's try to construct a decomposition.
    # The tensor is T_{i,j,k} = delta_{i,j} delta_{j,k} ? No, that's not right.
    # Matrix multiplication: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T_{i,j,k} = 1 if (i,j,k) is a valid multiplication term, i.e., always 1?
    # No, the tensor for matrix multiplication is defined as:
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to (AB)_{i,k}.
    # Actually, the standard definition is that the multiplication tensor mu_{n,m,p} has entries:
    # mu_{i,j,k} = 1 if (i,j,k) corresponds to a valid index combination.
    # For C = A B, C_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}.
    # The tensor is a 3-tensor of shape (n, m, p) where the entry at (i,j,k) is 1 if
    # the product A_{i,j} B_{j,k} is part of the sum for C_{i,k}.
    # Wait, the indices are (i, j, k) where i is row of A (and C), j is column of A (and row of B),
    # and k is column of B (and C).
    # So the tensor T has shape (n, m, p) and T[i,j,k] = 1 for all valid i,j,k.
    # No, that's not right either. The tensor is the structure tensor of the algebra.
    # For matrix multiplication, the structure constants are:
    # mu_{(i,j), (j,k), (i,k)} = 1.
    # In terms of a 3-tensor T of shape (n*m, m*p, p*n), T is sparse.
    # But the problem asks for U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # And the decomposition is: sum_r U[:,r] V[:,r]^T W[:,r]^T = T (as a 3-tensor).
    # Or rather, the bilinear form is: sum_r (U^T x) (V^T y) (W^T z) = T(x,y,z).
    # Here x is a vector of length n*m (flattened A), y is length m*p (flattened B),
    # and z is length p*n (flattened C^T or something).
    # The task says: "The output factor follows the official trace convention W[k*n+i, r]."
    # This suggests that the third factor W corresponds to the output matrix C,
    # but with indices ordered as k*n + i, where k is the column and i is the row.
    # So W is of shape (p*n, rank).
    
    # Let's verify the schoolbook solution:
    # rank = n*m*p = 45.
    # For each (i,j,k), we have a term A_{i,j} B_{j,k} C_{i,k}.
    # In the flattened vectors:
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    # So U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r = i*m*p + j*p + k.
    # This gives a valid decomposition of rank 45.
    
    # Can we do better?
    # For 2x2x2, Strassen's algorithm gives rank 7 instead of 8.
    # For 3x3x3, the best known rank is 23 (or 22? no, 23 is the current record).
    # For 3x3x4, the best known rank is... let's think.
    # The rank of mu_{n,m,p} is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15.
    # Also, the rank is at least the border rank, which is at least the max of the slice ranks.
    # For 3x3x5, the lower bound is 15.
    
    # Let's try to use a Strassen-like approach.
    # Strassen's 2x2x2 algorithm:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5-m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # This can be generalized. However, finding an optimal decomposition for 3x3x5
    # is a hard open problem in algebraic complexity theory.
    
    # Given the constraints of the problem (120s time limit, standard library + NumPy/SciPy),
    # it is unlikely that we can discover a new optimal decomposition from scratch.
    # However, we can try to use known decompositions for smaller cases and extend them.
    
    # Let's consider the 3x3x3 case. The best known rank is 23.
    # If we have a rank-23 decomposition for 3x3x3, can we extend it to 3x3x5?
    # The 3x3x5 problem can be seen as two 3x3x3 problems plus a 3x3x2 problem,
    # but that's not quite right.
    
    # Alternatively, we can try to use a heuristic search for a low-rank decomposition.
    # But that's computationally intensive.
    
    # Let's try a different approach: use the fact that the rank of mu_{n,m,p} is
    # at most the rank of mu_{n,m,p-1} + n*m.
    # This is because we can decompose the p-th column of B using the schoolbook method,
    # and use the existing decomposition for the first p-1 columns.
    # So rank(mu_{3,3,5}) <= rank(mu_{3,3,4}) + 9.
    # And rank(mu_{3,3,4}) <= rank(mu_{3,3,3}) + 9 = 23 + 9 = 32.
    # So rank(mu_{3,3,5}) <= 32 + 9 = 41.
    # This is better than 45, but still not optimal.
    
    # Actually, the bound is rank(mu_{n,m,p}) <= rank(mu_{n,m,p-1}) + n*m.
    # So:
    # rank(mu_{3,3,2}) <= rank(mu_{3,3,1}) + 9 = 3 + 9 = 12. (Schoolbook is 9, so this bound is loose.)
    # Wait, rank(mu_{3,3,1}) = 3 (since it's just the identity matrix).
    # rank(mu_{3,3,2}) <= 3 + 9 = 12. Schoolbook is 9.
    # rank(mu_{3,3,3}) <= 12 + 9 = 21. But the best known is 23, so this bound is not tight.
    
    # Let's try to construct a decomposition of rank 41 or lower.
    # But without a known optimal decomposition, this is difficult.
    
    # Given the time constraints, let's return the schoolbook decomposition of rank 45.
    # This is a valid solution, even if not optimal.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
