"""Tensor decomposition for 3x3x7 matrix multiplication using rank-revealing QR."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # The matrix multiplication tensor T_{i,j,k} where:
    # i in [0, n), j in [0, m), k in [0, p)
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # Tensor index: (i*m + j, j*p + k, k*n + i)
    
    # Build the full tensor as a 3D array, then flatten to a matrix
    # We want to find a rank-r decomposition:
    # T = sum_{r=1}^{R} u_r (outer) v_r (outer) w_r
    # where u_r is nm-dim, v_r is mp-dim, w_r is pn-dim
    
    # The trivial decomposition has rank nm*mp*pn / ... no wait.
    # The trivial (Kronecker) decomposition has rank n*m*p = 63.
    # Each term picks one (i,j,k) and contributes A_{i,j} * B_{j,k} to C_{i,k}.
    # So rank = n*m*p = 3*3*7 = 63.
    
    # Let's try to find a lower rank decomposition.
    # 
    # Approach: Flattened tensor matrix
    # T_{(i*m+j), (j*p+k), (k*n+i)} = 1 for the multiplication tensor
    # 
    # We can think of this as a bilinear map. Let's use the "mode-1 flattening" 
    # and try to find the minimum rank.
    #
    # Actually, let's think about this differently. The matrix multiplication 
    # tensor T ∈ R^{nm} ⊗ R^{mp} ⊗ R^{pn} where T_{a,b,c} = 1 if the indices
    # are consistent with multiplication, 0 otherwise.
    #
    # a = i*m + j, b = j*p + k, c = k*n + i
    # 
    # The tensor is non-zero when:
    # a % m == b // p (the j index matches)
    # and b % p == c // n (the k index matches)  
    # and c % n == a // m (the i index matches)
    
    # Let's build the tensor and use SVD to find a low-rank approximation,
    # but we need EXACT decomposition with integer/half-integer entries.
    
    # Strategy: Try rank = pn = 21 (the dimension of the output space)
    # This corresponds to expressing each output element as a linear combination
    # of the input products.
    
    # Actually, let me think about this more carefully.
    # C_{i,k} = sum_{j=0}^{m-1} A_{i,j} * B_{j,k}
    # 
    # For each (i,k) pair, C_{i,k} is a linear combination of the m products
    # A_{i,j}*B_{j,k} for j=0,...,m-1.
    # 
    # The total number of distinct products A_{i,j}*B_{j,k} is n*m*p = 63,
    # but they share indices. Specifically, for fixed i and k, we have m products.
    # 
    # The key insight: can we combine products from different (i,k) pairs?
    
    # Let's try a systematic approach:
    # 1. Build the full tensor
    # 2. Try to find exact low-rank decomposition using integer entries
    
    # Build tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find minimal rank exact decomposition.
    # 
    # For a tensor, the rank is the minimum R such that
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # 
    # We can try ranks from lower bounds up to 63.
    # Lower bound: max(nm, mp, pn) = max(9, 21, 21) = 21
    # Also, the "border rank" lower bound can be higher.
    
    # Let's try a greedy approach: use the structure of matrix multiplication.
    # 
    # Key observation: C_{i,k} depends only on A_{i,0..m-1} and B_{0..m-1,k}
    # So for each fixed i, the outputs C_{i,0}, C_{i,1}, ..., C_{i,p-1} 
    # depend on the row A_{i,*} and the column B_{*,k}.
    
    # Let's try to use the fact that for 3x3x7, we can potentially share
    # some computation between the 3 rows of A.
    
    # Alternative approach: direct search for low-rank exact decomposition
    # using the structure.
    
    # For now, let's construct the trivial rank-63 decomposition but try
    # to see if we can do better by exploiting symmetries.
    
    # Actually, let me try a different approach: 
    # Use the "Strassen-like" approach where we look for a decomposition
    # that shares multiplications.
    
    # For 3x3x7, the known lower bound for rank is... let me think.
    # For 2x2x2, rank is 7 (Strassen).
    # For 3x3x3, rank is at most 19 (Laderman) or maybe lower.
    # For 3x3x7, it's less well-studied.
    
    # Let me try to find an exact decomposition with rank < 63.
    
    # Approach: parameterize the decomposition and solve for integer entries.
    # 
    # Let R be the rank. We have:
    # U: nm x R, V: mp x R, W: pn x R
    # T = sum_r u_r * v_r * w_r (outer product sum)
    # 
    # This means for each (a, b, c):
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Let's try rank = 21 first (the minimum possible).
    
    # For rank R, we have R * (nm + mp + pn) = R * (9 + 21 + 21) = 51R variables
    # We have nm * mp * pn = 9 * 21 * 21 = 3969 equations (but most are 0).
    # Non-zero equations: n*m*p = 63.
    
    # So we have 51R variables and 63 non-trivial equations.
    # For R = 21: 1071 variables, 63 equations. Highly underdetermined.
    # But we need integer/half-integer entries.
    
    # Let's try a constructive approach.
    
    # For each output element C_{i,k} = sum_j A_{i,j} B_{j,k}
    # We can write this as:
    # C_{i,k} = sum_j (A_{i,j} * B_{j,k})
    # 
    # The trivial decomposition uses 3 multiplications per output, for 3*7=21 outputs,
    # total 63 multiplications.
    
    # Can we do better? Let's see if we can share multiplications.
    # 
    # For example, if we compute (A_{i,j} + A_{i',j'}) * (B_{j,k} + B_{j',k'}),
    # we get 4 terms but use 1 multiplication.
    
    # This is essentially the Strassen idea. Let me try to apply it here.
    
    # For 3x3 matrices, Laderman's algorithm uses 19 multiplications.
    # But we have 3x3 * 3x7, so 21 output elements.
    
    # Let me just try the trivial decomposition for now, since finding
    # a lower-rank exact integer decomposition is non-trivial.
    
    # Actually, let me try to be smarter. Let's use the following approach:
    # 
    # We know that the tensor has a certain structure. Let's try to find
    # a rank-r decomposition where r < 63 by exploiting the fact that
    # many entries are zero.
    
    # Build the non-zero pattern
    # T[a,b,c] = 1 iff a%m == b//p and b%p == c//n and c%n == a//m
    
    # Let's try rank = 42, 36, 30, 24, 21, etc.
    
    # For a quick test, let's just return the trivial rank-63 decomposition.
    # But first, let me check if there's a simple improvement.
    
    # Idea: For each column k, C[:,k] = A * B[:,k]
    # This is a 3x3 times 3x1 multiplication, which has rank at most 9.
    # But we have 7 columns, so naively 63.
    
    # However, if we can share multiplications across columns...
    
    # Let me try a different tactic: use numpy to find the minimal rank
    # by checking if a rank-r decomposition exists.
    
    # For now, return the trivial decomposition.
    rank = n * m * p  # 63
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
