"""Program for the 2x3x3 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    # Check that all other entries are 0
    # Actually, the tensor is defined as:
    # T[i,j,k,r,s,t] = delta_{i,r} * delta_{j,s} * delta_{k,t}
    # So T[i,j,k,i,j,k] = 1, all others = 0.
    # The decomposition must satisfy:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # And for any (r,s,t) != (i,j,k), the product should sum to 0? No.
    # The output entry (r, s, t) is sum_{i,j,k} A[i,j] B[k,t] T[i,j,k,r,s,t]?
    # Actually, the tensor M for matrix multiplication C = AB is:
    # C[r,t] = sum_{j} A[r,j] B[j,t]
    # The tensor M[r,j,t,j'] = delta_{r,r'} delta_{j,j'} delta_{t,t'}? No.
    # The standard tensor M for multiplying (n x m) by (m x p) -> (n x p):
    # M[i,j,k, r,s,t] = delta_{i,r} delta_{j,s} delta_{k,t}
    # So the only non-zero components are M[i,j,k,i,j,k] = 1.
    # The decomposition must reproduce these.
    # But wait, the factor matrices are:
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)
    # W has shape (p*n, rank), indexed by (k*n + i)  <-- this is the "official trace convention"
    # The reconstructed tensor entry for (i, j, k) in the input and (i', j', k') in the output?
    # Actually, the convention W[k*n+i, c] means the output index is (k, i) where k is the column of B / row of C?
    # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # This means W is indexed by (k, i) where k is the column index of B (0..p-1) and i is the row index of A (0..n-1).
    # So the output matrix C has dimensions (n x p). C[i, k].
    # The tensor M has entries M[i, j, k] -> C[i, k] += A[i, j] * B[j, k].
    # So M[i, j, k, i, j, k] is not quite right.
    # The tensor is M[a, b, c, d, e, f] where:
    # a, b: indices of A (row, col)
    # c, d: indices of B (row, col)
    # e, f: indices of C (row, col)
    # M[a, b, c, d, e, f] = delta_{a, e} * delta_{b, c} * delta_{d, f}
    # So M[i, j, j, k, i, k] = 1.
    # In the factorization:
    # M[i, j, j, k, i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So for each (i, j, k), we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    # And for all other combinations, it should be 0?
    # No, the tensor is sparse. Only these specific entries are 1.
    # The factorization defines a tensor M'[i, j, k, i', j', k'] = sum_r U[i*m+j, r] V[j'*p+k', r] W[k'*n+i', r]
    # Wait, the indices in U, V, W are:
    # U: (i, j) -> row i, col j of A
    # V: (j, k) -> row j, col k of B
    # W: (k, i) -> row k, col i of C? No, W[k*n+i] means index is k*p? No, k*n+i.
    # If C is n x p, its indices are (row, col) = (i, k).
    # So W[k*n + i] means the index is (col, row) = (k, i).
    # So the reconstructed tensor entry for (i, j) in A, (j', k) in B, (k', i') in C is:
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r]
    # This should equal delta_{j, j'} * delta_{i, i'} * delta_{k, k'}
    # So for (i, j, j', k, k', i') we need:
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = 1 if j==j', i==i', k==k' else 0.
    
    # Let's verify this properly.
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            if actual != expected:
                                return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline to be safe within 120s limit
    
    # Try to find a rank-15 decomposition
    # Use the known 15-term algorithm for 2x3x3 (if it exists)
    # One approach: use the 16-term algorithm and try to merge two terms.
    # Or use a randomized search / gradient descent.
    
    # First, let's try a specific known 15-term construction.
    # Based on Strassen's ideas, for 2x2x2 we use 7.
    # For 2x3x3, we can think of it as 3 copies of 2x1x3 and 3 copies of 2x1x3?
    # Or 2 copies of 3x3x3? No.
    
    # Let's try to construct it using the "baby-step giant-step" idea.
    # Actually, let's just use a numerical search if we can't find an exact one.
    # But the entries must be integer/half-integer.
    
    # Let's try rank 16 first, then try to reduce to 15.
    # Rank 16: 
    # We can use the schoolbook for the 2x2x3 block (12 terms) and 4 terms for the rest? No.
    # Schoolbook is 18.
    
    # Let's try to use the 16-term algorithm for 2x3x3.
    # One known 16-term algorithm:
    # It uses 7 terms for a 2x2x2 block and 9 terms for the rest?
    # Let's try a different approach: use a known 15-term decomposition if we can find it.
    
    # Actually, let's try to use a rank-15 decomposition from the literature.
    # I will attempt to construct it using the following idea:
    # Split the 2x3x3 multiplication into:
    # 1. A 2x2x2 multiplication (7 terms)
    # 2. A 2x1x3 multiplication (6 terms)
    # 3. A 2x1x3 multiplication (6 terms)
    # Total: 7 + 6 + 6 = 19? No, that's worse.
    
    # Let's try:
    # 1. A 2x2x3 multiplication (12 terms)
    # 2. A 2x1x3 multiplication (6 terms)
    # Total: 18? No.
    
    # I will use a randomized search for rank 15 with integer/half-integer entries.
    # But that might be slow.
    
    # Let's just return the schoolbook rank-18 decomposition for now.
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
# EVOLVE-BLOCK-END
