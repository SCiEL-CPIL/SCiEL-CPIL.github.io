"""
Rewritten program for Matrix Multiplication 2x3x7.
Approach: Programmatic construction of the schoolbook decomposition with verification.
"""

import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                # Index in V: j*p + k  (Note: V corresponds to the second input, usually B, 
                # but the index structure j*p+k implies row j, col k of B)
                # Index in W: k*n + i  (Note: W corresponds to the output C, 
                # but index k*n+i is unusual for C which is n x p. 
                # The task says W[k*n+i, r]. This maps (k,i) to a linear index.
                # This implies the output is indexed by (k,i)? 
                # Standard C is (i,j). Here the trace convention might be different.
                # Let's follow the task description strictly:
                # U[i*m+j, c]
                # V[j'*p+k, c]  <-- Wait, the comment in seed says V[j'*p+k, c]
                # W[k*n+i, c]
                # And the condition: sum_c ... = delta_{j, j'}
                
                # This condition implies that the tensor decomposes the identity map 
                # or the multiplication operator in a specific basis.
                
                # For the schoolbook decomposition of matrix multiplication:
                # C[i, j] = sum_k A[i, k] B[k, j]
                # The tensor is T = sum_{i,j,k} E_{i,j} \otimes E_{i,k} \otimes E_{k,j} ?
                # Let's map the indices from the task:
                # U index: i*m + j. This looks like A[i, j].
                # V index: j'*p + k. This looks like B[j', k]? Or B[k, j']?
                # If V is B, and B is m x p, index is row * p + col.
                # So V[j*p + k] corresponds to B[j, k].
                # W index: k*n + i. This looks like C[k, i]?
                # But C is n x p. So C[i, j] is standard.
                # If W[k*n + i] corresponds to C, then the output is indexed by (k, i).
                # This suggests the output matrix is p x n?
                # Or the convention is C[k, i] = sum_j A[i, j] B[j, k]?
                # If A is n x m, B is m x p, then C is n x p.
                # If the output index is k*n + i, and k ranges 0..p-1, i ranges 0..n-1.
                # Then the output is effectively indexed by (k, i).
                # This implies the multiplication is A (n x m) * B (m x p) = C (p x n)?
                # That would be A * B^T?
                
                # Let's assume the standard schoolbook mapping fits the indices:
                # Term for (i, j, k) contributes to:
                # U[i*m + j]
                # V[j*p + k]  (Assuming V index uses j as row, k as col)
                # W[k*n + i]
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Optional: Verification function (not called in construct, but present for robustness)
def verify_decomposition(decomp, n=2, m=3, p=7):
    U = np.array(decomp['U'])
    V = np.array(decomp['V'])
    W = np.array(decomp['W'])
    rank = U.shape[1]
    
    # Check dimensions
    if U.shape != (n * m, rank): return False
    if V.shape != (m * p, rank): return False
    if W.shape != (p * n, rank): return False
    
    # The condition from the seed comment:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j, j'}
    # This needs to hold for all i, j, j', k.
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1 if j == jp else 0
                    if val != expected:
                        return False
    return True

if __name__ == "__main__":
    # Self-test
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", verify_decomposition(res))
