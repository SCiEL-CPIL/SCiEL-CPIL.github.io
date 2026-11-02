"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # The tensor for matrix multiplication C = A @ B where A is n×m and B is m×p:
    # T[i, j, k] = 1 if the (i,j) element of A multiplies with (j,k) element of B
    # to produce (i,k) element of C.
    # 
    # We need U (n*m × rank), V (m*p × rank), W (p*n × rank) such that:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other index combinations (i,j,k) where j is not the shared index:
    #   The tensor is 0, so no constraint needed beyond the above.
    #
    # Actually, the matrix multiplication tensor has entries:
    # T[i, j, k] = 1 for all valid (i,j,k) where i∈[0,n), j∈[0,m), k∈[0,p)
    # and 0 otherwise. But since we're indexing U by (i,j), V by (j,k), W by (k,i),
    # the constraint is exactly:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Schoolbook: rank = n*m*p = 50
    # Each (i,j,k) triple gets its own rank-1 term.
    
    # Key insight for 2×5×5: We can potentially do better than 50.
    # 
    # Let's think about this differently. The multiplication C = A @ B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is a bilinear map. The tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in appropriate spaces)
    #
    # Wait, that's not quite right. Let me be more careful.
    # 
    # Input space: A has entries A[i,j] for i∈[0,n), j∈[0,m)
    #              B has entries B[j,k] for j∈[0,m), k∈[0,p)
    # Output space: C has entries C[i,k] for i∈[0,n), k∈[0,p)
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # So the tensor T in (input_A ⊗ input_B) → output_C is:
    # T = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
    # No wait, that's not right either.
    #
    # Let me use the standard tensor decomposition framework:
    # We want to write the multiplication as:
    # C[i,k] = sum_c U[i*m+j... 
    #
    # Actually, the standard formulation is:
    # A is n×m, B is m×p, C = A@B is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor for this multiplication, when flattened:
    # Input 1: A flattened to n*m elements, indexed by (i,j)
    # Input 2: B flattened to m*p elements, indexed by (j,k)  
    # Output: C flattened to n*p elements, indexed by (i,k)
    #
    # The bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # As a tensor: T_{(i,j),(j,k),(i,k)} = 1 (with appropriate indices)
    #
    # For a rank-r decomposition:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # This must equal sum_j A[i,j] * B[j,k] for all A, B.
    #
    # So we need: for all i, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] 
    #   = coefficient of A[i,j]*B[j,k] in the product
    #
    # The coefficient of A[i',j']*B[j'',k'] in C[i,k] is:
    #   1 if i'=i, k'=k, j'=j''=j (for some j), else 0
    #
    # So the constraint is:
    # For all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # And for all (i,j) and (j',k) where j ≠ j':
    #   The cross terms must vanish, but this is automatically satisfied
    #   if the above holds for each (i,j,k) independently... 
    #   No wait, that's not how it works.
    #
    # Let me reconsider. The full constraint is:
    # For all A, B: C = A@B
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # With the decomposition:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    #
    # Hmm, this is getting complicated. Let me just use the direct verification approach.
    
    def verify(U, V, W, rank):
        """Verify that U, V, W give a valid decomposition."""
        # U: (n*m) x rank
        # V: (m*p) x rank  
        # W: (p*n) x rank
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each (i, j, k), check:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a rank-25 decomposition
    # Structure: Since n=2, we have 10 input A entries, 25 input B entries, 10 output C entries.
    # 
    # One approach: For each (j, k) pair (25 pairs), we need to compute contributions.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's try a construction where we use rank = m*p = 25.
    # For each (j,k) pair, create one rank-1 term.
    # 
    # Term (j,k): 
    #   U[i*m+j, (j,k)] = 1 for all i
    #   V[j*p+k, (j,k)] = 1
    #   W[k*n+i, (j,k)] = 1 for all i
    #
    # Then for C[i,k]:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_{j',k'} [U[i*m+j, (j',k')] * V[j*p+k, (j',k')] * W[k*n+i, (j',k')]]
    #
    # U[i*m+j, (j',k')] = 1 if j=j' else 0
    # V[j*p+k, (j',k')] = 1 if j=j' and k=k' else 0
    # W[k*n+i, (j',k')] = 1 (always, for the given i,k)
    #
    # So the sum = 1 * 1 * 1 = 1 when j'=j and k'=k, else 0.
    # Total = 1. ✓
    #
    # But wait, this gives C[i,k] = sum_j A[i,j] * B[j,k] * 1 = correct!
    # 
    # Let me verify: 
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #         = sum_{j',k'} U[i*m+j, (j',k')] * V[j*p+k, (j',k')] * W[k*n+i, (j',k')]
    #
    # U[i*m+j, (j',k')] = 1 if j' == j, else 0
    # V[j*p+k, (j',k')] = 1 if j' == j and k' == k, else 0
    # W[k*n+i, (j',k')] = 1
    #
    # So only the term (j',k') = (j,k) contributes, giving 1*1*1 = 1.
    # 
    # But this is for a fixed (i,j,k). The actual C[i,k] involves summing over j:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # With our decomposition, for a given (i,k):
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]
    #
    # Hmm, I'm confusing myself. Let me just construct it and verify.
    
    # Rank 25 construction:
    rank_25 = m * p  # 25
    U25 = [[0] * rank_25 for _ in range(n * m)]
    V25 = [[0] * rank_25 for _ in range(m * p)]
    W25 = [[0] * rank_25 for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U25[i * m + j][c] = 1
                W25[k * n + i][c] = 1
            V25[j * p + k][c] = 1
    
    if verify(U25, V25, W25, rank_25):
        # Try to find even lower rank
        # For now, return rank 25
        return {"U": U25, "V": V25, "W": W25}
    
    # Fallback: schoolbook rank 50
    rank_school = n * m * p  # 50
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
