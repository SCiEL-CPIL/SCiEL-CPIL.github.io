import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Matrix multiplication tensor: T[i,j,k] = 1 if we compute C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to find a rank-r decomposition:
    #   T = sum_{r=1}^{rank} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the tensor entry T[i*m+j, j*p+k, k*n+i] = 1 (using the flattening convention)
    
    # The tensor T has entries:
    # T_flat[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # A rank-1 term is u ⊗ v ⊗ w where:
    #   (u ⊗ v ⊗ w)[a, b, c] = u[a] * v[b] * w[c]
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all (a,b,c) that don't correspond to valid (i,j,k): the sum = 0
    
    # Let's try a constructive approach using known results.
    # For 4x5x5 matrix multiplication, let's try to build a decomposition.
    
    # Approach: Use the fact that we can split the problem into sub-problems.
    # A (4x5) * B (5x5) = C (4x5)
    
    # Let's try a direct construction with rank as low as possible.
    
    # First, let's try the schoolbook: rank = 4*5*5 = 100
    # Then try to reduce using Strassen-like ideas.
    
    # Actually, let's try a more systematic approach:
    # We can view this as: C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # One simple decomposition: for each (i,j,k), use one rank-1 term:
    #   U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # This gives rank = 4*5*5 = 100.
    
    # Can we do better? Let's try to combine terms.
    
    # For fixed i,k: C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
    # Wait, that's not right. The tensor entry is T[i*m+j, j*p+k, k*n+i] = 1.
    
    # So for each triple (i,j,k), we need:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # And for all other (a,b,c) combinations (where a=i*m+j, b=j'*p+k', c=k''*n+i'' 
    # but j != j' or k != k'' or i != i''), we need the sum to be 0.
    
    # Let me think about this differently. The matrix multiplication tensor for n x m * m x p
    # has rank at most n*m*p (schoolbook).
    
    # For 4x5x5, let's try to find a better decomposition.
    
    # Strategy: Try to use a rank-r decomposition by solving a system of equations.
    # But that's NP-hard in general. Let's try a heuristic approach.
    
    # Let's try rank values from low to high and check if a decomposition exists.
    # We'll use a greedy/iterative approach.
    
    # First, let's construct the full tensor as a 3D array.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Now try to find a rank decomposition.
    # Start with schoolbook (rank 100) and try to reduce.
    
    # Actually, let's try a different approach: use known lower bounds and constructions.
    # The rank of the 4x5x5 matrix multiplication tensor is at least max(n, m, p) * something.
    # Known: rank of 2x2x2 is 7 (Strassen).
    
    # For 4x5x5, let's try to build a decomposition using block structure.
    
    # Split A into A1 (4x4) and A2 (4x1), B into B1 (4x5) and B2 (1x5).
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 * 4x5 -> use rank 7 (Strassen-like for 4x4x4) + 4 (for the extra column)
    # A2*B2: 4x1 * 1x5 -> rank 4 (outer product)
    
    # Wait, 4x4 * 4x5: we can split B1 into B1a (4x4) and B1b (4x1).
    # A1*B1a: 4x4 * 4x4 -> rank 7 (Strassen)
    # A1*B1b: 4x4 * 4x1 -> rank 4 (each row of A1 times the column)
    # A2*B1: 4x1 * 4x5 -> rank 4 (each scalar times the row)
    # A2*B2: 4x1 * 1x5 -> rank 4 (outer product)
    # Total: 7 + 4 + 4 + 4 = 19
    
    # But wait, I need to be careful. Let me construct this explicitly.
    
    # Let's try rank 19 first and see if it works.
    
    def try_rank(r):
        """Try to construct a rank-r decomposition."""
        # Initialize with zeros
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Try to fill in the decomposition
        # This is a heuristic approach - we'll try to assign values to minimize the error
        
        # For now, let's just return the schoolbook if r >= 100
        if r >= n * m * p:
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            return U, V, W
        
        return None
    
    # Let's try the block decomposition approach:
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x5) = [B1 (4x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x5
    #   Split B1 = [B1a (4x4) | B1b (4x1)]
    #   A1*B1a: 4x4 * 4x4 -> Strassen rank 7
    #   A1*B1b: 4x4 * 4x1 -> rank 4
    # A2*B2: 4x1 * 1x5 -> rank 4
    
    # Total: 7 + 4 + 4 = 15
    
    # Let's construct the Strassen decomposition for 4x4x4.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 4x4x4, we can use the same idea with 2x2 blocks.
    
    # Actually, Strassen's algorithm is for multiplying 2x2 matrices, giving rank 7.
    # For 4x4 matrices, we'd need to apply it recursively, but that gives rank 7^2 = 49 for 4x4x4.
    # That's worse than schoolbook (64).
    
    # So for 4x4x4, schoolbook has rank 64, and Strassen recursive gives 49.
    # But we can do better with other algorithms.
    
    # Let's just use schoolbook for now and return rank 100.
    
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
