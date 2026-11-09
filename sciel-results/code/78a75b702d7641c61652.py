"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Trivial decomposition (rank = n*m*p = 54)
    base_rank = n * m * p
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def verify_np(U, V, W):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-8:
                        return False
        return True
    
    best_rank = base_rank
    best_result = (best_U, best_V, best_W)
    
    # Strategy 1: Block-structured decomposition
    # View the 3x9 matrix as three 3x3 blocks
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each block b (0,1,2), B_block is 3x3
    # C[:, b*3:(b+1)*3] = A @ B_block
    # Each 3x3 multiplication can be done with rank 23 (Strassen)
    # But we need to share the A matrix across blocks
    # 
    # Actually, let's think differently. The tensor is:
    # T[i,j,k] = 1 for all (i,j,k)
    # We need sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
    #
    # Let's try a structured approach where we group by the middle index j.
    # For fixed j, we need: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,k
    # This means for each j, the matrices U_j (n x rank_j), V_j (rank_j x p), W_j (n x p)
    # satisfy: U_j @ V_j * W_j (element-wise) = ones(n,p)
    # 
    # Actually that's not quite right. Let me reconsider.
    #
    # The tensor decomposition means:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Let's try a different structured approach.
    # Group terms by which (i,j,k) they primarily contribute to.
    # 
    # Strategy 2: Try to find a low-rank decomposition using the structure
    # that each output C[i,k] depends on all j.
    #
    # Let's try random search with various rank values, using structured initialization.
    
    np.random.seed(42)
    
    # Try various rank values
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But actually the lower bound is the tensor rank, which is at least max(n*m, m*p, p*n)
    # For 2x3 * 3x9, let's try ranks from 27 upward
    
    # First, let's try a smarter structured approach
    # We can write the multiplication as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # 
    # This is a sum of 3 rank-1 terms in the (i,k) space for each j,
    # but they're coupled through the shared j index.
    
    # Let's try a direct optimization approach
    # For a given rank R, we want to find U (6xR), V (27xR), W (18xR)
    # such that for all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1
    
    # This is a system of 54 equations in 6*R + 27*R + 18*R = 51*R unknowns
    # For R >= 2, we have more unknowns than equations, so solutions may exist.
    
    # Let's try using numpy's least squares or a simple iterative method
    
    # Approach: Fix W to be identity-like, then solve for U and V
    # Or use a gradient-free search
    
    # Let's try a more systematic approach:
    # For R = 27 (the trivial lower bound), we can try to find a solution
    # where each rank-1 term contributes to a subset of (i,j,k) triples
    
    # Actually, let's just try random search more aggressively with better initialization
    
    # Try ranks from 30 down to 27
    for target_rank in range(30, 26, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try random initialization with various patterns
        for trial in range(200):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with random small integers
            for r in range(target_rank):
                U[:, r] = np.random.randint(-2, 3, size=n*m).astype(float)
                V[:, r] = np.random.randint(-2, 3, size=m*p).astype(float)
                W[:, r] = np.random.randint(-2, 3, size=p*n).astype(float)
            
            # Verify
            if verify_np(U, V, W):
                best_rank = target_rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
                break
    
    # Try a structured approach: use the fact that we can decompose
    # the multiplication into independent parts
    
    # For each pair (i, k), C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a dot product of length 3
    # We can use the identity: a*b = ((a+b)^2 - a^2 - b^2) / 4
    # But that increases the rank
    
    # Let's try a different structured approach:
    # Use the fact that 3 = 2 + 1, and use Strassen-like tricks
    
    # For multiplying 2x3 by 3x9:
    # Split the 3 middle dimension into 2+1
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x9, B2 is 1x9
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x9, can be done with rank 7 (Strassen for 2x2)
    # A2*B2: 2x1 * 1x9, rank 2 (just 2 products)
    # Total rank: 7 + 2 = 9? No, that's not how it works.
    
    # Actually, the rank of a sum of tensors is at most the sum of ranks,
    # but we need to be careful about the shared indices.
    
    # Let me try a completely different approach: use the known fact that
    # the tensor rank of matrix multiplication M_{n,m,p} is at most n*m*p
    # and try to find improvements using known results.
    
    # For 2x3 * 3x9, the schoolbook rank is 54.
    # Can we do better?
    
    # One approach: use the fact that 9 = 3*3, and view the problem as
    # multiplying a 2x3 matrix by a 3x3 matrix, then by a 3x3 matrix.
    # But that doesn't directly help.
    
    # Let's try a more aggressive random search with more trials and better initialization
    
    np.random.seed(123)
    
    # Try ranks from 50 down to 30
    for target_rank in range(50, 29, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try random initialization with various patterns
        for trial in range(100):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with random small integers
            for r in range(target_rank):
                U[:, r] = np.random.randint(-3, 4, size=n*m).astype(float)
                V[:, r] = np.random.randint(-3, 4, size=m*p).astype(float)
                W[:, r] = np.random.randint(-3, 4, size=p*n).astype(float)
            
            # Verify
            if verify_np(U, V, W):
                best_rank = target_rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
                break
    
    # Try a structured approach using the block structure
    # View B as three 3x3 blocks: B = [B0 | B1 | B2]
    # C = A * B = [A*B0 | A*B1 | A*B2]
    # Each A*Bb is a 2x3 * 3x3 multiplication
    # The tensor rank of 2x3 * 3x3 is at most 2*3*3 = 18
    # But we can potentially share terms between blocks
    
    # For each block b, we need:
    # sum_r U[i,j,r,b] * V[j,k,r,b] * W[i,k,r,b] = 1 for all i,j,k in block b
    # But the U and V matrices are shared across blocks in the j index
    
    # Actually, the tensor is:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,8}
    # We need to decompose this into rank-1 terms
    
    # Let's try a structured decomposition where we group by k mod 3
    # For k = 3*b + k', b in {0,1,2}, k' in {0,1,2}
    # T[i,j,3*b+k'] = 1
    
    # We can write:
    # sum_r U[i,j,r] * V[j,3*b+k',r] * W[3*b+k',i,r] = 1
    
    # Let's try to find a decomposition where each rank-1 term
    # contributes to a specific (b, k') pair
    
    # For each (b, k'), we need a decomposition of the 2x3 tensor
    # T_b[i,j] = 1 (for all i,j)
    # into rank-1 terms: U_b[i,j] * V_b[j] * W_b[i] = 1
    # This is trivially done with rank 6: U_b[i,j] = 1, V_b[j] = 1, W_b[i] = 1
    # But we can share the U and V matrices across different (b, k') pairs
    
    # Actually, let's just try a more systematic search
    
    # Try using a simple iterative method to find a low-rank decomposition
    # Start with the trivial decomposition and try to merge terms
    
    # For now, return the best result found
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
