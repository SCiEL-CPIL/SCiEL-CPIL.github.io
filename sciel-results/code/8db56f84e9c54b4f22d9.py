"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def time_remaining():
        return deadline - time.time()

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                    if val != expected:
                        return False
        return True

    # Strategy: Use a structured algebraic decomposition based on the tensor network approach.
    # Key insight: For 3x3 * 3x8, we can exploit the structure by:
    # 1. Grouping the 8 columns into 4 pairs
    # 2. Using a rank-7 decomposition for each 2x2 block (Strassen-like)
    # 3. Combining with the 3x3 row structure
    
    # The tensor T[(i,j),(j,k),(k,i)] = 1 if indices match
    # We want: T = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Approach: Use a hierarchical decomposition
    # For each column pair (k, k+1), we can use a 2-column Strassen-like method
    # This reduces the effective dimension
    
    # Let's try a direct construction using integer coefficients:
    # We'll use a rank-48 decomposition based on the following:
    # For each of the 3 middle indices j, and each of the 4 column pairs,
    # we use a rank-4 decomposition (2x2 Strassen has rank 7, but we can do better with structure)
    
    # Actually, let's try a more systematic approach:
    # The minimum rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we need to find an exact decomposition.
    
    # Try a rank-36 decomposition: 3 (rows) * 3 (cols) * 4 (column pairs) = 36
    # Each column pair can be handled with rank-4 terms
    
    # Construction: For column pair (k, k+1), define:
    # Term 1: U[i*m+j] = A[i,j], V[j*p+k] = B[j,k], W[k*n+i] = 1
    # Term 2: U[i*m+j] = A[i,j], V[j*p+k+1] = B[j,k+1], W[(k+1)*n+i] = 1
    # But this is just schoolbook for pairs, giving rank 36
    
    # To reduce further, use Strassen-like identities on the column dimension:
    # For 2 columns, we can use:
    # C[:,k] = A[:,0]*B[0,k] + A[:,1]*B[1,k] + A[:,2]*B[2,k]
    # C[:,k+1] = A[:,0]*B[0,k+1] + A[:,1]*B[1,k+1] + A[:,2]*B[2,k+1]
    
    # Define:
    # S0 = A[:,0]*(B[0,k] + B[0,k+1])
    # S1 = A[:,1]*(B[1,k] + B[1,k+1])  
    # S2 = A[:,2]*(B[2,k] + B[2,k+1])
    # T0 = A[:,0]*B[0,k]
    # T1 = A[:,1]*B[1,k]
    # T2 = A[:,2]*B[2,k]
    # Then C[:,k] = T0 + T1 + T2
    # C[:,k+1] = S0 + S1 + S2 - T0 - T1 - T2
    
    # This uses 6 terms per column pair instead of 6 (same), but we can combine across pairs
    
    # Better: Use a global Strassen-like approach
    # For the full 3x8 multiplication, we can use:
    # Let B be split into 4 column pairs: B1, B2, B3, B4
    # C = A * B = A * [B1 B2 B3 B4]
    
    # Define intermediate products:
    # P0 = A[:,0] * (B[0,0] + B[0,1] + B[0,2] + B[0,3] + B[0,4] + B[0,5] + B[0,6] + B[0,7])
    # This is getting complex. Let's try a different approach.
    
    # Use a rank-48 decomposition with the following structure:
    # For each row i (0,1,2), and each column k (0..7), we need A[i,j]*B[j,k]
    # We can use a 4x4 Strassen-like decomposition for the 3x3 * 3x4 part, then repeat
    
    # Actually, the simplest improvement: use rank-54 instead of 72
    # By using the identity: for each j, sum over k of A[i,j]*B[j,k]
    # We can factor out A[i,j] and use a rank-8 decomposition for the k-dimension
    
    # Let's try rank-48: 3 (i) * 3 (j) * (8/2) = 36, plus 12 extra for Strassen savings
    
    # Direct construction with rank-48:
    # Use the fact that for each fixed j, we have a 3x8 outer product
    # We can use a rank-4 decomposition for the 8-dimensional space using 4 basis vectors
    
    # Define 4 basis vectors for the 8 columns:
    # e1 = [1,1,0,0,0,0,0,0]
    # e2 = [1,-1,0,0,0,0,0,0]
    # e3 = [0,0,1,1,0,0,0,0]
    # e4 = [0,0,1,-1,0,0,0,0]
    # e5 = [0,0,0,0,1,1,0,0]
    # e6 = [0,0,0,0,1,-1,0,0]
    # e7 = [0,0,0,0,0,0,1,1]
    # e8 = [0,0,0,0,0,0,1,-1]
    
    # This gives rank 3*3*8 = 72 again. Not helpful.
    
    # Let's try a completely different approach: use known low-rank decompositions
    # For 3x3 * 3x3, the best known rank is 19 (Lader 1976, improved by others)
    # For 3x3 * 3x8, we can use: rank(3x3*3x3) + rank(3x3*3x5) or similar
    
    # Actually, let's just try to find a valid decomposition with rank < 72 using random search
    # with a smarter initialization
    
    # Start with schoolbook (rank 72)
    schoolbook_rank = n * m * p  # 72
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition using local search
    # Start from schoolbook and try to eliminate terms
    
    def try_reduce_rank(U, V, W, max_iter=5000):
        """Try to reduce the rank by finding redundant terms."""
        rank = len(U[0])
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Try removing a random term
            r_remove = np.random.randint(0, rank)
            
            # Create a copy with term r_remove zeroed out
            U_new = [row[:] for row in U]
            V_new = [row[:] for row in V]
            W_new = [row[:] for row in W]
            
            U_new[r_remove] = [0] * rank
            V_new[r_remove] = [0] * rank
            W_new[r_remove] = [0] * rank
            
            # Check if the decomposition still works
            if verify_decomposition(U_new, V_new, W_new):
                # Remove the term
                U = [row[:r_remove] + row[r_remove+1:] for row in U]
                V = [row[:r_remove] + row[r_remove+1:] for row in V]
                W = [row[:r_remove] + row[r_remove+1:] for row in W]
                rank -= 1
            else:
                # Try adjusting other terms to compensate
                # This is complex, so we just skip for now
                pass
        
        return U, V, W, rank
    
    # Try random initializations for lower ranks
    for target_rank in [64, 60, 56, 52, 48, 44, 40, 36, 32]:
        if time_remaining() < 10:
            break
        
        found = False
        for attempt in range(500):
            if time_remaining() < 5:
                break
            
            # Generate a random sparse decomposition
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with a structured pattern
            for r in range(target_rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
            
            # Add some randomness
            for r in range(target_rank):
                if np.random.random() < 0.3:
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
