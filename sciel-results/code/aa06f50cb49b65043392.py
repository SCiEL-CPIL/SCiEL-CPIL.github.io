"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Uses structured search exploiting the factorization 6 = 2*3.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline with margin
    
    n, m, p = DIMENSIONS
    
    # Create the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 36
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, column] = 1.0
                V_sb[j * p + k, column] = 1.0
                W_sb[k * n + i, column] = 1.0
                column += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Helper function to verify a decomposition
    def verify(U, V, W, tol=1e-10):
        rank = U.shape[1]
        # Check all valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(actual - expected) > tol:
                        return False
        
        # Check invalid positions: for each (a, b, c) where indices don't align
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        # Valid when i==i', j==j', k==k'
        # We need to check all invalid (a, b, c) tuples
        checked_invalid = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # This is valid, skip
                    pass
        
        # Check a random sample of invalid positions for efficiency
        for _ in range(200):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            is_valid = (i1 == i2) and (j1 == j2) and (k1 == k2)
            if not is_valid:
                actual = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(actual) > tol:
                    return False
        return True
    
    # Try to find a Kronecker-structured decomposition
    # The idea: if we can write the multiplication as a tensor product of smaller problems
    # For 2x3 * 3x6, we can try to exploit 6 = 2*3
    
    # Strategy: Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But we can potentially do better
    
    # First, try to construct a structured decomposition using the factorization
    # If 6 = 2*3, we can think of the 3x6 matrix as two 3x3 blocks
    # Then (2x3) * (3x6) = (2x3) * [(3x3) | (3x3)]
    # This is like multiplying 2x3 by two separate 3x3 matrices
    
    # For each half (3x3 multiplication), the schoolbook rank is 27
    # But we can share structure
    
    # Let's try a more systematic approach:
    # Use the fact that we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Try ranks from 12 (lower bound based on dimension products) upward
    # Actually, the lower bound is at least max(n*m, m*p, p*n) = 18
    # But let's start from a reasonable point
    
    # Use a smarter initialization based on the Kronecker structure
    for target_rank in range(12, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts decreases as rank increases
        max_attempts = max(5, 50 // max(1, target_rank - 11))
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization exploiting 6 = 2*3
            # Split the p-dimension into two halves of size 3
            half_p = 3
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Assign ranks to the two halves
            # Try different splits of the rank between the two halves
            if attempt < 3:
                # Even split
                r1 = target_rank // 2
                r2 = target_rank - r1
            elif attempt < 6:
                # 1/3 - 2/3 split
                r1 = target_rank // 3
                r2 = target_rank - r1
            else:
                # Random split
                r1 = random.randint(1, max(1, target_rank - 1))
                r2 = target_rank - r1
            
            # First half: k in 0..2
            # Second half: k in 3..5
            
            # For the first half, we need to decompose (2x3) * (3x3)
            # For the second half, we need to decompose (2x3) * (3x3)
            
            # Initialize with random values but with structure
            # Each rank element in the first half only has non-zero entries for k in 0..2
            # Each rank element in the second half only has non-zero entries for k in 3..5
            
            for r in range(r1):
                # First half
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = random.gauss(0, 1)
                for j in range(m):
                    for k in range(half_p):
                        V[j*p+k, r] = random.gauss(0, 1)
                for k in range(half_p):
                    for i in range(n):
                        W[k*n+i, r] = random.gauss(0, 1)
            
            for r in range(r1, target_rank):
                # Second half
                r2_idx = r - r1
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = random.gauss(0, 1)
                for j in range(m):
                    for k in range(half_p, p):
                        V[j*p+k, r] = random.gauss(0, 1)
                for k in range(half_p, p):
                    for i in range(n):
                        W[k*n+i, r] = random.gauss(0, 1)
            
            if verify(U, V, W):
                best_rank = target_rank
                best_decomp = (U.copy(), V.copy(), W.copy())
                found = True
                break
        
        if found:
            break
    
    if best_rank < rank_schoolbook:
        return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
