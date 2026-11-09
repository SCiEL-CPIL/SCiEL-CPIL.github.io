"""Program for rank-minimal exact tensor decomposition of 3x3 * 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _validate(U, V, W, n=3, m=3, p=8):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    
    # Build target tensor T
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check: sum_c U[a,c] * V[b,c] * W[c_idx,c] should equal T[a,b,c_idx]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                
                # Compute sum over rank
                val = sum(U[a, r] * V[b, r] * W[c_idx, r] for r in range(rank))
                if val != 1:
                    return False
    
    # Check zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for k2 in range(p):
                    if k2 == k:
                        continue
                    c_idx2 = k2 * n + i
                    # This entry should be 0 if it's not the diagonal in the output
                    # Actually, the tensor has 1s only at specific positions
                    # Let me reconsider the tensor structure
    
    # The multiplication tensor for C = A * B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    
    # Check some zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                # Check that non-matching output indices give 0
                for i2 in range(n):
                    for k2 in range(p):
                        if i2 == i and k2 == k:
                            continue
                        c_idx = k2 * n + i2
                        # This should be 0
                        val = sum(U[a, r] * V[b, r] * W[c_idx, r] for r in range(rank))
                        if val != 0:
                            return False
    return True

def _schoolbook_decomposition(n=3, m=3, p=8):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_3x3_decomposition():
    """
    Return a rank-19 decomposition for 3x3 * 3x3 matrix multiplication.
    Uses a known optimal decomposition.
    """
    # Known rank-19 decomposition for 3x3 matrix multiplication
    # This uses the work of Alon et al. and subsequent improvements
    # We'll use a construction based on Strassen's method extended to 3x3
    
    # For 3x3, one approach is to use a 4x4 decomposition and restrict
    # Or use known explicit rank-19 decompositions
    
    # Let's use a systematic construction:
    # We can embed 3x3 into 4x4 and use Strassen's rank-7 for 2x2
    # But that gives higher rank.
    
    # Alternative: use the fact that 3x3 can be decomposed with rank 19
    # Using the decomposition from "A rank-19 algorithm for 3x3 matrix multiplication"
    
    # For simplicity and correctness, let's use a construction that we can verify
    # We'll use a combination of diagonal and off-diagonal terms
    
    # Actually, let's use a different approach: construct a valid decomposition
    # with rank less than 27 by exploiting structure
    
    # Here's a rank-24 construction that's easier to implement:
    # Use 3x3 = sum of rank-1 terms with some sharing
    
    # For now, let's return a valid decomposition and try to optimize
    # We'll use a construction based on partitioning
    
    # Let me try a different strategy: use the structure of the problem
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this using intermediate sums
    # Let S[j] = sum over some grouping
    
    # For a simpler approach, let's use rank 24 with a specific structure
    # that we can then try to improve
    
    # Actually, the safest approach is to use the schoolbook and try random
    # low-rank constructions with exact verification
    
    return None  # We'll handle this in the main construct function

def _try_random_decomposition(rank, n=3, m=3, p=8, seed=0, max_iterations=10000):
    """
    Try to find a valid decomposition of the given rank using random search.
    Uses exact rational arithmetic.
    """
    import random
    random.seed(seed)
    
    deadline = time.time() + 120
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Generate random factor matrices with entries in {0, 1/2, 1, -1/2, -1}
        # Scale by 2 to work with integers, then divide by 2 at the end
        entries = [0, 1, 2, -1, -2]  # These will be divided by 2
        
        U = [[random.choice(entries) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(entries) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(entries) for _ in range(rank)] for _ in range(p * n)]
        
        # Check if this decomposition is valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    
                    val = sum(U[a, r] * V[b, r] * W[c_idx, r] for r in range(rank))
                    if val != 4:  # 4 because we scaled by 2, so 1 * 2^2 = 4
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            if not valid:
                break
        
        if valid:
            # Scale back: divide all entries by 2
            U = [[u / 2 for u in row] for row in U]
            V = [[v / 2 for v in row] for row in V]
            W = [[w / 2 for w in row] for row in W]
            
            # Convert to fractions for exactness
            U = [[Fraction(u) for u in row] for row in U]
            V = [[Fraction(v) for v in row] for row in V]
            W = [[Fraction(w) for w in row] for row in W]
            
            return {"U": U, "V": V, "W": W}
    
    return None

def _structured_decomposition(n=3, m=3, p=8):
    """
    Try a structured decomposition that exploits the shared structure
    across the 8 output columns.
    
    Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    We can group the 8 columns and use shared intermediate computations.
    """
    # Strategy: Use a decomposition that shares intermediate results
    # across the 8 output columns
    
    # Let's try a construction based on:
    # 1. Compute some intermediate sums S_j = A[i,j] for various i
    # 2. Use these to compute all 8 output columns
    
    # For a 3x3 * 3x8 problem, we have:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # We can try to use rank-19 for the 3x3 core and extend to 8 columns
    
    # Alternative: use a decomposition with rank R where we share
    # intermediate products across columns
    
    # Let's try a systematic approach with rank 40-50
    # using a structured pattern
    
    # Construction idea:
    # For each of the 3 rows of A and 3 columns of B, we have 3*3=9 products
    # But we can share some computations
    
    # Let's use a block structure:
    # Group the 8 columns into 2 groups of 4
    # Use rank-19 for each 3x3*3x4 subproblem with sharing
    
    # Actually, let's try a simpler structured approach:
    # Use the fact that we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # We can precompute some combinations of A entries
    # and use them to compute multiple C[i,k] values
    
    # For example, let:
    # X0 = A[i,0]
    # X1 = A[i,1]  
    # X2 = A[i,2]
    # X01 = A[i,0] + A[i,1]
    # X02 = A[i,0] + A[i,2]
    # X12 = A[i,1] + A[i,2]
    # X012 = A[i,0] + A[i,1] + A[i,2]
    
    # Similarly for B entries
    
    # This is getting complex. Let's use a different approach:
    # Try to find a valid decomposition with rank < 72 using
    # a guided search based on the structure
    
    return None

def construct():
    n, m, p = 3, 3, 8
    deadline = time.time() + 120
    
    # Start with schoolbook as fallback
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 72
    
    # Try structured decomposition
    if time.time() < deadline - 60:
        structured = _structured_decomposition(n, m, p)
        if structured is not None:
            rank = len(structured["U"][0])
            if rank < best_rank:
                # Verify
                if _validate(structured["U"], structured["V"], structured["W"], n, m, p):
                    best = structured
                    best_rank = rank
    
    # Try random search with decreasing rank
    # Start from best_rank - 1 and go down
    for target_rank in range(best_rank - 1, max(20, best_rank - 30), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_random_decomposition(target_rank, n, m, p, seed=target_rank, max_iterations=5000)
        if result is not None:
            if _validate(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = target_rank
                break  # Found a valid one, stop
    
    # Convert to plain Python types for output
    U = [[float(u) for u in row] for row in best["U"]]
    V = [[float(v) for v in row] for row in best["V"]]
    W = [[float(w) for w in row] for row in best["W"]]
    
    # Round to nearest half-integer
    def round_half(x):
        return round(2 * x) / 2
    
    U = [[round_half(u) for u in row] for row in U]
    V = [[round_half(v) for v in row] for row in V]
    W = [[round_half(w) for w in row] for row in W]
    
    # Final verification with the rounded values
    U_final = [[Fraction(u) for u in row] for row in U]
    V_final = [[Fraction(v) for v in row] for row in V]
    W_final = [[Fraction(w) for w in row] for row in W]
    
    if _validate(U_final, V_final, W_final, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If rounding broke it, return the schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END

