"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def schoolbook():
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

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U,V,W give a valid tensor decomposition."""
        rank = len(U[0])
        # Check all (a,b,c) triples
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    # Valid if j_a == j_b and k_b == k_c and i_a == i_c
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s - 1) > tol:
                            return False
                    else:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Strategy: Try to find lower rank using structured approaches
    # Key idea: The tensor for 2x3 * 3x7 can potentially be decomposed at lower rank
    # by sharing intermediate computations.
    
    # Approach 1: Try rank values from best_rank-1 down, using systematic search
    # with structured initializations
    
    # The tensor has structure: for each j (0,1,2), we have a 2x7 sub-problem
    # C[:,k] contributions from A[:,j]*B[j,k]
    
    # Let's try a parametric family:
    # For each j, define vectors u_j[i] and v_j[k] such that
    # the contribution to C[i,k] from j is u_j[i]*v_j[k]*A[i,j]*B[j,k]
    # But that's just the schoolbook with weights.
    
    # Better: use the fact that we can combine terms across j.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Try: introduce auxiliary variables. For each k, define:
    # X_k = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]  (= C[0,k])
    # Y_k = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]  (= C[1,k])
    # These are independent, so no sharing between i=0 and i=1 for same k.
    
    # But we can share across k for same i,j.
    # A[i,j]*B[j,k] for fixed i,j varies with k.
    
    # Let's try a different approach: use known results.
    # For M_{2x3} * M_{3x7}, the border rank and tensor rank are studied.
    # The trivial bound is 42. Can we do better?
    
    # Try a constructive approach: group the 42 terms.
    # For each j (3 choices), and each (i,k) pair (14 choices), we have one term.
    # Total 42 terms.
    
    # Idea: For fixed j, the 14 terms A[i,j]*B[j,k] for i in {0,1}, k in {0..6}
    # form a 2x7 matrix outer product structure. This has rank 14 as a tensor
    # in the (i,k) space, but we need to combine across j.
    
    # Let's try a randomized search with better initialization:
    # Start from schoolbook and try to merge columns.
    
    def try_merge(schoolbook_U, schoolbook_V, schoolbook_W, n, m, p):
        """Try to merge pairs of schoolbook columns to reduce rank."""
        rank = n * m * p
        U = [row[:] for row in schoolbook_U]
        V = [row[:] for row in schoolbook_V]
        W = [row[:] for row in schoolbook_W]
        
        # Try merging column c1 and c2 by finding weights
        # If we can set U[a][c2] = alpha * U[a][c1] for all a,
        # V[b][c2] = alpha * V[b][c1] for all b,
        # W[c][c2] = alpha * W[c][c1] for all c,
        # then column c2 is redundant.
        # More generally, we can try to express one column as a linear combo.
        
        # Simpler: try to find if any two columns are "compatible" for merging
        # with a single weight.
        
        # Actually, let's try a different approach: 
        # Try rank r by selecting r columns from the 42 and seeing if we can
        # adjust them to satisfy the constraints.
        
        pass
    
    # Approach 2: Direct constructive search for specific rank values
    # Use the fact that the problem has small dimensions.
    
    # Let's try to construct a rank-36 decomposition (saving 6).
    # Idea: For each j, instead of 14 separate terms, use 12 by sharing.
    # This doesn't quite work directly.
    
    # Let's try a more systematic approach: use integer linear algebra.
    # The constraints are:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For invalid (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # This is a system of polynomial equations. Let's try to solve it for small rank.
    
    # Given the time constraints, let's try a focused random search with 
    # structured initialization for ranks 36-41.
    
    def structured_random_search(rank, max_iters=200, seed=42):
        """Random search with structured initialization."""
        random.seed(seed)
        entries = [-1, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 1]
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None
            
            # Initialize with schoolbook-like structure but with some zeros
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Assign each of the 42 terms to one of the rank columns
            # with some probability of sharing
            assignments = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Choose a column for this term
                        col = random.randint(0, rank - 1)
                        assignments.append((i, j, k, col))
            
            # Now, for each assignment, set the appropriate entries
            # But if multiple terms map to the same column, we need to check consistency
            for (i, j, k, col) in assignments:
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # We need U[a][col] * V[b][col] * W[c][col] to contribute 1
                # to the sum for this term, and 0 to all other terms.
                # This is tricky. Let's just try random values.
                U[a][col] = random.choice(entries)
                V[b][col] = random.choice(entries)
                W[c][col] = random.choice(entries)
            
            # Add some random noise to other entries
            for a in range(n * m):
                for r in range(rank):
                    if random.random() < 0.1:
                        U[a][r] = random.choice(entries)
            for b in range(m * p):
                for r in range(rank):
                    if random.random() < 0.1:
                        V[b][r] = random.choice(entries)
            for c in range(p * n):
                for r in range(rank):
                    if random.random() < 0.1:
                        W[c][r] = random.choice(entries)
            
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try ranks from 41 down to 30
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10.0:
            break
        result = structured_random_search(r, max_iters=100, seed=r * 7919)
        if result is not None:
            best = result
            best_rank = r
            break

    # Also try a pure random search for the best rank found so far
    def pure_random_search(rank, max_iters=500, seed=42):
        random.seed(seed)
        entries = [-1, -0.5, 0, 0, 0, 0, 0, 0.5, 1]
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None
            U = [[random.choice(entries) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(entries) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(entries) for _ in range(rank)] for _ in range(p * n)]
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        return None

    # If we found something better, try to go even lower
    if best_rank < n * m * p:
        for r in range(best_rank - 1, 20, -1):
            if time.time() > deadline - 10.0:
                break
            result = pure_random_search(r, max_iters=200, seed=r * 104729)
            if result is not None:
                best = result
                best_rank = r
                break

    # Final verification
    if verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    else:
        # Fallback to schoolbook
        return schoolbook()

# EVOLVE-BLOCK-END
