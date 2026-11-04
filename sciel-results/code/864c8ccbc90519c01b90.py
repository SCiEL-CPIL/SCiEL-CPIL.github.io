import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    # Schoolbook decomposition (baseline)
    rank_sb = n * m * p  # 100
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W form a valid decomposition."""
        r = U.shape[1]
        # Check all valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1.0) > tol:
                        return False
        # Check some invalid entries (sampled)
        # For a full check we'd need all (a,b,c) triples, but that's n*m * m*p * p*n = 100*25*20 = 50000
        # Let's check a subset
        for a in range(0, n*m, 3):
            for b in range(0, m*p, 5):
                for c in range(0, p*n, 7):
                    # Check if this is a valid triple
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j != j2 or k != k2 or i != i2:
                        s = 0.0
                        for rr in range(r):
                            s += U[a, rr] * V[b, rr] * W[c, rr]
                        if abs(s) > tol:
                            return False
        return True
    
    def full_verify(U, V, W, tol=1e-10):
        """Full verification of all entries."""
        r = U.shape[1]
        # Check all valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1.0) > tol:
                        return False
        # Check all invalid entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        continue  # valid, already checked
                    s = 0.0
                    for rr in range(r):
                        s += U[a, rr] * V[b, rr] * W[c, rr]
                    if abs(s) > tol:
                        return False
        return True
    
    # Strategy 1: Try to find a decomposition with rank close to known bounds
    # Theoretical lower bound for 4x5x5: at least max(4*5, 5*5, 5*4) = 25
    # Strassen-like constructions might give better than 100
    
    # Strategy 2: Use a structured search with small integer/half-integer values
    # Try ranks from a reasonable lower bound up to schoolbook
    
    # First, let's try a greedy/local optimization approach
    # Start from schoolbook and try to merge/eliminate terms
    
    # Actually, let's try a different approach:
    # Use the fact that we can combine Strassen blocks
    
    # For 4x5x5, we can view it as combining smaller multiplications
    # Let's try a block structure
    
    # Try ranks in a range, using random initialization + local search
    min_rank_to_try = 60  # Start from a reasonable lower bound
    max_attempts_per_rank = 2
    
    for target_rank in range(min_rank_to_try, rank_sb + 1):
        if time_left() < 5:
            break
        if target_rank >= best_rank:
            continue
        
        for attempt in range(max_attempts_per_rank):
            if time_left() < 3:
                break
            
            # Initialize with random values, then try to optimize
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Use a simple gradient-free optimization
            # Minimize the error over all (a,b,c) triples
            best_err = float('inf')
            for step in range(50):
                if time_left() < 3:
                    break
                    
                # Compute error
                err = 0.0
                # Valid entries
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(target_rank):
                                s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            err += (s - 1.0) ** 2
                # Sample invalid entries
                for a in range(0, n*m, 4):
                    for b in range(0, m*p, 4):
                        for c in range(0, p*n, 4):
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            k2 = c // n
                            i2 = c % n
                            if j != j2 or k != k2 or i != i2:
                                s = 0.0
                                for rr in range(target_rank):
                                    s += U[a, rr] * V[b, rr] * W[c, rr]
                                err += s ** 2
                
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        break
                
                # Random perturbation
                U += np.random.randn(n*m, target_rank) * 0.1
                V += np.random.randn(m*p, target_rank) * 0.1
                W += np.random.randn(p*n, target_rank) * 0.1
            
            if best_err < 1e-10:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
    
    # Final: ensure we have a valid solution
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
