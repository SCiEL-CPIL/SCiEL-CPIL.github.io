"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

# We work with the matrix multiplication tensor for C = A * B
# A is n x m, B is m x p, C is n x p
# The tensor T has indices (i, j, k) for A[i,j], B[j,k], C[i,k]
# Rank decomposition: T = sum_{r=1}^{R} U[:,r] * V[:,r] * W[:,r]
# where U is (n*m) x R, V is (m*p) x R, W is (p*n) x R
# The trace convention for W is W[k*n+i, r]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    For each output element C[i,k] = sum_j A[i,j] * B[j,k]
    In the tensor decomposition, the coefficient of A[i,j]*B[j,k] in C[i,k] must be 1.
    
    The decomposition gives: C[i,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    For this to equal sum_j A[i,j]*B[j,k], we need:
    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta_{i,j,k} 
    where delta_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k] (always true)
    Actually, the tensor T_{i,j,k} = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]B[j,k]
    
    So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other combinations (which don't exist in the tensor), it should be 0.
    But since we only have these indices, we just check all (i,j,k) triples.
    """
    R = len(U[0]) if len(U) > 0 else 0
    # Check all (i, j, k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Faster verification using numpy.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    R = U.shape[1]
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.array(U[i*m+j]), np.array(V[j*p+k]) * np.array(W[k*n+i]))
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_greedy_rank_reduction(n, m, p, max_rank, time_limit, rng):
    """
    Try to find a lower-rank decomposition using a greedy approach.
    Start from schoolbook and try to merge/remove terms.
    """
    deadline = time.time() + time_limit
    best = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try random perturbations and local searches
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Start with a random subset of terms and try to complete
        current_rank = rng.randint(max(30, best_rank // 2), min(best_rank - 1, max_rank))
        
        U = np.zeros((n*m, current_rank), dtype=float)
        V = np.zeros((m*p, current_rank), dtype=float)
        W = np.zeros((p*n, current_rank), dtype=float)
        
        # Initialize with random values
        for r in range(current_rank):
            U[:, r] = rng.uniform(-1, 1, n*m)
            V[:, r] = rng.uniform(-1, 1, m*p)
            W[:, r] = rng.uniform(-1, 1, p*n)
        
        # Try to solve for the tensor entries
        # This is a non-convex problem, so we use a simple iterative approach
        # For each (i,j,k), we want sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Use a simple gradient-free approach: try to adjust one factor at a time
        improved = True
        max_iter = 50
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Check current error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = abs(s - 1)
                        max_err = max(max_err, err)
            
            if max_err < 1e-6:
                # Convert to exact fractions
                U_frac = [[Fraction(int(round(x*1000000)), 1000000) if abs(x) > 1e-10 else 0 for x in row] for row in U]
                V_frac = [[Fraction(int(round(x*1000000)), 1000000) if abs(x) > 1e-10 else 0 for x in row] for row in V]
                W_frac = [[Fraction(int(round(x*1000000)), 1000000) if abs(x) > 1e-10 else 0 for x in row] for row in W]
                
                # Verify with exact arithmetic
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r] for r in range(current_rank))
                            if s != 1:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best = {"U": U_frac, "V": V_frac, "W": W_frac}
                break
            
            # Gradient step: for each (i,j,k) with error, adjust factors
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = s - 1
                        if abs(err) > 1e-8:
                            # Adjust W[k*n+i] slightly
                            grad_W = U[i*m+j] * V[j*p+k]
                            W[k*n+i] -= 0.01 * err * grad_W / (np.dot(grad_W, grad_W) + 1e-10)
                            break
                    break
                break
        
        # Check if we got a valid solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    max_err = max(max_err, abs(s - 1))
        
        if max_err < 1e-6:
            # Try to get exact solution
            U_frac = [[Fraction(int(round(x*10000000)), 10000000) if abs(x) > 1e-10 else 0 for x in row] for row in U]
            V_frac = [[Fraction(int(round(x*10000000)), 10000000) if abs(x) > 1e-10 else 0 for x in row] for row in V]
            W_frac = [[Fraction(int(round(x*10000000)), 10000000) if abs(x) > 1e-10 else 0 for x in row] for row in W]
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r] for r in range(current_rank))
                        if s != 1:
                            valid = False
                            break
                if not valid:
                    break
            if valid and current_rank < best_rank:
                best_rank = current_rank
                best = {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return best

def _try_kronecker_substitution(n, m, p, time_limit, rng):
    """
    Try Kronecker substitution approach: embed the problem into a larger field
    and use polynomial identity testing.
    """
    deadline = time.time() + time_limit
    
    # For small dimensions, try to find structured decompositions
    # The key insight: if we can find a low-rank decomposition over F_2 or another field,
    # we can lift it to the rationals.
    
    # Try to find a decomposition using random linear combinations
    # Start with the schoolbook and try to combine terms
    
    # For 3x5x7, the schoolbook rank is 105.
    # Known lower bounds: max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # Upper bound: n*m*p = 105
    
    # Try random projections
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Try rank R
        for R in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]:
            if time.time() > deadline:
                break
            
            # Random initialization
            U = rng.uniform(-1, 1, (n*m, R))
            V = rng.uniform(-1, 1, (m*p, R))
            W = rng.uniform(-1, 1, (p*n, R))
            
            # Try to minimize the error
            # Use a simple iterative method
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Compute errors
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            errors[(i,j,k)] = s - 1
                
                max_err = max(abs(e) for e in errors.values())
                if max_err < 1e-5:
                    # Try to get exact solution
                    break
                
                # Gradient descent step
                lr = 0.001
                for (i,j,k), err in errors.items():
                    if abs(err) > 1e-6:
                        # Update W
                        gW = U[i*m+j] * V[j*p+k]
                        W[k*n+i] -= lr * err * gW
                        # Update U
                        gU = V[j*p+k] * W[k*n+i]
                        U[i*m+j] -= lr * err * gU
                        # Update V
                        gV = U[i*m+j] * W[k*n+i]
                        V[j*p+k] -= lr * err * gV
            
            # Check final error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-5:
                # Try to round to exact values
                U_r = np.round(U * 1000000) / 1000000
                V_r = np.round(V * 1000000) / 1000000
                W_r = np.round(W * 1000000) / 1000000
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                            if abs(s - 1) > 1e-4:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Convert to fractions
                    U_frac = [[Fraction(int(round(x*1000000)), 1000000) if abs(x) > 1e-10 else 0 for x in row] for row in U_r]
                    V_frac = [[Fraction(int(round(x*1000000)), 1000000) if abs(x) > 1e-10 else 0 for x in row] for row in V_r]
                    W_frac = [[Fraction(int(round(x*1000000)), 1000000) if abs(x) > 1e-10 else 0 for x in row] for row in W_r]
                    
                    # Final exact verification
                    valid_exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r] for r in range(R))
                                if s != 1:
                                    valid_exact = False
                                    break
                        if not valid_exact:
                            break
                    
                    if valid_exact:
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook as fallback
    best = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = random.Random(42)
    
    # Try Kronecker substitution approach
    if time.time() < deadline - 30:
        result = _try_kronecker_substitution(n, m, p, deadline - time.time() - 10, rng)
        if result is not None:
            best = result
            best_rank = len(result["U"][0]) if len(result["U"]) > 0 else best_rank
    
    # Try greedy rank reduction
    if time.time() < deadline - 10:
        result = _try_greedy_rank_reduction(n, m, p, best_rank, deadline - time.time() - 5, rng)
        if result is not None:
            r = len(result["U"][0]) if len(result["U"]) > 0 else float('inf')
            if r < best_rank:
                best = result
                best_rank = r
    
    # Final safety check
    if time.time() > deadline - 2:
        return best
    
    return best
# EVOLVE-BLOCK-END
