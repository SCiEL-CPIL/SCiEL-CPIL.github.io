import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a tensor: T[i,j,k] should equal 1 for all valid (i,j,k) since
    # each triple contributes exactly once to the product.
    # Actually, the tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where the coefficient is 1 for all (i,j,k) in the range.
    # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_block_decomposition(n, m, p):
    """Try block-based decomposition that might share computation."""
    best = None
    best_rank = n * m * p
    
    # Try splitting n into blocks
    for n1 in range(1, n):
        n2 = n - n1
        # Rank contribution: n1*m*p + n2*m*p = n*m*p (no gain)
        # But if we can share, maybe less
        # Try: compute C1 = A1*B and C2 = A2*B separately
        # This is still n*m*p, no gain unless we exploit structure
        
    # Try splitting m into blocks
    for m1 in range(1, m):
        m2 = m - m1
        # A[:, :m1] * B[:m1, :] + A[:, m1:] * B[m1:, :]
        # Rank: n*m1*p + n*m2*p = n*m*p (no gain)
        
    # Try a different approach: exploit that p = 5 and n = 4
    # Maybe use the fact that 4 = 2+2 or 5 = 3+2
    
    # Try rank = n*p + m*p - something (Strassen-like)
    # For 2x2: rank 7 instead of 8
    # For 4x5*5x5, try various structured patterns
    
    # Pattern: treat as (2+2)x5 * 5x5
    # Block A = [[A11, A12], [A21, A22]] where each is 2x5
    # C = [[A11*B, A12*B], [A21*B, A22*B]]
    # Each 2x5*5x5 costs 2*5*5 = 50, total 200 (worse)
    
    # Better: try to find if there's a pattern with rank < 100
    # The lower bound for 4x5x5 is at least max(n*p, m*p, n*m) = max(20, 25, 20) = 25
    # But practical lower bounds are higher
    
    # Try a random search for small rank reductions
    # Start from schoolbook and try to merge terms
    
    return best

def _greedy_merge(U, V, W, n, m, p, max_iterations=100):
    """Try to merge pairs of terms to reduce rank."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for iteration in range(max_iterations):
        merged = False
        for a in range(rank):
            for b in range(a + 1, rank):
                # Try to see if term a and term b can be merged
                # by finding a linear combination that still works
                # This is a heuristic - check if removing both and adding
                # a single term works
                
                # Check if the sum of contributions from a and b
                # can be represented by a single term
                # This requires solving a system, which is expensive
                # Skip for now - just try simple cases
                
                # Simple case: if U[:,a] == U[:,b] and V[:,a] == V[:,b]
                # then we can merge W terms
                if np.allclose(U[:, a], U[:, b]) and np.allclose(V[:, a], V[:, b]):
                    # Merge: W[:,a] += W[:,b], remove column b
                    W[:, a] += W[:, b]
                    U = np.delete(U, b, axis=1)
                    V = np.delete(V, b, axis=1)
                    W = np.delete(W, b, axis=1)
                    rank -= 1
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try greedy merging
    if time.time() < deadline - 30:
        U_m, V_m, W_m = _greedy_merge(U, V, W, n, m, p)
        if U_m.shape[1] < best_rank:
            if _verify(U_m, V_m, W_m, n, m, p):
                best_U, best_V, best_W = U_m, V_m, W_m
                best_rank = U_m.shape[1]
    
    # Try random perturbations and local search
    if time.time() < deadline - 20:
        random.seed(42)
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try a slightly perturbed version of schoolbook
            # and see if we can find a lower rank solution
            
            # Alternative: try specific known patterns
            # For 4x5*5x5, try rank = 95, 90, 85, etc.
            # by introducing small perturbations and checking
            
            # This is unlikely to work with random perturbations
            # since we need exact integer/half-integer solutions
            pass
    
    # Try a structured approach: exploit the 5x5 substructure
    if time.time() < deadline - 15:
        # The 5x5 part might have special structure
        # Try to decompose the 5x5 multiplication part with lower rank
        # and combine with the 4x5 part
        
        # For 5x5*5x5, schoolbook is 125, but we only need 5x5 output
        # for each of the 4 rows, so total is 4*125 = 500? No.
        
        # Actually, the full tensor is 4*5*5 = 100 terms.
        # Each term corresponds to one (i,j,k) triple.
        
        # Try: can we share computation between different i values?
        # If two rows of A are identical, we could share, but in general they're not.
        
        # For exact decomposition, we need to handle arbitrary A and B,
        # so we can't assume any structure in A or B.
        
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
