"""Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # Schoolbook baseline: rank = n*m*p = 72
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # For each i,j,k check sum over r
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using tensor operations."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        # For each (i,j,k): sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # We can compute this as: for each r, the outer product contributes
        # Let's compute element-wise
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured low-rank decompositions
    # Strategy: try various rank values and search for valid decompositions
    
    # First, try to find if rank < 72 is possible
    # Use a greedy/structured approach
    
    # Try rank values from known lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But that's a very loose bound. For matrix mult, rank >= max(nm, mp, pn)
    
    # Let's try a systematic search for small improvements first
    # Then try more aggressive searches
    
    # Approach 1: Try to find a decomposition with rank slightly below 72
    # by merging some schoolbook terms
    
    # Approach 2: Use a random search with structured initialization
    
    # Let's try a pattern-based approach:
    # The key constraint is that for each (i,j,k), the sum over r must equal 1.
    
    # Try rank values from 60 down to see if we can find something
    # But random search in high dimensions is hard.
    
    # Better: use the fact that we can try to "compress" the schoolbook
    # by finding linear dependencies.
    
    # For now, let's try a few structured patterns
    
    # Pattern: try rank = 60, 54, 48, etc. with random initialization + verification
    # This is unlikely to work quickly, but let's try a few quick attempts
    
    # Actually, for this small case, let's try a more clever approach:
    # Use the fact that the tensor has structure and try to find a low-rank
    # approximation that is exact.
    
    # Try: can we do better than 72? The answer is likely yes.
    # Known: rank of 3x3x3 matmul is 23 (Strassen-like), 4x4x4 is ~34
    # For 3x4x6, let's estimate: maybe around 50-60?
    
    # Let's try a randomized search for ranks below 72
    # Use random matrices and check if they can be "fixed" to be exact
    
    # Actually, the cleanest approach for exact decomposition:
    # We need integer/half-integer entries. Let's search over small integer grids.
    
    # For a given rank r, we need to find U, V, W with entries in {0, 1/2, 1, 3/2, 2, ...}
    # such that the tensor equation holds.
    
    # This is a constraint satisfaction problem. For small r, we can try:
    # 1. Random initialization with small integers
    # 2. Check if valid
    # 3. If not, try to fix violations
    
    # Let's try a simple random search for ranks from 60 down to 40
    random.seed(42)
    
    # Try a few quick random attempts for various ranks
    for target_rank in range(68, 40, -4):
        if time.time() > deadline - 20:
            break
        
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small integers
            U = np.random.randint(0, 3, size=(nm, target_rank))
            V = np.random.randint(0, 3, size=(mp, target_rank))
            W = np.random.randint(0, 3, size=(pn, target_rank))
            
            if verify_fast(U, V, W):
                best_rank = target_rank
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    # Try a more structured approach: start from schoolbook and try to merge
    if time.time() < deadline - 30:
        # Try to find if we can reduce rank by finding linear combinations
        # that preserve the tensor equation
        
        # Schoolbook has rank 72. Can we express some columns as linear combos?
        # This requires the columns to be linearly dependent in a way that
        # preserves the bilinear form.
        
        # For now, let's try a different structured pattern:
        # Group terms by i and try to factor
        
        # Try rank = 64 with a structured pattern
        for target_rank in [64, 60, 56, 52, 48]:
            if time.time() > deadline - 20:
                break
            
            # Try a structured initialization:
            # Use blocks corresponding to (i, j) pairs for U, etc.
            # but with some sharing
            
            # Simple structured attempt:
            # For each i, use a shared set of r_i columns
            # Total rank = sum_i r_i where r_i >= m*p/n or something
            
            # Let's just try random with more attempts for these ranks
            for attempt in range(100):
                if time.time() > deadline - 20:
                    break
                
                U = np.random.randint(0, 4, size=(nm, target_rank))
                V = np.random.randint(0, 4, size=(mp, target_rank))
                W = np.random.randint(0, 4, size=(pn, target_rank))
                
                if verify_fast(U, V, W):
                    best_rank = target_rank
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    # Final: try a few more structured patterns if time permits
    if time.time() < deadline - 10:
        # Try half-integer entries
        for target_rank in range(60, 40, -4):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Use half-integers: multiply by 2 to work with integers
                U = np.random.randint(0, 6, size=(nm, target_rank))
                V = np.random.randint(0, 6, size=(mp, target_rank))
                W = np.random.randint(0, 6, size=(pn, target_rank))
                
                # Check with scaling: sum U*V*W should be 8 (since 2*2*2=8)
                Ur = U.reshape(n, m, target_rank)
                Vr = V.reshape(m, p, target_rank)
                Wr = W.reshape(p, n, target_rank)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:])
                            if abs(s - 8.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert back to half-integers
                    best_rank = target_rank
                    best = {
                        "U": (U / 2.0).tolist(),
                        "V": (V / 2.0).tolist(),
                        "W": (W / 2.0).tolist()
                    }
                    break
    
    return best
