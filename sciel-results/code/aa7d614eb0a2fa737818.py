import numpy as np
import time
import sys
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    start_time = time.time()
    time_limit = 135.0  # Leave 15s margin before 150s kill
    
    def check_decomposition_exact(U, V, W, rank):
        """Check if decomposition is exact using Fraction arithmetic."""
        U_frac = [[Fraction(U[i, t]) for t in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, t]) for t in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W[i, t]) for t in range(rank)] for i in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for t in range(rank):
                        val += U_frac[i*m+j][t] * V_frac[j*p+k][t] * W_frac[k*n+i][t]
                    if val != 1:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank 72."""
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
        
        return {"U": U, "V": V, "W": W}, rank
    
    def try_merge_terms(U, V, W, rank):
        """
        Try to merge pairs of terms in the decomposition to reduce rank.
        Returns a new decomposition if successful.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try all pairs of columns to see if they can be merged
        for t1 in range(rank):
            if time.time() - start_time > time_limit:
                return None
            for t2 in range(t1 + 1, rank):
                # Check if merging these two terms is possible
                # We need to find if there exist values such that the combined
                # term still satisfies all constraints
                
                # For each triple, we have:
                # U[i*m+j, t1]*V[j*p+k, t1]*W[k*n+i, t1] + 
                # U[i*m+j, t2]*V[j*p+k, t2]*W[k*n+i, t2] = 1
                # for all (i,j,k) where both terms contribute
                
                # In schoolbook, each term only contributes to one triple
                # So merging is only possible if the two terms contribute to
                # the same triple, which they don't in schoolbook.
                # 
                # Instead, let's try a different approach: look for a rank
                # reduction by finding a linear dependency.
                
                # Try to express one column as a combination of others
                # This is complex, so let's try a simpler approach:
                # Randomly perturb and see if we can reduce rank
                
                pass
        
        return None
    
    def structured_search(target_rank):
        """
        Structured search for a decomposition of target_rank.
        Uses the fact that we can try to find decompositions with
        specific patterns (e.g., some entries are 0, ±1, ±1/2).
        """
        if time.time() - start_time > time_limit:
            return None
        
        np.random.seed(42)
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try random initializations with structured patterns
        for attempt in range(20000):
            if time.time() - start_time > time_limit:
                return None
            
            # Generate random entries from a small set of half-integers
            scale = 0.5
            U = (np.random.randint(0, 7, size=(n*m, target_rank)) - 3) * scale
            V = (np.random.randint(0, 7, size=(m*p, target_rank)) - 3) * scale
            W = (np.random.randint(0, 7, size=(p*n, target_rank)) - 3) * scale
            
            # Quick float check
            valid = True
            for (i, j, k) in triples:
                val = 0.0
                for t in range(target_rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Exact verification
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                if check_decomposition_exact(U_arr, V_arr, W_arr, target_rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    def local_search_from_schoolbook():
        """
        Start from schoolbook and try to reduce rank via local moves.
        """
        if time.time() - start_time > time_limit:
            return None
        
        U, V, W, rank = schoolbook_decomposition()
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try to eliminate one term at a time by adjusting others
        for target_rank in range(rank - 1, max(rank - 15, 50), -1):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a valid decomposition of target_rank
            # by searching in a neighborhood of the schoolbook solution
            result = structured_search(target_rank)
            if result is not None:
                return result
        
        return None
    
    # Try structured search for various ranks
    for target_rank in [66, 67, 68, 69, 70, 71]:
        if time.time() - start_time > time_limit:
            break
        
        result = structured_search(target_rank)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook_decomposition()[0]
