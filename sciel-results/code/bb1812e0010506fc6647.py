"""
Matrix multiplication 3x4 by 4x5: rank-minimal exact tensor decomposition.

Strategy:
- The bilinear map (A, B) -> A*B has 3*4*5 = 60 trilinear coefficients.
- A rank-r decomposition uses r terms, each with entries in U (12 x r), V (20 x r), W (15 x r).
- The constraints are: for each (i, j, k), sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.

- We search for the minimum r such that there exist integer (or half-integer) U, V, W satisfying these.
- For 3x4x5, the schoolbook is 60. We try to find a decomposition of rank < 60.

Approach:
1. Start with schoolbook (rank 60) as fallback.
2. Try to find lower rank using structured search with small integer coefficients.
3. Use a greedy algorithm: iteratively pick a term (column) that satisfies the most remaining constraints.
4. Verify exactness with integer arithmetic.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
DEADLINE = 140  # seconds (safety margin before 150s limit)
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U and U[0] else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[0]) != rank:
            return False
    
    # Check all (i,j,k) equations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _try_greedy(n, m, p, rank, max_attempts=100):
    """
    Try to find a rank-r decomposition using a greedy approach.
    We build the decomposition column by column, trying to satisfy as many (i,j,k) equations as possible.
    """
    for attempt in range(max_attempts):
        if _time_left() < 5:
            return None
        
        # Initialize with zeros
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Track which (i,j,k) equations are still unsatisfied
        # eq_status[i][j][k] = current sum for this equation
        eq_status = [[[0 for k in range(p)] for j in range(m)] for i in range(n)]
        unsatisfied = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    unsatisfied.add((i, j, k))
        
        # For each column, try to find a combination of U, V, W entries that helps satisfy equations
        for c in range(rank):
            if _time_left() < 2:
                return None
            
            if not unsatisfied:
                break
            
            # Try random small integer values for this column
            best_combo = None
            best_score = -1
            best_eq_satisfied = 0
            
            # Sample a few random combinations
            for trial in range(20):
                if _time_left() < 1:
                    break
                
                # Randomly choose which (i,j,k) to target
                target_ijs = list(unsatisfied)
                random.shuffle(target_ijs)
                
                # For this column, pick a pattern
                # We want U[i*m+j], V[j*p+k], W[k*n+i] to be non-zero for some (i,j,k)
                # and the product to contribute to satisfying equations
                
                # Simple heuristic: pick a random (i,j,k) and set the three entries to 1
                if not target_ijs:
                    break
                    
                i, j, k = target_ijs[0]
                
                # Try setting U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1
                # This adds 1 to eq_status[i][j][k]
                
                # Check if this is useful (i.e., doesn't overshoot)
                if eq_status[i][j][k] < 1:
                    # This would help
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    eq_status[i][j][k] += 1
                    if eq_status[i][j][k] == 1:
                        unsatisfied.discard((i, j, k))
                    break
            
            # If no useful column found, try a different approach
            # Set random small values
            if U[i*m+j][c] == 0 and V[j*p+k][c] == 0 and W[k*n+i][c] == 0:
                # Pick a random unsatisfied equation and try to help it
                if unsatisfied:
                    i, j, k = random.choice(list(unsatisfied))
                    # Try different sign combinations
                    for sign in [1, -1]:
                        if eq_status[i][j][k] + sign == 1:
                            U[i*m+j][c] = sign
                            V[j*p+k][c] = 1
                            W[k*n+i][c] = 1
                            eq_status[i][j][k] += sign
                            if eq_status[i][j][k] == 1:
                                unsatisfied.discard((i, j, k))
                            break
        
        # Check if all equations are satisfied
        if not unsatisfied:
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured(n, m, p, rank):
    """
    Try a more structured approach: use known patterns for matrix multiplication.
    For example, we can try to group terms by output position (i,k).
    """
    if _time_left() < 5:
        return None
    
    # For each output position (i,k), we need:
    # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # But the standard decomposition requires:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k)
    
    # Let's try a different approach: use the fact that for fixed (i,k),
    # the sum over j should give 1 for the correct terms.
    
    # Actually, let's just try a simpler heuristic:
    # Assign each (i,j,k) to a unique column, but try to share columns when possible.
    
    # For now, this is just the schoolbook approach with rank = n*m*p
    # We need a better method to reduce rank.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = 20
    # But in practice, it's likely higher. Let's try from 30 up to 59.
    
    for r in range(30, rank_schoolbook):
        if _time_left() < 10:
            break
        
        # Try greedy approach
        result = _try_greedy(n, m, p, r, max_attempts=50)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_rank = r
                best_result = result
                break
        
        # Try structured approach
        result = _try_structured(n, m, p, r)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_rank = r
                best_result = result
                break
    
    return best_result
