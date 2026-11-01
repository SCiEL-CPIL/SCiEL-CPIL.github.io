"""
Combined program for matrix multiplication 2x5x5 rank minimization.
Keeps the schoolbook decomposition as a guaranteed-valid fallback (rank 50)
and adds a bounded, structured search for lower-rank exact decompositions
using integer/half-integer entries, with a hard deadline guard.
"""
import time
import random
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Uses structured search starting from schoolbook, trying to merge terms
    and find lower-rank exact decompositions with integer/half-integer entries.
    Falls back to schoolbook rank 50 if no improvement found within deadline.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the schoolbook decomposition as guaranteed fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    fallback_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    # Check if we have time to search
    if time.time() > deadline - 5.0:
        return fallback_result
    
    # Validate a decomposition
    def _validate(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by merging pairs of schoolbook terms
    # Two terms (c1, c2) can be merged if there exist scalars a,b such that
    # for all (i,j,k): a*U1[i,j]*V1[j,k]*W1[k,i] + b*U2[i,j]*V2[j,k]*W2[k,i] 
    # contributes correctly. This is complex; instead try direct rank reduction.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Strategy 1: Try ranks from 49 down, using structured initialization
    # based on modifying the schoolbook solution
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 15), -1):
        if time.time() > deadline - 5.0:
            break
        
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Start from schoolbook and try to eliminate one term by adjusting others
            # or use a mixed approach
            
            # Approach: Use a dense set of values and try to solve
            values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Initialize with a mix: some terms from schoolbook pattern, some random
            # Keep first (target_rank - 5) terms close to schoolbook structure
            base_terms = min(target_rank - 3, schoolbook_rank)
            
            for c in range(base_terms):
                # Map column c to a schoolbook-like pattern
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n * m)) % p
                if i_idx < n and j_idx < m and k_idx < p:
                    U[i_idx * m + j_idx][c] = 1.0
                    V[j_idx * p + k_idx][c] = 1.0
                    W[k_idx * n + i_idx][c] = 1.0
            
            # Randomly perturb the remaining columns and some existing ones
            for c in range(target_rank):
                if random.random() < 0.3:
                    # Randomize this column
                    for row in range(n * m):
                        if random.random() < 0.5:
                            U[row][c] = random.choice(values)
                    for row in range(m * p):
                        if random.random() < 0.5:
                            V[row][c] = random.choice(values)
                    for row in range(p * n):
                        if random.random() < 0.5:
                            W[row][c] = random.choice(values)
            
            if _validate(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_result is not None:
            if time.time() > deadline - 10.0:
                break
    
    # Strategy 2: Try a completely different structured approach
    # For 2x5x5, exploit the small n=2 dimension
    # Each term c has: U[i,j,c] for i in {0,1}, j in {0..4}
    #                    V[j,k,c] for j in {0..4}, k in {0..4}  
    #                    W[k,i,c] for k in {0..4}, i in {0,1}
    
    if best_result is None and time.time() < deadline - 10.0:
        # Try ranks with a focus on the structure
        for target_rank in range(48, 40, -1):
            if time.time() > deadline - 5.0:
                break
            
            for attempt in range(30):
                if time.time() > deadline - 5.0:
                    break
                
                values = [-1.0, -0.5, 0.0, 0.5, 1.0]
                
                U = [[0.0] * target_rank for _ in range(n * m)]
                V = [[0.0] * target_rank for _ in range(m * p)]
                W = [[0.0] * target_rank for _ in range(p * n)]
                
                # Dense initialization with low density
                for c in range(target_rank):
                    # Each term should contribute to multiple (i,j,k) triples
                    # U row: i*m+j, V row: j*p+k, W row: k*n+i
                    u_rows = random.sample(range(n*m), k=min(3, n*m))
                    v_rows = random.sample(range(m*p), k=min(3, m*p))
                    w_rows = random.sample(range(p*n), k=min(3, p*n))
                    
                    for r in u_rows:
                        U[r][c] = random.choice(values)
                    for r in v_rows:
                        V[r][c] = random.choice(values)
                    for r in w_rows:
                        W[r][c] = random.choice(values)
                
                if _validate(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            
            if best_result is not None:
                if time.time() > deadline - 10.0:
                    break
    
    if best_result is not None:
        return best_result
    
    return fallback_result
# EVOLVE-BLOCK-END
