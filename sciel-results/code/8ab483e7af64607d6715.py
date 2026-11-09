import time
import random
import numpy as np

def is_latin(sq):
    n = len(sq)
    sq_np = np.array(sq)
    for i in range(n):
        if len(np.unique(sq_np[i])) != n:
            return False
        if len(np.unique(sq_np[:, i])) != n:
            return False
    return True

def is_orthogonal(sq1, sq2):
    sq1_np = np.array(sq1)
    sq2_np = np.array(sq2)
    pairs = sq1_np * 10 + sq2_np
    return len(np.unique(pairs)) == 100

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50
    
    # Base Latin Square
    base = [[(i+j)%n for j in range(n)] for i in range(n)]
    
    # Try to find a second square
    # Generate random Latin squares by permuting rows/cols of base
    # and check orthogonality
    
    best_result = [base]
    
    # Try random permutations
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        # Random row and column permutations
        row_perm = list(range(n))
        col_perm = list(range(n))
        random.shuffle(row_perm)
        random.shuffle(col_perm)
        
        # Apply to base
        cand = [[base[row_perm[i]][col_perm[j]] for j in range(n)] for i in range(n)]
        
        if is_orthogonal(base, cand):
            best_result = [base, cand]
            break
            
    return best_result
