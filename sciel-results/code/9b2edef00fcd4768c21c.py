import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # For this specific task, q=3, n=6, radius=1
    # But we write it generally enough to handle the specific case efficiently.
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (coverage sets)
    # ball[i] is a list of indices of words within distance 'radius' of word i
    balls = []
    for i, w in enumerate(all_words):
        ball = []
        for j, v in enumerate(all_words):
            # Calculate Hamming distance
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                ball.append(j)
        balls.append(ball)
        
        # Time check during precomputation (should be fast, but just in case)
        if time.time() > deadline:
             # Fallback to simple construction if precomputation is too slow (unlikely for n=6)
             return _fallback_construct(q, n, radius)

    # Formulate ILP
    # Minimize sum(x_i)
    c = np.ones(num_words)
    
    # Constraints: For every word i, sum of x_j for j in ball(i) >= 1
    # This is equivalent to: A x >= 1, where A is the incidence matrix.
    # A[i, j] = 1 if j is in ball(i), else 0.
    
    # We can construct the linear constraint matrix A.
    # Since A is symmetric (distance is symmetric), A[i,j] = 1 iff dist(i,j) <= radius.
    # We can build A efficiently.
    A = np.zeros((num_words, num_words), dtype=int)
    for i in range(num_words):
        for j in balls[i]:
            A[i, j] = 1
            
    # Lower bound for constraints is 1, upper bound is infinity
    lb = np.ones(num_words)
    ub = np.full(num_words, np.inf)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(num_words)
    
    # Bounds for variables
    var_bounds = Bounds(0, 1)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={
                'time_limit': 120, # Hard limit inside solver
                'mip_rel_gap': 1e-6,
                'verbose': 0
            }
        )
    except Exception:
        return _fallback_construct(q, n, radius)
    
    if res.status == 0: # Optimal
        x = res.x
    elif res.status == 1: # Feasible (time limit reached)
        x = res.x
    else:
        return _fallback_construct(q, n, radius)
        
    # Extract codewords
    codewords = []
    for i in range(num_words):
        if x[i] > 0.5: # Binary threshold
            codewords.append(list(all_words[i]))
            
    # Final time check
    if time.time() > deadline:
        # If we are too close to the limit, return what we have or fallback
        # But extracting is fast.
        pass
        
    return codewords

def _fallback_construct(q, n, radius):
    # The original seed program's logic
    # Direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return tetracode
    if n < 4:
        # Handle small n recursively or brute force if needed, but task is n=6
        # For n=6, this won't be called unless ILP fails
        pass
        
    # If n > 4, extend tetracode
    # The seed used: [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # This yields 9 * 3**(n-4) words.
    # For n=6, this is 9 * 9 = 81 words.
    
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Fallback for n < 4 (not expected for this task)
    # Simple greedy or brute force
    all_words = list(itertools.product(range(q), repeat=n))
    selected = []
    covered = set()
    for w in all_words:
        if w not in covered:
            selected.append(list(w))
            # Add all words within radius to covered
            for v in all_words:
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    covered.add(v)
    return selected

# EVOLVE-BLOCK-END
