import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 128
    # A is a subset of [0, 128]
    # Target: every t in [128, 256] must be expressible as a+b with a<=b, a,b in A
    
    # Fast evaluation using convolution
    # If we have indicator vector v of length 129 (indices 0..128),
    # then the number of ordered pairs (a,b) with a+b=t is (v*v)[t]
    # For unordered pairs with a<=b: 
    #   if a != b: count_ordered/2
    #   if a == b (i.e., t is even and t/2 in A): count_ordered includes 1 for (a,a)
    # So unordered_count(t) = (ordered_count(t) + (1 if t even and t//2 in A else 0)) / 2
    
    def evaluate(A_set):
        """Return the maximum multiplicity over t in [128,256], and coverage info."""
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            v[x] = 1
        # Convolution gives ordered pair counts
        conv = np.convolve(v, v, mode='full')
        # conv[t] = number of ordered pairs (a,b) with a+b=t, a,b in [0,128]
        # We need unordered pairs with a<=b
        # unordered(t) = (conv[t] + diag(t)) / 2 where diag(t) = 1 if t even and t//2 in A
        max_mult = 0
        covered = True
        for t in range(128, 257):
            if t >= len(conv):
                break
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    # Initial solution: use the seed as starting point, then improve
    # Seed: range(64, 129)
    A = set(range(64, 129))
    best_A = set(A)
    best_score, best_covered = evaluate(A)
    if not best_covered:
        # Need to fix coverage
        pass
    
    # Actually let's think more carefully about initial solution
    # We need coverage of [128, 256]
    # If A = [0, 128], then sums range from 0 to 256
    # But we need each t in [128,256] to have at least one representation
    
    # Let's use a smarter initial solution
    # Start with a dense set and try to sparsify
    A = set(range(0, 129))  # all elements
    score, covered = evaluate(A)
    best_A = set(A)
    best_score = score
    best_covered = covered
    
    # Simulated annealing
    import random
    random.seed(42)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    current_A = set(A)
    current_score, current_covered = evaluate(current_A)
    
    if not current_covered:
        # This shouldn't happen with full set
        pass
    
    iteration = 0
    while time.time() < deadline and T > T_min:
        iteration += 1
        # Generate a move: add or remove a random element
        if random.random() < 0.5 and len(current_A) > 1:
            # Remove
            x = random.choice(sorted(current_A))
            new_A = current_A - {x}
        else:
            # Add
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A = current_A | {x}
        
        new_score, new_covered = evaluate(new_A)
        
        # Score function: penalize lack of coverage heavily
        if new_covered:
            new_cost = new_score
        else:
            new_cost = new_score + 1000  # heavy penalty for uncovered
        
        if current_covered:
            current_cost = current_score
        else:
            current_cost = current_score + 1000
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_A = new_A
            current_score = new_score
            current_covered = new_covered
            
            if current_covered and (not best_covered or current_score < best_score):
                best_A = set(current_A)
                best_score = current_score
                best_covered = True
        
        T *= cooling
    
    # Ensure best is valid
    if not best_covered:
        # Fallback to a known valid solution
        best_A = set(range(64, 129))
        # Verify
        s, c = evaluate(best_A)
        if not c:
            best_A = set(range(0, 129))
    
    return sorted(best_A)
