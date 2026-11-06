import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin for 60s limit
    
    n = 128
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    
    # Fast evaluation using convolution with incremental updates
    def evaluate(A_set):
        """Return (max_multiplicity, is_covered) for target interval [128,256]."""
        v = np.zeros(257, dtype=np.int32)
        for x in A_set:
            v[x] = 1
        conv = np.convolve(v, v, mode='full')
        # conv[t] = ordered pairs (a,b) with a+b=t
        # unordered(t) = (conv[t] + diag) / 2 where diag=1 if t even and t//2 in A
        max_mult = 0
        covered = True
        for t in range(MIN_SUM, MAX_SUM + 1):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            elif unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    # Start with a good initial solution
    A = set(range(0, 129))
    score, covered = evaluate(A)
    best_A = set(A)
    best_score = score
    best_covered = covered
    
    random.seed(42)
    
    # Simulated annealing with faster evaluation
    T = 20.0
    T_min = 0.01
    cooling = 0.9995
    
    current_A = set(A)
    current_score, current_covered = evaluate(current_A)
    
    # Precompute convolution for current state
    v = np.zeros(257, dtype=np.int32)
    for x in current_A:
        v[x] = 1
    conv = np.convolve(v, v, mode='full')
    
    def get_unordered_multiplicities(conv, A_set):
        """Get unordered multiplicities for all target sums."""
        mults = np.zeros(NUM_SUMS, dtype=np.int32)
        for i, t in enumerate(range(MIN_SUM, MAX_SUM + 1)):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            mults[i] = (ordered + diag) // 2
        return mults
    
    while time.time() < deadline:
        # Generate a move: add or remove a random element
        if random.random() < 0.6 and len(current_A) > 5:
            # Remove
            x = random.choice(tuple(current_A))
            new_A = current_A - {x}
            # Update convolution: subtract contributions of x
            v[x] -= 1
            conv = np.convolve(v, v, mode='full')
        else:
            # Add
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A = current_A | {x}
            # Update convolution: add contributions of x
            v[x] += 1
            conv = np.convolve(v, v, mode='full')
        
        new_mults = get_unordered_multiplicities(conv, new_A)
        new_score = int(np.max(new_mults))
        new_covered = bool(np.all(new_mults > 0))
        
        # Cost function: heavily penalize uncovered
        if new_covered:
            new_cost = float(new_score)
        else:
            new_cost = float(new_score) + 10000.0
        
        if current_covered:
            current_cost = float(current_score)
        else:
            current_cost = float(current_score) + 10000.0
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_A = new_A
            current_score = new_score
            current_covered = new_covered
            
            if current_covered and (not best_covered or current_score < best_score):
                best_A = set(current_A)
                best_score = current_score
                best_covered = True
        else:
            # Revert the convolution change
            if x in new_A and x not in current_A:
                v[x] -= 1
            elif x in current_A and x not in new_A:
                v[x] += 1
            conv = np.convolve(v, v, mode='full')
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Ensure best is valid
    if not best_covered:
        best_A = set(range(0, 129))
        s, c = evaluate(best_A)
        if not c:
            best_A = set(range(64, 129))
    
    return sorted(best_A)
