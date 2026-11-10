import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    TARGETS = np.arange(128, 257)
    
    def compute_max_mult(A):
        """Compute max multiplicity over targets [128,256] for set A."""
        if len(A) == 0:
            return float('inf'), False
        B = np.zeros(257, dtype=np.float64)
        for a in A:
            B[a] = 1.0
        # Convolution to get ordered pair counts
        conv = np.convolve(B, B, mode='full')
        # conv[t] = number of ordered pairs (a,b) in A x A with a+b=t
        max_mult = 0
        covered = True
        for t in TARGETS:
            ordered = conv[t]
            # Adjust for unordered
            if t % 2 == 0:
                half = t // 2
                if 0 <= half <= N and B[half] > 0:
                    unordered = (ordered + 1) / 2.0
                else:
                    unordered = ordered / 2.0
            else:
                unordered = ordered / 2.0
            if unordered < 0.5:  # not covered (should be >= 1)
                covered = False
                return float('inf'), False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    def try_move(A, move_type, idx=None):
        """Try a move and return new A if valid."""
        A_set = set(A)
        if move_type == 'add':
            # Add a random element not in A
            candidates = [x for x in range(N+1) if x not in A_set]
            if not candidates:
                return None
            new_elem = random.choice(candidates)
            return A + [new_elem]
        elif move_type == 'remove':
            if len(A) <= 1:
                return None
            new_A = A[:]
            new_A.pop(random.randint(0, len(A)-1))
            return new_A
        elif move_type == 'swap':
            if len(A) == 0:
                return None
            candidates = [x for x in range(N+1) if x not in A_set]
            if not candidates:
                return None
            new_A = A[:]
            new_A[random.randint(0, len(A)-1)] = random.choice(candidates)
            return new_A
        return None
    
    # Initial solution: try a structured approach
    # Start with evenly spaced elements
    random.seed(42)
    
    # Try different initial sizes
    best_A = None
    best_mult = float('inf')
    
    # Start with a reasonable initial set
    # Try k elements evenly spread
    for k in [20, 22, 24, 26, 28, 30]:
        # Evenly spaced in [0, 128]
        A = list(set(int(round(i * N / (k-1))) for i in range(k)))
        mult, covered = compute_max_mult(A)
        if covered and mult < best_mult:
            best_mult = mult
            best_A = A[:]
    
    if best_A is None:
        # Fallback: use seed
        return list(range(64, 129))
    
    current_A = best_A[:]
    current_mult = best_mult
    
    # Simulated annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    step = 0
    while time.time() < deadline and T > min_T:
        step += 1
        
        # Choose move type
        r = random.random()
        if r < 0.3:
            move_type = 'add'
        elif r < 0.6:
            move_type = 'remove'
        else:
            move_type = 'swap'
        
        new_A = try_move(current_A, move_type)
        if new_A is None:
            continue
        
        new_mult, new_covered = compute_max_mult(new_A)
        
        if not new_covered:
            # Reject uncovered solutions
            T *= cooling
            continue
        
        if new_mult <= current_mult:
            # Accept (improvement or equal)
            current_A = new_A
            current_mult = new_mult
            if current_mult < best_mult:
                best_mult = current_mult
                best_A = current_A[:]
        else:
            # Deterioration: accept with probability exp(-delta/T)
            delta = new_mult - current_mult
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_A = new_A
                    current_mult = new_mult
        
        T *= cooling
    
    # Return the best found
    if best_A is not None and best_mult < float('inf'):
        return sorted(set(best_A))
    
    # Fallback
    return list(range(64, 129))
