import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Target range [128, 256]
    # counts[i] corresponds to target 128 + i
    # i in [0, 128]
    
    # 1. Greedy Initialization
    A = []
    A_set = set()
    counts = np.zeros(129, dtype=int)
    
    # Precompute contribution of adding x to a set S
    # We'll just compute it on the fly for simplicity in init, 
    # but for SA we use the fast swap logic.
    
    # Greedy: Add elements to cover all targets
    # We want to minimize peak, so we prefer adding elements that 
    # don't increase peak too much, but cover new targets.
    
    # Initial candidate pool
    candidates = list(range(129))
    
    while True:
        if not np.all(counts > 0):
            # Find best candidate
            best_x = -1
            best_newly_covered = -1
            best_peak = float('inf')
            best_counts = None
            
            for x in candidates:
                if x in A_set:
                    continue
                
                # Calculate delta
                delta = np.zeros(129, dtype=int)
                for y in A:
                    t = x + y
                    if 128 <= t <= 256:
                        delta[t - 128] += 1
                t_2x = 2 * x
                if 128 <= t_2x <= 256:
                    delta[t_2x - 128] += 1
                
                new_counts = counts + delta
                newly_covered = np.sum((counts == 0) & (new_counts > 0))
                new_peak = np.max(new_counts)
                
                # Prioritize covering new targets, then lower peak
                if newly_covered > best_newly_covered or (newly_covered == best_newly_covered and new_peak < best_peak):
                    best_newly_covered = newly_covered
                    best_peak = new_peak
                    best_x = x
                    best_counts = new_counts
            
            if best_x == -1:
                # Should not happen if A can be expanded
                break
            
            A.append(best_x)
            A_set.add(best_x)
            counts = best_counts
        else:
            break
        
        # Check time
        if time.time() > deadline - 30:
            break

    # Ensure validity
    if not np.all(counts > 0):
        # Fallback: Add random until valid
        while not np.all(counts > 0) and len(A) < 129:
            x = random.randint(0, 128)
            if x not in A_set:
                A.append(x)
                A_set.add(x)
                # Recalculate counts
                counts = np.zeros(129, dtype=int)
                for a in A:
                    for b in A:
                        if a <= b:
                            t = a + b
                            if 128 <= t <= 256:
                                counts[t - 128] += 1
                if time.time() > deadline - 30:
                    break

    if not np.all(counts > 0):
        # Absolute fallback
        A = list(range(129))
        A_set = set(A)
        counts = np.zeros(129, dtype=int)
        for a in A:
            for b in A:
                if a <= b:
                    t = a + b
                    if 128 <= t <= 256:
                        counts[t - 128] += 1

    current_peak = int(np.max(counts))
    best_peak = current_peak
    best_A = list(A)
    best_counts = counts.copy()
    
    # 2. Local Search (Simulated Annealing)
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    
    # Helper to evaluate swap u -> v
    def evaluate_swap(u, v):
        # u in A, v not in A
        # Returns new_counts, new_peak, valid
        delta = np.zeros(129, dtype=int)
        
        # Remove u
        # Pairs involving u: (u, a) for all a in A
        for a in A:
            t = u + a
            if 128 <= t <= 256:
                delta[t - 128] -= 1
        
        # Add v
        # Pairs involving v in new set: (v, a) for a in A \ {u}, and (v, v)
        for a in A:
            if a == u:
                continue
            t = v + a
            if 128 <= t <= 256:
                delta[t - 128] += 1
        
        t_2v = 2 * v
        if 128 <= t_2v <= 256:
            delta[t_2v - 128] += 1
            
        new_counts = counts + delta
        if np.any(new_counts < 0):
            return None, None, False
        if not np.all(new_counts > 0):
            return None, None, False
            
        new_peak = int(np.max(new_counts))
        return new_counts, new_peak, True

    while time.time() < deadline - 2:
        if T < min_T:
            break
            
        # Random swap
        if len(A) == 0:
            break
            
        u_idx = random.randint(0, len(A) - 1)
        u = A[u_idx]
        v = random.randint(0, 128)
        
        if v in A_set:
            continue
            
        new_counts, new_peak, valid = evaluate_swap(u, v)
        
        if not valid:
            T *= cooling
            continue
            
        delta_peak = new_peak - current_peak
        # Objective: minimize peak, then sum of squares (variance)
        old_obj = int(np.sum(counts**2))
        new_obj = int(np.sum(new_counts**2))
        delta_obj = new_obj - old_obj
        
        accept = False
        if delta_peak < 0:
            accept = True
        elif delta_peak == 0:
            if delta_obj < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_obj / T)
        else:
            # Peak increased
            # Accept with low probability to escape local minima
            accept = random.random() < np.exp(-delta_peak * 5 / T)
            
        if accept:
            # Commit swap
            A[u_idx] = v
            A_set.remove(u)
            A_set.add(v)
            counts = new_counts
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(A)
                best_counts = counts.copy()
                
        T *= cooling
        
    return best_A
