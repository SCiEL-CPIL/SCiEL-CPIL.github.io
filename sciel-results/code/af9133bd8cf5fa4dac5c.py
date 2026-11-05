import random
import time
import math

def calc_max_disc_and_witnesses(signs, n=256, limit=None):
    """
    Calculates max discrepancy and returns the witnesses (progressions) achieving it.
    If limit is provided and exceeded, returns early with the current best witness.
    """
    max_disc = 1
    best_witnesses = []
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p: min_p = curr
                if curr > max_p: max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc, [(d, r, min_p, max_p)]
                best_witnesses = [(d, r, min_p, max_p)]
            elif disc == max_disc:
                best_witnesses.append((d, r, min_p, max_p))
    return max_disc, best_witnesses

def get_indices_from_witnesses(witnesses, n=256):
    """Extract unique indices from a list of witnesses."""
    indices = set()
    for d, r, min_p, max_p in witnesses:
        idx = r
        while idx < n:
            indices.add(idx)
            idx += d
    return list(indices)

def greedy_repair(signs, n=256, deadline=None):
    """Greedy repair: repeatedly flip the sign that most reduces max discrepancy."""
    current_disc = calc_max_disc(signs)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        best_gain = 0
        best_idx = -1
        # Try flipping each position and see if it reduces the discrepancy
        for i in range(n):
            signs[i] *= -1
            new_disc = calc_max_disc(signs, n, limit=current_disc)
            signs[i] *= -1
            gain = current_disc - new_disc
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0 and best_gain > 0:
            signs[best_idx] *= -1
            current_disc -= best_gain
            improved = True
    return current_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Initial solution
    signs = [random.choice((-1, 1)) for _ in range(n)]
    
    # Initial full calculation
    current_disc, current_witnesses = calc_max_disc_and_witnesses(signs, n)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Targeted Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # 1. Select move based on current witnesses
        if not current_witnesses:
            # Fallback to random if no witnesses found (should not happen)
            i = random.randint(0, n - 1)
        else:
            hot_indices = get_indices_from_witnesses(current_witnesses, n)
            if not hot_indices:
                i = random.randint(0, n - 1)
            else:
                i = random.choice(hot_indices)
                
        # 2. Flip sign
        signs[i] *= -1
        
        # 3. Evaluate new state
        # Use limit to speed up rejection
        limit = current_disc + int(T * 5) + 1
        new_disc, new_witnesses = calc_max_disc_and_witnesses(signs, n, limit=limit)
        
        delta = new_disc - current_disc
        
        # 4. Accept or Reject
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_disc = new_disc
            current_witnesses = new_witnesses # Update witnesses for next move selection
            
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            # Reject: Revert sign
            signs[i] *= -1
            # Witnesses remain valid as state is unchanged
            
        # 5. Cool
        T *= cooling
        if T < T_min:
            T = T_min

    return best_signs
