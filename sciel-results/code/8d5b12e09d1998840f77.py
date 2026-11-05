import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total = 3 ** n
    powers = [3 ** p for p in range(n)]

    # --- Precompute neighbor (ball) indices for every word: shape (total, 13) ---
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            base = i - d[p] * powers[p]
            for alt in range(3):
                if alt == d[p]:
                    continue
                neighbors[i, k] = base + alt * powers[p]
                k += 1

    def to_word(idx):
        return [(idx // powers[p]) % 3 for p in range(n)]

    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    # --- Initial code: tetracode direct sum (81 words) ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_idx = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    current_code = np.array(sorted(set(init_idx)), dtype=np.int32)
    code_set = set(int(x) for x in current_code)

    # coverage counts: number of chosen words within distance 1 of each word
    coverage = np.zeros(total, dtype=np.int32)
    for c in current_code:
        coverage[neighbors[c]] += 1

    best_code = current_code.copy()
    best_size = len(best_code)

    start_time = time.time()
    deadline = start_time + 115.0

    def greedy_remove(code_arr, cov, code_set_local):
        """Repeatedly remove any word whose whole ball is covered >=2 times."""
        while True:
            if time.time() > deadline:
                break
            # vectorized: for each chosen word, min coverage over its ball
            mins = np.min(cov[neighbors[code_arr]], axis=1)
            removable = np.where(mins >= 2)[0]
            if removable.size == 0:
                break
            # Remove the one with the highest min coverage (most redundant)
            # To keep it simple and fast, just remove the first one found or random
            # Removing the one with max mins is better for stability
            max_min_idx = removable[np.argmax(mins[removable])]
            c = int(code_arr[max_min_idx])
            cov[neighbors[c]] -= 1
            code_set_local.discard(c)
            code_arr = np.delete(code_arr, max_min_idx)
        return code_arr

    # Phase 1: aggressive greedy removal from the 81-word start
    current_code = greedy_remove(current_code, coverage, code_set)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = current_code.copy()

    # Phase 2: Simulated Annealing with Greedy Removal
    # State: current_code, coverage, code_set
    # Move: Add a word, then greedy_remove.
    # Energy: len(current_code)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute gain for speed? 
    # gain[i] = number of words in neighbors[i] that are under-covered (coverage == 1)
    # We need to update this or compute on the fly.
    # Computing on the fly is O(total * 13) which is ~9000 ops. Very fast.
    
    random.seed(42)
    np.random.seed(42)
    
    last_improvement_time = start_time
    
    while time.time() < deadline - 5.0:
        # Calculate gain for all words: how many "critical" (coverage==1) words they cover
        under = (coverage == 1).astype(np.int8)
        gain = np.sum(under[neighbors], axis=1)
        
        # Mask out words already in the code
        gain_in_code = gain.copy()
        for c in code_set:
            gain_in_code[c] = -1
            
        # Candidate selection:
        # 1. Top K gain words
        # 2. Random word (for exploration)
        
        # Find top candidates
        # We want to add a word that covers the most under-covered words.
        # If no under-covered words exist, we are in a local minimum for this metric.
        
        max_gain = np.max(gain_in_code)
        
        if max_gain <= 0:
            # No word improves the "under-covered" count directly.
            # Try a random move or a swap.
            # For now, just pick a random word to kick the system.
            candidate = random.randrange(total)
            if candidate in code_set:
                continue
        else:
            # Pick randomly from top 5 gain words to diversify
            top_k = 5
            # Get indices of top K
            # argsort is O(N log N). 729 log 729 is small.
            sorted_indices = np.argsort(gain_in_code)[::-1]
            top_indices = sorted_indices[:top_k]
            candidate = int(random.choice(top_indices))
        
        # Save state
        saved_code = current_code.copy()
        saved_cov = coverage.copy()
        saved_set = set(code_set)
        saved_size = len(current_code)
        
        # Apply move: Add candidate
        coverage[neighbors[candidate]] += 1
        code_set.add(candidate)
        current_code = np.append(current_code, candidate)
        
        # Greedy remove
        current_code = greedy_remove(current_code, coverage, code_set)
        
        new_size = len(current_code)
        delta = new_size - saved_size
        
        # Acceptance Criteria
        if delta < 0:
            # Improvement
            accept = True
            last_improvement_time = time.time()
            if new_size < best_size:
                best_size = new_size
                best_code = current_code.copy()
        elif delta == 0:
            # Neutral
            accept = random.random() < 0.5
        else:
            # Worsening
            # Accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            # Keep the new state
            pass
        else:
            # Revert
            current_code = saved_code
            coverage = saved_cov
            code_set = saved_set
        
        # Cooling
        T = max(T * cooling, T_min)
        
        # If stuck for too long, restart temperature or perturb
        if time.time() - last_improvement_time > 2.0:
            # Perturb: Randomly swap a few words
            # Or just reset T to a higher value
            T = 0.5
            last_improvement_time = time.time()
            
    # Final polish
    if time.time() < deadline - 2.0:
        current_code = greedy_remove(current_code, coverage, code_set)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()

    return [to_word(int(x)) for x in best_code]
