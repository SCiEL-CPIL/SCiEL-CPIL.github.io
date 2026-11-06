"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_data():
    """Precompute AP membership as arrays for vectorized delta computation.
    Returns (ap_indices, ap_pos_idx) where ap_indices[i] are the 4 positions of each AP
    and ap_pos_idx[pos] is the list of AP indices containing position pos."""
    # Build list of all APs as (a, d) pairs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            aps.append((a, d))
    
    num_aps = len(aps)
    ap_indices = np.array([[a, a+d, a+2*d, a+3*d] for a, d in aps], dtype=np.int32)
    
    # For each position, which APs contain it
    ap_pos_idx = [[] for _ in range(N)]
    for idx, (a, d) in enumerate(aps):
        for k in range(4):
            pos = a + k * d
            ap_pos_idx[pos].append(idx)
    ap_pos_idx = [np.array(lst, dtype=np.int32) for lst in ap_pos_idx]
    
    return ap_indices, ap_pos_idx

def _delta_flip_vec(c, i, ap_indices, ap_pos_idx):
    """Compute change in mono-4AP count if we flip position i, vectorized."""
    ap_idxs = ap_pos_idx[i]
    if len(ap_idxs) == 0:
        return 0
    # Get all 4 positions for these APs
    positions = ap_indices[ap_idxs]  # shape (num_aps, 4)
    vals = c[positions]  # shape (num_aps, 4)
    
    # Count mono APs before flip
    was_mono = (vals[:, 0] == vals[:, 1]) & (vals[:, 0] == vals[:, 2]) & (vals[:, 0] == vals[:, 3])
    
    # After flip: position i has flipped
    # For each AP, find which position index (0-3) corresponds to position i
    # We need to flip that position's value
    # Determine which of the 4 positions is i
    pos_match = (positions == i)  # shape (num_aps, 4)
    # Flip the value at the matching position
    new_vals = vals.copy()
    new_vals[pos_match] = 1 - new_vals[pos_match]
    
    is_mono = (new_vals[:, 0] == new_vals[:, 1]) & (new_vals[:, 0] == new_vals[:, 2]) & (new_vals[:, 0] == new_vals[:, 3])
    
    delta = np.count_nonzero(is_mono) - np.count_nonzero(was_mono)
    return int(delta)

def _all_deltas(c, ap_indices, ap_pos_idx):
    """Compute delta for all positions at once."""
    deltas = np.zeros(N, dtype=np.int32)
    for i in range(N):
        deltas[i] = _delta_flip_vec(c, i, ap_indices, ap_pos_idx)
    return deltas

def _local_search_delta(c, time_limit, ap_indices, ap_pos_idx):
    """Improve coloring via single-flip local search using vectorized delta."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            delta = _delta_flip_vec(c, i, ap_indices, ap_pos_idx)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            break
    return c, current

def _two_flip_search(c, time_limit, ap_indices, ap_pos_idx):
    """Try pairs of flips to escape local optima."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break
        for i in range(N):
            if time.time() - start > time_limit:
                break
            # Try flipping i
            c[i] = 1 - c[i]
            for j in range(i + 1, N):
                if time.time() - start > time_limit:
                    break
                c[j] = 1 - c[j]
                # Quick count change: compute delta for both flips
                # We need to be careful: flipping i then j is not the same as sum of individual deltas
                # due to interactions. Let's just count.
                cnt = _count_mono4(c)
                if cnt < current:
                    current = cnt
                    improved = True
                else:
                    c[j] = 1 - c[j]  # undo j
            # Undo i
            c[i] = 1 - c[i]
    return c, current

def _crossover(p1, p2, rng):
    """Uniform crossover."""
    mask = rng.integers(0, 2, size=N).astype(np.int8)
    child = np.where(mask == 1, p1, p2)
    return child.copy()

def _mutate(c, rng, rate=0.05):
    """Bit-flip mutation."""
    flips = rng.random(N) < rate
    c[flips] = 1 - c[flips]
    return c

def _perturb(c, rng, num_flips):
    """Flip exactly num_flips random positions."""
    c = c.copy()
    positions = rng.choice(N, size=num_flips, replace=False)
    c[positions] = 1 - c[positions]
    return c

def construct():
    start_time = time.time()
    
    # Precompute AP data for vectorized delta
    ap_indices, ap_pos_idx = _precompute_ap_data()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # Local search from Thue-Morse
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 10:
        c, cnt = _local_search_delta(c, min(20, time_remaining - 10), ap_indices, ap_pos_idx)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # Multi-restart local search with perturbations
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 1: Multi-restart from best with varied perturbations
    restarts_done = 0
    max_restarts = 30
    while time.time() - start_time < DEADLINE - 30 and restarts_done < max_restarts:
        restarts_done += 1
        # Perturb the best solution
        num_flips = rng.integers(2, 8)
        c_pert = _perturb(best_c, rng, num_flips)
        c_pert, cnt_pert = _local_search_delta(c_pert, 2.0, ap_indices, ap_pos_idx)
        if cnt_pert < best_count:
            best_count = cnt_pert
            best_c = c_pert.copy()
    
    # Phase 2: Genetic Algorithm
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 15:
        pop_size = 16
        elite_size = 3
        mutation_rate = 0.04
        
        # Initialize population
        population = [best_c.copy()]
        scores = [best_count]
        
        # Add perturbed versions of best
        for _ in range(3):
            num_flips = rng.integers(3, 10)
            c_pert = _perturb(best_c, rng, num_flips)
            c_pert, cnt_pert = _local_search_delta(c_pert, 1.5, ap_indices, ap_pos_idx)
            population.append(c_pert)
            scores.append(cnt_pert)
        
        # Add random solutions with local search
        for _ in range(pop_size - 4):
            c_rand = rng.integers(0, 2, size=N).astype(np.int8)
            c_rand, cnt_rand = _local_search_delta(c_rand, 1.0, ap_indices, ap_pos_idx)
            population.append(c_rand)
            scores.append(cnt_rand)
        
        for i in range(pop_size):
            if scores[i] < best_count:
                best_count = scores[i]
                best_c = population[i].copy()
        
        # GA loop
        while time.time() - start_time < DEADLINE - 5:
            # Sort by score
            sorted_idx = np.argsort(scores)
            population = [population[i].copy() for i in sorted_idx]
            scores = [scores[i] for i in sorted_idx]
            
            # Elitism
            new_population = [population[i].copy() for i in range(elite_size)]
            new_scores = [scores[i] for i in range(elite_size)]
            
            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Tournament selection
                idxs = rng.integers(0, pop_size, size=3)
                best_idx1 = idxs[np.argmin([scores[i] for i in idxs])]
                idxs2 = rng.integers(0, pop_size, size=3)
                best_idx2 = idxs2[np.argmin([scores[i] for i in idxs2])]
                
                p1 = population[best_idx1]
                p2 = population[best_idx2]
                
                child = _crossover(p1, p2, rng)
                child = _mutate(child, rng, mutation_rate)
                
                time_left = DEADLINE - (time.time() - start_time)
                ls_time = min(1.0, max(0.1, time_left / 10))
                child, cnt_child = _local_search_delta(child, ls_time, ap_indices, ap_pos_idx)
                
                new_population.append(child)
                new_scores.append(cnt_child)
            
            population = new_population
            scores = new_scores
            
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < best_count:
                best_count = scores[min_score_idx]
                best_c = population[min_score_idx].copy()
    
    # Final: try a few more perturbations of the best if time remains
    elapsed = time.time() - start_time
    if DEADLINE - elapsed > 5:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 2:
                break
            num_flips = rng.integers(1, 5)
            c_pert = _perturb(best_c, rng, num_flips)
            c_pert, cnt_pert = _local_search_delta(c_pert, 1.0, ap_indices, ap_pos_idx)
            if cnt_pert < best_count:
                best_count = cnt_pert
                best_c = c_pert.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
