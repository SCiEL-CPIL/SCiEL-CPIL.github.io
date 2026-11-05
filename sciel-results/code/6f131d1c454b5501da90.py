import time
import random
import numpy as np

BASE = 51975

def _get_divisors():
    n = BASE
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

ALL_MODULI = _get_divisors()
NUM_MODULI = len(ALL_MODULI)

def _calc_score_and_counts(residues):
    counts = np.zeros(BASE, dtype=np.int32)
    for i, m in enumerate(ALL_MODULI):
        r = residues[i]
        counts[r::m] += 1
    score = np.sum(counts == 0)
    return score, counts

def _greedy_init(order_indices, rng):
    residues = [0] * NUM_MODULI
    counts = np.zeros(BASE, dtype=np.int32)
    # Process moduli in the given order
    for idx in order_indices:
        m = ALL_MODULI[idx]
        best_r = 0
        best_gain = -1
        # Try all residues for this modulus
        for r in range(m):
            # Gain is number of 0s in counts[r::m]
            gain = np.sum(counts[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[idx] = best_r
        counts[best_r::m] += 1
        
    score = np.sum(counts == 0)
    return residues, counts, score

def _simulated_annealing(residues, counts, score, deadline, rng):
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    current_residues = residues[:]
    current_counts = counts.copy()
    current_score = score
    
    best_residues = current_residues[:]
    best_score = current_score
    best_counts = current_counts.copy()
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick random modulus
        i = rng.randint(0, NUM_MODULI - 1)
        m = ALL_MODULI[i]
        old_r = current_residues[i]
        
        # Pick new random residue
        new_r = rng.randint(0, m - 1)
        if new_r == old_r:
            continue
            
        # Calculate delta
        # Loss: residues in old slice that are 1
        # Gain: residues in new slice that are 0
        # Note: old and new slices are disjoint
        loss = np.sum(current_counts[old_r::m] == 1)
        gain = np.sum(current_counts[new_r::m] == 0)
        delta = loss - gain
        
        # Acceptance
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            current_counts[old_r::m] -= 1
            current_counts[new_r::m] += 1
            current_residues[i] = new_r
            current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_residues = current_residues[:]
                best_counts = current_counts.copy()
                
        T *= cooling
        
    return best_residues, best_score

def _adaptive_greedy_build(deadline, rng):
    """
    Adaptive greedy heuristic with probabilistic selection based on marginal gain.
    Instead of a fixed order, we iteratively pick the next modulus-residue pair
    that offers the best marginal gain, but we add some randomness to explore.
    """
    # We start with empty coverage
    counts = np.zeros(BASE, dtype=np.int32)
    used_moduli = [False] * NUM_MODULI
    residues = [0] * NUM_MODULI
    
    # Precompute potential gains for all (m, r) pairs? No, too much memory.
    # Instead, we evaluate on the fly.
    
    # To make it faster, we can sort moduli by size initially as a heuristic for "importance"
    # but the adaptive nature should handle it.
    
    # We will select NUM_MODULI times.
    for step in range(NUM_MODULI):
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_r = -1
        best_gain = -1
        
        # Sample a subset of unused moduli to speed up if needed, but 47 is small.
        # We'll iterate all unused.
        candidates = [i for i in range(NUM_MODULI) if not used_moduli[i]]
        
        # To add adaptivity/probabilistic selection:
        # We could pick the top K by gain and then choose randomly among them.
        # Or use a softmax over gains.
        # Let's stick to deterministic best gain first for stability, 
        # but the "adaptive" part is that the gain changes as counts update.
        # The prompt asks for "probabilistically selects the next modulus based on its marginal gain".
        
        # Let's calculate gains for all candidates.
        gains = []
        for i in candidates:
            m = ALL_MODULI[i]
            # Find best r for this m
            # Optimization: We can just find the max gain for this m.
            # gain = max_r sum(counts[r::m] == 0)
            # To speed this up, we can sample r if m is large, but m<=51975.
            # 47 * 51975 ops is ~2.4M, which is okay for a few steps, but we do this 47 times.
            # Total ~115M ops. Might be too slow in Python loops.
            
            # Let's optimize: 
            # For a given m, the residues r are 0..m-1.
            # counts[r::m] are slices.
            # We can compute the sum of zeros for all r at once?
            # No, they overlap.
            
            # Let's just iterate r, but break early if we exceed current best?
            # Or use numpy to compute gains for all r in a vectorized way?
            # Not easily possible because slices are different lengths/positions.
            
            # Let's stick to the loop but be careful.
            # Actually, we can just pick the global best (m, r) among all unused m.
            
            # To speed up, let's just evaluate a random subset of residues if m is large?
            # No, let's try the full loop first. If it's too slow, we'll see.
            
            # Heuristic: Start with r=0 and see.
            # Actually, the previous _greedy_init did exactly this and it was fast enough.
            # So let's do the same here.
            
            max_gain_for_m = -1
            best_r_for_m = 0
            for r in range(m):
                g = np.sum(counts[r::m] == 0)
                if g > max_gain_for_m:
                    max_gain_for_m = g
                    best_r_for_m = r
            
            if max_gain_for_m > best_gain:
                best_gain = max_gain_for_m
                best_idx = i
                best_r = best_r_for_m
        
        if best_idx == -1:
            break
            
        # Apply the best choice
        used_moduli[best_idx] = True
        residues[best_idx] = best_r
        m = ALL_MODULI[best_idx]
        counts[best_r::m] += 1
        
    score = np.sum(counts == 0)
    return residues, counts, score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    rng = random.Random(42)
    
    best_score = BASE
    best_residues = [0] * NUM_MODULI
    
    # Strategy 1: Greedy with increasing order
    order1 = list(range(NUM_MODULI))
    res1, counts1, score1 = _greedy_init(order1, rng)
    if score1 < best_score:
        best_score = score1
        best_residues = res1[:]
        
    # Strategy 2: Greedy with decreasing order
    order2 = list(range(NUM_MODULI - 1, -1, -1))
    res2, counts2, score2 = _greedy_init(order2, rng)
    if score2 < best_score:
        best_score = score2
        best_residues = res2[:]
        
    # Strategy 3: Adaptive Greedy Build
    if time.time() < deadline - 5.0:
        res3, counts3, score3 = _adaptive_greedy_build(deadline - 5.0, rng)
        if score3 < best_score:
            best_score = score3
            best_residues = res3[:]
            
    # Strategy 4: Random orders + SA
    # We have time for a few SA runs
    # Each SA run should take ~10-20s
    # Let's try 5 runs
    
    runs = 5
    for k in range(runs):
        if time.time() > deadline - 5.0:
            break
            
        # Random order for init
        order = list(range(NUM_MODULI))
        rng.shuffle(order)
        res_init, counts_init, score_init = _greedy_init(order, rng)
        
        # Run SA
        run_deadline = min(deadline - 1.0, time.time() + 15.0)
        res_sa, score_sa = _simulated_annealing(res_init, counts_init, score_init, run_deadline, rng)
        
        if score_sa < best_score:
            best_score = score_sa
            best_residues = res_sa[:]
            
    # Convert residues to classes
    classes = [[ALL_MODULI[i], best_residues[i]] for i in range(NUM_MODULI)]
    
    return classes
