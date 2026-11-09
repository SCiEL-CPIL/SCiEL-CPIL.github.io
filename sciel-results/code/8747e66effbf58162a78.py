"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
_DEADLINE = 45  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
# Pre-compute expected sorted masks for each possible popcount
_EXPECTED_MASKS = {}
def _init_expected_masks():
    """Pre-compute expected sorted masks for all possible popcounts."""
    for k in range(N + 1):
        if k == 0:
            _EXPECTED_MASKS[k] = 0
        else:
            _EXPECTED_MASKS[k] = (1 << k) - 1
_init_expected_masks()

def _popcount(mask):
    """Count set bits in mask."""
    return bin(mask).count('1')

def _verify_fast(network):
    """Fast verification using zero-one principle with pre-computed expected masks."""
    for mask in range(1 << N):
        for i, j in network:
            vi = (mask >> i) & 1
            vj = (mask >> j) & 1
            if vi > vj:
                mask |= (1 << i)
                mask &= ~(1 << j)
            elif vi < vj:
                mask &= ~(1 << i)
                mask |= (1 << j)
        k = _popcount(mask)
        if mask != _EXPECTED_MASKS[k]:
            return False
    return True

def _verify_numpy(network):
    """Vectorized verification using NumPy for speed."""
    n_comps = len(network)
    if n_comps == 0:
        return True
    
    # Build arrays for comparator indices
    comp_i = np.array([c[0] for c in network], dtype=np.int32)
    comp_j = np.array([c[1] for c in network], dtype=np.int32)
    
    # All 2^N masks
    masks = np.arange(1 << N, dtype=np.int32)
    
    # Process each comparator
    for idx in range(n_comps):
        i = comp_i[idx]
        j = comp_j[idx]
        vi = (masks >> i) & 1
        vj = (masks >> j) & 1
        
        # Where vi > vj (1,0): swap to (0,1)
        swap_mask = (vi > vj)
        masks[swap_mask] |= (1 << i)
        masks[swap_mask] &= ~(1 << j)
        
        # Where vi < vj (0,1): swap to (1,0)
        swap_mask = (vi < vj)
        masks[swap_mask] &= ~(1 << i)
        masks[swap_mask] |= (1 << j)
    
    # Check if all masks are sorted
    popcounts = np.array([bin(m).count('1') for m in masks], dtype=np.int32)
    expected = np.array([_EXPECTED_MASKS[k] for k in popcounts], dtype=np.int32)
    return np.all(masks == expected)

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _find_redundant_indices(network):
    """Find indices of comparators that appear more than once (redundant candidates)."""
    from collections import Counter
    counter = Counter(network)
    redundant = []
    for idx, comp in enumerate(network):
        if counter[comp] > 1:
            redundant.append(idx)
    return redundant

def construct():
    start_time = time.time()
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed
    if not _verify_fast(network):
        return best
    
    # Phase 1: Greedy removal passes (fast, deterministic)
    passes = 0
    max_passes = 15
    
    while passes < max_passes:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
        
        # Strategy 1: Try removing redundant comparators first
        redundant = _find_redundant_indices(network)
        for idx in sorted(redundant, reverse=True):
            if time.time() - start_time > _DEADLINE - 10:
                break
            candidate = list(network)
            candidate.pop(idx)
            if _verify_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
        
        # Strategy 2: Reverse-order removal
        for attempt in range(len(network) - 1, -1, -1):
            if time.time() - start_time > _DEADLINE - 10:
                break
            candidate = list(network)
            candidate.pop(attempt)
            if _verify_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
        
        # Strategy 3: Forward-order removal
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE - 10:
                break
            candidate = list(network)
            candidate.pop(i)
            if _verify_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
        
        # Check if we made any progress this pass
        if len(network) == best_len:
            # No progress, try one more pass with shuffled order
            if time.time() - start_time > _DEADLINE - 10:
                break
            indices = list(range(len(network)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > _DEADLINE - 10:
                    break
                candidate = list(network)
                candidate.pop(idx)
                if _verify_fast(candidate):
                    network = candidate
                    best = list(candidate)
                    best_len = len(best)
        
        passes += 1
    
    # Phase 2: Simulated annealing to escape local minima
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        # Use numpy verification for speed in SA
        current = list(network)
        current_len = len(current)
        sa_best = list(best)
        sa_best_len = best_len
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        
        # Pre-compute all possible comparators
        all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
        all_comps_set = set(all_comps)
        
        steps = 0
        max_steps = 50000
        
        while T > T_min and steps < max_steps:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            steps += 1
            step_type = random.random()
            
            if step_type < 0.4:
                # Try removing a random comparator
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    candidate = list(current)
                    candidate.pop(idx)
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
                    else:
                        # Accept with probability exp(-delta/T) where delta = 1 (worse)
                        if random.random() < np.exp(-1.0 / T):
                            current = candidate
            elif step_type < 0.7:
                # Try swapping two comparators
                if len(current) >= 2:
                    i, j = random.sample(range(len(current)), 2)
                    candidate = list(current)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
            elif step_type < 0.85:
                # Try replacing a comparator with a random one
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    new_comp = random.choice(all_comps)
                    candidate = list(current)
                    candidate[idx] = new_comp
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
                    else:
                        if random.random() < np.exp(-1.0 / T):
                            current = candidate
            else:
                # Try inserting a random comparator (if we have room)
                if len(current) < 4096:
                    new_comp = random.choice(all_comps)
                    idx = random.randint(0, len(current))
                    candidate = list(current)
                    candidate.insert(idx, new_comp)
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
                    else:
                        # Insertion makes it worse (longer), accept with low prob
                        if random.random() < np.exp(-2.0 / T):
                            current = candidate
            
            T *= cooling
        
        # Update best if SA found something better
        if sa_best_len < best_len:
            best = sa_best
            best_len = sa_best_len
    
    return best
# EVOLVE-BLOCK-END
