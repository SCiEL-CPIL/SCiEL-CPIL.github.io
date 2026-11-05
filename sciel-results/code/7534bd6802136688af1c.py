import time
import random
import numpy as np

N = 16
TOTAL_BITS = 1 << N  # 65536

# Precompute sorted masks: for input x, sorted output has top popcount(x) bits set.
def _build_sorted_masks():
    masks = np.zeros(TOTAL_BITS, dtype=np.uint64)
    for x in range(TOTAL_BITS):
        cnt = bin(x).count('1')
        if cnt > 0:
            m = ((1 << cnt) - 1) << (N - cnt)
        else:
            m = 0
        masks[x] = m
    return masks

SORTED_MASKS = _build_sorted_masks()

def _build_initial_wires():
    """Build initial wire bit-vectors for all 65536 inputs."""
    wires = [0] * N
    for i in range(N):
        w = 0
        for k in range(TOTAL_BITS):
            if (k >> i) & 1:
                w |= (1 << k)
        wires[i] = w
    return wires

INITIAL_WIRES = _build_initial_wires()
FULL_MASK = (1 << TOTAL_BITS) - 1

def _verify(nets):
    """Verify network sorts all inputs using bit-parallel simulation."""
    if not nets:
        return False
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
        # compare-exchange: min to i, max to j
        # swap_mask is 1 where wire i is 1 and wire j is 0
        swap_mask = wires[i] & (~wires[j]) & FULL_MASK
        wires[i] ^= swap_mask
        wires[j] ^= swap_mask
    
    # Check all 65536 inputs
    # We can vectorize the check or just loop. Looping 65536 * 16 is ~1M ops, fast enough.
    # But we can do it faster.
    # For each k, check if the bit k in wires[i] matches SORTED_MASKS[k] bit i.
    # Actually, let's just construct the final value for each k and compare.
    
    # Optimization: Check if the set of 1s in wires is the same as in SORTED_MASKS.
    # Since it's a sorting network, the number of 1s is preserved.
    # So we just need to check if the pattern matches.
    
    for k in range(TOTAL_BITS):
        val = 0
        for i in range(N):
            if (wires[i] >> k) & 1:
                val |= (1 << i)
        if val != SORTED_MASKS[k]:
            return False
    return True

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    result = []
    # Compare-exchange between halves
    for i in range(n // 2):
        result.append((i, i + n // 2))
    # Recursively merge odd and even positions
    for (i, j) in _batcher_odd_even_merge(n // 2):
        result.append((2 * i + 1, 2 * j + 1))
    for (i, j) in _batcher_odd_even_merge(n // 2):
        result.append((2 * i, 2 * j))
    return result

def _batcher_sort(n):
    """Batcher's odd-even merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    result = []
    # Sort halves
    result.extend(_batcher_sort(half))
    result.extend([(i + half, j + half) for (i, j) in _batcher_sort(half)])
    # Odd-even merge
    result.extend(_batcher_odd_even_merge(n))
    return result

def construct():
    """Search for a minimal sorting network on 16 wires using Simulated Annealing."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Start with Batcher's sort (147 comparators)
    # This is a valid starting point and has good structure for reduction
    current_net = _batcher_sort(N)
    
    # Verify start
    if not _verify(current_net):
        # Fallback to OET if Batcher fails (shouldn't happen)
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = list(current_net)
    best_len = len(current_net)
    best_time = time.time()
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.9995 # Cooling rate
    
    random.seed(42) # For reproducibility during development, remove for final if needed? No, keep it stable.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        trial_net = list(current_net)
        
        if move_type < 0.4:
            # Delete
            if len(trial_net) > 0:
                idx = random.randint(0, len(trial_net) - 1)
                trial_net.pop(idx)
        elif move_type < 0.8:
            # Insert
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = random.randint(0, len(trial_net))
            trial_net.insert(pos, (i, j))
        else:
            # Swap adjacent
            if len(trial_net) > 1:
                idx = random.randint(0, len(trial_net) - 2)
                trial_net[idx], trial_net[idx+1] = trial_net[idx+1], trial_net[idx]
        
        # Verify trial
        if _verify(trial_net):
            # Accept if valid (and we want to minimize size, so we always accept valid smaller ones, 
            # but SA logic usually applies to cost. Here cost = -size or just size? 
            # We want to minimize size. 
            # If trial is valid, it's a good candidate.
            # We should accept it if it's smaller, or with some probability if it's larger?
            # Actually, if it's valid, it's a "success". 
            # If we accept larger valid networks, we might wander. 
            # Better: Only accept if len(trial_net) < len(current_net) OR with prob exp(-delta/T) where delta = len(trial)-len(current).
            # Since we want to minimize, delta = len(trial) - len(current).
            # If delta < 0 (smaller), always accept.
            # If delta > 0 (larger), accept with prob exp(-delta/T).
            
            delta = len(trial_net) - len(current_net)
            
            if delta < 0:
                current_net = trial_net
                if len(current_net) < best_len:
                    best_net = list(current_net)
                    best_len = len(current_net)
                    best_time = time.time()
            else:
                # Accept larger network with probability
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_net = trial_net
        else:
            # Invalid network. Reject.
            pass
            
        # Cool down
        T *= alpha
        
        # Restart logic: If no improvement for a while, perturb best_net
        if time.time() - best_time > 10.0:
            # Perturb best_net to escape local minima
            current_net = list(best_net)
            # Add some random noise
            for _ in range(5):
                if random.random() < 0.5:
                    # Delete
                    if len(current_net) > 0:
                        idx = random.randint(0, len(current_net) - 1)
                        current_net.pop(idx)
                else:
                    # Insert
                    i, j = random.sample(range(N), 2)
                    if i > j: i, j = j, i
                    pos = random.randint(0, len(current_net))
                    current_net.insert(pos, (i, j))
            
            # Reset temperature slightly to allow exploration
            T = max(T, 0.5)
            best_time = time.time() # Reset timer for this restart attempt? No, keep tracking global best time.
            # Actually, let's just keep the best_net as the anchor.
            
    return best_net
