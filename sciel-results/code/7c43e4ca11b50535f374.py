import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N

def _precompute_inputs():
    mat = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        half = 1 << i
        period = 2 * half
        pattern = np.concatenate([np.zeros(half, dtype=np.uint8), np.ones(half, dtype=np.uint8)])
        mat[i] = np.tile(pattern, NUM_INPUTS // period)
    return mat

INPUTS = _precompute_inputs()

def _simulate(comparators):
    mat = INPUTS.copy()
    for (i, j) in comparators:
        mi = mat[i] & mat[j]
        mj = mat[i] | mat[j]
        mat[i] = mi
        mat[j] = mj
    return mat

def _is_valid(comparators):
    mat = _simulate(comparators)
    for i in range(N - 1):
        if np.any(mat[i] > mat[i + 1]):
            return False
    return True

def _batcher_merge(n):
    """
    Generates Batcher's Odd-Merge Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    # Split into two halves
    k = n // 2
    # Sort the two halves
    # Note: Batcher's sort splits into n/2 and n/2 (if even) or (n-1)/2 and (n+1)/2
    # For simplicity and correctness, let's use the standard recursive definition.
    # Standard Batcher's Odd-Merge:
    # 1. Sort first half and second half.
    # 2. Merge them.
    
    # To handle odd n, we split into floor(n/2) and ceil(n/2)
    left_n = n // 2
    right_n = n - left_n
    
    left_net = _batcher_merge(left_n)
    # Shift right_net indices by left_n
    right_net = [(i + left_n, j + left_n) for (i, j) in _batcher_merge(right_n)]
    
    # Merge phase
    # The merge network for m and k wires (m <= k)
    # Here we merge left_n and right_n.
    # The standard merge network consists of:
    # 1. Odd-even merge of the two sorted lists.
    
    def _odd_even_merge(m, k):
        # m and k are sizes, m <= k
        # Returns comparators for merging two sorted lists of size m and k
        # The wires are 0..m-1 and m..m+k-1
        if m == 0 or k == 0:
            return []
        if m == 1 and k == 1:
            return [(0, 1)]
        
        # Recursive odd-even merge
        # Step 1: Merge odd elements and even elements
        # The odd-even merge network is built recursively.
        # It compares (i, i + k) for appropriate i?
        # Actually, the structure is:
        # 1. Apply odd-even merge to the odd-indexed wires and even-indexed wires of the combined sequence?
        # No, the standard definition:
        # Merge(m, k):
        #   if m == 0 or k == 0: return []
        #   if m == 1 and k == 1: return [(0, 1)]
        #   m1 = m // 2, k1 = k // 2
        #   # Merge the "odd" parts and "even" parts
        #   # The "odd" part of the first list (indices 0, 2, ...) and "even" part of second list?
        #   # This is tricky to implement from scratch correctly.
        
        # Alternative: Use Bitonic Sort for the initial seed, it's easier to implement correctly.
        # Or use a known small network and pad?
        # Let's just use Bitonic Sort for N=16 and add 2 wires?
        # Bitonic sort for N=18 is not standard.
        
        # Let's use a simpler seed: **Odd-Even Transposition** is 153.
        # **Bitonic Sort** for N=16 is 64.
        # We can take Bitonic Sort for N=16 (wires 0-15) and add wires 16, 17.
        # To sort 18, we can:
        # 1. Sort 0-15 (Bitonic).
        # 2. Compare 16, 17.
        # 3. Merge the sorted 16-wire block with the 2-wire block?
        # This is effectively a merge of 16 and 2.
        # Merge(16, 2) using Batcher's merge?
        
        # Let's just implement a generic **Batcher's Merge** correctly.
        # Reference: Batcher's Odd-Merge Sort.
        # The merge network for two sorted lists of length m and k (m <= k) is:
        # 1. If m=1, k=1: compare (0,1).
        # 2. Otherwise:
        #    Let m1 = m//2, m2 = m - m1
        #    Let k1 = k//2, k2 = k - k1
        #    The merge network consists of:
        #    - Merge(m1, k1) on the "odd" wires?
        #    - Merge(m2, k2) on the "even" wires?
        #    - Then a final set of comparators.
        
        # This is getting complex. Let's use a simpler, robust seed:
        # **Random Dense Network** + Pruning.
        # Generate a network with, say, 200 random comparators.
        # Check if it sorts. If not, keep adding random comparators until it does.
        # Then prune.
        
        return []

def _generate_seed():
    """
    Generates a valid sorting network seed.
    Uses a randomized approach to find a valid dense network, then returns it.
    """
    # Start with a known valid network: Odd-Even Transposition
    # It's safe, though large.
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Try to improve the seed by randomizing and re-validating?
    # No, let's just use it. It's 153 comparators.
    # We can also try to construct a Bitonic sort for N=16 and merge with 2.
    # But OET is fine for a seed.
    
    # Let's try to make a denser, more "mixed" seed to help escape local minima.
    # Shuffle the comparators? No, order matters.
    # Let's just return OET.
    return net

def _greedy_remove(net, deadline):
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Randomize the order of removal attempts to avoid bias
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                return net
            trial = net[:idx] + net[idx+1:]
            if _is_valid(trial):
                net = trial
                changed = True
                break
    return net

def _optimize(net, deadline):
    best_net = net
    best_size = len(net)
    
    while time.time() < deadline:
        current_net = net
        current_size = len(net)
        
        # Try to improve current_net using local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try Removal
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                trial = current_net[:idx] + current_net[idx+1:]
                if _is_valid(trial):
                    current_net = trial
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            
            if improved:
                continue
            
            # 2. If stuck with removal, try Swap/Move to escape local minimum
            # Swap two comparators
            if len(current_net) >= 2:
                i, j = random.sample(range(len(current_net)), 2)
                trial = current_net.copy()
                trial[i], trial[j] = trial[j], trial[i]
                if _is_valid(trial):
                    # Accept swap even if size is same, to explore
                    current_net = trial
                    improved = True
                    continue
            
            # Move a comparator
            if len(current_net) >= 2:
                i, j = random.sample(range(len(current_net)), 2)
                trial = current_net.copy()
                comp = trial.pop(i)
                trial.insert(j, comp)
                if _is_valid(trial):
                    current_net = trial
                    improved = True
                    continue
            
            # 3. If still stuck, try adding a random comparator (to enable future removals)
            # This increases size, but might lead to a better local minimum.
            # Limit this to prevent infinite growth.
            if len(current_net) < best_size + 10:
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = random.randint(0, len(current_net))
                trial = current_net.copy()
                trial.insert(pos, (i, j))
                if _is_valid(trial):
                    # Accept with some probability or if it leads to improvement later?
                    # For now, just accept to explore.
                    current_net = trial
                    improved = True
                    continue
        
        # Update best
        if len(current_net) < best_size:
            best_net = current_net
            best_size = len(current_net)
            net = current_net
        else:
            # If no improvement, restart from best_net with a perturbation?
            # Or just continue from current_net.
            # Let's reset to best_net to ensure we don't drift too far.
            net = best_net

    return best_net

def construct():
    start_time = time.time()
    # 90s wall clock limit. Keep 5s buffer.
    deadline = start_time + 85
    
    net = _generate_seed()
    
    if not _is_valid(net):
        # Fallback
        return net
    
    # Initial greedy removal
    net = _greedy_remove(net, deadline)
    
    # Main optimization loop
    net = _optimize(net, deadline)
    
    return net
