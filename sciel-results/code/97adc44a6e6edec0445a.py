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

def _odd_even_transposition():
    net = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                net.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                net.append((i, i + 1))
    return net

def _batcher_sort(n):
    """Generate Batcher's odd-even mergesort network for n inputs."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    if n % 2 == 1:
        # Odd case: sort first n-1, then merge with last element
        net = _batcher_sort(n - 1)
        # Merge the (n-1)-sorted with the 1 element
        for i in range(n - 1):
            net.append((i, n - 1))
        return net
    
    # Even case
    half = n // 2
    left = _batcher_sort(half)
    right = _batcher_sort(half)
    
    # Offset right comparators by half
    right = [(i + half, j + half) for (i, j) in right]
    
    net = left + right
    
    # Merge the two halves
    net += _batcher_merge(half, half)
    
    return net

def _batcher_merge(m, n):
    """Generate Batcher's odd-even merge network to merge m and n sorted sequences."""
    if m == 0 or n == 0:
        return []
    if m == 1 and n == 1:
        return [(0, 1)]
    
    net = []
    
    # Odd-even merge
    if m <= n:
        for i in range(min(m, n)):
            net.append((i, m + i))
        # Recursively merge
        if m > 1:
            net += _batcher_merge(m // 2, n // 2)
            net += [(i + m // 2, i + m // 2 + n // 2) for i in range(m // 2)]
            net += _batcher_merge(m - m // 2, n - n // 2)
    else:
        for i in range(n):
            net.append((i, n + i))
        if n > 1:
            net += _batcher_merge(n // 2, m // 2)
            net += [(i + n // 2, i + n // 2 + m // 2) for i in range(n // 2)]
            net += _batcher_merge(n - n // 2, m - n // 2)
    
    return net

def _akl_sort(n):
    """AKL (Abelsson-Knuth) sorting network - a good starting point."""
    # Use Batcher as a solid baseline, then improve
    return _batcher_sort(n)

def _remove_redundant(net, deadline):
    """Aggressively remove redundant comparators."""
    net = list(net)
    
    # Multiple passes of removal
    for pass_num in range(20):
        if time.time() > deadline:
            break
        
        changed = True
        while changed:
            changed = False
            # Try removing from end to start
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline:
                    return net
                if len(net) <= 50:  # Stop if we get really small
                    break
                trial = net[:idx] + net[idx+1:]
                if _is_valid(trial):
                    net = trial
                    changed = True
                    break
    
    return net

def _simulated_annealing(net, deadline):
    """Simulated annealing to explore comparator space more thoroughly."""
    net = list(net)
    best = net[:]
    best_len = len(net)
    
    # Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    current_len = len(net)
    
    while time.time() < deadline:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random comparator
            if len(net) <= 50:
                continue
            idx = random.randint(0, len(net) - 1)
            removed = net.pop(idx)
            new_len = len(net)
            
            if _is_valid(net):
                # Accept
                if new_len < best_len:
                    best = net[:]
                    best_len = new_len
            else:
                # Reject: put it back
                net.insert(idx, removed)
        
        elif move_type < 0.6:
            # Add a random comparator
            if len(net) >= 200:
                continue
            comp = random.choice(all_comps)
            idx = random.randint(0, len(net))
            net.insert(idx, comp)
            new_len = len(net)
            
            if _is_valid(net):
                # Accept
                if new_len < best_len:
                    best = net[:]
                    best_len = new_len
            else:
                # Reject: remove it
                net.pop(idx)
        
        elif move_type < 0.85:
            # Move a comparator to a different position
            if len(net) < 2:
                continue
            idx = random.randint(0, len(net) - 1)
            new_idx = random.randint(0, len(net) - 1)
            if idx == new_idx:
                continue
            
            comp = net.pop(idx)
            net.insert(new_idx, comp)
            
            if _is_valid(net):
                # Accept
                if len(net) < best_len:
                    best = net[:]
                    best_len = len(net)
            else:
                # Reject: put it back
                net.pop(new_idx)
                net.insert(idx, comp)
        
        else:
            # Swap two comparators
            if len(net) < 2:
                continue
            i1 = random.randint(0, len(net) - 1)
            i2 = random.randint(0, len(net) - 1)
            if i1 == i2:
                continue
            
            net[i1], net[i2] = net[i2], net[i1]
            
            if _is_valid(net):
                # Accept
                if len(net) < best_len:
                    best = net[:]
                    best_len = len(net)
            else:
                # Reject: swap back
                net[i1], net[i2] = net[i2], net[i1]
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best

def _local_search(net, deadline):
    """Local search: try swapping adjacent comparators, moving comparators, etc."""
    net = list(net)
    best = net
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try moving a comparator to a different position
        idx = random.randint(0, len(net) - 1)
        new_idx = random.randint(0, len(net) - 1)
        if idx == new_idx:
            continue
        
        comp = net.pop(idx)
        net.insert(new_idx, comp)
        
        if _is_valid(net):
            if len(net) < len(best):
                best = net[:]
        
        net.pop(new_idx)
        net.insert(idx, comp)
        
        # Try swapping two comparators
        if len(net) >= 2:
            i1 = random.randint(0, len(net) - 1)
            i2 = random.randint(0, len(net) - 1)
            if i1 != i2:
                net[i1], net[i2] = net[i2], net[i1]
                if _is_valid(net):
                    if len(net) < len(best):
                        best = net[:]
                net[i1], net[i2] = net[i2], net[i1]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second budget, 10 second safety margin
    
    # Start with Batcher's odd-even mergesort
    net = _batcher_sort(N)
    
    if not _is_valid(net):
        # Fallback to odd-even transposition
        net = _odd_even_transposition()
    
    # Aggressive removal first
    net = _remove_redundant(net, deadline)
    
    # Simulated annealing for deeper exploration
    if time.time() < deadline - 10:
        net = _simulated_annealing(net, deadline - 5)
    
    # Final cleanup
    net = _remove_redundant(net, deadline)
    
    return net
