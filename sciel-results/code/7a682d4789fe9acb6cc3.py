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

def _greedy_remove_redundant(nets, deadline):
    """
    Try to remove redundant comparators one by one.
    A comparator is redundant if the network still sorts without it.
    """
    nets = list(nets)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(nets) - 1, -1, -1):  # Remove from end to preserve order
            if time.time() >= deadline:
                break
            # Try removing this comparator
            trial = nets[:idx] + nets[idx+1:]
            if _verify(trial):
                nets = trial
                changed = True
                break  # Restart from beginning since network changed
    return nets

def _local_search_improve(nets, deadline):
    """
    Local search: try removing, swapping, and inserting comparators.
    """
    nets = list(nets)
    best_net = list(nets)
    best_len = len(nets)
    last_improve = time.time()
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline and T >= T_min:
        trial = list(nets)
        move_type = random.random()
        
        if move_type < 0.5:
            # Delete
            if len(trial) > 0:
                idx = random.randint(0, len(trial) - 1)
                trial.pop(idx)
        elif move_type < 0.8:
            # Insert
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = random.randint(0, len(trial))
            trial.insert(pos, (i, j))
        else:
            # Swap adjacent
            if len(trial) > 1:
                idx = random.randint(0, len(trial) - 2)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
        
        if _verify(trial):
            delta = len(trial) - len(nets)
            if delta < 0:
                nets = trial
                if len(nets) < best_len:
                    best_net = list(nets)
                    best_len = len(nets)
                    last_improve = time.time()
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    nets = trial
        T *= alpha
        
        # Restart if stuck
        if time.time() - last_improve > 5.0:
            nets = list(best_net)
            T = max(T, 0.5)
            last_improve = time.time()
    
    return best_net

def construct():
    """
    Construct a minimal sorting network on 16 wires.
    Strategy:
    1. Start with Batcher's sort
    2. Greedy removal of redundant comparators
    3. Local search to further optimize
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Start with Batcher's sort
    current_net = _batcher_sort(N)
    
    # Verify start
    if not _verify(current_net):
        # Fallback to OET
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = list(current_net)
    best_len = len(current_net)
    
    # Phase 1: Greedy removal
    greedy_deadline = start_time + 30.0
    current_net = _greedy_remove_redundant(current_net, greedy_deadline)
    if len(current_net) < best_len:
        best_net = list(current_net)
        best_len = len(current_net)
    
    # Phase 2: Local search with remaining time
    if time.time() < deadline:
        current_net = _local_search_improve(current_net, deadline)
        if len(current_net) < best_len:
            best_net = list(current_net)
            best_len = len(current_net)
    
    return best_net
