"""Size-efficient sorting network search for 16 wires."""
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
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
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

def _odd_even_transposition():
    """Standard odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

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

def _remove_redundant(nets, deadline):
    """Remove redundant comparators via iterative deletion."""
    net = list(nets)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if _verify(candidate):
                net = candidate
                improved = True
    return net

def construct():
    """Search for a minimal sorting network on 16 wires."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin for safety
    
    # Start with known good networks
    candidates = []
    
    # Odd-even transposition: 120 comparators
    oet = _odd_even_transposition()
    candidates.append(oet)
    
    # Batcher's sort
    batcher = _batcher_sort(N)
    candidates.append(batcher)
    
    # Find the best starting point
    best = None
    best_len = float('inf')
    for net in candidates:
        if time.time() > deadline - 5.0:
            break
        if _verify(net):
            if len(net) < best_len:
                best = list(net)
                best_len = len(net)
    
    if best is None:
        return oet  # Fallback to odd-even transposition
    
    # Local search: iterative removal
    current = best
    current_len = best_len
    
    # Phase 1: Aggressive removal
    while time.time() < deadline - 10.0:
        prev_len = current_len
        current = _remove_redundant(current, deadline - 10.0)
        current_len = len(current)
        
        if current_len == prev_len:
            break
        if current_len < best_len:
            best = list(current)
            best_len = current_len
    
    # Phase 2: Try adding comparators to escape local optima
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Add a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            
            trial = list(current) + [(i, j)]
            if _verify(trial):
                # Try to remove from this new network
                trial = _remove_redundant(trial, deadline - 5.0)
                if len(trial) < best_len:
                    best = list(trial)
                    best_len = len(trial)
                    current = trial
    
    # Final cleanup
    if time.time() < deadline - 2.0:
        best = _remove_redundant(best, deadline - 2.0)
    
    return best
