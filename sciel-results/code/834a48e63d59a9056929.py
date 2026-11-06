"""Size-efficient sorting networks on 17 wires."""
import time
import random
import numpy as np

N = 17
MASK = (1 << N) - 1

# Precompute the 131072 Boolean inputs as a 2D array of uint8
# rows: 131072, cols: 17
# x[b, w] = 1 if bit w of b is set
_ALL_BITS = np.arange(1 << N, dtype=np.uint64)[:, None]
_WIRE_BITS = (1 << np.arange(N, dtype=np.uint64))[None, :]
X0 = ((_ALL_BITS & _WIRE_BITS) != 0).astype(np.uint8)  # shape (131072, 17)

def _apply_comp(X, i, j):
    """Apply comparator (i,j): min to i, max to j. X is (131072, 17) uint8."""
    a = X[:, i].copy()
    b = X[:, j].copy()
    X[:, i] = np.minimum(a, b)
    X[:, j] = np.maximum(a, b)
    return X

def _apply_comp_mask(mask, i, j):
    """Apply comparator (i,j) to a single bitmask (int). Returns new mask."""
    a = (mask >> i) & 1
    b = (mask >> j) & 1
    if a == b:
        return mask
    # a=1,b=0 -> clear i, set j ; a=0,b=1 -> set i, clear j
    if a == 1 and b == 0:
        mask ^= (1 << i) | (1 << j)
    else:  # a=0, b=1
        mask ^= (1 << i) | (1 << j)
    return mask

def _is_sorted_mask(mask):
    """Check if bitmask is sorted: bits set from MSB down (i.e., all 1s then all 0s).
    A sorted 17-bit mask has form: top k bits are 1, rest 0.
    So for some k in [0,17], mask == (1<<17 - 1) ^ (1<<k - 1) if we consider
    wire 16 as MSB... actually let's think: after sorting, the largest values
    are on the highest-indexed wires? No — the comparator puts min on i, max on j
    where i<j. So after sorting, wire 0 has the smallest, wire 16 has the largest.
    For a Boolean input, sorted means: there exists k such that wires 0..k-1 are 0
    and wires k..16 are 1.
    So mask should be: (MASK ^ ((1 << k) - 1)) for some k, i.e., lower k bits 0, upper 17-k bits 1.
    Equivalently: mask >> k == (1 << (N - k)) - 1 and (mask & ((1<<k)-1)) == 0.
    Simpler: the mask should be of the form 000...0111...111.
    """
    # A mask is sorted iff it has no pattern '10' in the wire order (0 to 16).
    # i.e., once we see a 0, all subsequent must be 0.
    # Equivalent: mask & (mask - 1) == 0 is NOT right (that's power of 2).
    # Let's just check: for each position, if bit is 0, all higher bits must be 0.
    # Fast way: the mask must be of the form ((1 << (N - k)) - 1) << k for some k.
    # There are only 18 such masks. Precompute them.
    # Actually even simpler: count trailing zeros. If mask has t trailing zeros,
    # then it's sorted iff mask == ((1 << (N - t)) - 1) << t.
    if mask == 0:
        return True
    # Count trailing zeros
    tz = 0
    m = mask
    while (m & 1) == 0:
        tz += 1
        m >>= 1
    # Now m should be all 1s for the remaining N - tz bits
    expected = ((1 << (N - tz)) - 1) << tz
    return mask == expected

def _validate_network_fast(comps):
    """Validate using bitmask simulation over all 131072 inputs.
    Returns True if network sorts all inputs."""
    # For each input mask, simulate the network
    # To speed up: represent the state of all 131072 masks in a numpy array of uint32
    masks = _ALL_BITS.astype(np.uint32).copy()
    for i, j in comps:
        a = (masks >> i) & 1
        b = (masks >> j) & 1
        # Where a=1, b=0: swap bits
        swap = (a == 1) & (b == 0)
        if np.any(swap):
            bit_i = np.uint32(1) << np.uint32(i)
            bit_j = np.uint32(1) << np.uint32(j)
            masks[swap] ^= (bit_i | bit_j)
    # Check all are sorted
    # For each mask, check if it's of the form 000...0111...1
    # A mask is sorted iff: for the number of set bits k, the mask equals
    # ((1 << k) - 1) << (N - k)
    # Count set bits
    popcount = np.zeros(len(masks), dtype=np.uint8)
    m = masks.copy()
    for _ in range(N):
        popcount += (m & 1).astype(np.uint8)
        m >>= 1
    # Expected sorted mask for each popcount k: ((1<<k)-1) << (N-k)
    # k ranges 0..17
    expected_masks = np.array([((1 << k) - 1) << (N - k) for k in range(N + 1)], dtype=np.uint32)
    # For each mask, compute expected based on its popcount
    expected = expected_masks[popcount]
    return np.all(masks == expected)

def _validate_network_mask(comps, max_checks=None):
    """Validate by checking a subset of masks for speed during search."""
    if max_checks is None or max_checks >= (1 << N):
        return _validate_network_fast(comps)
    # Check a random subset
    rng = np.random.default_rng(42)
    sample = rng.choice(1 << N, size=max_checks, replace=False).astype(np.uint32)
    masks = sample.copy()
    for i, j in comps:
        a = (masks >> i) & 1
        b = (masks >> j) & 1
        swap = (a == 1) & (b == 0)
        if np.any(swap):
            bit_i = np.uint32(1) << np.uint32(i)
            bit_j = np.uint32(1) << np.uint32(j)
            masks[swap] ^= (bit_i | bit_j)
    popcount = np.zeros(len(masks), dtype=np.uint8)
    m = masks.copy()
    for _ in range(N):
        popcount += (m & 1).astype(np.uint8)
        m >>= 1
    expected_masks = np.array([((1 << k) - 1) << (N - k) for k in range(N + 1)], dtype=np.uint32)
    expected = expected_masks[popcount]
    return np.all(masks == expected)

def _odd_even_transposition():
    """Standard odd-even transposition network for N wires: N^2/2 comparators."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def _bubble_sort_network():
    """Bubble sort network: N*(N-1)/2 comparators."""
    comps = []
    for i in range(N):
        for j in range(N - 1 - i):
            comps.append((j, j + 1))
    return comps

def _remove_redundant(comps, deadline):
    """Try to remove redundant comparators one at a time (from the end)."""
    comps = list(comps)
    n = len(comps)
    # Try removing each comparator and check validity
    # To be fast, check with a subset first
    check_size = min(4096, 1 << N)
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline - 2.0:
            break
        candidate = comps[:idx] + comps[idx + 1:]
        if _validate_network_mask(candidate, max_checks=check_size):
            # Full validation
            if _validate_network_fast(candidate):
                comps = candidate
    return comps

def _simulated_annealing(comps, deadline, rng):
    """Simulated annealing to reduce comparator count."""
    best = list(comps)
    best_len = len(best)
    current = list(comps)
    current_len = len(current)
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    check_size = 2048
    steps = 0
    
    while time.time() < deadline - 3.0 and T > T_min:
        steps += 1
        # Move: try removing a random comparator
        if len(current) > 63:
            move_type = rng.random()
            if move_type < 0.6 and len(current) > 1:
                # Remove a random comparator
                idx = rng.integers(0, len(current))
                candidate = current[:idx] + current[idx + 1:]
            elif move_type < 0.8:
                # Swap two adjacent comparators
                if len(current) < 2:
                    T *= cooling
                    continue
                idx = rng.integers(0, len(current) - 1)
                candidate = current.copy()
                candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            else:
                # Replace a random comparator with a random valid one
                if len(current) < 1:
                    T *= cooling
                    continue
                idx = rng.integers(0, len(current))
                i = rng.integers(0, N - 1)
                j = rng.integers(i + 1, N)
                candidate = current.copy()
                candidate[idx] = (i, j)
        else:
            # Already at or below lower bound, just try swaps
            if len(current) < 2:
                T *= cooling
                continue
            idx = rng.integers(0, len(current) - 1)
            candidate = current.copy()
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
        
        # Evaluate candidate
        if not _validate_network_mask(candidate, max_checks=check_size):
            T *= cooling
            continue
        
        # Full validation
        if not _validate_network_fast(candidate):
            T *= cooling
            continue
        
        new_len = len(candidate)
        if new_len <= current_len:
            current = candidate
            current_len = new_len
            if new_len < best_len:
                best = candidate
                best_len = new_len
        else:
            # Accept with probability exp(-(new_len - current_len)/T)
            delta = new_len - current_len
            prob = np.exp(-delta / max(T, 1e-9))
            if rng.random() < prob:
                current = candidate
                current_len = new_len
        
        T *= cooling
    
    return best, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with odd-even transposition
    comps = _odd_even_transposition()
    # Verify it's valid
    assert _validate_network_fast(comps), "Odd-even transposition should be valid"
    
    best = comps
    best_len = len(comps)
    
    # Phase 1: Redundancy removal
    if time.time() < deadline - 10.0:
        best = _remove_redundant(best, deadline - 10.0)
        best_len = len(best)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(12345)
        best, best_len = _simulated_annealing(best, deadline - 5.0, rng)
    
    # Phase 3: Final redundancy removal on the result
    if time.time() < deadline - 2.0:
        best = _remove_redundant(best, deadline - 2.0)
    
    return best
