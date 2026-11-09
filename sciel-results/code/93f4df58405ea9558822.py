"""Size-efficient sorting networks on 17 wires using bitmask-based zero-one principle verification."""
import time

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
TIME_LIMIT = 80.0  # seconds, leaving margin for the 90s wall-clock limit


def _make_bitmasks():
    """Create the 17 bitmask integers representing each wire's state across all 131072 Boolean inputs.
    Wire i is 1 in input x iff bit i of x is set. So bitmask_i has bit x set iff (x >> i) & 1 == 1.
    """
    masks = []
    for i in range(N):
        m = 0
        step = 1 << (i + 1)
        half = 1 << i
        # For each block of size step, the lower half has bit i = 0, upper half has bit i = 1
        # Build using pattern repetition
        # Pattern: 0*half then 1*half, repeated step/step times = 2^(N-i-1) times
        # Total bits: 2^N
        pattern = ((1 << half) - 1) << half  # lower half 0s, upper half 1s
        # Repeat pattern (1 << (N - i - 1)) times
        count = 1 << (N - i - 1)
        m = 0
        for _ in range(count):
            m = (m << step) | pattern
        masks.append(m)
    return masks


def _verify_sort(masks):
    """Check if after applying the network, wire 0 is all 0s and wire N-1 is all 1s.
    For a correct sorting network on Boolean inputs, this is sufficient (zero-one principle).
    Actually we need: output wire i should have exactly i ones... no.
    For Boolean inputs, a sorting network is correct iff for every input, the output is sorted.
    The output is sorted iff wire 0 = 0 and wire N-1 = 1 for ALL inputs? No.
    
    Actually, the zero-one principle says: a network sorts all inputs iff it sorts all 0-1 inputs.
    A 0-1 input is sorted iff all 0s come before all 1s. The number of 1s is preserved.
    So for a given input with k ones, the sorted output has wires 0..N-k-1 = 0 and wires N-k..N-1 = 1.
    
    To check correctness: for each possible k (0 to N), the set of inputs with exactly k ones must produce
    output where wires N-k..N-1 are 1 and wires 0..N-k-1 are 0.
    
    But there's a simpler check: the network is correct iff for every input, the output is non-decreasing.
    For Boolean values, non-decreasing means once a 1 appears, all subsequent wires are 1.
    This is equivalent to: for each adjacent pair (i, i+1), we never have wire i = 1 and wire i+1 = 0.
    So: for all i in 0..N-2, (mask[i] & ~mask[i+1]) == 0.
    
    Even simpler: the total number of 1s in each output wire across all inputs must match the input.
    Actually the simplest full check: for each input x, the output must be sorted.
    
    Let me use the adjacent-pair check: for all i, (out[i] & ~out[i+1]) == 0.
    """
    for i in range(N - 1):
        if (masks[i] & ~masks[i + 1]) != 0:
            return False
    return True


def _apply_comparator(masks, i, j):
    """Apply compare-exchange on wires i and j: min goes to i, max goes to j.
    For Boolean: min(a,b) = a & b, max(a,b) = a | b.
    Returns new masks list (copies affected wires).
    """
    a = masks[i]
    b = masks[j]
    masks[i] = a & b
    masks[j] = a | b
    return masks


def construct():
    start_time = time.time()
    
    # Generate seed network: odd-even transposition
    # For N wires, N phases. Phase p (0-indexed): compare (p%2, p%2+2, ...)
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify seed
    masks = _make_bitmasks()
    for i, j in network:
        _apply_comparator(masks, i, j)
    assert _verify_sort(masks), "Seed network should be correct"
    
    # Greedy removal: try to remove each comparator and check if network still sorts
    # We'll do multiple passes
    best_network = list(network)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5.0:
            break
        
        made_progress = False
        # Try removing each comparator
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT - 3.0:
                break
            
            # Remove comparator at idx
            trial = best_network[:idx] + best_network[idx + 1:]
            
            # Verify
            masks = _make_bitmasks()
            valid = True
            for i, j in trial:
                a = masks[i]
                b = masks[j]
                masks[i] = a & b
                masks[j] = a | b
            
            # Check sorted
            for i in range(N - 1):
                if (masks[i] & ~masks[i + 1]) != 0:
                    valid = False
                    break
            
            if valid:
                best_network = trial
                made_progress = True
        
        if not made_progress:
            break
    
    return best_network


# Pre-compute masks for speed
_MASKS = _make_bitmasks()


def _verify_network_fast(net):
    """Fast verification using pre-computed masks."""
    masks = list(_MASKS)
    for i, j in net:
        a = masks[i]
        b = masks[j]
        masks[i] = a & b
        masks[j] = a | b
    for i in range(N - 1):
        if (masks[i] & ~masks[i + 1]) != 0:
            return False
    return True


def construct():
    start_time = time.time()
    
    # Generate seed network: odd-even transposition
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify seed
    assert _verify_network_fast(network), "Seed network should be correct"
    
    best_network = list(network)
    
    # Greedy removal: try to remove each comparator and check if network still sorts
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5.0:
            break
        
        made_progress = False
        # Try removing each comparator from back to front
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT - 3.0:
                break
            
            # Remove comparator at idx
            trial = best_network[:idx] + best_network[idx + 1:]
            
            if _verify_network_fast(trial):
                best_network = trial
                made_progress = True
        
        if not made_progress:
            break
    
    return best_network
