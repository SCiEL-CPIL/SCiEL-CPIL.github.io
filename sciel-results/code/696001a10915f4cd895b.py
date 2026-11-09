"""Search-based constructor for size-efficient sorting networks on 18 wires."""
import time
import random

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_network(net, state):
    """Apply comparator network to a state (list of 0/1). Returns sorted state if correct."""
    for i, j in net:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_net(net):
    """Check if network sorts all 2^18 boolean inputs."""
    for mask in range(1 << N):
        state = [(mask >> i) & 1 for i in range(N)]
        for i, j in net:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        zeros = 0
        for v in state:
            if v == 0:
                zeros += 1
        for k in range(zeros):
            if state[k] != 0:
                return False
        for k in range(zeros, N):
            if state[k] != 1:
                return False
    return True

def _pack_bits_to_u64(net):
    """Pack comparator network into a compact representation for fast verification.
    Each comparator (i, j) is encoded as i*18 + j (since i < j < 18, this fits in 8 bits).
    We store the network as a list of integers for fast iteration."""
    return [i * N + j for i, j in net]

def _calc_popcount_u64(x):
    """Calculate popcount of a Python integer."""
    return bin(x).count('1')

def _verify_net_packed(packed_net):
    """Verify a packed network sorts all 2^18 boolean inputs using bit-parallel operations.
    Returns True if the network is valid."""
    for mask in range(1 << N):
        state = mask
        for enc in packed_net:
            i = enc // N
            j = enc % N
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state = (state & ~(1 << i)) | (1 << j)
        # Check sorted: all 0s at low indices, all 1s at high indices
        ones = _calc_popcount_u64(state)
        expected = ((1 << ones) - 1) << (N - ones)
        if state != expected:
            return False
    return True

def _verify_net_packed_partial(packed_net, masks):
    """Verify a packed network on a subset of masks. Returns number of failures."""
    failures = 0
    for mask in masks:
        state = mask
        for enc in packed_net:
            i = enc // N
            j = enc % N
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state = (state & ~(1 << i)) | (1 << j)
        ones = _calc_popcount_u64(state)
        expected = ((1 << ones) - 1) << (N - ones)
        if state != expected:
            failures += 1
    return failures

def _depth(net):
    """Compute parallel depth (longest path) of the network."""
    n = len(net)
    if n == 0:
        return 0
    return n

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _bubble_sort_network():
    """Generate a bubble sort network (adjacent comparators, multiple passes)."""
    net = []
    for _ in range(N):
        for i in range(N - 1):
            net.append((i, i + 1))
    return net

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    if n & (n - 1) != 0:
        return _odd_even_transposition()
    
    net = []
    for size in range(2, n + 1):
        if size & (size - 1) == 0:
            for step in range(1, size, 2):
                for i in range(0, n, size * 2):
                    for j in range(i, i + size - step):
                        a = j
                        b = j + step
                        net.append((a, b))
    return net

def _random_network(num_comparators, rng):
    """Generate a random network with a given number of comparators."""
    net = []
    for _ in range(num_comparators):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        net.append((i, j))
    return net

def _critical_inputs(net, sample_size=5000, rng=None):
    """Find a set of critical (unsatisfied) inputs for the given network.
    Returns a list of masks that are not sorted by the network."""
    if rng is None:
        rng = random.Random(42)
    
    # First try all inputs if feasible, otherwise sample
    critical = []
    all_masks = list(range(1 << N))
    rng.shuffle(all_masks)
    
    for mask in all_masks[:sample_size]:
        state = mask
        for i, j in net:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state = (state & ~(1 << i)) | (1 << j)
        ones = _calc_popcount_u64(state)
        expected = ((1 << ones) - 1) << (N - ones)
        if state != expected:
            critical.append(mask)
    
    return critical

def _hamming_distance_error(net, mask):
    """Compute the Hamming distance between the network's output and the sorted target for a given mask."""
    state = mask
    for i, j in net:
        bi = (state >> i) & 1
        bj = (state >> j) & 1
        if bi > bj:
            state = (state & ~(1 << i)) | (1 << j)
    ones = _calc_popcount_u64(state)
    expected = ((1 << ones) - 1) << (N - ones)
    return _calc_popcount_u64(state ^ expected)

def _total_hamming_error(net, masks):
    """Compute total Hamming distance error across a set of masks."""
    total = 0
    for mask in masks:
        state = mask
        for i, j in net:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state = (state & ~(1 << i)) | (1 << j)
        ones = _calc_popcount_u64(state)
        expected = ((1 << ones) - 1) << (N - ones)
        total += _calc_popcount_u64(state ^ expected)
    return total

def _greedy_add(net, critical_masks, max_comparators=100, rng=None):
    """Iteratively add the comparator that maximizes reduction in total Hamming distance.
    Returns the improved network."""
    if rng is None:
        rng = random.Random(123)
    
    current = list(net)
    current_packed = _pack_bits_to_u64(current)
    
    # Get current error
    current_error = _total_hamming_error(current, critical_masks)
    
    # Generate candidate comparators
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for _ in range(max_comparators):
        best_pair = None
        best_error = current_error
        best_reduction = 0
        
        # Sample a subset of candidate pairs for efficiency
        sample_size = min(len(all_pairs), 200)
        candidates = rng.sample(all_pairs, sample_size)
        
        for i, j in candidates:
            # Check if this comparator is already in the network
            already_in = any(ci == i and cj == j for ci, cj in current)
            if already_in:
                continue
            
            candidate = current + [(i, j)]
            error = _total_hamming_error(candidate, critical_masks)
            reduction = current_error - error
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_error = error
                best_pair = (i, j)
        
        if best_pair is None or best_reduction <= 0:
            break
        
        current.append(best_pair)
        current_error = best_error
    
    return current

def _local_search(net, critical_masks, max_iterations=2000, deadline=None, rng=None):
    """Perform local search via random comparator swaps and insertions.
    Guided by total Hamming distance error."""
    if rng is None:
        rng = random.Random(777)
    
    current = list(net)
    current_packed = _pack_bits_to_u64(current)
    current_error = _total_hamming_error(current, critical_masks)
    best = list(current)
    best_error = current_error
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Random perturbation
        new_net = list(current)
        move_type = rng.random()
        
        if move_type < 0.4 and len(new_net) > 0:
            # Remove a random comparator
            idx = rng.randint(0, len(new_net) - 1)
            new_net.pop(idx)
        elif move_type < 0.7 and len(new_net) > 1:
            # Swap two comparators
            i, j = rng.sample(range(len(new_net)), 2)
            new_net[i], new_net[j] = new_net[j], new_net[i]
        elif move_type < 0.9:
            # Insert a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(new_net))
            new_net.insert(pos, (i, j))
        else:
            # Move a comparator to a random position
            if len(new_net) > 1:
                idx = rng.randint(0, len(new_net) - 1)
                comp = new_net.pop(idx)
                new_pos = rng.randint(0, len(new_net))
                new_net.insert(new_pos, comp)
        
        # Evaluate
        new_error = _total_hamming_error(new_net, critical_masks)
        
        # Accept if better or with some probability for exploration
        if new_error < current_error:
            current = new_net
            current_error = new_error
            if new_error < best_error:
                best = list(current)
                best_error = new_error
        elif new_error <= current_error + 2:
            # Accept with some probability for exploration
            if rng.random() < 0.3:
                current = new_net
                current_error = new_error
    
    return best

def _remove_redundant(net, deadline=None):
    """Try to remove redundant comparators from the network."""
    current = list(net)
    best = list(current)
    
    for iteration in range(100):
        if deadline and time.time() > deadline:
            break
        if not current:
            break
        
        improved = False
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if deadline and time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_net_packed(_pack_bits_to_u64(candidate)):
                current = candidate
                best = list(current)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Start with the seed network
    seed_net = _odd_even_transposition()
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Verify seed
    if not _verify_net_packed(_pack_bits_to_u64(seed_net)):
        raise ValueError("Seed network is invalid")
    
    deadline = start_time + DEADLINE
    
    # Generate critical inputs from the seed network
    rng = random.Random(42)
    critical_masks = _critical_inputs(seed_net, sample_size=3000, rng=rng)
    
    # If no critical inputs, the seed is already perfect (unlikely)
    if not critical_masks:
        # Try to reduce the network
        reduced = _remove_redundant(seed_net, deadline=deadline)
        if len(reduced) < best_len:
            best_net = reduced
            best_len = len(reduced)
        return best_net
    
    # Phase 1: Greedy addition to find a good starting point
    greedy_net = _greedy_add(seed_net, critical_masks, max_comparators=50, rng=rng)
    
    # Verify greedy result
    if _verify_net_packed(_pack_bits_to_u64(greedy_net)):
        if len(greedy_net) < best_len:
            best_net = greedy_net
            best_len = len(greedy_net)
    else:
        # If greedy didn't produce a valid network, try local search from seed
        greedy_net = seed_net
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 10:
        improved = _local_search(greedy_net, critical_masks, max_iterations=3000, deadline=deadline - 5, rng=rng)
        
        # Verify improved network
        if _verify_net_packed(_pack_bits_to_u64(improved)):
            if len(improved) < best_len:
                best_net = improved
                best_len = len(improved)
        else:
            # Try to fix by removing problematic comparators
            fixed = _remove_redundant(improved, deadline=deadline - 5)
            if _verify_net_packed(_pack_bits_to_u64(fixed)):
                if len(fixed) < best_len:
                    best_net = fixed
                    best_len = len(fixed)
    
    # Phase 3: Try to remove redundant comparators from best
    if time.time() < deadline - 5:
        reduced = _remove_redundant(best_net, deadline=deadline)
        if _verify_net_packed(_pack_bits_to_u64(reduced)):
            if len(reduced) < best_len:
                best_net = reduced
                best_len = len(reduced)
    
    # Final verification
    if not _verify_net_packed(_pack_bits_to_u64(best_net)):
        # Fallback to seed
        best_net = seed_net
    
    if len(best_net) > 4096:
        best_net = best_net[:4096]
    
    return best_net

# EVOLVE-BLOCK-END
