"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
DEADLINE = time.time() + 85  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_comparator_to_bits(bits, i, j):
    """Apply comparator (i,j) to a 18-bit integer. Returns new bits."""
    bi = (bits >> i) & 1
    bj = (bits >> j) & 1
    if bi > bj:
        # swap: clear i, set j
        bits ^= (1 << i) | (1 << j)
    return bits

def _verify_network(comparators):
    """Verify that the network sorts all 2^18 boolean inputs.
    Uses the 0/1 principle: it suffices to check all 2^N inputs.
    For speed, we check all 2^18 = 262144 inputs using vectorized numpy."""
    # Represent each input as an 18-bit integer
    # We need to check that after applying all comparators, the output is sorted
    # A boolean vector is sorted iff all bits 0..17 are non-decreasing, i.e., once a 1 appears, all subsequent are 1.
    # Equivalently: for all k in 0..16, if bit k is 1 then bit k+1 must be 1.
    # Or: the output must be of the form 0...01...1
    
    inputs = np.arange(1 << N, dtype=np.uint32)
    # Apply each comparator
    for (i, j) in comparators:
        mask_i = np.uint32(1) << i
        mask_j = np.uint32(1) << j
        # bit i > bit j means bit i is 1 and bit j is 0
        bi = (inputs & mask_i) != 0
        bj = (inputs & mask_j) != 0
        # swap where bi and not bj
        swap_mask = bi & (~bj)
        # XOR both bits
        inputs = np.where(swap_mask, inputs ^ mask_i ^ mask_j, inputs)
    
    # Check sorted: for each input, it should be of form 0^k 1^(N-k)
    # i.e., there exists k such that bits 0..k-1 are 0 and bits k..N-1 are 1
    # Equivalently: if we look at the binary representation, it should be all zeros followed by all ones
    # Check: for each position k from 0 to N-2, if bit k is 1 then bit k+1 must be 1
    # Or simpler: the value should equal ((1 << N) - 1) - ((1 << k) - 1) for some k, i.e., it's a suffix of 1s
    # Actually, sorted means: the integer, when viewed as bits 0..17, has the property that
    # bit k <= bit k+1 for all k. This means the pattern is 0...01...1.
    # Such numbers are exactly: 0, 1<<17, (1<<17)+(1<<16), ..., (1<<18)-1
    # i.e., values where the set bits form a contiguous suffix at the high end.
    
    # Check: for each input value v, verify it's sorted
    # v is sorted iff for all k in 0..N-2: bit k of v <= bit k+1 of v
    # Equivalent: v & (v >> 1) == v >> 1 is NOT the right check.
    # Let me think: v is sorted (non-decreasing bits) iff v has form 0...01...1
    # This means: v = ((1 << N) - 1) ^ ((1 << k) - 1) for some k in 0..N
    # i.e., v = (1 << N) - 1 - ((1 << k) - 1) = (1 << N) - (1 << k)
    # Wait no. If bits are 0,0,...,0,1,1,...,1 with k zeros and N-k ones:
    # value = sum_{i=k}^{N-1} (1 << i) = (1 << N) - (1 << k)
    # So valid sorted values are: 0, 1<<17, (1<<18 - 1<<16), ..., (1<<18)-1
    # i.e., (1 << N) - (1 << k) for k = 0, 1, ..., N
    # When k=0: (1<<18) - 1 = all ones (N ones, 0 zeros)
    # When k=1: (1<<18) - 2 = all ones except bit 0 is 0
    # When k=18: 0 - 0... wait (1<<18) - (1<<18) = 0
    
    # So the set of valid sorted outputs is { (1 << N) - (1 << k) for k in range(N+1) }
    valid = set((1 << N) - (1 << k) for k in range(N + 1))
    
    # Check all inputs
    for v in inputs:
        if v not in valid:
            return False
    return True

def _verify_network_fast(comparators):
    """Faster verification using the 0/1 principle with numpy vectorization."""
    inputs = np.arange(1 << N, dtype=np.uint32)
    for (i, j) in comparators:
        mask_i = np.uint32(1) << i
        mask_j = np.uint32(1) << j
        bi = (inputs & mask_i) != 0
        bj = (inputs & mask_j) != 0
        swap_mask = bi & (~bj)
        inputs = np.where(swap_mask, inputs ^ mask_i ^ mask_j, inputs)
    
    # Check if each input is a valid sorted pattern
    # Valid: v = (1<<N) - (1<<k) for k in 0..N
    # Equivalently: v == 0 or (v & (v+1)) == 0 and v < (1<<N) and v has trailing zeros then ones
    # Actually simpler: v is sorted iff v == 0 or (v | (v+1)) == (1<<N)-1 ... no
    # Let's use: v is sorted iff for the binary representation, all 1s are to the left of all 0s
    # i.e., v has form: some number of 1s at high bits, then 0s at low bits
    # v & (v >> 1) should have the same set of bits as v shifted... 
    
    # Simplest vectorized check:
    # v is sorted iff v == ((1<<N) - 1) ^ ((1<<k) - 1) for some k
    # i.e., v == (1<<N) - 1 - ((1<<k) - 1) = (1<<N) - (1<<k)
    # So v + (1<<k) == (1<<N) for some k
    # Equivalently: v has exactly one transition from 1 to 0 (or all 1s or all 0s)
    # Check: popcount of v must equal popcount of (v & (v+1)) ... no
    
    # Vectorized: check that v & (v + 1) == 0 is NOT right.
    # Let me just check: v is of form (1<<N)-(1<<k) iff v | ((1<<N)-1 - v) == (1<<N)-1 and 
    # the 1s in v are a contiguous suffix.
    # 
    # Easiest: for each v, check if it's in the valid set. Use numpy:
    # valid_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.uint32)
    # Then check membership. But that's O(N*2^N) which is fine for N=18.
    
    valid_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.uint32)
    # Check: for each input, is it in valid_values?
    # Use broadcasting: inputs[:, None] == valid_values[None, :]
    # That's 2^18 * 19 = ~5M comparisons, fast enough.
    matches = (inputs[:, None] == valid_values[None, :])
    if not matches.any(axis=1).all():
        return False
    return True

def _count_violations(comparators):
    """Count how many of the 2^18 inputs are not sorted after the network."""
    inputs = np.arange(1 << N, dtype=np.uint32)
    for (i, j) in comparators:
        mask_i = np.uint32(1) << i
        mask_j = np.uint32(1) << j
        bi = (inputs & mask_i) != 0
        bj = (inputs & mask_j) != 0
        swap_mask = bi & (~bj)
        inputs = np.where(swap_mask, inputs ^ mask_i ^ mask_j, inputs)
    
    valid_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.uint32)
    matches = (inputs[:, None] == valid_values[None, :])
    not_sorted = ~matches.any(axis=1)
    return int(not_sorted.sum())

def _make_seed_network():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_remove(comparators, max_iters=50):
    """Try to remove comparators one at a time if the network still sorts."""
    comps = list(comparators)
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        removed = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network(candidate):
                comps = candidate
                removed = True
                break
        if not removed:
            break
    return comps

def _add_comparator(comparators, i, j):
    """Add comparator (i,j) at the end."""
    return comparators + [(i, j)]

def _remove_comparator(comparators, idx):
    """Remove comparator at index idx."""
    return comparators[:idx] + comparators[idx+1:]

def _swap_comparators(comparators, idx1, idx2):
    """Swap two comparators."""
    comps = list(comparators)
    comps[idx1], comps[idx2] = comps[idx2], comps[idx1]
    return comps

def _simulated_annealing(start_comps, budget=60):
    """Simulated annealing to minimize comparator count."""
    comps = list(start_comps)
    current_count = len(comps)
    best_comps = list(comps)
    best_count = current_count
    
    T = 10.0
    cooling = 0.995
    start_time = time.time()
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < start_time + budget and T > 0.01:
        # Random move
        move_type = random.random()
        
        if move_type < 0.3 and len(comps) > 1:
            # Remove a random comparator
            idx = random.randint(0, len(comps) - 1)
            new_comps = _remove_comparator(comps, idx)
            delta = -1
        elif move_type < 0.6:
            # Add a random comparator
            i, j = random.choice(all_pairs)
            new_comps = _add_comparator(comps, i, j)
            delta = 1
        elif move_type < 0.8:
            # Swap two comparators
            if len(comps) >= 2:
                idx1 = random.randint(0, len(comps) - 1)
                idx2 = random.randint(0, len(comps) - 1)
                if idx1 != idx2:
                    new_comps = _swap_comparators(comps, idx1, idx2)
                    delta = 0
                else:
                    T *= cooling
                    continue
            else:
                T *= cooling
                continue
        else:
            # Replace a comparator
            if len(comps) > 0:
                idx = random.randint(0, len(comps) - 1)
                i, j = random.choice(all_pairs)
                new_comps = list(comps)
                new_comps[idx] = (i, j)
                delta = 0
            else:
                T *= cooling
                continue
        
        new_count = len(new_comps)
        # Acceptance: always accept if fewer comparators, otherwise with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            comps = new_comps
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_comps = list(comps)
        
        T *= cooling
    
    return best_comps, best_count

def _local_search_remove(comparators, budget=30):
    """Aggressively try to remove comparators."""
    comps = list(comparators)
    start_time = time.time()
    
    improved = True
    while improved and time.time() < start_time + budget:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > start_time + budget:
                break
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network(candidate):
                comps = candidate
                improved = True
                break
    
    return comps

def _bidirectional_reachability_search(budget=40):
    """
    Try to find a small sorting network using a search approach.
    Start from a known good network and try to reduce it.
    Also try constructing from scratch using a greedy approach.
    """
    start_time = time.time()
    
    # Start from seed
    seed = _make_seed_network()
    
    # First, try aggressive removal
    reduced = _local_search_remove(seed, budget=min(budget, 20))
    
    # Then simulated annealing
    if time.time() < start_time + budget:
        remaining = budget - (time.time() - start_time)
        if remaining > 5:
            sa_result, sa_count = _simulated_annealing(reduced, budget=remaining)
            if sa_count < len(reduced):
                reduced = sa_result
    
    # Final greedy pass
    if time.time() < start_time + budget:
        reduced = _local_search_remove(reduced, budget=max(0, budget - (time.time() - start_time)))
    
    return reduced

def _construct_from_known_pattern():
    """Try a known smaller network structure for 18 wires."""
    # Use a batcher-style or AKS-inspired construction
    # For now, use odd-even merge as a base and try to optimize
    # Odd-even merge sort network for 18 wires
    def odd_even_merge_sort(n, offset=0):
        """Generate odd-even merge sort network for n wires starting at offset."""
        if n <= 1:
            return []
        comps = []
        # Split into halves
        half = n // 2
        # Recursively sort each half
        comps += odd_even_merge_sort(half, offset)
        comps += odd_even_merge_sort(n - half, offset + half)
        # Merge
        comps += odd_even_merge(half, n - half, offset)
        return comps
    
    def odd_even_merge(n1, n2, offset):
        """Merge two sorted sequences of length n1 and n2."""
        if n1 == 0:
            return []
        if n2 == 0:
            return []
        comps = []
        # Odd-even merge
        # Compare elements at positions offset+i and offset+n1+i for i in 0..min(n1,n2)-1
        m = min(n1, n2)
        for i in range(m):
            comps.append((offset + i, offset + n1 + i))
        # Recursively merge the odd and even subsequences
        # This is complex; let's use a simpler approach
        # Just do a full merge network
        return comps
    
    # This is getting complex. Let's just use the seed and optimize.
    return _make_seed_network()

# EVOLVE-BLOCK-END

def construct():
    """Main entry point. Returns a list of (i, j) comparators that sort all 18-input sequences."""
    start_time = time.time()
    
    # Start with seed network
    network = _make_seed_network()
    
    # Phase 1: Aggressive local search removal
    if time.time() < DEADLINE - 30:
        network = _local_search_remove(network, budget=min(30, DEADLINE - 30 - (time.time() - start_time)))
    
    # Phase 2: Simulated annealing
    remaining = DEADLINE - 10 - (time.time() - start_time)
    if remaining > 10:
        sa_net, sa_count = _simulated_annealing(network, budget=min(remaining, 50))
        if sa_count < len(network):
            network = sa_net
    
    # Phase 3: Final greedy removal
    remaining = DEADLINE - 5 - (time.time() - start_time)
    if remaining > 2:
        network = _local_search_remove(network, budget=max(0, remaining))
    
    # Verify the final network
    if not _verify_network(network):
        # Fallback to seed
        network = _make_seed_network()
    
    return network
