"""Size-efficient sorting networks on 18 wires - structural approach using known sub-networks."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, safety margin before 90s kill


def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comparators.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
    return comparators


def _batcher_odd_even_merge(n):
    """
    Generate a Batcher odd-even merge network for n inputs (n must be power of 2).
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    # Split into two halves
    half = n // 2
    left = _batcher_odd_even_merge(half)
    right = [ (i + half, j + half) for (i, j) in _batcher_odd_even_merge(half) ]
    
    comparators = left + right
    
    # Odd-even merge: merge the two sorted halves
    for k in range(half):
        for i in range(k, half, 2):
            # Compare (i, half + i)
            comparators.append((i, half + i))
            if i + 1 < half:
                # Compare (i+1, half + i)
                comparators.append((i + 1, half + i))
    
    return comparators


def _batcher_sort_network(n):
    """Generate a full Batcher sorting network for n inputs (n must be power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _batcher_sort_network(half)
    right = [ (i + half, j + half) for (i, j) in _batcher_sort_network(half) ]
    
    comparators = left + right
    
    # Odd-even merge to combine the two sorted halves
    merge = _batcher_odd_even_merge(n)
    # The merge function above already includes the recursive sorts, so we need a clean merge
    # Let's redo this properly
    
    return _batcher_sort_network_clean(n)


def _batcher_sort_network_clean(n):
    """Clean implementation of Batcher's bitonic sort."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For bitonic sort, we need a different approach
    # Use the standard recursive construction
    return _bitonic_sort(n)


def _bitonic_sort(n):
    """Bitonic sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half
    left = _bitonic_sort(half)
    # Sort second half
    right = [ (i + half, j + half) for (i, j) in _bitonic_sort(half) ]
    
    comparators = left + right
    
    # Now we have two sorted halves. Create a bitonic sequence and merge.
    # To create bitonic: sort first half ascending, second half descending
    # Then apply bitonic merge
    
    # For simplicity, let's use a different well-known network
    return _known_16_wire_network() if n == 16 else _batcher_merge_sort(n)


def _batcher_merge_sort(n):
    """Batcher's merge sort network."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _batcher_merge_sort(half)
    right = [ (i + half, j + half) for (i, j) in _batcher_merge_sort(half) ]
    
    comparators = left + right
    
    # Merge the two sorted halves using odd-even merge
    comparators += _odd_even_merge(half)
    
    return comparators


def _odd_even_merge(half):
    """Odd-even merge network for merging two sorted halves of size 'half'."""
    comparators = []
    for k in range(half):
        for i in range(k, half, 2):
            comparators.append((i, half + i))
            if i + 1 < half:
                comparators.append((i + 1, half + i))
    return comparators


def _known_16_wire_network():
    """
    Return a known good sorting network for 16 wires.
    Using a well-constructed network based on known results.
    """
    # Use a Batcher-based network for 16 wires
    # Batcher's bitonic sort for 16 uses 60 comparators
    # Let's construct it properly
    
    comparators = []
    
    # Phase 1: Sort groups of 2
    for i in range(0, 16, 2):
        comparators.append((i, i + 1))
    
    # Phase 2: Sort groups of 4 (merge two sorted pairs)
    for i in range(0, 16, 4):
        comparators.append((i, i + 2))
        comparators.append((i + 1, i + 3))
    
    # Phase 3: Sort groups of 8 (merge two sorted quads)
    for i in range(0, 16, 8):
        # Merge first two quads
        comparators.append((i, i + 4))
        comparators.append((i + 1, i + 5))
        comparators.append((i + 2, i + 6))
        comparators.append((i + 3, i + 7))
        # Internal comparisons
        comparators.append((i + 1, i + 4))
        comparators.append((i + 2, i + 5))
        comparators.append((i + 3, i + 6))
    
    # Phase 4: Sort all 16 (merge two sorted eights)
    for i in range(8):
        comparators.append((i, i + 8))
    for i in range(1, 8, 2):
        comparators.append((i, i + 8))
        comparators.append((i + 1, i + 8))
    for i in range(2, 8, 4):
        comparators.append((i, i + 8))
        comparators.append((i + 1, i + 8))
        comparators.append((i + 2, i + 8))
        comparators.append((i + 3, i + 8))
    
    # This is getting complex. Let me use a simpler approach:
    # Just use the odd-even transposition and optimize from there
    return _odd_even_transposition_network(16)


def _verify_network_subset(network, inputs):
    """Verify network on a subset of inputs."""
    if not network:
        return len(inputs) == 0
    
    n_inputs = len(inputs)
    # Work with copies
    arr = inputs.copy()
    
    for (i, j) in network:
        bit_i = (arr >> i) & 1
        bit_j = (arr >> j) & 1
        swap_mask = bit_j & (~bit_i)
        arr = arr ^ (swap_mask << i) ^ (swap_mask << j)
    
    # Check if all are sorted: sorted bitmask has form 2^N - 2^k
    # i.e., all 0s followed by all 1s
    for val in arr:
        # A sorted value in N bits: 0...011...1
        # This means val = 2^N - 2^k for some k, or val = 0 or val = 2^N - 1
        # Actually: sorted means for each position, if bit at pos is 0, all higher bits are 0
        # Equivalent: the binary representation is of the form 0^a 1^b
        if val == 0 or val == (1 << N) - 1:
            continue
        # Check: find the position where 0s turn to 1s
        # Count trailing zeros and leading ones
        # Simpler: check that there's no pattern 1 followed by 0
        # i.e., (val >> 1) & val should not have any 1s where val has 0s above
        # Actually: val is sorted iff (val & (val >> 1)) == 0 is NOT the right check
        # Let me think: sorted means 0...011...1
        # This means: for all i < j, if bit_i = 1 then bit_j = 1
        # Equivalently: once we see a 1, all higher bits must be 1
        # Check: let k be the number of trailing 1s. Then bits 0..k-1 are 1, bits k..N-1 are 0
        # So val = 2^k - 1 for some k? No, that's 1...10...0 which is reversed.
        # Sorted ascending: 0...011...1 means low bits are 0, high bits are 1
        # So val = (2^N - 1) - (2^k - 1) = 2^N - 2^k where k = number of leading 0s
        # Or equivalently: val + 1 is a power of 2 (if val != 0)
        if val != 0:
            v = val + 1
            if v & (v - 1) != 0:
                return False
    return True


def _verify_network_full(network, all_inputs):
    """Verify network on all 2^N inputs."""
    return _verify_network_subset(network, all_inputs)


def _apply_and_check(network, all_inputs):
    """Apply network and check all outputs are sorted."""
    if not network:
        # Check if all inputs are already sorted
        return _verify_network_subset([], all_inputs)
    
    arr = all_inputs.copy()
    
    for (i, j) in network:
        bit_i = (arr >> i) & 1
        bit_j = (arr >> j) & 1
        swap_mask = bit_j & (~bit_i)
        arr = arr ^ (swap_mask << i) ^ (swap_mask << j)
    
    # Check sortedness
    for val in arr:
        if val == 0:
            continue
        if val == (1 << N) - 1:
            continue
        v = val + 1
        if v & (v - 1) != 0:
            return False
    return True


def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Start with a good initial network
    # Use odd-even transposition as seed, then aggressively optimize
    initial_network = _odd_even_transposition_network(N)
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    # Verify initial
    if not _apply_and_check(best_network, all_inputs):
        return best_network
    
    rng = random.Random(12345)
    
    def try_remove_comparator(network, idx, all_inputs):
        """Try removing comparator at index idx. Returns True if network still valid."""
        if idx < 0 or idx >= len(network):
            return False
        candidate = network[:idx] + network[idx+1:]
        return _apply_and_check(candidate, all_inputs)
    
    def greedy_remove(network, all_inputs, max_passes=5):
        """Greedy removal of redundant comparators."""
        net = list(network)
        
        for _ in range(max_passes):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            removed_any = False
            # Try removing from the end first (often easier to remove)
            for idx in range(len(net) - 1, -1, -1):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                if try_remove_comparator(net, idx, all_inputs):
                    net = net[:idx] + net[idx+1:]
                    removed_any = True
            
            if not removed_any:
                # Try forward pass
                for idx in range(len(net)):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    
                    if try_remove_comparator(net, idx, all_inputs):
                        net = net[:idx] + net[idx+1:]
                        removed_any = True
            
            if not removed_any:
                break
        
        return net
    
    # Phase 1: Initial greedy removal
    best_network = greedy_remove(best_network, all_inputs, max_passes=10)
    best_count = len(best_network)
    
    # Phase 2: Local search - try reordering and removing
    no_improvement = 0
    max_no_improve = 30
    
    while time.time() - start_time < DEADLINE - 10:
        # Create a perturbation
        net = list(best_network)
        pert_type = rng.choice(['swap_adj', 'move', 'remove_and_reinsert', 'reverse_segment'])
        
        if pert_type == 'swap_adj' and len(net) > 1:
            i = rng.randint(0, len(net) - 2)
            net[i], net[i+1] = net[i+1], net[i]
        elif pert_type == 'move' and len(net) > 2:
            i = rng.randint(0, len(net) - 1)
            j = rng.randint(0, len(net) - 1)
            if i != j:
                elem = net.pop(i)
                net.insert(j, elem)
        elif pert_type == 'remove_and_reinsert' and len(net) > 2:
            i = rng.randint(0, len(net) - 1)
            elem = net.pop(i)
            j = rng.randint(0, len(net))
            net.insert(j, elem)
        elif pert_type == 'reverse_segment' and len(net) > 3:
            i = rng.randint(0, len(net) - 4)
            j = rng.randint(i + 2, len(net) - 1)
            net[i:j+1] = reversed(net[i:j+1])
        
        # Check validity quickly
        if not _apply_and_check(net, all_inputs):
            no_improvement += 1
            continue
        
        # Try to reduce further
        reduced = greedy_remove(net, all_inputs, max_passes=3)
        
        if len(reduced) < best_count:
            best_network = reduced
            best_count = len(best_network)
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement > max_no_improve:
            no_improvement = 0
    
    # Final polish
    if time.time() - start_time < DEADLINE - 3:
        final = greedy_remove(best_network, all_inputs, max_passes=5)
        if len(final) < best_count:
            best_network = final
    
    return best_network
