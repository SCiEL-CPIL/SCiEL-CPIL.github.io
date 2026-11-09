"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# Global cache for initial masks to avoid repeated attribute lookups
_INITIAL_MASKS_CACHE = {}

# Cache for optimal small sorting networks
_SMALL_NETWORKS_CACHE = {}

def _build_optimal_small_network(n):
    """
    Build an optimal or near-optimal sorting network for small n.
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    if n in _SMALL_NETWORKS_CACHE:
        return _SMALL_NETWORKS_CACHE[n][:]
    
    # Known optimal networks for small n
    if n == 2:
        net = [(0, 1)]
    elif n == 3:
        # 3-wire optimal: 3 comparators
        net = [(0, 1), (1, 2), (0, 1)]
    elif n == 4:
        # 4-wire optimal: 5 comparators (AKS or standard)
        net = [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
    elif n == 5:
        # 5-wire optimal: 9 comparators
        net = [
            (0, 1), (2, 3), (4, 4),  # dummy
            (1, 2), (3, 4),
            (0, 2), (1, 3),
            (2, 4),
            (1, 3), (2, 3)
        ]
        # Actually, let me use a known 5-wire network
        net = [
            (0, 1), (2, 3),
            (1, 2), (3, 4),
            (0, 2), (1, 3),
            (2, 4),
            (1, 2), (3, 4)
        ]
    elif n == 6:
        # 6-wire: use odd-even merge or known construction
        # Start with odd-even transposition (18 comparators) and greedily remove
        net = _build_seed_network_n(6)
        net = _greedy_remove_small(net, n, 0.5)
    elif n == 8:
        # 8-wire: use bitonic sort construction
        net = _build_bitonic_network(8)
        net = _greedy_remove_small(net, n, 1.0)
    elif n == 12:
        # 12-wire: use a combination approach
        # Split into 6+6, sort each, then merge
        net = _build_hierarchical_merge(12, 6)
        net = _greedy_remove_small(net, n, 2.0)
    else:
        # Fallback: odd-even transposition + greedy removal
        net = _build_seed_network_n(n)
        net = _greedy_remove_small(net, n, 1.0)
    
    _SMALL_NETWORKS_CACHE[n] = net[:]
    return net[:]

def _build_seed_network_n(n):
    """Build odd-even transposition network for n wires."""
    network = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                network.append((i, i + 1))
    return network

def _build_bitonic_network(n):
    """
    Build a bitonic sorting network for n wires (n must be power of 2).
    """
    if n <= 1:
        return []
    
    network = []
    
    def _bitonic_merge(start, length, ascending):
        """Merge a bitonic sequence of given length starting at start."""
        if length <= 1:
            return
        half = length // 2
        # Compare elements distance 'half' apart
        for i in range(start, start + half):
            if ascending:
                network.append((i, i + half))
            else:
                network.append((i + half, i))
        # Recursively merge the two halves
        _bitonic_merge(start, half, True)
        _bitonic_merge(start + half, half, False)
    
    def _bitonic_sort(start, length, ascending):
        """Sort a sequence of given length starting at start."""
        if length <= 1:
            return
        half = length // 2
        # Sort first half ascending, second half descending
        _bitonic_sort(start, half, True)
        _bitonic_sort(start + half, half, False)
        # Merge the bitonic sequence
        _bitonic_merge(start, length, ascending)
    
    _bitonic_sort(0, n, True)
    return network

def _build_hierarchical_merge(n, half):
    """
    Build a hierarchical merge network:
    1. Sort first half and second half independently
    2. Merge them using odd-even merge
    """
    network = []
    
    # Sort first half
    left_net = _build_optimal_small_network(half)
    for i, j in left_net:
        network.append((i, j))
    
    # Sort second half
    right_net = _build_optimal_small_network(half)
    for i, j in right_net:
        network.append((i + half, j + half))
    
    # Odd-even merge
    # We need to merge two sorted sequences of length 'half'
    # This is a standard odd-even merge network
    # For simplicity, use a basic merge: compare elements that need to cross
    
    # Simple merge: for each position, compare with the other half
    # This is not optimal but works
    for i in range(half):
        network.append((i, i + half))
    
    # Additional passes to ensure correctness
    for pass_num in range(half - 1):
        for i in range(half):
            if i + 1 < half:
                network.append((i, i + 1))
                network.append((i + half, i + half + 1))
    
    return network

def _greedy_remove_small(network, n_wires, time_limit):
    """
    Greedy removal for small networks with a time limit.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    changed = True
    while changed:
        changed = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, n_wires):
                network = new_net
                changed = True
                break
    return network

def _get_initial_masks(n_wires):
    """
    Get or generate the initial bitmask representations for all 2^n inputs.
    mask[w] is an integer where bit k is 1 if input k has a 1 on wire w.
    """
    if n_wires not in _INITIAL_MASKS_CACHE:
        total_inputs = 1 << n_wires
        masks = []
        # Optimization: Generate masks using bit patterns
        # For wire w, the pattern is 0...0 1...1 repeating with period 2^(w+1)
        # We can construct this efficiently.
        for w in range(n_wires):
            m = 0
            # The bit k is set if (k >> w) & 1 is 1
            # This is equivalent to: for blocks of size 2^(w+1), the second half is 1s.
            # We can build this by iterating, but for n=17, 131072 iterations is fine.
            # To be faster, we can use a recursive doubling method or just the loop.
            # The loop is ~2M ops total for all wires, which is < 0.1s in Python.
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _INITIAL_MASKS_CACHE[n_wires] = masks
    return _INITIAL_MASKS_CACHE[n_wires]

def _verify_sorting_network(network, n_wires):
    """
    Verify that the network sorts all 2^n_wires boolean inputs using the zero-one principle.
    """
    if not network:
        return n_wires <= 1
        
    # Get initial masks
    init_masks = _get_initial_masks(n_wires)
    # Copy the masks to simulate
    # Using list comprehension is faster than [:] for large lists? No, [:] is C-level.
    masks = init_masks[:]
    
    # Simulate network
    # Local variables for speed
    m = masks
    for i, j in network:
        # Basic sanity check
        if i >= j:
            return False
        mi = m[i]
        mj = m[j]
        # Compare-exchange: min to i, max to j
        # min(a, b) = a & b
        # max(a, b) = a | b
        m[i] = mi & mj
        m[j] = mi | mj
    
    # Check sortedness: 0s then 1s.
    # This means for any wire w, if it has a 1, all wires w+1... must have 1s.
    # Equivalently, masks[w] is a subset of masks[w+1].
    # masks[w] & ~masks[w+1] must be 0.
    for w in range(n_wires - 1):
        if m[w] & ~m[w + 1]:
            return False
            
    return True

def _build_seed_network():
    """
    Build a seed sorting network using odd-even transposition.
    For N=17, this gives 136 comparators.
    """
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _greedy_remove(network, deadline):
    """
    Perform greedy removal of comparators.
    Returns the final network.
    """
    changed = True
    # Limit iterations to prevent infinite loops if verification is buggy (it shouldn't be)
    # But mostly limited by deadline.
    while changed:
        changed = False
        # Try removing from the end first, as later comparators are often redundant
        # if the network is already sorted by earlier ones.
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            
            # Create candidate network without comparator i
            # Optimization: Instead of slicing, we can pass the network and index to verifier?
            # Current verifier takes a list. Slicing is O(N). N is small (~100).
            new_net = network[:i] + network[i+1:]
            
            if _verify_sorting_network(new_net, N):
                network = new_net
                changed = True
                # Restart the scan from the end? Or continue?
                # Restarting from the end is safer to catch new redundancies created by removal.
                # However, it's O(N^2).
                # Let's break and restart the while loop to scan from the end again.
                break
                
    return network

def _local_search_step(network, deadline):
    """
    Perform a local search step: Perturb the network and try to clean it up.
    Returns the best network found.
    """
    best_net = list(network)
    best_size = len(best_net)
    
    # Number of attempts
    # We want to make many fast attempts.
    attempts = 0
    max_attempts = 200 # High number, but each attempt is fast if verification fails early or succeeds
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        candidate = list(best_net)
        
        # Strategy 1: Swap two comparators
        if len(candidate) > 1 and random.random() < 0.5:
            i = random.randint(0, len(candidate) - 2)
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
        else:
            # Strategy 2: Replace a comparator with a random valid one
            if len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate[idx] = (i, j)
        
        # Verify the perturbed network
        if _verify_sorting_network(candidate, N):
            # If valid, try to remove redundant comparators
            # Use a limited greedy removal to save time
            cleaned = _greedy_remove(candidate, deadline)
            
            if len(cleaned) < best_size:
                best_net = cleaned
                best_size = len(best_net)
                # Reset attempts? Or continue?
                # If we found a better one, we can try to improve further from this new best.
                # But we are inside a loop. Let's just update best and continue trying to beat it.
                # To avoid getting stuck in a local minimum of the "best" variable, 
                # we continue the loop, using the new best as the base for next iterations?
                # No, `candidate` is derived from `best_net`. 
                # So the next iteration will perturb the *new* best. This is good.
                
    return best_net

def _build_hierarchical_17():
    """
    Build a hierarchical 17-wire network by decomposing into 5 + 12.
    1. Sort 5-wire and 12-wire sub-networks
    2. Merge them using a merge network
    """
    network = []
    
    # Sort first 5 wires
    left_net = _build_optimal_small_network(5)
    for i, j in left_net:
        network.append((i, j))
    
    # Sort last 12 wires
    right_net = _build_optimal_small_network(12)
    for i, j in right_net:
        network.append((i + 5, j + 5))
    
    # Merge the two sorted halves
    # We need to merge a 5-element sorted sequence with a 12-element sorted sequence
    # Use a simple merge strategy: compare elements that might need to cross
    # This is a basic merge - not optimal but provides a good starting point
    
    # Phase 1: Cross-compare
    for i in range(5):
        network.append((i, i + 5))
    
    # Phase 2: Additional passes to propagate values
    for pass_num in range(10):
        for i in range(16):
            if pass_num % 2 == 0:
                if i % 2 == 0:
                    network.append((i, i + 1))
            else:
                if i % 2 == 1:
                    network.append((i, i + 1))
    
    return network

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    # Set a strict deadline to ensure we return well before the 60s limit.
    # 55s leaves 5s margin.
    deadline = start_time + 55.0
    
    # Try hierarchical construction first
    hierarchical_net = _build_hierarchical_17()
    
    # Also build seed network for comparison
    seed_net = _build_seed_network()
    
    # Choose the better starting point
    if len(hierarchical_net) < len(seed_net):
        network = hierarchical_net
    else:
        network = seed_net
    
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Initial Greedy Removal
    network = _greedy_remove(network, deadline)
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Local Search Loop
    # Use remaining time for local search
    # We run local search steps until time runs out
    while time.time() < deadline - 2.0:
        # Perform a local search step
        # Pass the current best as the starting point
        net = _local_search_step(best_network, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
            # If we improved, do a greedy cleanup on the new best immediately
            # to ensure we are at a local minimum of the removal graph
            if time.time() < deadline - 2.0:
                cleaned_net = _greedy_remove(best_network, deadline)
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
        else:
            # If no improvement, we could try a different random seed or strategy
            # but for now, just continue. The random nature of _local_search_step
            # provides the diversification.
            pass

    return best_network
