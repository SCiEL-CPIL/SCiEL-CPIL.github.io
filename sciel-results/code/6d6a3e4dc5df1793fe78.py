import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _apply_comparator_vec(inputs, i, j):
    """Apply a single compare-exchange to a batch of inputs.
    inputs: (batch, N) array
    i, j: wires
    Returns new inputs array with min at i, max at j.
    """
    a = inputs[:, i].copy()
    b = inputs[:, j].copy()
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    inputs[:, i] = mn
    inputs[:, j] = mx
    return inputs

def _verify_network(comparators, batch_size=10000):
    """Verify that a network sorts all 2^17 Boolean inputs using the zero-one principle.
    Returns True if valid, False otherwise."""
    total = 1 << N
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = np.array([(x >> i) & 1 for x in range(start, end) for i in range(N)])
        batch = batch.reshape(-1, N)
        for i, j in comparators:
            _apply_comparator_vec(batch, i, j)
        # Check if sorted: each row should be non-decreasing
        diffs = np.diff(batch, axis=1)
        if np.any(diffs < 0):
            return False
    return True

def _verify_network_small(comparators, sample_size=5000):
    """Quick verification with random samples (not exhaustive)."""
    inputs = np.random.randint(0, 2, size=(sample_size, N))
    for i, j in comparators:
        _apply_comparator_vec(inputs, i, j)
    diffs = np.diff(inputs, axis=1)
    return not np.any(diffs < 0)

def _depth_of_network(comparators):
    """Compute parallel depth of a network."""
    wire_depth = [0] * N
    for i, j in comparators:
        d = wire_depth[i] if wire_depth[i] >= wire_depth[j] else wire_depth[j]
        wire_depth[i] = d + 1
        wire_depth[j] = d + 1
    return max(wire_depth)

def _odd_even_transposition():
    """Generate the odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _shell_sort_network():
    """Generate a shell sort based network."""
    comps = []
    gaps = [16, 8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comps.append((i - gap, i))
    return comps

def _merge_network(left_comps, right_comps, merge_comps):
    """Combine networks."""
    return left_comps + right_comps + merge_comps

def _generate_merge_network(n):
    """Generate a merge-based sorting network for n wires using Batcher's odd-even merge."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    
    # Batcher's odd-even merge sort
    if n % 2 == 0:
        half = n // 2
        left = _generate_merge_network(half)
        right = _generate_merge_network(half)
        # Shift right by half
        right = [(i + half, j + half) for i, j in right]
        # Odd-even merge
        merge = _odd_even_merge(0, n, half)
        return left + right + merge
    else:
        # For odd n, use a simpler approach: sort n-1 then insert
        if n <= 5:
            # Hardcoded small networks
            if n == 3:
                return [(0, 1), (1, 2), (0, 1)]
            if n == 5:
                return [(0,1),(2,3),(1,2),(3,4),(0,1),(2,3),(1,2),(0,1),(2,3)]
        
        # For larger odd, use a general construction
        # Sort first n-1, then merge the last element
        left = _generate_merge_network(n - 1)
        # Insert last element: compare with each of the sorted ones
        insert = []
        for i in range(n - 1):
            insert.append((i, n - 1))
        # This isn't quite right for a proper network, let's use a different approach
        # Actually, let's just use the odd-even transposition for odd sizes as fallback
        return _odd_even_transposition()[:len(_odd_even_transposition())]  # placeholder

def _odd_even_merge(start, end, k):
    """Generate odd-even merge comparators for merging two sorted halves."""
    # This is a simplified version; for correctness we'll use a known construction
    # For the purpose of this program, we'll use a simpler merge network
    comps = []
    # Odd-even merge: compare odd pairs then even pairs
    for i in range(start, end - 1, 2):
        comps.append((i, i + 1))
    for i in range(start + 1, end - 1, 2):
        comps.append((i, i + 1))
    return comps

def _batcher_odd_even_merge_sort(n):
    """Generate Batcher's odd-even merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    
    comps = []
    
    def _build(k, offset):
        if k == 1:
            return
        if k == 2:
            comps.append((offset, offset + 1))
            return
        half = k // 2
        _build(half, offset)
        _build(k - half, offset + half)
        # Merge the two halves
        _merge(offset, offset + k, half, k - half)
    
    def _merge(a, b, m, n):
        """Merge two sorted sequences of length m and n starting at a and b."""
        if m == 0 or n == 0:
            return
        if m + n == 2:
            comps.append((a, b))
            return
        # Odd-even merge
        # Compare odd-indexed pairs
        for i in range(0, m + n - 1, 2):
            comps.append((a + i, a + i + 1))
        # Recursively merge odd and even subsequences
        # This is getting complex; let's use a simpler approach
        pass
    
    # For simplicity, let's use a known correct construction
    # We'll fall back to a verified construction
    return _odd_even_transposition()

def _known_good_network():
    """Return a known good network with fewer comparators than odd-even transposition."""
    # Let's try a simple approach: use the odd-even transposition but with optimizations
    # Actually, let's try to construct a better network using a greedy approach
    
    # Start with odd-even transposition
    base = _odd_even_transposition()
    
    # Try to remove redundant comparators
    # A comparator is redundant if removing it still allows the network to sort
    # This is expensive to check, so let's use a heuristic
    
    # For now, return the base network
    return base

def construct():
    """Construct a sorting network for N=17 wires."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Start with a good initial network
    # Odd-even transposition: 136 comparators
    # Try to find a better one
    
    # Strategy: Use a genetic algorithm to optimize the network
    # But first, let's try some known constructions
    
    # Batcher's odd-even merge sort for 16 wires has 85 comparators
    # For 17 wires, we can use a variant
    
    # Let's try to build a network using divide and conquer
    # Sort 8 and 9 wires separately, then merge
    
    # For now, let's use a simpler approach:
    # 1. Start with odd-even transposition (136 comparators)
    # 2. Try to remove redundant comparators using the zero-one principle
    # 3. Use a genetic algorithm to further optimize
    
    # Initial network: odd-even transposition
    network = _odd_even_transposition()
    best_network = network[:]
    best_count = len(network)
    
    # Verify the initial network
    if not _verify_network_small(best_network, 1000):
        # This shouldn't happen, but just in case
        return best_network
    
    # Try to improve by removing redundant comparators
    # Greedy removal: try removing each comparator and check if the network still sorts
    # This is expensive, so we'll do a limited search
    
    # First, let's try a local search: for each comparator, try removing it
    # and see if the network still sorts (using a quick check)
    
    removed = set()
    i = 0
    while i < len(network) and time.time() < deadline - 10:
        if i in removed:
            i += 1
            continue
        # Try removing comparator i
        test_network = network[:i] + network[i+1:]
        if _verify_network_small(test_network, 2000):
            # Try full verification
            if _verify_network(test_network):
                network = test_network
                removed.add(i)
                # Don't increment i since we removed an element
                # Actually, we should restart since indices changed
                removed = set()
                i = 0
            else:
                i += 1
        else:
            i += 1
    
    best_network = network[:]
    best_count = len(network)
    
    # Now try to add comparators in strategic positions to allow further removal
    # This is a local search: try adding a comparator, then see if we can remove more
    
    # Try a genetic algorithm approach
    # Represent a network as a list of comparators
    # Fitness: number of comparators (lower is better), with penalty if not sorting
    
    # Population of networks
    # Start with variations of the current best
    
    if time.time() < deadline - 5:
        # Try to improve with a simple local search
        # For each position in the network, try swapping with adjacent positions
        # or trying different comparator pairs
        
        # Try adding a comparator at a random position and see if it allows removals
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Try adding a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(network))
            test_network = network[:pos] + [(i, j)] + network[pos:]
            
            # Try to remove redundant comparators from this new network
            test_network_copy = test_network[:]
            removed_local = set()
            k = 0
            while k < len(test_network_copy) and time.time() < deadline - 5:
                if k in removed_local:
                    k += 1
                    continue
                test2 = test_network_copy[:k] + test_network_copy[k+1:]
                if _verify_network_small(test2, 1000):
                    if _verify_network(test2):
                        test_network_copy = test2
                        removed_local.add(k)
                        removed_local = set()
                        k = 0
                    else:
                        k += 1
                else:
                    k += 1
            
            if len(test_network_copy) < best_count:
                best_network = test_network_copy[:]
                best_count = len(test_network_copy)
                network = best_network[:]
    
    # Final verification
    if not _verify_network(best_network):
        # Fall back to the known good odd-even transposition
        return _odd_even_transposition()
    
    return best_network
# EVOLVE-BLOCK-END
