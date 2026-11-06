"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85.0  # safety margin from 90s

# Precompute all Boolean inputs as a 2D array
def _build_all_inputs():
    bits = np.arange(NUM_INPUTS, dtype=np.uint8)
    arr = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for w in range(N):
        arr[:, w] = (bits >> w) & 1
    return arr

_ALL_INPUTS = _build_all_inputs()

def _expected_sorted(arr):
    """Compute expected sorted output for all inputs."""
    ones = arr.sum(axis=1)
    result = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for w in range(N):
        result[:, w] = (ones >= (N - w)).astype(np.uint8)
    return result

_EXPECTED = _expected_sorted(_ALL_INPUTS)

def _verify_network(comparators):
    """Verify that the network sorts all Boolean inputs. Returns True if correct."""
    arr = _ALL_INPUTS.copy()
    for (i, j) in comparators:
        ai = arr[:, i]
        aj = arr[:, j]
        mask = ai > aj
        if np.any(mask):
            arr[:, i] = np.where(mask, aj, ai)
            arr[:, j] = np.where(mask, ai, aj)
    return np.array_equal(arr, _EXPECTED)

def _depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    wire_depth = [0] * N
    max_d = 0
    for (i, j) in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        max_d = max(max_d, d)
        wire_depth[i] = d
        wire_depth[j] = d
    return max_d

def _odd_even_transposition():
    """Generate odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _odd_even_merge_sort():
    """Generate odd-even merge sort network for N wires."""
    def _odd_even_merge(n, a, b):
        """Generate comparators for odd-even merge of two sorted arrays of size n."""
        if n == 1:
            return [(a, b)]
        # Split into halves
        half = n // 2
        # Recursively merge left halves and right halves
        # But odd-even merge is specific:
        # Actually, standard odd-even merge:
        # 1. Odd-even merge the left halves (size n/2) and right halves (size n/2)
        # 2. Odd-even merge the cross pairs
        
        # Let's use the standard recursive definition:
        # O(n) = [odd-even merge of (a..a+n/2-1) and (b..b+n/2-1)]
        #        [odd-even merge of (a+n/2..a+n-1) and (b+n/2..b+n-1)]
        #        [compare-exchange (a+i, b+i) for i in 0..n/2-1]
        
        comps = []
        # Left half merge
        comps.extend(_odd_even_merge(half, a, b))
        # Right half merge
        comps.extend(_odd_even_merge(half, a + half, b + half))
        # Cross compare-exchanges
        for i in range(half):
            comps.append((a + i, b + i))
        
        return comps

    def _odd_even_sort(n, a):
        """Generate comparators for odd-even sort of array starting at a, size n."""
        if n == 1:
            return []
        half = n // 2
        # Recursively sort left and right halves
        comps = []
        comps.extend(_odd_even_sort(half, a))
        comps.extend(_odd_even_sort(half, a + half))
        # Merge the two sorted halves
        comps.extend(_odd_even_merge(half, a, a + half))
        return comps

    return _odd_even_sort(N, 0)

def construct():
    start_time = time.time()
    
    # Try both base networks and pick the better one after initial reduction
    # First, let's try odd-even merge sort as it's known to be better
    base_oms = _odd_even_merge_sort()
    base_oe = _odd_even_transposition()
    
    # Verify both
    valid_oms = _verify_network(base_oms)
    valid_oe = _verify_network(base_oe)
    
    # Start with the smaller valid one
    if valid_oms and (not valid_oe or len(base_oms) < len(base_oe)):
        best = base_oms.copy()
    else:
        best = base_oe.copy()
    
    if not _verify_network(best):
        # Fallback to transposition if something went wrong
        best = _odd_even_transposition()
        if not _verify_network(best):
            return best
    
    best_count = len(best)
    
    # Local search to reduce comparator count
    rng = random.Random(42)
    
    while time.time() - start_time < DEADLINE - 5.0:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(best)):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network(candidate):
                best = candidate
                best_count = len(best)
                improved = True
                break  # restart removal search
        
        if not improved:
            # Try swapping adjacent disjoint comparators
            found_swap = False
            for idx in range(len(best) - 1):
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                c1, c2 = best[idx], best[idx+1]
                # Check if disjoint
                if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
                    candidate = best.copy()
                    candidate[idx], candidate[idx+1] = c2, c1
                    if _verify_network(candidate):
                        best = candidate
                        found_swap = True
                        break
            
            if not found_swap:
                # Try random perturbation: add a random comparator, then try to remove
                if len(best) < 4096:
                    # Add a random comparator
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    pos = rng.randint(0, len(best))
                    candidate = best[:pos] + [(i, j)] + best[pos:]
                    if _verify_network(candidate):
                        # Now try to remove some comparators from this larger network
                        for rem_idx in range(len(candidate)):
                            if time.time() - start_time > DEADLINE - 5.0:
                                break
                            test = candidate[:rem_idx] + candidate[rem_idx+1:]
                            if _verify_network(test):
                                best = test
                                best_count = len(best)
                                improved = True
                                break
        
        if not improved and time.time() - start_time > DEADLINE - 10.0:
            break
    
    return best
