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
# Expected sorted output: all 1s then all 0s (since we sort in non-decreasing order? No, min to i, max to j)
# Actually with compare-exchange writing min to i and max to j, the network sorts in non-decreasing order
# So sorted output for a Boolean input with k ones: first k wires are 1, rest are 0? No...
# min goes to wire i (lower index), max to wire j (higher index)
# So after full sort, wire 0 has the minimum, wire N-1 has the maximum
# For Boolean: wire 0 = 0 (if there's at least one 0), ..., wire N-1 = 1 (if there's at least one 1)
# Actually: sorted in non-decreasing order means wire 0 <= wire 1 <= ... <= wire N-1
# For Boolean inputs, this means all 0s come first, then all 1s
# So for an input with k ones: wires 0..N-k-1 are 0, wires N-k..N-1 are 1

def _expected_sorted(arr):
    """Compute expected sorted output for all inputs."""
    # Count number of 1s in each input
    ones = arr.sum(axis=1)  # shape (NUM_INPUTS,)
    result = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for w in range(N):
        # Wire w is 1 if ones > (N - 1 - w), i.e., ones >= N - w
        result[:, w] = (ones >= (N - w)).astype(np.uint8)
    return result

_EXPECTED = _expected_sorted(_ALL_INPUTS)

def _verify_network(comparators):
    """Verify that the network sorts all Boolean inputs. Returns True if correct."""
    arr = _ALL_INPUTS.copy()
    for (i, j) in comparators:
        # Compare-exchange: min to i, max to j
        ai = arr[:, i]
        aj = arr[:, j]
        # Where ai > aj, swap
        mask = ai > aj
        if np.any(mask):
            arr[:, i] = np.where(mask, aj, ai)
            arr[:, j] = np.where(mask, ai, aj)
    return np.array_equal(arr, _EXPECTED)

def _depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # Use topological depth: each comparator depends on the latest use of its wires
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

def construct():
    start_time = time.time()
    
    # Start with odd-even transposition
    best = _odd_even_transposition()
    best_count = len(best)
    
    # Verify initial network
    if not _verify_network(best):
        # Should not happen, but safety
        return best
    
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
            # Try swapping adjacent comparators
            found_swap = False
            for idx in range(len(best) - 1):
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                candidate = best.copy()
                candidate[idx], candidate[idx+1] = best[idx+1], best[idx]
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
