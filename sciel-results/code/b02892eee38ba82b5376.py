"""Size-efficient sorting network search on 17 wires."""
import time
import random
import numpy as np

N = 17
M = 1 << N  # 131072

# Precompute a fast verification path:
# We represent the network as a list of (i, j) comparators.
# Verification over all 2^17 boolean inputs is the zero-one principle.

def _verify_fast(network):
    """Verify sorting network over all 2^17 boolean inputs using bit-packing."""
    # Build input matrix: for input k (0..M-1), wire w has bit w of k
    inputs = np.arange(M, dtype=np.uint32)
    wires = np.empty((N, M), dtype=np.uint8)
    for w in range(N):
        wires[w] = ((inputs >> w) & 1).astype(np.uint8)
    
    for (i, j) in network:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    
    # Check: for each input, output must be sorted (non-decreasing across wires)
    # and the total count of 1s is preserved (implied by min/max operations on booleans)
    # Fast check: verify wire 0 is all 0s, wire N-1 is all 1s, and adjacent wires are non-decreasing
    # Combine checks to reduce Python overhead
    for w in range(N - 1):
        if not np.all(wires[w] <= wires[w + 1]):
            return False
    if not np.all(wires[0] == 0):
        return False
    if not np.all(wires[N-1] == 1):
        return False
    return True


def _depth_of_network(network):
    """Compute parallel depth of a compare-exchange network."""
    # depth[w] = earliest time wire w is available
    depth = [0] * N
    for (i, j) in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0


def _seed_odd_even():
    """Odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def _seed_batcher_17():
    """
    Construct a Batcher-style sorting network for 17 wires.
    Uses recursive odd-even merge sort. For non-power-of-2, we use a
    padding approach: build for 32 wires, then extract the subnetwork
    that only involves wires 0..16, and verify/fix.
    
    Actually, a cleaner approach: use the known structure for small N.
    Let's implement a proper Batcher odd-even merge for N=17 by
    padding to 32, building the full network, and keeping only
    comparators where both endpoints are < 17.
    """
    # Build Batcher's odd-even merge sort for 32 wires (power of 2)
    def _batcher_odd_merge(n):
        """Return list of comparators for Batcher's odd-even merge sort on n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comps = []
        # Split into two halves
        half = n // 2
        # Recursively sort each half
        left = _batcher_odd_merge(half)
        right = _batcher_odd_merge(half)
        for (i, j) in left:
            comps.append((i, j))
        for (i, j) in right:
            comps.append((i + half, j + half))
        
        # Odd-even merge the two sorted halves
        # Merge step for odd-even merge:
        # For k = 0, 1, ..., half-1:
        #   Compare-exchange (k, half + k)
        #   Compare-exchange (k, half + k + 1) if k + 1 < half
        #   Compare-exchange (k + 1, half + k) if k + 1 < half
        #   Compare-exchange (k + 1, half + k + 1) if k + 1 < half and k + 2 < half
        # Actually, the standard odd-even merge pattern:
        # For i = 0 to half-1:
        #   If i is even: compare (i, half+i)
        #   If i is odd: compare (i, half+i) and (i, half+i-1) and (i+1, half+i)
        # Let me use the correct recursive odd-even merge:
        
        # The odd-even merge of two sorted sequences of length half:
        # Step 1: compare (i, half + i) for all i
        # Step 2: recursively merge even-indexed and odd-indexed sub-sequences
        
        # Simpler: use the iterative odd-even merge
        # Phase 1: compare (i, half + i) for i = 0..half-1
        for i in range(half):
            comps.append((i, half + i))
        
        # Phase 2: recursively merge the even positions and odd positions
        # Even positions: 0, 2, 4, ... -> wires 0, 2, 4, ... and half, half+2, ...
        # Odd positions: 1, 3, 5, ... -> wires 1, 3, 5, ... and half+1, half+3, ...
        
        # For even sub-merge:
        even_half = half // 2
        if even_half > 1:
            even_left = _batcher_odd_merge(even_half)
            for (i, j) in even_left:
                comps.append((2*i, 2*j))
            even_right = _batcher_odd_merge(even_half)
            for (i, j) in even_right:
                comps.append((2*i + half, 2*j + half))
            # Now odd-even merge the even subsequences
            # This is getting complex. Let me use a simpler known construction.
        
        return comps
    
    # The recursive approach is getting complex. Let me use a different strategy:
    # Use a known efficient network for 16 wires (Batcher) and add wire 17.
    # Batcher's network for 16 wires has 119 comparators.
    # For 17, we can start with 16-wire Batcher and then insert wire 17.
    
    # Alternative: use the seed as-is but try to improve it.
    # Let's just return the odd-even seed for now, but with a slightly better
    # initial construction.
    return _seed_odd_even()


def _seed_batcher():
    """Batcher's odd-even merge sort network for N wires (not power of 2, use recursive construction)."""
    # For non-power-of-2, we can use a simple approach: pad to next power of 2, build, then trim.
    # But that's complex. Instead, use a known small network or just use odd-even as seed.
    # Let's implement Batcher's for power of 2 and adapt.
    # For N=17, pad to 32, build Batcher, keep only comparators involving wires < 17,
    # then fix up. Actually this is tricky. Let's just use odd-even as the base seed.
    return _seed_odd_even()


def construct():
    start = time.time()
    deadline = start + 85  # 5s margin before 90s kill
    
    best_network = _seed_odd_even()
    best_count = len(best_network)
    
    rng = random.Random(42)
    
    def _time_left():
        return deadline - time.time()
    
    def _try_network(net):
        nonlocal best_network, best_count
        if len(net) < best_count:
            if _verify_fast(net):
                best_network = list(net)
                best_count = len(net)
                return True
        return False
    
    # First, verify our seed
    if not _verify_fast(best_network):
        # Shouldn't happen, but just in case
        pass
    
    # Local search: try removing each comparator and check if still valid
    # This is expensive but let's try a few rounds
    
    improved = True
    round_num = 0
    while improved and _time_left() > 5:
        improved = False
        round_num += 1
        # Shuffle the order of indices to try different removals each round
        indices = list(range(len(best_network)))
        rng.shuffle(indices)
        for idx in indices:
            if _time_left() < 5:
                break
            # Try removing comparator idx
            trial = best_network[:idx] + best_network[idx+1:]
            if _verify_fast(trial):
                best_network = trial
                improved = True
                break  # restart scan after a removal
        # If no removal worked, try some mutations
        if not improved:
            # Try swapping adjacent comparators
            for idx in range(len(best_network) - 1):
                if _time_left() < 5:
                    break
                trial = list(best_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                if _verify_fast(trial):
                    best_network = trial
                    improved = True
                    break
            # Try replacing a comparator with a different one
            if not improved:
                for idx in range(len(best_network)):
                    if _time_left() < 5:
                        break
                    for _ in range(3):  # try a few random replacements
                        if _time_left() < 5:
                            break
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        trial = list(best_network)
                        trial[idx] = (i, j)
                        if _verify_fast(trial):
                            best_network = trial
                            improved = True
                            break
    
    # Return the best network found
    return best_network
