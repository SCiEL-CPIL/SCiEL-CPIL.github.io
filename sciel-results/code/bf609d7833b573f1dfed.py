"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators, n=N):
    """Vectorized verification using numpy for speed."""
    masks = np.arange(1 << n, dtype=np.uint8)
    wires = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        wires[:, i] = (masks >> i) & 1
    
    for i, j in comparators:
        swap_mask = wires[:, i] > wires[:, j]
        if np.any(swap_mask):
            wires[swap_mask, i] = wires[swap_mask, i] ^ 1
            wires[swap_mask, j] = wires[swap_mask, j] ^ 1
    
    diffs = np.diff(wires, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _generate_batcher(n):
    """Generate Batcher's Odd-Even Merge Sort network."""
    if n == 1:
        return []
    
    def _merge(k):
        if k == 1:
            return [(0, 1)]
        k_half = k // 2
        even_comps = []
        for i, j in _merge(k_half):
            mi = 2 * i if i < k_half else 2 * (i - k_half) + k
            mj = 2 * j if j < k_half else 2 * (j - k_half) + k
            even_comps.append((mi, mj))
        odd_comps = []
        for i, j in _merge(k_half):
            mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
            mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
            odd_comps.append((mi, mj))
        final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
        return even_comps + odd_comps + final_comps
    
    k = n // 2
    left = _generate_batcher(k)
    right = [(i + k, j + k) for i, j in _generate_batcher(k)]
    merge = _merge(k)
    return left + right + merge

def _aggressive_reduction(comparators, n=N, time_limit=50):
    """
    Aggressively reduce comparator count using:
    1. Repeated removal
    2. Insert-then-remove cycles
    3. Reordering to enable more removals
    """
    start_time = time.time()
    current = list(comparators)
    best_count = len(current)
    best_network = list(current)
    
    while time.time() - start_time < time_limit:
        # Phase 1: Try to remove any comparator
        removed_any = False
        for idx in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate, n):
                current = candidate
                best_count = len(current)
                best_network = list(current)
                removed_any = True
                break
        
        if removed_any:
            continue
        
        # Phase 2: Try reordering (move a comparator to a different position)
        reordered = False
        for idx in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            comp = current[idx]
            for new_pos in range(len(current)):
                if new_pos == idx:
                    continue
                if time.time() - start_time > time_limit:
                    break
                candidate = current[:idx] + current[idx+1:]
                candidate = candidate[:new_pos] + [comp] + candidate[new_pos:]
                # Check if this ordering is valid (it should be if the set is valid)
                # But reordering can affect whether we can remove others
                # Actually for a fixed set of comparators, any order that's a valid topological 
                # sort works. Let's just try it and see if it enables removals.
                if _verify_network_fast(candidate, n):
                    # Try to remove something from this new ordering
                    for ridx in range(len(candidate)):
                        if time.time() - start_time > time_limit:
                            break
                        candidate2 = candidate[:ridx] + candidate[ridx+1:]
                        if _verify_network_fast(candidate2, n):
                            current = candidate2
                            best_count = len(current)
                            best_network = list(current)
                            reordered = True
                            break
                if reordered:
                    break
            if reordered:
                break
        
        if reordered:
            continue
        
        # Phase 3: Try inserting a comparator then removing a different one
        inserted = False
        for pos in range(len(current) + 1):
            if time.time() - start_time > time_limit:
                break
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    new_comp = (i, j)
                    if new_comp in current:
                        continue
                    candidate = current[:pos] + [new_comp] + current[pos:]
                    if _verify_network_fast(candidate, n):
                        for ridx in range(len(candidate)):
                            if time.time() - start_time > time_limit:
                                break
                            if ridx == pos:
                                continue
                            candidate2 = candidate[:ridx] + candidate[ridx+1:]
                            if _verify_network_fast(candidate2, n):
                                current = candidate2
                                best_count = len(current)
                                best_network = list(current)
                                inserted = True
                                break
                    if inserted:
                        break
                if inserted:
                    break
            if inserted:
                break
        
        if not inserted and not reordered:
            # No improvement possible with current strategies
            break
    
    return best_network

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Start with Batcher's network (63 comparators)
    network = _generate_batcher(N)
    
    # Verify
    if not _verify_network_fast(network, N):
        raise ValueError("Batcher network failed verification")
    
    # Aggressive reduction
    network = _aggressive_reduction(network, N, time_limit=48)
    
    # Final verification
    if not _verify_network_fast(network, N):
        network = _generate_batcher(N)
    
    return network

# EVOLVE-BLOCK-END
