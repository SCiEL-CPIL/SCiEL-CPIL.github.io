"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_sorted_mask_set(n):
    """Precompute the set of sorted masks (all 0s then all 1s)."""
    sorted_masks = set()
    for k in range(n + 1):
        mask = ((1 << (n - k)) - 1) << k
        sorted_masks.add(mask)
    return sorted_masks

_SORTED_MASKS = _build_sorted_mask_set(N)

def _vectorized_verify(comparators, n=N):
    """Vectorized verification using NumPy - processes all 2^n inputs at once."""
    if not comparators:
        return n <= 1
    
    num_inputs = 1 << n
    masks = np.arange(num_inputs, dtype=np.uint64)
    bit_positions = np.arange(n, dtype=np.uint64)
    wires = ((masks[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)
    
    for i, j in comparators:
        wi = wires[:, i]
        wj = wires[:, j]
        swap_mask = wi > wj
        wires[:, i] = np.where(swap_mask, wj, wi)
        wires[:, j] = np.where(swap_mask, wi, wj)
    
    diffs = wires[:, 1:].astype(np.int8) - wires[:, :-1].astype(np.int8)
    if np.any(diffs == -1):
        return False
    return True

def _fast_verify_network(comparators, n=N):
    """Faster verification using bit manipulation with precomputed sorted masks."""
    sorted_masks = _SORTED_MASKS
    for mask in range(1 << n):
        state = mask
        for i, j in comparators:
            if (state >> i) & 1 == 0 and (state >> j) & 1 == 1:
                state ^= (1 << i) | (1 << j)
        if state not in sorted_masks:
            return False
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _shell_sort_network(n=N):
    """Generate a shell-sort based sorting network using gap sequences."""
    comparators = []
    gaps = [9, 5, 3, 2, 1]
    for gap in gaps:
        for i in range(gap, n):
            comparators.append((i - gap, i))
    return comparators

def _efficient_merge(n1, n2, base=0):
    """Efficient merge of two sorted sequences of lengths n1 and n2.
    First sequence at [base, base+n1), second at [base+n1, base+n1+n2).
    Uses a divide-and-conquer approach."""
    comps = []
    if n1 == 0 or n2 == 0:
        return comps
    
    # For small sizes, use direct comparison
    if n1 + n2 <= 4:
        for i in range(n1):
            for j in range(n2):
                a = base + i
                b = base + n1 + j
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
        return comps
    
    # Divide and conquer merge
    # Split both sequences in half
    n1_half = n1 // 2
    n2_half = n2 // 2
    
    # The median of the combined sequence is around position (n1+n2)//2
    # Use a recursive approach:
    # Compare the middle elements to determine which half to recurse into
    
    # Actually, for a sorting network, the standard efficient merge is:
    # 1. Compare corresponding elements
    # 2. Recursively merge the results
    
    m = min(n1, n2)
    # Compare (base+i, base+n1+i) for i in range(m)
    for i in range(m):
        comps.append((base + i, base + n1 + i))
    
    # After this, positions base..base+m-1 contain the smaller of each pair
    # and positions base+n1..base+n1+m-1 contain the larger
    # But these aren't necessarily sorted within themselves
    
    # The remaining elements:
    # If n1 > m: positions base+m..base+n1-1 are the tail of first sequence (sorted)
    # If n2 > m: positions base+n1+m..base+n1+n2-1 are the tail of second sequence (sorted)
    
    # We need to merge:
    # - The m "min" elements (positions base..base+m-1) with the tail of the longer sequence
    # - The m "max" elements with... 
    
    # This is getting complicated. Let me use a simpler but still efficient approach:
    # Just do the pairwise comparisons but in a structured way.
    
    # For now, return the simple version and rely on the search to optimize
    return comps

def _merge_sort_network(n=N):
    """Generate a merge-sort based sorting network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    def _build(n, base):
        if n <= 1:
            return []
        if n == 2:
            return [(base, base + 1)]
        
        mid = n // 2
        left = _build(mid, base)
        right = _build(n - mid, base + mid)
        
        # Merge the two sorted halves
        # Use a simple but correct merge
        merge_comps = []
        for i in range(mid):
            for j in range(n - mid):
                a = base + i
                b = base + mid + j
                if a < b:
                    merge_comps.append((a, b))
                else:
                    merge_comps.append((b, a))
        
        return left + right + merge_comps
    
    return _build(n, 0)

def _batch_removal_search(comparators, n=N, max_attempts=1000, use_vectorized=True):
    """Try to remove comparators one at a time, greedily keeping valid removals."""
    current = list(comparators)
    verify_fn = _vectorized_verify if use_vectorized else _fast_verify_network
    removed = 0
    attempts = 0
    deadline = time.time() + 25  # 25s budget for this phase
    
    while attempts < max_attempts and time.time() < deadline:
        if len(current) <= 72:
            break
        # Try removing each comparator in random order
        indices = list(range(len(current)))
        random.shuffle(indices)
        found_removal = False
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if verify_fn(candidate, n):
                current = candidate
                removed += 1
                found_removal = True
                break
            attempts += 1
        if not found_removal:
            break
    
    return current, removed

def _local_search(comparators, n=N, max_iterations=100000, use_vectorized=True):
    """Local search with removal, swap, and mutation operations."""
    current = list(comparators)
    verify_fn = _vectorized_verify if use_vectorized else _fast_verify_network
    best = list(current)
    best_len = len(current)
    deadline = time.time() + 30  # 30s budget
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 72:
            break
        
        mod_type = random.random()
        
        if mod_type < 0.5 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if verify_fn(candidate, n):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        elif mod_type < 0.75:
            # Try swapping two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if verify_fn(candidate, n):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        else:
            # Try replacing a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            a = random.randint(0, N - 1)
            b = random.randint(0, N - 1)
            if a == b:
                b = (a + 1) % N
            if a > b:
                a, b = b, a
            candidate = list(current)
            candidate[idx] = (a, b)
            if verify_fn(candidate, n):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
    
    return best

def construct():
    deadline = time.time() + 85  # 5 second safety margin before 90s kill
    
    # Try multiple starting networks and pick the best
    candidates = [
        _odd_even_transposition(N),
        _shell_sort_network(N),
    ]
    
    # Add merge sort network if it's not too large
    try:
        ms_net = _merge_sort_network(N)
        if len(ms_net) <= 200:
            candidates.append(ms_net)
    except:
        pass
    
    # Verify and pick the smallest valid candidate
    best = None
    best_len = float('inf')
    for cand in candidates:
        if not _vectorized_verify(cand, N):
            continue
        if len(cand) < best_len:
            best = list(cand)
            best_len = len(cand)
    
    if best is None:
        # Fallback to odd-even transposition
        best = _odd_even_transposition(N)
        best_len = len(best)
    
    if time.time() > deadline:
        return best
    
    random.seed(42)
    current = list(best)
    
    # Phase 1: Aggressive removal search using vectorized verification
    if time.time() < deadline - 10:
        current, removed = _batch_removal_search(current, N, max_attempts=1000, use_vectorized=True)
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    # Phase 2: Fine-grained local search
    if time.time() < deadline - 3:
        current = _local_search(current, N, max_iterations=100000, use_vectorized=True)
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    # Final verification
    if not _vectorized_verify(best, N):
        # Fallback
        return _odd_even_transposition(N)
    
    return best
# EVOLVE-BLOCK-END
