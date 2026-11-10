import time
import itertools

def _subset_sums_set(seq):
    """Return the set of all subset sums for seq using iterative doubling."""
    sums = {0}
    for x in seq:
        sums = sums | {s + x for s in sums}
    return sums

def _check_validity(seq):
    """Check if seq has distinct subset sums (all 2^n sums are unique)."""
    n = len(seq)
    return len(_subset_sums_set(seq)) == (1 << n)

def _greedy_construct(n):
    """
    Greedy construction: build sequence element by element.
    At each step, choose the smallest positive integer not already used
    that maintains distinct subset sums.
    This naturally produces powers of 2: 1, 2, 4, 8, ...
    """
    seq = []
    used = set()
    current_sums = {0}
    
    for i in range(n):
        candidate = 1
        while True:
            if candidate not in used:
                # Check if adding candidate maintains distinct subset sums
                new_sums = current_sums | {s + candidate for s in current_sums}
                if len(new_sums) == len(current_sums) * 2:
                    seq.append(candidate)
                    used.add(candidate)
                    current_sums = new_sums
                    break
            candidate += 1
    
    return seq

def _local_search_improve(seq, deadline):
    """
    Try to improve the sequence by reducing elements.
    Use a local search approach: try to reduce each element to a smaller value.
    """
    n = len(seq)
    best = list(seq)
    best_max = max(best)
    
    # Sort indices by value (largest first) to prioritize reducing large elements
    indices = sorted(range(n), key=lambda i: -best[i])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in indices:
            current_val = best[idx]
            if current_val <= 1:
                continue
            # Try to reduce this element
            others = [best[j] for j in range(n) if j != idx]
            others_set = set(others)
            
            # Try values from current_val-1 down to 1
            for new_val in range(current_val - 1, 0, -1):
                if new_val in others_set:
                    continue
                candidate = list(best)
                candidate[idx] = new_val
                if _check_validity(candidate):
                    best = candidate
                    best_max = max(best)
                    improved = True
                    break
    
    return best, best_max

def _multi_start_search(n, deadline):
    """
    Try multiple starting points and local search to find better solutions.
    Start from slightly perturbed versions of the greedy solution.
    """
    base_seq = _greedy_construct(n)
    best_seq = list(base_seq)
    best_max = max(base_seq)
    
    # Try reducing the largest element first with a wider search
    max_idx = best_seq.index(max(best_seq))
    others = [best_seq[j] for j in range(n) if j != max_idx]
    others_set = set(others)
    
    # The largest element must satisfy: for any subset S of others,
    # sum(S) + new_val != sum(S') for any other subset S'
    # This is complex, so just try values and check
    for trial in range(best_seq[max_idx] - 1, 0, -1):
        if time.time() > deadline:
            break
        if trial in others_set:
            continue
        candidate = list(best_seq)
        candidate[max_idx] = trial
        if _check_validity(candidate):
            best_seq = candidate
            best_max = trial
            break
    
    # Continue with full local search
    if time.time() < deadline:
        best_seq, best_max = _local_search_improve(best_seq, deadline)
    
    return best_seq, best_max

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    Uses greedy construction followed by local search improvement.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    
    # Step 1: Greedy construction (produces powers of 2)
    seq = _greedy_construct(n)
    best_max = max(seq)
    
    # Verify
    if not _check_validity(seq):
        # Fallback to powers of 2
        seq = [1 << j for j in range(n)]
        best_max = seq[-1]
    
    # Step 2: Multi-start search with local search
    if time.time() < deadline:
        seq, best_max = _multi_start_search(n, deadline)
    
    return seq
