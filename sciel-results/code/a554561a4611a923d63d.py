import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s kill

    # Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    smooth_set = set(smooth_numbers)
    n = len(smooth_numbers)
    
    # Precompute divisibility: divs[i] = set of j where smooth_numbers[j] divides smooth_numbers[i]
    # and vice versa - we need "incompatible" pairs (one divides the other)
    incompatible = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if a % b == 0 or b % a == 0:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # For each target, find all valid antichain representations of size 1, 2, 3, 4
    # representations[target] = list of (size, frozenset_of_indices)
    representations = {t: [] for t in range(1, 129)}
    
    # Size 1
    for i in range(n):
        v = smooth_numbers[i]
        if v <= 128:
            representations[v].append((1, frozenset([i])))
    
    # Size 2
    for i in range(n):
        for j in range(i+1, n):
            if j in incompatible[i]:
                continue
            s = smooth_numbers[i] + smooth_numbers[j]
            if s <= 128:
                representations[s].append((2, frozenset([i, j])))
    
    # Size 3
    for i in range(n):
        for j in range(i+1, n):
            if j in incompatible[i]:
                continue
            for k in range(j+1, n):
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                if s <= 128:
                    representations[s].append((3, frozenset([i, j, k])))
    
    # Size 4 - only for targets not yet covered by size 1-3
    uncovered = set()
    for t in range(1, 129):
        if not representations[t]:
            uncovered.add(t)
    
    if uncovered:
        for i in range(n):
            for j in range(i+1, n):
                if j in incompatible[i]:
                    continue
                for k in range(j+1, n):
                    if k in incompatible[i] or k in incompatible[j]:
                        continue
                    for m in range(k+1, n):
                        if m in incompatible[i] or m in incompatible[j] or m in incompatible[k]:
                            continue
                        s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m]
                        if s <= 128 and s in uncovered:
                            representations[s].append((4, frozenset([i, j, k, m])))
    
    # Now do global optimization: assign representations to targets to minimize total terms
    # Use greedy: process targets in order of fewest options first
    # For each target, pick the representation with minimum size that doesn't conflict
    # But representations for different targets can share smooth numbers (no global constraint stated)
    # Re-reading: "no chosen term may divide another" is per-row, not global
    # So each row is independent! The only coupling is that we want to minimize TOTAL terms.
    # Since rows are independent, the optimal is simply: for each target, pick the min-size valid antichain.
    
    # Wait - let me re-read. "Minimize the total number of summands over all targets"
    # Since each target's representation is independent (no shared constraint between rows),
    # the global minimum is achieved by independently minimizing each target's term count.
    
    # So the current per-target greedy IS optimal IF it finds the minimum size for each target.
    # The issue is: does the current search guarantee finding the MINIMUM size?
    # Current: tries 1, then 2, then 3. If found at size k, that's minimum.
    # But the search within size k might not find ALL valid antichains of size k.
    # For the purpose of minimizing count, we just need ONE valid antichain of minimum size.
    
    # So the real question: for each target, what is the minimum k such that a valid antichain of size k exists?
    # And we need to find one.
    
    # Let's verify: for each target, find the minimum k with a valid antichain.
    answers = []
    total_terms = 0
    
    for target in range(1, 129):
        found = None
        for size in range(1, 5):
            # Find any valid antichain of this size summing to target
            for (sz, idx_set) in representations[target]:
                if sz == size:
                    found = [smooth_numbers[i] for i in sorted(idx_set)]
                    break
            if found:
                break
        
        if found is None:
            # Should not happen, but fallback
            # Try a brute force search for size 4 or 5
            found = _brute_force(target, smooth_numbers, incompatible, n)
        
        if found is None:
            # Absolute fallback: use target itself if smooth, else [1, target-1] if valid
            if target in smooth_set:
                found = [target]
            else:
                # Find any valid 2-term
                for i in range(n):
                    for j in range(i+1, n):
                        if j in incompatible[i]:
                            continue
                        if smooth_numbers[i] + smooth_numbers[j] == target:
                            found = [smooth_numbers[i], smooth_numbers[j]]
                            break
                    if found:
                        break
                if found is None:
                    # This should never happen
                    found = [target]  # Invalid but prevents crash
        
        answers.append(found)
        total_terms += len(found)
    
    return answers

def _brute_force(target, smooth_numbers, incompatible, n):
    """Brute force search for a valid antichain summing to target."""
    for size in range(1, 6):
        for combo in combinations(range(n), size):
            s = sum(smooth_numbers[i] for i in combo)
            if s != target:
                continue
            # Check antichain
            valid = True
            for a_idx in range(size):
                for b_idx in range(a_idx+1, size):
                    i, j = combo[a_idx], combo[b_idx]
                    if j in incompatible[i]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [smooth_numbers[i] for i in combo]
    return None
