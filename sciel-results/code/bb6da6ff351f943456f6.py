# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    
    Uses a greedy algorithm with multiple orderings and local improvement.
    """
    m = 205
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)
    
    # For efficiency, precompute for each candidate whether it conflicts with each other
    # conflict[a][b] = True if (a-b) mod m or (b-a) mod m is in forbidden
    # But since we're doing greedy, let's just check on the fly
    
    def is_conflict(a, b):
        d = (a - b) % m
        return d in forbidden
    
    def greedy(order):
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                if is_conflict(candidate, s):
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    # Try multiple orderings
    best_subset = []
    
    # Ordering 1: natural order 0..204
    subset1 = greedy(range(m))
    if len(subset1) > len(best_subset):
        best_subset = subset1
    
    # Ordering 2: reverse order
    subset2 = greedy(range(m - 1, -1, -1))
    if len(subset2) > len(best_subset):
        best_subset = subset2
    
    # Ordering 3: sorted by number of conflicts (fewest conflicts first)
    # Count how many other elements each candidate conflicts with
    conflict_count = [0] * m
    for a in range(m):
        for b in range(a + 1, m):
            if is_conflict(a, b):
                conflict_count[a] += 1
                conflict_count[b] += 1
    
    order3 = sorted(range(m), key=lambda x: conflict_count[x])
    subset3 = greedy(order3)
    if len(subset3) > len(best_subset):
        best_subset = subset3
    
    # Ordering 4: random-ish deterministic shuffle based on golden ratio
    golden = 0.6180339887498949
    order4 = sorted(range(m), key=lambda x: (x * golden) % 1)
    subset4 = greedy(order4)
    if len(subset4) > len(best_subset):
        best_subset = subset4
    
    # Local improvement: try to add elements that were rejected
    # For each element not in best_subset, check if it can be added
    current = set(best_subset)
    improved = list(best_subset)
    
    for candidate in range(m):
        if candidate in current:
            continue
        ok = True
        for s in improved:
            if is_conflict(candidate, s):
                ok = False
                break
        if ok:
            improved.append(candidate)
            current.add(candidate)
    
    if len(improved) > len(best_subset):
        best_subset = improved
    
    # Try another round of local improvement
    current = set(best_subset)
    improved = list(best_subset)
    for candidate in range(m):
        if candidate in current:
            continue
        ok = True
        for s in improved:
            if is_conflict(candidate, s):
                ok = False
                break
        if ok:
            improved.append(candidate)
            current.add(candidate)
    
    if len(improved) > len(best_subset):
        best_subset = improved
    
    # Try a more sophisticated approach: start from empty and add the element
    # that allows the most future additions (look-ahead greedy)
    def greedy_lookahead():
        selected = []
        remaining = list(range(m))
        while remaining:
            best_candidate = None
            best_score = -1
            for idx, candidate in enumerate(remaining):
                # Check if candidate can be added
                ok = True
                for s in selected:
                    if is_conflict(candidate, s):
                        ok = False
                        break
                if not ok:
                    continue
                # Score: how many remaining elements would still be compatible
                score = 0
                for other_idx in range(len(remaining)):
                    if other_idx == idx:
                        continue
                    other = remaining[other_idx]
                    if not is_conflict(candidate, other):
                        # Check if other is compatible with all selected
                        other_ok = True
                        for s in selected:
                            if is_conflict(other, s):
                                other_ok = False
                                break
                        if other_ok:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        return selected
    
    subset_la = greedy_lookahead()
    if len(subset_la) > len(best_subset):
        best_subset = subset_la
    
    # Local improvement on lookahead result
    current = set(best_subset)
    improved = list(best_subset)
    for candidate in range(m):
        if candidate in current:
            continue
        ok = True
        for s in improved:
            if is_conflict(candidate, s):
                ok = False
                break
        if ok:
            improved.append(candidate)
            current.add(candidate)
    
    if len(improved) > len(best_subset):
        best_subset = improved
    
    return np.array(sorted(best_subset), dtype=np.int64)


# EVOLVE-BLOCK-END
