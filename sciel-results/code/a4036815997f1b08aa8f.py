"""Rewritten program: greedy + local search with swaps and deadline guard."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 192
K = 3
MAX_SIZE = 48
TIME_LIMIT = 75.0  # safety margin before 90s kill


def _valid_candidate(x, selected_set, selected_list):
    """Check if x can be added to selected without violating the constraint."""
    # For every a in candidate = selected + [x], no a divides sum of any 3 distinct OTHER elements
    # We need to check:
    # 1. For each existing a in selected: does a divide sum of any 3 from (selected without a) + [x]?
    #    This includes triples that include x and triples that don't (already valid).
    # 2. For x itself: does x divide sum of any 3 from selected?
    
    # Check x: x must not divide sum of any 3 distinct elements from selected
    for t in combinations(selected_list, K):
        if sum(t) % x == 0:
            return False
    
    # Check each existing a: a must not divide sum of any 3 distinct elements from (selected - {a}) + {x}
    # The new triples involving x are: (x, u, v) where u,v in selected, u != a, v != a, u != v
    # Also need to check triples (u, v, w) all from selected - {a} — those were already checked when a was added
    # So we only need to check triples that include x.
    for i, a in enumerate(selected_list):
        # Triples containing x and two elements from selected (excluding a)
        others = [v for j, v in enumerate(selected_list) if j != i]
        for t in combinations(others, K - 1):
            if (x + t[0] + t[1]) % a == 0:
                return False
    return True


def _is_valid(selected_list):
    """Full validation of a set."""
    n = len(selected_list)
    for i, a in enumerate(selected_list):
        others = [v for j, v in enumerate(selected_list) if j != i]
        for t in combinations(others, K):
            if sum(t) % a == 0:
                return False
    return True


def _greedy_construct():
    """Greedy construction from large to small."""
    selected = []
    selected_set = set()
    for x in range(N, 0, -1):
        if _valid_candidate(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected


def _local_search(initial, deadline):
    """Local search with swaps to improve set size."""
    best = list(initial)
    best_set = set(best)
    best_size = len(best)
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any missing element
        missing = [x for x in range(N, 0, -1) if x not in best_set]
        for x in missing:
            if _valid_candidate(x, best_set, best):
                best.append(x)
                best_set.add(x)
                best_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        # Prioritize removing small elements (they're more likely to cause divisibility issues)
        for i in range(len(best) - 1, -1, -1):
            if time.time() >= deadline:
                break
            a = best[i]
            # Try to find an element x not in best that can replace a
            for x in range(N, 0, -1):
                if x in best_set:
                    continue
                # Check if best without a plus x is valid
                # Quick check: x must not divide sum of any 3 from (best - {a})
                others = [v for j, v in enumerate(best) if j != i]
                ok = True
                for t in combinations(others, K):
                    if sum(t) % x == 0:
                        ok = False
                        break
                if not ok:
                    continue
                # Check each b in best - {a}: b must not divide sum of any 3 from (best - {a} - {b}) + {x}
                # Triples involving x: (x, u, v) where u,v in best - {a} - {b}
                for j, b in enumerate(best):
                    if j == i:
                        continue
                    others_b = [v for k, v in enumerate(best) if k != i and k != j]
                    for t in combinations(others_b, K - 1):
                        if (x + t[0] + t[1]) % b == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # Perform swap
                    best[i] = x
                    best_set.discard(a)
                    best_set.add(x)
                    improved = True
                    break
            if improved:
                break
        
        # If no improvement, try removing an element that's causing issues
        if not improved:
            # Try removing the smallest element and see if we can add two
            if len(best) > 1:
                # Find the element that participates in the most "near-miss" violations
                # Simple heuristic: remove smallest
                a = min(best)
                idx = best.index(a)
                best.pop(idx)
                best_set.discard(a)
                improved = True
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Phase 1: Greedy construction
    selected = _greedy_construct()
    
    # Phase 2: Local search
    if time.time() < deadline - 2.0:
        selected = _local_search(selected, deadline)
    
    # Ensure validity
    if not _is_valid(selected):
        # Fallback: rebuild greedily
        selected = _greedy_construct()
    
    return selected


# EVOLVE-BLOCK-END
