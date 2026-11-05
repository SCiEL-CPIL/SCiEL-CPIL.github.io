import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    def is_bad_triple(a, b, c):
        return gcd(a, b) == gcd(a, c) == gcd(b, c)
    
    # Precompute all bad triples
    numbers = list(range(1, 129))
    n = len(numbers)
    
    # Build conflict structure: for each pair, which third elements create bad triples
    # bad_triples is a set of frozensets of size 3
    bad_triple_set = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_bad_triple(numbers[i], numbers[j], numbers[k]):
                    bad_triple_set.add(frozenset((i, j, k)))
    
    # Precompute for each pair (i,j), the set of k that form a bad triple
    pair_conflicts = {}
    for triple in bad_triple_set:
        elems = list(triple)
        for a in range(3):
            for b in range(a+1, 3):
                i, j = elems[a], elems[b]
                k = elems[3-a-b]
                key = (min(i,j), max(i,j))
                if key not in pair_conflicts:
                    pair_conflicts[key] = set()
                pair_conflicts[key].add(k)
    
    def count_conflicts(selected_indices):
        """Count bad triples in the selected set."""
        s = set(selected_indices)
        count = 0
        for i in s:
            for j in s:
                if i >= j:
                    continue
                key = (min(i,j), max(i,j))
                if key in pair_conflicts:
                    for k in pair_conflicts[key]:
                        if k in s and k > j:
                            count += 1
        return count
    
    def is_valid(selected_indices):
        return count_conflicts(selected_indices) == 0
    
    # Greedy construction: add elements in order of "safety"
    # Score each element by how many conflicts it would create
    def greedy_build(order):
        selected = set()
        for idx in order:
            # Check if adding idx creates any bad triple
            conflict = False
            for i in selected:
                key = (min(i, idx), max(i, idx))
                if key in pair_conflicts:
                    for k in pair_conflicts[key]:
                        if k in selected:
                            conflict = True
                            break
                if conflict:
                    break
            if not conflict:
                selected.add(idx)
        return selected
    
    # Try different orderings
    best_set = set()
    
    # Ordering 1: ascending
    s1 = greedy_build(list(range(n)))
    if len(s1) > len(best_set):
        best_set = s1
    
    # Ordering 2: descending
    s2 = greedy_build(list(range(n-1, -1, -1)))
    if len(s2) > len(best_set):
        best_set = s2
    
    # Ordering 3: by smallest prime factor (diverse SPFs first)
    def spf(x):
        if x % 2 == 0: return 2
        if x % 3 == 0: return 3
        if x % 5 == 0: return 5
        if x % 7 == 0: return 7
        if x % 11 == 0: return 11
        if x % 13 == 0: return 13
        if x % 17 == 0: return 17
        if x % 19 == 0: return 19
        if x % 23 == 0: return 23
        if x % 29 == 0: return 29
        if x % 31 == 0: return 31
        return x
    order3 = sorted(range(n), key=lambda i: (spf(numbers[i]), numbers[i]))
    s3 = greedy_build(order3)
    if len(s3) > len(best_set):
        best_set = s3
    
    # Local search: try to add removed elements or swap
    def local_search(selected, max_iters=50000):
        selected = set(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Try adding each non-selected element
            for idx in range(n):
                if idx in selected:
                    continue
                # Check if adding idx is valid
                conflict = False
                for i in selected:
                    key = (min(i, idx), max(i, idx))
                    if key in pair_conflicts:
                        for k in pair_conflicts[key]:
                            if k in selected:
                                conflict = True
                                break
                    if conflict:
                        break
                if not conflict:
                    selected.add(idx)
                    improved = True
                    break
            # Try swap: remove one, add another
            if not improved:
                for remove_idx in list(selected):
                    for add_idx in range(n):
                        if add_idx in selected:
                            continue
                        new_set = selected - {remove_idx}
                        # Check if add_idx is valid in new_set
                        conflict = False
                        for i in new_set:
                            key = (min(i, add_idx), max(i, add_idx))
                            if key in pair_conflicts:
                                for k in pair_conflicts[key]:
                                    if k in new_set:
                                        conflict = True
                                        break
                            if conflict:
                                break
                        if not conflict:
                            selected = new_set | {add_idx}
                            improved = True
                            break
                    if improved:
                        break
        return selected
    
    best_set = local_search(best_set)
    
    # Convert back to values
    result = [numbers[i] for i in sorted(best_set)]
    
    # Final validation
    for a, b, c in combinations(result, 3):
        if gcd(a,b) == gcd(a,c) == gcd(b,c):
            # Should not happen, but if it does, remove one
            break
    
    return result
