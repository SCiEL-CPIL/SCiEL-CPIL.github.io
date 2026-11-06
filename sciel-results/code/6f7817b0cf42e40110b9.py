"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[gcd(a, b) for b in range(1, n + 1)] for a in range(1, n + 1)]
    
    # Find all "bad triples": three distinct elements a, b, c where
    # gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We'll use a set of frozensets for fast lookup
    bad_triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            gij = gcd_mat[i][j]
            for k in range(j + 1, n):
                if gcd_mat[i][k] == gij and gcd_mat[j][k] == gij:
                    bad_triples.add((i, j, k))
    
    # For fast conflict checking, for each pair (i,j), store which k's would form a bad triple
    # conflict_with[i][j] = set of k's such that {i,j,k} is a bad triple
    conflict_with = [[set() for _ in range(n)] for _ in range(n)]
    for (i, j, k) in bad_triples:
        conflict_with[i][j].add(k)
        conflict_with[j][i].add(k)
        conflict_with[i][k].add(j)
        conflict_with[k][i].add(j)
        conflict_with[j][k].add(i)
        conflict_with[k][j].add(i)
    
    # Also track for each element, which pairs it conflicts with
    # For a given set S, an element x can be added if for all pairs (a,b) in S,
    # x is not in conflict_with[a][b]
    
    def is_valid_addition(selected, x):
        """Check if x can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if x in conflict_with[selected[i]][selected[j]]:
                    return False
        return True
    
    def is_valid_set(selected):
        """Check if the entire set is valid (no bad triples)."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                for k in range(j + 1, len(selected)):
                    if (selected[i], selected[j], selected[k]) in bad_triples:
                        return False
        return True
    
    # Greedy construction with multiple orderings
    def greedy_construct(order):
        selected = []
        for x in order:
            if is_valid_addition(selected, x):
                selected.append(x)
        return selected
    
    # Local search: try to swap out elements to allow more in
    def local_search(selected, max_iterations=100000):
        selected = list(selected)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            # Try 1-for-1 swaps: remove one element, try to add another
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                # Try to add any element not in remaining
                for x in range(n):
                    if x in remaining:
                        continue
                    if is_valid_addition(remaining, x):
                        new_selected = remaining + [x]
                        # Try to add more greedily
                        for y in range(n):
                            if y in new_selected:
                                continue
                            if is_valid_addition(new_selected, y):
                                new_selected.append(y)
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            # Try 1-for-2 swaps: remove one element, try to add two
            if not improved:
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    removed = selected[i]
                    remaining = selected[:i] + selected[i+1:]
                    # Try to add two elements
                    candidates = [x for x in range(n) if x not in remaining and is_valid_addition(remaining, x)]
                    for x in candidates:
                        if time.time() > deadline:
                            break
                        temp = remaining + [x]
                        for y in candidates:
                            if y == x:
                                continue
                            if is_valid_addition(temp, y):
                                new_selected = temp + [y]
                                # Try to add more greedily
                                for z in range(n):
                                    if z in new_selected:
                                        continue
                                    if is_valid_addition(new_selected, z):
                                        new_selected.append(z)
                                if len(new_selected) > len(selected):
                                    selected = new_selected
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        return selected
    
    # Try multiple greedy orderings
    best = []
    
    # Standard order
    best = greedy_construct(elements)
    best = local_search(best, 50000)
    
    # Reverse order
    if time.time() < deadline:
        s = greedy_construct(elements[::-1])
        s = local_search(s, 50000)
        if len(s) > len(best):
            best = s
    
    # Try ordering by number of conflicts (least conflicts first)
    if time.time() < deadline:
        # Count how many bad triples each element is in
        triple_count = [0] * n
        for (i, j, k) in bad_triples:
            triple_count[i] += 1
            triple_count[j] += 1
            triple_count[k] += 1
        order = sorted(range(n), key=lambda x: triple_count[x])
        s = greedy_construct(order)
        s = local_search(s, 50000)
        if len(s) > len(best):
            best = s
    
    # Try ordering by number of conflicts (most conflicts first)
    if time.time() < deadline:
        order = sorted(range(n), key=lambda x: -triple_count[x])
        s = greedy_construct(order)
        s = local_search(s, 50000)
        if len(s) > len(best):
            best = s
    
    # Try random orderings
    import random
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s, 30000)
        if len(s) > len(best):
            best = s
    
    # Final polish: try to add any remaining elements
    for x in range(n):
        if time.time() > deadline:
            break
        if x not in best and is_valid_addition(best, x):
            best.append(x)
    
    # Convert back to actual numbers (1-indexed)
    result = [x + 1 for x in best]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
