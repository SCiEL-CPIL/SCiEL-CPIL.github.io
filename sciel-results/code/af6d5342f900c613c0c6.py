import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all equal pairwise gcds."""
        return g[a][b] == g[a][c] == g[b][c]
    
    def can_add(S, x, S_set):
        """Check if x can be added to set S without creating a bad triple."""
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if g[a][b] == g[a][x] == g[b][x]:
                    return False
        return True
    
    def greedy_build(order):
        """Build a valid set by processing numbers in the given order."""
        selected = []
        selected_set = set()
        for x in order:
            if x in selected_set:
                continue
            if can_add(selected, x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    def local_search(selected, max_iters=200):
        """Try to improve the set by swaps: remove one element, try to add others."""
        selected = list(selected)
        selected_set = set(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each element and see if we can add a better one
            for i in range(len(selected)):
                remove_idx = i
                candidate = selected[remove_idx]
                new_set = selected[:remove_idx] + selected[remove_idx+1:]
                new_set_set = set(new_set)
                # Try to add each number not in new_set
                for x in numbers:
                    if x in new_set_set:
                        continue
                    if can_add(new_set, x, new_set_set):
                        # We found a replacement or addition
                        if x != candidate:
                            # Swap: remove candidate, add x
                            selected = new_set + [x]
                            selected_set = new_set_set | {x}
                            improved = True
                            break
                        else:
                            # Can re-add same element, no improvement
                            break
                if improved:
                    break
        return selected
    
    # Strategy 1: Greedy by ascending order (natural)
    best = greedy_build(numbers)
    
    # Strategy 2: Greedy by descending order
    best2 = greedy_build(numbers[::-1])
    if len(best2) > len(best):
        best = best2
    
    # Strategy 3: Randomized restarts
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = numbers[:]
        random.shuffle(order)
        candidate = greedy_build(order)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 4: Greedy prioritizing numbers with fewer conflicts
    # Compute degree: for each number, count how many bad triples it participates in
    degrees = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad_triple(a, b, c):
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
    
    # Sort by degree (ascending - fewer conflicts first)
    order_by_degree = sorted(numbers, key=lambda x: degrees[x])
    candidate = greedy_build(order_by_degree)
    if len(candidate) > len(best):
        best = candidate
    
    # Strategy 5: Sort by degree descending (more conflicts first, avoid them)
    order_by_degree_desc = sorted(numbers, key=lambda x: -degrees[x])
    candidate = greedy_build(order_by_degree_desc)
    if len(candidate) > len(best):
        best = candidate
    
    # Local search on the best found
    best = local_search(best, max_iters=100)
    
    # More randomized restarts with local search
    random.seed(123)
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        order = numbers[:]
        random.shuffle(order)
        candidate = greedy_build(order)
        candidate = local_search(candidate, max_iters=50)
        if len(candidate) > len(best):
            best = candidate
    
    # Final local search pass with more iterations
    if time.time() < deadline - 5:
        best = local_search(best, max_iters=200)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
