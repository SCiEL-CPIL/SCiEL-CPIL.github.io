"""Search for a 12-element ambient set minimizing the largest Sidon subset."""
import time
import random
import itertools
import sys

DEADLINE = 85.0  # safety margin under 90s wall-clock

def _largest_sidon_subset(arr):
    """Return the size of the largest subset of arr with all unordered pair sums unique (incl. diagonals).
    Uses max-clique on graph where edge = pair sums don't conflict with any other pair.
    Actually simpler: a subset S is Sidon iff for all distinct pairs (a,b),(c,d) with a<=b, c<=d,
    not (a+b == c+d). Equivalently, the set of all sums {a+b : a,b in S, a<=b} has size C(|S|+1,2).
    We find max |S| by checking subsets in decreasing size order, or via branch-and-bound.
    """
    n = len(arr)
    # Precompute all pair sums (a<=b)
    pair_sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            pair_sums.setdefault(s, []).append((i, j))
    
    # Two pairs (i,j) and (k,l) conflict if they share a sum and are not the same pair
    # Build conflict: pair (i,j) conflicts with (k,l) if i+j == k+l and (i,j) != (k,l)
    # A subset S is Sidon iff no two distinct pairs in S share a sum.
    # Equivalently: for each sum, at most one pair from S.
    # This is equivalent to: the pairs of S form a matching in the "sum-conflict" sense.
    
    # Alternative: S is Sidon iff the multiset of pair sums has all distinct values.
    # We want max |S| such that all C(|S|+1,2) pair sums are distinct.
    
    # Branch and bound: try subsets in order of decreasing size
    # For n=12, max possible is 12, check down.
    # Use a smarter approach: greedy + local search, or exact via backtracking.
    
    # Exact: backtracking with pruning
    # Order elements by index
    best = 1
    used_pairs = set()  # (i,j) with i<=j included
    
    def backtrack(start, current, current_size):
        nonlocal best
        if current_size > best:
            best = current_size
        # Try to add elements
        for i in range(start, n):
            # Check if adding i conflicts
            conflict = False
            for j in current:
                if j > i:
                    continue
                # pair (j, i) or (i, j) with i<=j
                a, b = (j, i) if j <= i else (i, j)
                s = arr[a] + arr[b]
                # Check if this sum is already used by another pair in current
                # We need: for all (p,q) in current with p<=q, arr[p]+arr[q] != s
                # unless (p,q) == (a,b)
                for (p, q) in used_pairs:
                    if p == a and q == b:
                        continue
                    if arr[p] + arr[q] == s:
                        conflict = True
                        break
            if conflict:
                continue
            # Add i
            for j in current:
                a, b = (j, i) if j <= i else (i, j)
                used_pairs.add((a, b))
            current.append(i)
            backtrack(i + 1, current, current_size + 1)
            current.pop()
            for j in current:
                a, b = (j, i) if j <= i else (i, j)
                used_pairs.discard((a, b))
    
    backtrack(0, [], 0)
    return best

def _evaluate_set(arr):
    """Return the largest Sidon subset size for arr."""
    return _largest_sidon_subset(arr)

def construct():
    start_time = time.time()
    
    # Initial candidate: try to build a set with many collisions
    # Strategy: use arithmetic progressions or structured sets
    # A set like {0, 1, 2, ..., 11} has many collisions.
    # Let's try various structured sets and random search.
    
    best_set = list(range(12))
    best_score = _evaluate_set(best_set)
    
    # Try some structured candidates
    candidates = []
    # Arithmetic progressions
    for base in range(0, 120, 2):
        for step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            s = [base + i * step for i in range(12)]
            if max(s) <= 128:
                candidates.append(s)
    # Multiples of small base
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        s = [i * base for i in range(12)]
        if max(s) <= 128:
            candidates.append(s)
    # Mix of small and large
    for small in range(0, 10):
        for gap in range(5, 20):
            s = [small + i * gap for i in range(12)]
            if max(s) <= 128:
                candidates.append(s)
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() - start_time > DEADLINE:
            break
        score = _evaluate_set(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
    
    # Random search
    rng = random.Random(42)
    iterations = 0
    while time.time() - start_time < DEADLINE:
        iterations += 1
        # Generate random set
        s = sorted(rng.sample(range(0, 129), 12))
        score = _evaluate_set(s)
        if score < best_score:
            best_score = score
            best_set = s[:]
            # Local search from here
            improved = True
            while improved and time.time() - start_time < DEADLINE:
                improved = False
                for i in range(12):
                    for new_val in range(max(0, best_set[i] - 3), min(129, best_set[i] + 4)):
                        if new_val in best_set and new_val != best_set[i]:
                            continue
                        old_val = best_set[i]
                        best_set[i] = new_val
                        best_set.sort()
                        new_score = _evaluate_set(best_set)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_set[i] = old_val
                            best_set.sort()
    
    return best_set

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Size: {len(result)}, Distinct: {len(set(result))}, Range: [0,128]")
    print(f"Largest Sidon subset: {_largest_sidon_subset(result)}")
