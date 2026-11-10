"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85  # 5s safety margin under 90s kill
    N = 128
    # Precompute GCD matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad_pairs: for each vertex v, list of (a,b) pairs that with v form a bad triple
    bad_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd(a+1, b+1)
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd(a+1, c+1) == ga and gcd(b+1, c+1) == ga:
                    bad_pairs[c].append((a,b))
                    bad_pairs[a].append((b,c))
                    bad_pairs[b].append((a,c))

    # Convert to sets for faster membership testing
    bad_pairs_set = [set() for _ in range(N)]
    for v in range(N):
        for (a, b) in bad_pairs[v]:
            if a > b:
                a, b = b, a
            bad_pairs_set[v].add((a, b))

    def is_valid_subset(subset):
        """Check if subset has no bad triple."""
        s = set(subset)
        for v in s:
            for (a, b) in bad_pairs_set[v]:
                if a in s and b in s:
                    return False
        return True

    def can_add(v, selected_set):
        """Check if vertex v can be added to selected_set."""
        for (a, b) in bad_pairs_set[v]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid."""
        selected = []
        selected_set = set()
        for v in order:
            if can_add(v, selected_set):
                selected.append(v)
                selected_set.add(v)
        return selected

    def local_improve(subset):
        """Try to improve subset by adding vertices that can fit."""
        s = set(subset)
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in s:
                    continue
                if can_add(v, s):
                    s.add(v)
                    improved = True
        return list(s)

    def swap_improve(subset):
        """Try to improve by removing one vertex and adding one or two others."""
        s = set(subset)
        improved = True
        while improved:
            improved = False
            # Try single swap: remove v, try to add u
            for v in list(s):
                s_without_v = s - {v}
                for u in range(N):
                    if u in s_without_v:
                        continue
                    if can_add(u, s_without_v):
                        s = s_without_v | {u}
                        improved = True
                        break
                if improved:
                    break
            # Try double add after removing one: remove v, try to add u and w
            if not improved:
                for v in list(s):
                    s_without_v = s - {v}
                    added_any = False
                    for u in range(N):
                        if u in s_without_v:
                            continue
                        if can_add(u, s_without_v):
                            s_with_u = s_without_v | {u}
                            for w in range(N):
                                if w in s_with_u:
                                    continue
                                if can_add(w, s_with_u):
                                    s = s_with_u | {w}
                                    improved = True
                                    added_any = True
                                    break
                            if added_any:
                                break
                    if improved:
                        break
        return list(s)

    best = []
    random.seed(42)

    # Try multiple random orderings with greedy + local improve
    for trial in range(400):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Also try sorted orderings and other heuristics
    for order in [list(range(N)), list(range(N-1, -1, -1)),
                  sorted(range(N), key=lambda x: len(bad_pairs_set[x])),
                  sorted(range(N), key=lambda x: -len(bad_pairs_set[x]))]:
        if time.time() > deadline:
            break
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Additional randomized local search with perturbation
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        # Do a few rounds of random perturbation
        for _ in range(4):
            if time.time() > deadline:
                break
            if not candidate:
                break
            # Remove a random element
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
            candidate = local_improve(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Try swap-based improvement on the best solution found
    if best:
        improved = swap_improve(best)
        if len(improved) > len(best):
            best = improved
        # Try swap improvement on a few other good solutions
        for trial in range(30):
            if time.time() > deadline:
                break
            order = list(range(N))
            random.shuffle(order)
            candidate = greedy_construction(order)
            candidate = local_improve(candidate)
            if len(candidate) >= len(best) - 3:  # Only try if close to best
                improved = swap_improve(candidate)
                if len(improved) > len(best):
                    best = improved

    if not best:
        # Fallback: single greedy pass
        best = greedy_construction(list(range(N)))
        best = local_improve(best)

    # Final verification
    if not is_valid_subset(best):
        # Should not happen, but fallback to a known valid set
        best = [1]  # Single element is always valid

    # Convert back to 1-based
    return [v + 1 for v in sorted(best)]
# EVOLVE-BLOCK-END
