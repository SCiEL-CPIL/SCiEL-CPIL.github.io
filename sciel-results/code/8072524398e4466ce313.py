"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    deadline = time.time() + 72  # safety margin under 90s kill
    N = 128

    # Precompute GCD matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad_pairs: bad_pairs[v] = list of (a,b) such that (v,a,b) is a bad triple
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

    # Precompute lookup: bad_pair_lookup[v][u] = list of w such that (u,w) in bad_pairs[v]
    bad_pair_lookup = [{} for _ in range(N)]
    for v in range(N):
        for (a, b) in bad_pairs[v]:
            if a not in bad_pair_lookup[v]:
                bad_pair_lookup[v][a] = []
            bad_pair_lookup[v][a].append(b)
            if b not in bad_pair_lookup[v]:
                bad_pair_lookup[v][b] = []
            bad_pair_lookup[v][b].append(a)

    # Precompute degree (number of bad pairs involving v)
    degree = [len(bad_pairs[v]) for v in range(N)]

    def compute_blocked(selected_set):
        """Compute blocked counts for all vertices not in selected_set."""
        blocked = [0] * N
        sel_list = list(selected_set)
        for v in sel_list:
            for u in range(N):
                if u == v or u in selected_set:
                    continue
                if v in bad_pair_lookup[u]:
                    for w in bad_pair_lookup[u][v]:
                        if w in selected_set:
                            blocked[u] += 1
        return blocked

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid, using blocked counts."""
        selected = []
        selected_set = set()
        blocked = [0] * N
        for v in order:
            if v in selected_set:
                continue
            if blocked[v] == 0:
                selected.append(v)
                selected_set.add(v)
                for u in range(N):
                    if u == v or u in selected_set:
                        continue
                    if v in bad_pair_lookup[u]:
                        for w in bad_pair_lookup[u][v]:
                            if w in selected_set:
                                blocked[u] += 1
        return selected

    def local_improve(subset):
        """Try to improve subset by adding vertices that can fit. Multiple passes with different orders."""
        s = set(subset)
        blocked = compute_blocked(s)
        improved = True
        while improved:
            improved = False
            # Try adding in natural order first
            for v in range(N):
                if v in s:
                    continue
                if blocked[v] == 0:
                    s.add(v)
                    improved = True
                    for u in range(N):
                        if u == v or u in s:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s:
                                    blocked[u] += 1
            # Try again in reverse order to catch different opportunities
            for v in range(N-1, -1, -1):
                if v in s:
                    continue
                if blocked[v] == 0:
                    s.add(v)
                    improved = True
                    for u in range(N):
                        if u == v or u in s:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s:
                                    blocked[u] += 1
        return list(s)

    def swap_improve(subset, rng, max_swaps=20):
        """Try swap: remove one element, then greedily re-add. Limit swaps for speed."""
        s = set(subset)
        if not s:
            return list(s)
        best_size = len(s)
        best_set = None
        elements = list(s)
        rng.shuffle(elements)
        for rem in elements[:max_swaps]:
            if time.time() > deadline:
                break
            s2 = s - {rem}
            blocked2 = compute_blocked(s2)
            # Add in natural order
            for v in range(N):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            # Second pass in reverse
            for v in range(N-1, -1, -1):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = s2
        if best_set is not None:
            return list(best_set)
        return list(s)

    def branch_and_bound(initial_set, best_size, deadline):
        """
        B&B: try to find a larger independent set than best_size.
        Uses the initial_set as starting point and explores removals/additions.
        Upper bound: current_size + number of unblocked vertices not in current set.
        """
        s = set(initial_set)
        blocked = compute_blocked(s)
        cur_size = len(s)
        
        # Upper bound: cur_size + count of v not in s with blocked[v]==0
        ub = cur_size + sum(1 for v in range(N) if v not in s and blocked[v] == 0)
        if ub <= best_size:
            return list(s), best_size
        
        # Try to add all unblocked vertices
        for v in range(N):
            if v not in s and blocked[v] == 0:
                s.add(v)
                cur_size += 1
                for u in range(N):
                    if u == v or u in s:
                        continue
                    if v in bad_pair_lookup[u]:
                        for w in bad_pair_lookup[u][v]:
                            if w in s:
                                blocked[u] += 1
        
        if cur_size > best_size:
            best_size = cur_size
        
        # Now try removing one vertex at a time and re-optimizing
        elements = list(s)
        rng_local = random.Random(99)
        rng_local.shuffle(elements)
        
        for rem in elements:
            if time.time() > deadline:
                break
            s2 = s - {rem}
            blocked2 = compute_blocked(s2)
            # Greedily add all possible
            for v in range(N):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            if len(s2) > best_size:
                best_size = len(s2)
                s = s2
                blocked = blocked2
                cur_size = len(s)
                # Update upper bound
                ub = cur_size + sum(1 for v in range(N) if v not in s and blocked[v] == 0)
                if ub <= best_size:
                    break
        
        return list(s), best_size

    def targeted_polish(subset, deadline):
        """
        Try targeted removals: remove pairs of vertices that are 'conflicting'
        in the sense that removing both might allow adding more than 2 others.
        """
        s = set(subset)
        blocked = compute_blocked(s)
        cur_size = len(s)
        best_size = cur_size
        best_s = set(s)
        
        # Find vertices that are blocked
        blocked_vertices = [v for v in range(N) if v not in s and blocked[v] > 0]
        
        # For each blocked vertex, find which pairs in s are blocking it
        # Try removing those pairs and see if we gain
        rng_local = random.Random(77)
        rng_local.shuffle(blocked_vertices)
        
        for bv in blocked_vertices[:40]:  # increased from 30
            if time.time() > deadline:
                break
            # Find which pairs in s block bv
            blocking_pairs = []
            for (a, b) in bad_pairs[bv]:
                if a in s and b in s:
                    blocking_pairs.append((a, b))
            
            if not blocking_pairs:
                continue
            
            # Try removing one element from each blocking pair
            for (a, b) in blocking_pairs[:8]:  # increased from 5
                if time.time() > deadline:
                    break
                # Try removing a
                s2 = s - {a}
                blocked2 = compute_blocked(s2)
                for v in range(N):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                # Second pass
                for v in range(N-1, -1, -1):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                if len(s2) > best_size:
                    best_size = len(s2)
                    best_s = set(s2)
                # Try removing b
                s2 = s - {b}
                blocked2 = compute_blocked(s2)
                for v in range(N):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                # Second pass
                for v in range(N-1, -1, -1):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                if len(s2) > best_size:
                    best_size = len(s2)
                    best_s = set(s2)
        
        return list(best_s), best_size

    # Main search
    random.seed(42)
    rng = random.Random(12345)
    best = []
    best_size = 0

    # Phase 1: Multiple greedy + local_improve
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    # Sorted orderings
    for order in [list(range(N)), list(range(N-1, -1, -1)),
                  sorted(range(N), key=lambda x: degree[x]),
                  sorted(range(N), key=lambda x: -degree[x])]:
        if time.time() > deadline:
            break
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    # Phase 2: Swap improvement
    for trial in range(40):  # increased from 30
        if time.time() > deadline:
            break
        if not best:
            break
        improved = swap_improve(best, rng, max_swaps=25)  # increased from 15
        if len(improved) > best_size:
            best_size = len(improved)
            best = improved
        else:
            # Try a different randomization before giving up
            if trial < 20:
                continue
            else:
                break

    # Phase 3: B&B style exploration
    if best and time.time() < deadline - 5:
        improved, new_size = branch_and_bound(best, best_size, deadline - 2)
        if new_size > best_size:
            best_size = new_size
            best = improved

    # Phase 4: Targeted polish
    if best and time.time() < deadline - 3:
        improved, new_size = targeted_polish(best, deadline - 1)
        if new_size > best_size:
            best_size = new_size
            best = improved

    # Phase 5: More randomized restarts with perturbation
    for trial in range(60):  # increased from 50
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        for _ in range(3):  # increased from 2
            if time.time() > deadline:
                break
            if not candidate:
                break
            candidate = swap_improve(candidate, rng, max_swaps=15)  # increased from 10
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    # Phase 6: Final shake-and-polish - remove random subsets and re-optimize
    if best and time.time() < deadline - 5:
        rng_final = random.Random(54321)
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            # Remove a random subset of 2-4 elements
            k = rng_final.randint(2, 4)
            elems = list(best)
            rng_final.shuffle(elems)
            to_remove = set(elems[:k])
            s2 = set(best) - to_remove
            blocked2 = compute_blocked(s2)
            # Greedily re-add
            for v in range(N):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            # Second pass
            for v in range(N-1, -1, -1):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            if len(s2) > best_size:
                best_size = len(s2)
                best = list(s2)

    if not best:
        best = greedy_construction(list(range(N)))
        best = local_improve(best)

    return [v + 1 for v in sorted(best)]
# EVOLVE-BLOCK-END
