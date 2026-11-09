"""Conflict-graph based sum-free subset search."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    from collections import defaultdict

    t_start = time.time()
    DEADLINE = 112.0  # safety margin under the 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_conflict_graph(S0):
        """Build conflict graph: edge (i,j) means i and j cannot both be in sum-free set.
        Conflict if i+j is in S0, or 2*i == j (i.e., i is half of j).
        Also self-conflict if 2*i is in S0 (can't have both i and 2i).
        """
        n = len(S0)
        S0_set = set(S0)
        idx = {v: i for i, v in enumerate(S0)}
        
        # conflicts[i] = set of j such that i and j conflict
        conflicts = [set() for _ in range(n)]
        
        for i in range(n):
            a = S0[i]
            # Check 2a in S0
            if 2 * a in idx:
                j = idx[2 * a]
                conflicts[i].add(j)
                conflicts[j].add(i)
            # Check a + b in S0 for all b != a
            for j in range(i + 1, n):
                b = S0[j]
                s = a + b
                if s in idx:
                    k = idx[s]
                    # a + b = s means {a,b,s} violates if all three present
                    # So a conflicts with s, and b conflicts with s
                    if k != i:
                        conflicts[i].add(k)
                        conflicts[k].add(i)
                    if k != j:
                        conflicts[j].add(k)
                        conflicts[k].add(j)
        
        return conflicts

    def greedy_independent_set(conflicts, n, order=None):
        """Greedy: pick vertices in given order, add if no conflict with already picked."""
        if order is None:
            order = list(range(n))
        picked = set()
        for v in order:
            if v in picked:
                continue
            # Check if v conflicts with any already picked
            ok = True
            for u in conflicts[v]:
                if u in picked:
                    ok = False
                    break
            if ok:
                picked.add(v)
        return picked

    def greedy_by_conflict_degree(conflicts, n):
        """Order by ascending conflict degree (pick least-conflicting first)."""
        degrees = [len(conflicts[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: degrees[i])
        return greedy_independent_set(conflicts, n, order)

    def local_search_conflict(conflicts, n, S_init_set, deadline, max_passes=200):
        """Local search: try removing one element and re-greedy-ing, or swapping."""
        S = set(S_init_set)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            
            # Try removing each element in S and see if we can add more
            for x in list(S):
                if time.time() > deadline:
                    break
                S2 = S - {x}
                # Try to add elements not in S2 that don't conflict
                added = 0
                for y in range(n):
                    if y in S2:
                        continue
                    ok = True
                    for u in conflicts[y]:
                        if u in S2:
                            ok = False
                            break
                    if ok:
                        S2.add(y)
                        added += 1
                if len(S2) > len(S):
                    S = S2
                    improved = True
            
            if not improved:
                break
        return S

    def sum_free_check(S):
        """Verify sum-free property."""
        S = set(S)
        for a in S:
            for b in S:
                if a + b in S:
                    return False
        return True

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S0 = instance(size, seed)
        n = len(S0)
        S0_set = set(S0)
        
        # Build conflict graph
        conflicts = build_conflict_graph(S0)
        
        # Strategy 1: Greedy by conflict degree
        cand1 = greedy_by_conflict_degree(conflicts, n)
        
        # Strategy 2: Greedy by absolute value (small first)
        order_abs = sorted(range(n), key=lambda i: abs(S0[i]))
        cand2 = greedy_independent_set(conflicts, n, order_abs)
        
        # Strategy 3: Greedy by value (large first)
        order_val = sorted(range(n), key=lambda i: S0[i], reverse=True)
        cand3 = greedy_independent_set(conflicts, n, order_val)
        
        # Strategy 4: Greedy by value (small first)
        order_val2 = sorted(range(n), key=lambda i: S0[i])
        cand4 = greedy_independent_set(conflicts, n, order_val2)
        
        # Strategy 5: Odd numbers only
        odd_order = [i for i in range(n) if S0[i] % 2 != 0]
        cand5 = greedy_independent_set(conflicts, n, odd_order)
        
        # Local search from best initial candidates
        candidates = [cand1, cand2, cand3, cand4, cand5]
        best = max(candidates, key=len)
        
        # Apply local search to best few candidates
        for cand in sorted(candidates, key=len, reverse=True)[:3]:
            if time.time() > DEADLINE - 15:
                break
            improved = local_search_conflict(conflicts, n, cand, DEADLINE - 5, max_passes=100)
            if len(improved) > len(best):
                best = improved
        
        # Convert back to values
        best_vals = sorted([S0[i] for i in best])
        
        # Final verification
        if not sum_free_check(best_vals):
            # Fallback: just use odd numbers
            best_vals = sorted([x for x in S0 if x % 2 != 0])
        
        results.append(best_vals)
    
    return results
# EVOLVE-BLOCK-END
