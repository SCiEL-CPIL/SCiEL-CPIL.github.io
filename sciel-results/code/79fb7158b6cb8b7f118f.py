import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    sys.setrecursionlimit(10000)

    def is_valid(subset):
        reachable = 1
        for x in subset:
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1:
                return False
            reachable = new_reachable
        return True

    def build_conflict_graph():
        """
        Build a conflict graph where vertices are 1-100.
        Edge (u,v) exists if there's a subset of {u,v} that sums to 200,
        or more generally, if including both u and v makes it impossible
        to avoid a subset sum of 200 with some other elements.
        
        For simplicity, we'll use a pairwise conflict: u and v conflict if
        u + v = 200, or if there exists w such that u + v + w = 200 where
        w is in [1,100]. But this gets complex. Let's use a simpler proxy:
        u and v conflict if u + v >= 200 (since we can't use them together
        without risk), or if they're "close" to forming 200 with small additions.
        
        Actually, let's be more precise: u and v conflict if there exists a subset
        S of [1,100] \ {u,v} such that sum(S) + u + v = 200, OR sum(S) + u = 200,
        OR sum(S) + v = 200. This is too broad.
        
        Let's use: u and v conflict if u + v = 200, OR if |u + v - 200| is small
        enough that a single element could complete the sum.
        """
        n = 100
        conflict = [[False] * (n + 1) for _ in range(n + 1)]
        
        # Direct pairs: u + v = 200
        for u in range(1, n + 1):
            v = 200 - u
            if 1 <= v <= n and v != u:
                conflict[u][v] = True
                conflict[v][u] = True
        
        # Pairs that can form 200 with one more element: u + v + w = 200
        # This means u + v < 200 and 200 - u - v is in [1,100]
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                w = 200 - u - v
                if 1 <= w <= n and w != u and w != v:
                    conflict[u][v] = True
                    conflict[v][u] = True
        
        return conflict

    def find_bad_subset(subset):
        """
        Finds a subset of 'subset' that sums to 200.
        Returns the subset as a set, or None if no such subset exists.
        """
        # DP to track reachable sums and the element that achieved them
        # reachable_sum -> set of elements used? No, that's too big.
        # We just need ONE subset.
        # dp[s] = list of elements in a subset summing to s?
        # Better: dp[s] = (prev_sum, element) to backtrack.
        
        # Since max sum is 200, we can use an array of size 201.
        # parent[s] = (prev_s, x)
        parent = [None] * 201
        reachable = [False] * 201
        reachable[0] = True
        
        for x in subset:
            # Iterate downwards to avoid using x multiple times in the same step?
            # Actually, we are building a subset of the input set.
            # Standard 0/1 knapsack style.
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    if not reachable[s + x]:
                        reachable[s + x] = True
                        parent[s + x] = (s, x)
        
        if not reachable[200]:
            return None
        
        # Backtrack
        s = 200
        bad_subset = set()
        while parent[s] is not None:
            prev_s, x = parent[s]
            bad_subset.add(x)
            s = prev_s
        return bad_subset

    def repair(subset):
        """
        Removes elements from subset until it is valid.
        """
        subset = set(subset)
        while not is_valid(subset):
            bad = find_bad_subset(subset)
            if bad is None:
                break # Should not happen if is_valid is False
            
            # Try removing each element in the bad subset
            removed = False
            # Sort by value descending? Or random?
            # Let's try removing the largest element first, as it might be more "unique" in sums
            for x in sorted(bad, reverse=True):
                candidate = subset - {x}
                if is_valid(candidate):
                    subset = candidate
                    removed = True
                    break
            
            if not removed:
                # Fallback: remove the largest element in the bad subset
                x = max(bad)
                subset = subset - {x}
        
        return subset

    def greedy(order, existing=None):
        if existing is None:
            existing = set()
        reachable = 1
        selected = set(existing)
        for x in order:
            if x in selected:
                continue
            new_reachable = reachable | (reachable << x)
            if not ((new_reachable >> 200) & 1):
                selected.add(x)
                reachable = new_reachable
        return selected

    def greedy_min_degree(conflict, deadline):
        """
        Greedy minimum-degree heuristic for maximum independent set.
        At each step, pick the vertex with the fewest conflicts among remaining
        vertices, add it to the solution, and remove it and its neighbors.
        """
        remaining = set(range(1, 101))
        selected = set()
        
        while remaining and time.time() < deadline:
            # Find vertex with minimum degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and conflict[v][u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            
            if best_v is None:
                break
            
            selected.add(best_v)
            # Remove best_v and all its neighbors from remaining
            neighbors = {u for u in remaining if conflict[best_v][u]}
            remaining.discard(best_v)
            remaining -= neighbors
        
        return selected

    def local_search(best, deadline):
        """Improve via swap and add moves"""
        best = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements
            for x in range(1, 101):
                if x in best:
                    continue
                candidate = best | {x}
                if is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            if time.time() >= deadline:
                break
            # Try swapping: remove one, add one
            for remove_x in sorted(best):
                if time.time() >= deadline:
                    break
                remaining = best - {remove_x}
                for add_x in range(1, 101):
                    if add_x in remaining:
                        continue
                    candidate = remaining | {add_x}
                    if is_valid(candidate):
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
        return best

    # Generate diverse initial solutions
    random.seed(42)
    candidates = []

    # Strategy 1: Greedy from top
    candidates.append(greedy(range(100, 0, -1)))

    # Strategy 2: Greedy from bottom
    candidates.append(greedy(range(1, 101)))

    # Strategy 3: Greedy from middle out
    mid_order = list(range(50, 0, -1)) + list(range(51, 101))
    candidates.append(greedy(mid_order))

    # Strategy 4: Greedy prioritizing large numbers first
    candidates.append(greedy(sorted(range(1, 101), key=lambda x: -x)))

    # Strategy 5: Modular construction (Mod 3, non-zero)
    mod3_set = set(x for x in range(1, 101) if x % 3 != 0)
    repaired_mod3 = repair(mod3_set)
    candidates.append(repaired_mod3)

    # Strategy 6: Modular construction (Mod 4, 1,2)
    mod4_set = set(x for x in range(1, 101) if x % 4 in [1, 2])
    repaired_mod4 = repair(mod4_set)
    candidates.append(repaired_mod4)

    # Strategy 7-10: Random orderings
    for i in range(10):
        order = list(range(1, 101))
        random.shuffle(order)
        candidates.append(greedy(order))

    # Strategy 11: Graph-based minimum degree heuristic
    if time.time() < DEADLINE - 10.0:
        conflict = build_conflict_graph()
        graph_solution = greedy_min_degree(conflict, DEADLINE - 8.0)
        # Verify and repair if needed
        if is_valid(graph_solution):
            candidates.append(graph_solution)
        else:
            repaired_graph = repair(graph_solution)
            candidates.append(repaired_graph)

    # Pick best initial
    best = max(candidates, key=len)
    best_size = len(best)

    # Local search to improve
    best = local_search(best, DEADLINE - 2.0)

    # Try multiple local search restarts from different initial solutions
    if time.time() < DEADLINE - 5.0:
        for cand in sorted(candidates, key=len, reverse=True)[:8]:
            if time.time() >= DEADLINE - 3.0:
                break
            if len(cand) < best_size:
                continue
            improved = local_search(set(cand), DEADLINE - 2.0)
            if len(improved) > len(best):
                best = improved
                best_size = len(best)

    # Final attempt: try to add any remaining elements
    for x in range(1, 101):
        if time.time() >= DEADLINE - 1.0:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate
    
    # Second pass: try adding elements again after first pass
    for x in range(1, 101):
        if time.time() >= DEADLINE - 0.5:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate

    # Final verification
    assert len(best) > 0
    assert all(1 <= x <= 100 for x in best)
    assert len(best) == len(set(best))

    # Verify no subset sums to 200
    reachable = 1
    for x in sorted(best):
        new_reachable = reachable | (reachable << x)
        if (new_reachable >> 200) & 1:
            # Should not happen - return best effort
            return list(best)
        reachable = new_reachable

    return list(best)
