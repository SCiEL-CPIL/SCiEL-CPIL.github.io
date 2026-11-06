"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def max_independent_set(adj, n, time_limit=30.0):
        """
        Find a maximum independent set using branch and bound.
        adj: list of sets, adj[i] = set of neighbors of vertex i
        n: number of vertices
        Returns the set of vertex indices in the maximum independent set.
        """
        start_time = time.time()
        best_set = set()
        best_size = 0

        # Degree ordering for better branching
        degrees = [len(adj[i]) for i in range(n)]

        # Simple greedy coloring upper bound
        def greedy_color_bound(candidates):
            """Compute a greedy coloring upper bound on MIS size for candidates."""
            if not candidates:
                return 0
            # Greedy coloring
            color_class = []
            uncolored = list(candidates)
            num_colors = 0
            while uncolored:
                num_colors += 1
                # Pick a vertex with minimum degree in remaining
                v = min(uncolored, key=lambda x: len(adj[x] & set(uncolored)))
                color_class.append(v)
                uncolored = [u for u in uncolored if u != v and u not in adj[v]]
            return num_colors

        def branch(candidates, current_set):
            nonlocal best_set, best_size
            if time.time() - start_time > time_limit:
                return

            if not candidates:
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                return

            # Upper bound: if current + max possible <= best, prune
            ub = len(current_set) + greedy_color_bound(candidates)
            if ub <= best_size:
                return

            # Pick a vertex to branch on - use degree heuristic
            v = min(candidates, key=lambda x: len(adj[x] & set(candidates)))

            # Branch 1: include v
            new_candidates = [u for u in candidates if u != v and u not in adj[v]]
            current_set.add(v)
            branch(new_candidates, current_set)
            current_set.remove(v)

            # Branch 2: exclude v
            new_candidates = [u for u in candidates if u != v]
            branch(new_candidates, current_set)

        branch(list(range(n)), set())
        return best_set

    def solve_one(S, seed, time_budget=25.0):
        n = len(S)
        S_set = set(S)
        elem_to_idx = {x: i for i, x in enumerate(S)}

        # Build conflict graph: edge between i and j if they can't coexist
        # i and j conflict if:
        # 1. 2*S[i] in S_set (i conflicts with itself via doubling)
        # 2. S[i] + S[j] in S_set (i+j = some element)
        # 3. S[i] - S[j] in S_set (i-j = some element, i.e., j+i=S[j]+S[i]... wait)
        # Actually: a+b=c means conflict between a and b if c is in set, 
        # but also conflict between a and c if b is in set, etc.
        # For independent set: no three elements a,b,c with a+b=c
        # So conflict between i and j if:
        #   - S[i] + S[j] in S_set (then can't have both i and j if their sum is present)
        #   - S[i] - S[j] in S_set (then S[j] + S[i] = S[i]... no)
        # Let me think again. a+b=c. The triple (a,b,c) is forbidden.
        # In the conflict graph for MIS, we want no edge between any pair that 
        # could form a forbidden triple. But it's not pairwise - it's a hypergraph.
        # 
        # Actually, the standard reduction: create a graph where i~j if 
        # S[i]+S[j] is in the set OR S[i]-S[j] is in the set OR 2*S[i] is in the set.
        # Wait, that's not quite right either.
        #
        # Let me reconsider. A set is sum-free if no a+b=c. 
        # For MIS formulation: vertices are elements. We need to select a subset 
        # with no a+b=c. This is NOT simply an independent set in a pairwise graph 
        # because the constraint is ternary.
        #
        # However, we can use a different approach: the conflict graph where 
        # i and j are adjacent if S[i]+S[j] is in S, or S[i]-S[j] is in S, or 
        # S[j]-S[i] is in S, or 2*S[i] is in S. Then any independent set in this 
        # graph is sum-free. This is a valid (but possibly not tight) formulation.
        #
        # Actually, let's verify: if I have an independent set in this graph, 
        # can I have a+b=c? If a+b=c, then a and b are adjacent (since a+b=c in S), 
        # so they can't both be in the independent set. Good.
        # But this might be too restrictive. Let's just use it and see.

        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                a, b = S[i], S[j]
                s = a + b
                d1 = a - b
                d2 = b - a
                # Conflict if any of these are in S
                if s in S_set or d1 in S_set or d2 in S_set or 2*a in S_set or 2*b in S_set:
                    adj[i].add(j)
                    adj[j].add(i)
            # Self-conflict: if 2*S[i] in S, then i can't be with itself, 
            # but since we're picking distinct elements, this means i is "bad" 
            # in the sense that if we pick i, we can't pick the element 2*S[i] 
            # (if it exists). This is already handled by the pairwise check above 
            # when j is the index of 2*S[i].

        # Run maximum independent set
        start = time.time()
        mis_indices = max_independent_set(adj, n, time_limit=time_budget)
        result = [S[i] for i in mis_indices]

        # Fallback: if MIS didn't find anything good, use odds
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > len(result):
            result = odds

        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            remaining = deadline - time.time()
            # Allocate more time to larger instances
            if size <= 60:
                budget = min(20.0, remaining - 5)
            elif size <= 120:
                budget = min(40.0, remaining - 5)
            else:
                budget = min(60.0, remaining - 5)
            
            result = solve_one(S, seed, time_budget=max(5.0, budget))
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)

    return results
# EVOLVE-BLOCK-END
