"""Combined: Schur-graph spectral bisection + complete conflict local search for sum-free subsets."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 118.0  # safety margin before 120s kill

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
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if adding x to S_set maintains sum-free property.
        Three types of conflicts:
        1. 2*x in S_set (x+x=2x)
        2. exists c in S_set with c-x in S_set (i.e., (c-x)+x=c)
        3. exists a in S_set with x-a in S_set (i.e., a+(x-a)=x)
        """
        if 2 * x in S_set:
            return False
        for c in S_set:
            if c - x in S_set:
                return False
        for a in S_set:
            if x - a in S_set:
                return False
        return True

    def greedy_build(elements, order=None):
        """Greedy construction: add elements in given order if they don't create conflicts."""
        if order is None:
            order = list(elements)
        S_set = set()
        result = []
        for x in order:
            if can_add(x, S_set):
                result.append(x)
                S_set.add(x)
        return result

    def local_search_improve(elements, initial_subset, max_rounds=100, time_limit=None):
        """Local search: iteratively try to add elements that fit."""
        current = set(initial_subset)
        remaining = [x for x in elements if x not in current]

        for _ in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            added = False
            for x in remaining[:]:
                if can_add(x, current):
                    current.add(x)
                    remaining.remove(x)
                    added = True
            if not added:
                break
        return current

    def swap_improve(elements, current, max_rounds=30, time_limit=None):
        """Try to improve by removing one element and adding two others."""
        current_set = set(current)
        remaining = [x for x in elements if x not in current_set]

        for _ in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            for rem in list(current_set):
                if time_limit is not None and time.time() > time_limit:
                    break
                temp_set = current_set - {rem}
                added_count = 0
                new_elements = []
                for x in remaining:
                    if len(new_elements) >= 2:
                        break
                    if can_add(x, temp_set):
                        temp_set.add(x)
                        new_elements.append(x)
                        added_count += 1
                if added_count > 1:
                    current_set = temp_set
                    remaining = [x for x in elements if x not in current_set]
                    improved = True
                    break
            if not improved:
                break
        return current_set

    def schur_graph_spectral_bisection(elements, time_limit=None):
        """Build Schur graph and use spectral bisection to find large independent set.
        
        In the Schur graph, vertices are elements. An edge (u,v) exists if u and v 
        "conflict" - meaning they cannot both be in a sum-free set. Specifically:
        - If u+v = w for some w in elements, then u and v conflict (having both means w can't be included,
          but actually the constraint is: if u,v,w all in set, violation. So u,v don't directly conflict
          with each other, but they conflict with w.)
        
        Actually, let me reconsider. The sum-free constraint is: no a+b=c with a,b,c in S.
        This is a 3-uniform hypergraph constraint, not a graph constraint.
        
        However, we can create a derived graph: for each triple (a,b,c) with a+b=c,
        we can say that at most 2 of {a,b,c} can be in S. This is like a "not-all-three" constraint.
        
        For spectral bisection, let's think of it differently:
        - Create a graph where edge (x,y) means x+y is in elements (so x and y together force x+y out)
        - Also edge (x, 2x) if 2x is in elements (x+x=2x)
        
        Then a large independent set in this graph would be a set where no two elements sum to another in the set.
        But that's not quite sum-free either...
        
        Let me use a simpler approach: build the conflict graph where edge (i,j) means
        elements[i] + elements[j] is in the set OR elements[i] = 2*elements[j] (or vice versa).
        Then find a large independent set.
        
        Actually, the correct graph for this: 
        - Vertex per element
        - Edge between x and y if x+y is in the set (then having x and y means x+y must be excluded)
        - Edge between x and 2x if both in set (then having x means 2x must be excluded)
        
        An independent set in this graph: no two vertices connected by an edge are both selected.
        But this doesn't directly give sum-free. Let me think again...
        
        For sum-free: we need that for any a,b in S, a+b not in S.
        Equivalently: if a+b=c with a,b,c in the universe, then not all of {a,b,c} can be in S.
        
        One way to get a sum-free set: pick a subset T such that for every a,b in T, a+b is NOT in T.
        
        Let me try a different graph formulation:
        - For each element z in the set, create constraints: if a+b=z, then not (a in S AND b in S AND z in S)
        - This is a 3-SAT-like constraint
        
        For spectral methods, let's use a heuristic: 
        - Build a graph where edge (x,y) with weight 1 if x+y is in elements
        - Find a max-cut or use spectral partitioning
        - One side of the partition should be sum-free (approximately)
        
        Actually, let's use a simpler and more direct approach:
        - For each element x, define its "conflict set" C(x) = {y in elements : x+y in elements} ∪ {2x if 2x in elements}
        - Build a graph with edges (x, y) for y in C(x)
        - Find a large independent set in this graph using greedy or spectral methods
        
        An independent set in this graph means: for any x in the set, no y in the set has x+y in elements.
        But we also need: for any a,b in the set, a+b not in set. This is the same condition!
        Because if a+b is in the set, then there's an edge (a, b) (since b is in C(a) because a+b is in elements).
        So an independent set in this conflict graph IS sum-free!
        
        Wait, let me verify: 
        - C(x) = {y : x+y in elements} ∪ {2x if 2x in elements}
        - Edge (x,y) if y in C(x) or x in C(y)
        - Independent set S: for any x,y in S, no edge between them
        - This means: for any x,y in S, y not in C(x) and x not in C(y)
        - y not in C(x) means x+y not in elements (or x+y not in the universe)
        - So for any x,y in S, x+y not in elements, hence x+y not in S. ✓
        
        Great! So the conflict graph's independent sets are exactly the sum-free subsets.
        Now I need to find a large independent set in this graph.
        
        For n up to 240, I can use:
        1. Greedy: repeatedly pick vertex with minimum degree, add to independent set, remove it and its neighbors
        2. Spectral: use eigenvectors of the adjacency matrix to guide a partition
        3. Local search on the independent set
        
        Let me implement greedy + local search on the conflict graph.
        """
        n = len(elements)
        elem_set = set(elements)
        
        # Build conflict graph adjacency
        # adj[i] = set of j such that elements[i] and elements[j] conflict
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = elements[i]
            # x + y in elements for some y
            for j in range(n):
                y = elements[j]
                if x + y in elem_set:
                    adj[i].add(j)
                    adj[j].add(i)
            # 2x in elements
            if 2 * x in elem_set:
                # Find index of 2x
                for j in range(n):
                    if elements[j] == 2 * x:
                        adj[i].add(j)
                        adj[j].add(i)
                        break
        
        # Greedy independent set: pick vertex with min degree, remove it and neighbors
        # Try multiple orderings
        best_independent = set()
        best_size = 0
        
        def greedy_independent(order):
            """Given an ordering of vertices, greedily build independent set."""
            used = [False] * n
            indep = []
            for v in order:
                if used[v]:
                    continue
                # Check if v can be added (no neighbor already in indep)
                can_add = True
                for u in adj[v]:
                    if u in set(indep):
                        can_add = False
                        break
                if can_add:
                    indep.append(v)
                    used[v] = True
                    for u in adj[v]:
                        used[u] = True
            return set(indep)
        
        # Try different orderings
        orders = [
            list(range(n)),
            list(range(n, 0, -1)),
            sorted(range(n), key=lambda i: len(adj[i])),  # min degree first
            sorted(range(n), key=lambda i: -len(adj[i])),  # max degree first
            sorted(range(n), key=lambda i: elements[i]),  # by value
            sorted(range(n), key=lambda i: -elements[i]),
            sorted(range(n), key=lambda i: abs(elements[i])),
            sorted(range(n), key=lambda i: -abs(elements[i])),
        ]
        
        for order in orders:
            indep = greedy_independent(order)
            if len(indep) > best_size:
                best_size = len(indep)
                best_independent = indep
        
        # Convert back to elements
        best_elements = [elements[i] for i in best_independent]
        
        # Local search to improve
        current = set(best_elements)
        current = local_search_improve(elements, current, max_rounds=50, time_limit=time_limit)
        
        return current

    def solve_instance(elements, time_budget):
        """Find a large sum-free subset of elements using multiple strategies."""
        n = len(elements)
        if n == 0:
            return []

        best_subset = set()
        best_size = 0

        # Strategy 1: Parity (odd numbers) - always sum-free
        odds = {x for x in elements if x % 2 != 0}
        best_subset = odds
        best_size = len(odds)

        # Strategy 2: Greedy by various orderings
        for order in [
            sorted(elements, key=lambda x: -abs(x)),
            sorted(elements, key=lambda x: x),
            sorted(elements, key=lambda x: -x),
            sorted(elements, key=lambda x: abs(x)),
            list(elements),
            sorted(elements, key=lambda x: x % 2 != 0),  # odds first
        ]:
            S = greedy_build(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best_subset = set(S)

        # Strategy 3: Schur graph spectral/greedy bisection
        if time.time() < t0 + DEADLINE - 5:
            schur_result = schur_graph_spectral_bisection(elements, time_limit=t0 + DEADLINE - 3)
            if len(schur_result) > best_size:
                best_size = len(schur_result)
                best_subset = schur_result

        # Strategy 4: Local search from best so far
        if time.time() < t0 + DEADLINE - 5:
            ls_result = local_search_improve(elements, best_subset, max_rounds=200, time_limit=t0 + DEADLINE - 3)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_subset = ls_result

        # Strategy 5: Swap improvement
        if time.time() < t0 + DEADLINE - 5:
            swapped = swap_improve(elements, best_subset, max_rounds=20, time_limit=t0 + DEADLINE - 3)
            if len(swapped) > best_size:
                best_size = len(swapped)
                best_subset = swapped
            # Local search after swap
            if time.time() < t0 + DEADLINE - 3:
                final_ls = local_search_improve(elements, best_subset, max_rounds=100, time_limit=t0 + DEADLINE - 2)
                if len(final_ls) > best_size:
                    best_size = len(final_ls)
                    best_subset = final_ls

        # Strategy 6: Random restarts with local search
        rng = Random(12345)
        for restart in range(10):
            if time.time() > t0 + DEADLINE - 5:
                break
            # Start with random subset of about 30-50% of elements
            p = 0.3 + rng.random() * 0.2
            random_subset = [x for x in elements if rng.random() < p]
            # Clean up to make sum-free
            cleaned = greedy_build(random_subset, random_subset)
            # Local search
            improved = local_search_improve(elements, cleaned, max_rounds=50, time_limit=t0 + DEADLINE - 3)
            if len(improved) > best_size:
                best_size = len(improved)
                best_subset = improved

        # Final swap improvement
        if time.time() < t0 + DEADLINE - 3:
            swapped = swap_improve(elements, best_subset, max_rounds=15, time_limit=t0 + DEADLINE - 2)
            if len(swapped) > best_size:
                best_size = len(swapped)
                best_subset = swapped

        # Final verification
        if not is_sum_free(best_subset):
            # Fallback to parity
            best_subset = {x for x in elements if x % 2 != 0}

        return sorted(best_subset)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - t0 > DEADLINE - 8:
            # Not enough time; return parity for remaining
            elems = instance(size, seed)
            results.append(sorted([x for x in elems if x % 2 != 0]))
            continue
        elems = instance(size, seed)
        budget = min(DEADLINE - 8 - (time.time() - t0), 35.0)
        result = solve_instance(elems, budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
