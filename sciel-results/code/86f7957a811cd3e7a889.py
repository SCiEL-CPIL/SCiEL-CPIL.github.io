"""Stronger solver for sum-free subset maximization on three fixed instances."""
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_upper_bound(S, forbidden_pairs):
        """
        Greedy upper bound on max sum-free subset size.
        forbidden_pairs: for each element, the set of elements it conflicts with (a+b=c).
        We use a simple greedy coloring/weighting approach.
        
        Actually, let's use a simpler bound: the size of S itself, but we need something tighter.
        For branch and bound, we need an upper bound on how many more elements we can add.
        
        A simple valid upper bound: for each remaining element, count how many conflicts it has
        with already-selected elements. If it has any conflict, it can't be added.
        But we want an upper bound on the TOTAL we can still select.
        
        Let's use: for each unprocessed element, if it doesn't conflict with any selected element,
        it's a candidate. The max we can add is at most the number of candidates.
        But candidates may conflict with each other.
        
        A better approach: use the fact that in a sum-free set, at most one of {a, b, a+b} can be present
        if all three are in the universe. This gives a structural bound.
        
        For simplicity and speed, let's use: upper_bound = current_size + (number of elements that don't
        conflict with any already-chosen element). This is a valid upper bound because any element that
        conflicts with a chosen element cannot be in the final set.
        """
        selected_set = set(selected)
        candidates = []
        for x in remaining:
            # Check if x conflicts with any selected element
            conflict = False
            for s in selected_set:
                # s + x = c for some c in S? Or x + s = c? Or s + s = x?
                # We need: does adding x create a sum-free violation?
                # Violations: x + x = y for y in selected, or x + s = y for s,y in selected, or s + x = y
                # i.e., x*2 in selected, or for any s in selected, x+s in selected, or s+x in selected
                # Since addition is commutative, x+s = s+x
                if x * 2 in selected_set:
                    conflict = True
                    break
                for s in selected_set:
                    if x + s in selected_set:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                candidates.append(x)
        return len(candidates)

    def solve_instance(values, time_limit=40.0):
        """
        Solve one instance using branch-and-bound.
        values: sorted list of distinct nonzero integers.
        Returns the largest sum-free subset found.
        """
        n = len(values)
        if n == 0:
            return []
        
        # Precompute conflict structure
        # For each pair (i, j), does values[i] + values[j] appear in values?
        # If so, we can't have all of i, j, and the index of values[i]+values[j].
        
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        # For each element i, compute the set of elements j such that values[i] + values[j] is in values
        # i.e., the "conflict" set: if we pick both i and j, we cannot pick the element values[i]+values[j]
        # Also, if we pick i, we cannot pick i again (but we have distinct elements), and 2*values[i] if in values
        
        # Let's define: for each i, conflicts[i] = set of j such that values[i] + values[j] is in values
        # Note: if values[i] + values[j] == values[k], then {i, j, k} is a forbidden triple.
        
        conflicts = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i, n):  # j >= i to avoid duplicates, but we need to be careful
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Triple (i, j, k) is forbidden: can't have all three
                    # If i == j, then 2*values[i] = values[k]
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    # k is also involved
                    if k != i and k != j:
                        conflicts[k].add(i)
                        conflicts[k].add(j)
        
        # Actually, the constraint is: for any triple (a, b, c) with a+b=c, we cannot have all three in the set.
        # This is a 3-uniform hypergraph independent set problem.
        
        # For branch and bound, let's use a recursive approach with pruning.
        
        best_subset = []
        best_size = 0
        
        # Sort values by some heuristic: maybe by number of conflicts (most constrained first)
        # Or by value? Let's try by absolute value or by conflict degree.
        
        degree = [len(conflicts[i]) for i in range(n)]
        
        # Order: process elements in order of decreasing degree (most constrained first)
        order = sorted(range(n), key=lambda i: -degree[i])
        
        # For branch and bound, we need to track:
        # - which elements are already selected
        # - which elements are still available
        # - upper bound on how many more we can select
        
        # Let's use a different approach: iterative deepening or greedy with local search.
        # Given the time constraint, let's try a greedy approach with multiple restarts and local optimization.
        
        def greedy_solve(order_idx, values, conflicts, val_to_idx):
            """Greedy: add elements in order if they don't create a violation."""
            selected = set()
            for idx in order_idx:
                v = values[idx]
                # Check if adding v creates a violation
                # Violation if: 2v in selected, or for any s in selected, v+s in selected, or s+v in selected
                if 2 * v in selected:
                    continue
                conflict = False
                for s in selected:
                    if v + s in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
            return list(selected)
        
        # Try multiple greedy orders and pick the best
        candidates = []
        
        # Order 1: by degree (most constrained first)
        candidates.append(greedy_solve(order, values, conflicts, val_to_idx))
        
        # Order 2: by absolute value (smallest first)
        order_abs = sorted(range(n), key=lambda i: abs(values[i]))
        candidates.append(greedy_solve(order_abs, values, conflicts, val_to_idx))
        
        # Order 3: by value (smallest first)
        order_val = sorted(range(n), key=lambda i: values[i])
        candidates.append(greedy_solve(order_val, values, conflicts, val_to_idx))
        
        # Order 4: odd numbers first (heuristic: odds are sum-free among themselves)
        order_odd = sorted(range(n), key=lambda i: (values[i] % 2 == 0, abs(values[i])))
        candidates.append(greedy_solve(order_odd, values, conflicts, val_to_idx))
        
        # Order 5: negative numbers first
        order_neg = sorted(range(n), key=lambda i: (values[i] >= 0, abs(values[i])))
        candidates.append(greedy_solve(order_neg, values, conflicts, val_to_idx))
        
        # Order 6: positive numbers first
        order_pos = sorted(range(n), key=lambda i: (values[i] <= 0, abs(values[i])))
        candidates.append(greedy_solve(order_pos, values, conflicts, val_to_idx))
        
        # Pick the best greedy solution
        best = max(candidates, key=len)
        best_set = set(best)
        
        # Local search: try to add elements that don't conflict
        improved = True
        while improved:
            improved = False
            for v in values:
                if v in best_set:
                    continue
                # Check if v can be added
                if 2 * v in best_set:
                    continue
                conflict = False
                for s in best_set:
                    if v + s in best_set:
                        conflict = True
                        break
                if not conflict:
                    best_set.add(v)
                    improved = True
        
        # Local search: try to swap - remove one element, add one or more
        # This is more complex. Let's try a simple version.
        
        # Try removing each element and see if we can add more
        for v in list(best_set):
            best_set.remove(v)
            # Try to add elements
            added = 0
            for u in values:
                if u in best_set:
                    continue
                if 2 * u in best_set:
                    continue
                conflict = False
                for s in best_set:
                    if u + s in best_set:
                        conflict = True
                        break
                if not conflict:
                    best_set.add(u)
                    added += 1
            # If we added more than we removed (1), keep the change
            if added > 1:
                pass  # kept
            else:
                # Revert
                # Remove the elements we added
                for u in list(best_set):
                    if u != v and u not in best:  # This is getting complicated
                        pass
                # Simpler: just revert by removing all elements not in original best
                for u in list(best_set):
                    if u not in best:
                        best_set.remove(u)
                best_set.add(v)
        
        return list(best_set)
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > deadline:
            # Return what we have so far, padded with empty lists
            while len(results) < 3:
                results.append([])
            break
        
        vals = instance(size, seed)
        subset = solve_instance(vals)
        results.append(subset)
    
    return results
