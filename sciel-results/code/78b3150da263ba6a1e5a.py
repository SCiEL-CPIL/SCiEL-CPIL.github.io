"""Rewritten program: branch-and-bound max independent set on Schur conflict graph + multi-start local search."""
import time
import numpy as np
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_conflicts(values):
        """Build conflict bitmasks. conflict[i] has bit j set if values[i] and values[j] cannot both be in a sum-free set.
        Two elements x,y conflict if x+y is in values OR 2x is in values OR 2y is in values (i.e., x+x in values or y+y in values).
        Actually: sum-free means no a+b=c with a,b,c in S. So x,y in S conflict if:
          - x+y in S (then x+y=x+y, but x+y must also be in S; so if x+y is in values, x and y conflict)
          - 2x in S (then x+x=2x, so x conflicts with itself? No, x+x=2x means if both x and 2x are in S, conflict. But x and x are same element.)
          Wait: the constraint is no a+b=c with a,b,c all in S. So if x,y in S and x+y in S, that's a violation.
          Also if x in S and 2x in S, then x+x=2x is a violation.
          So x and y (x≠y) conflict if x+y is in values.
          x conflicts with itself if 2x is in values (but we only care about pairs in the independent set, and an element can't be "paired with itself" in the set - actually a=b is allowed in the constraint, so x+x=2x means if x and 2x are both in S, it's a violation. So x and 2x conflict as a pair.)
        """
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        val_set = set(values)
        # conflict[i] is a bitset of j such that values[i] and values[j] cannot coexist
        conflicts = [0] * n
        for i in range(n):
            x = values[i]
            # x conflicts with y if x+y in val_set (y != x handled, but if x+y==x then y=0, excluded)
            # x conflicts with 2x if 2x in val_set (pair {x, 2x} is bad because x+x=2x)
            # Also if 2x in val_set, then x and 2x conflict
            target = x + x
            if target in val_to_idx:
                j = val_to_idx[target]
                if j != i:
                    conflicts[i] |= (1 << j)
            for j in range(n):
                if i == j:
                    continue
                y = values[j]
                s = x + y
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # x + y = s, all three in values. So x and y conflict (can't both be in S if s is in S)
                    # But wait: the violation is a+b=c with a,b,c in S. If x,y,s all in S, violation.
                    # So x and y conflict if x+y is in values.
                    conflicts[i] |= (1 << j)
        return conflicts

    def bnb_max_independent_set(conflicts, n, time_budget):
        """Branch and bound for maximum independent set using bitsets.
        Returns (size, bitmask) of the best independent set found.
        """
        if n == 0:
            return 0, 0
        if n > 64:
            # Use Python integers as arbitrary precision bitsets
            pass
        
        best_size = [0]
        best_mask = [0]
        start_t = time.time()
        
        def elapsed():
            return time.time() - start_t
        
        def lower_bound(candidates_mask):
            """Lower bound: greedy coloring / degree-based estimate.
            Simple: use the fact that in any independent set, no two adjacent vertices.
            A quick lower bound: repeatedly pick the vertex with min degree in the candidate subgraph, add to set, remove it and its neighbors.
            But that's actually finding an independent set, not a bound.
            For an upper bound on what's left: we need to bound how many more we can add.
            Use: the size of a maximal independent set in the remaining graph is an upper bound? No.
            Better: use a coloring-based bound. Color the candidate subgraph; the number of colors is an upper bound on the independent set size (since an independent set can take at most one per color).
            But coloring is expensive. 
            Alternative: use the bound that the independent set size <= candidates_count.
            Or: use a greedy packing: the minimum degree d in the subgraph gives that we can add at most candidates_count / (d+1) more? Not quite.
            Let's use a simple bound: count of candidates. It's weak but fast.
            Actually, a better bound: the size of a maximum independent set in the subgraph is at most the number of vertices in the subgraph. That's trivial.
            For pruning to work, we need a tighter bound. Let's use: 
            In the candidate subgraph, let m = number of edges. A vertex cover has size at most... this is getting complex.
            Let's just use candidates_count as the upper bound for now, and rely on the branching to prune.
            Actually, a simple but useful bound: the independent set in the remaining candidates can have size at most the number of candidates. Combined with current_size, if current_size + candidate_count <= best_size, prune.
            """
            return candidates_mask.bit_count() if hasattr(int, 'bit_count') else bin(candidates_mask).count('1')
        
        def greedy_max_independent_set(candidates_mask):
            """Find a maximal independent set greedily (for lower bound / initial solution)."""
            result = 0
            remaining = candidates_mask
            while remaining:
                # Pick vertex with minimum degree in remaining subgraph
                best_v = -1
                best_deg = n + 1
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    # Degree of v in remaining subgraph
                    deg = (conflicts[v] & remaining).bit_count() if hasattr(int, 'bit_count') else bin(conflicts[v] & remaining).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp ^= lsb
                if best_v < 0:
                    break
                result |= (1 << best_v)
                # Remove best_v and all its neighbors from remaining
                remaining &= ~(conflicts[best_v] | (1 << best_v))
            return result
        
        def search(current_mask, current_size, candidates_mask):
            if elapsed() > time_budget:
                return
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            
            if candidates_mask == 0:
                return
            
            # Upper bound: current_size + number of candidates
            cand_count = candidates_mask.bit_count() if hasattr(int, 'bit_count') else bin(candidates_mask).count('1')
            if current_size + cand_count <= best_size[0]:
                return
            
            # If we can't beat best, prune
            # Pick a pivot vertex to reduce branching
            # Choose pivot p with maximum degree in the candidate subgraph
            pivot = -1
            max_deg = -1
            temp = candidates_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (conflicts[v] & candidates_mask).bit_count() if hasattr(int, 'bit_count') else bin(conflicts[v] & candidates_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                temp ^= lsb
            
            if pivot < 0:
                # No candidates
                return
            
            # Branch on neighbors of pivot in candidates: at least one of them must be excluded
            # Or: either include pivot, or exclude all neighbors of pivot
            # Standard B&B for MIS: pick a vertex v, branch on include v or exclude v.
            # But with pivot: the neighbors of pivot in the candidate set form a set where at least one must be in the independent set? No, at least one must be EXCLUDED from the independent set (since they're all adjacent to pivot, if pivot is included, all neighbors excluded; if pivot excluded, neighbors may still be included).
            # Actually, the standard trick: for a maximal independent set, either it contains pivot, or it contains at least one neighbor of pivot.
            # So we branch: (1) include pivot, (2) exclude pivot but include one of its neighbors.
            # This reduces branching factor.
            
            # Option 1: include pivot
            new_mask = current_mask | (1 << pivot)
            new_cands = candidates_mask & ~(conflicts[pivot] | (1 << pivot))
            search(new_mask, current_size + 1, new_cands)
            
            if elapsed() > time_budget:
                return
            
            # Option 2: exclude pivot, but we must include at least one neighbor of pivot
            # So for each neighbor u of pivot in candidates, branch on including u
            # But this can be many branches. Instead, we can just exclude pivot and continue.
            # Simpler: just branch on include/exclude pivot.
            # Let's do: exclude pivot
            new_cands2 = candidates_mask & ~(1 << pivot)
            search(current_mask, current_size, new_cands2)
        
        # Initial solution via greedy
        init_set = greedy_max_independent_set((1 << n) - 1)
        init_size = init_set.bit_count() if hasattr(int, 'bit_count') else bin(init_set).count('1')
        best_size[0] = init_size
        best_mask[0] = init_set
        
        search(0, 0, (1 << n) - 1)
        
        return best_size[0], best_mask[0]

    def local_search_1swap(conflicts, n, initial_mask, time_budget):
        """Local search: try 1-swap moves (remove one, add one) to improve."""
        best_mask = initial_mask
        best_size = best_mask.bit_count() if hasattr(int, 'bit_count') else bin(best_mask).count('1')
        start_t = time.time()
        
        def elapsed():
            return time.time() - start_t
        
        improved = True
        while improved and elapsed() < time_budget:
            improved = False
            # Try all 1-swap moves
            current = best_mask
            for i in range(n):
                if elapsed() > time_budget:
                    break
                if not (current & (1 << i)):
                    continue
                # Remove i
                without_i = current & ~(1 << i)
                # Try to add each j not in current
                for j in range(n):
                    if elapsed() > time_budget:
                        break
                    if current & (1 << j):
                        continue
                    # Check if j can be added to without_i
                    if conflicts[j] & without_i:
                        continue
                    # j is compatible with without_i
                    new_mask = without_i | (1 << j)
                    new_size = new_mask.bit_count() if hasattr(int, 'bit_count') else bin(new_mask).count('1')
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        improved = True
                        current = best_mask
                        break
                if improved:
                    break
        return best_size, best_mask

    def solve_one(values, time_budget):
        n = len(values)
        if n == 0:
            return []
        if n <= 2:
            return list(values)
        
        conflicts = build_conflicts(values)
        
        # For small n, use exact B&B
        if n <= 50:
            size, mask = bnb_max_independent_set(conflicts, n, time_budget)
            result = [values[i] for i in range(n) if mask & (1 << i)]
            return result
        
        # For larger n, use multi-start local search
        best_size = 0
        best_mask = 0
        rng = Random(12345)
        
        # Start with odd numbers
        odd_mask = 0
        for i, v in enumerate(values):
            if v % 2 != 0:
                odd_mask |= (1 << i)
        best_size = odd_mask.bit_count() if hasattr(int, 'bit_count') else bin(odd_mask).count('1')
        best_mask = odd_mask
        
        # Multiple random starts
        num_starts = max(3, int(time_budget / 2))
        for s in range(num_starts):
            if time.time() > start_time + time_budget + 5:
                break
            # Random initial set: include each vertex with probability 0.5, then fix conflicts
            init_mask = 0
            for i in range(n):
                if rng.random() < 0.5:
                    init_mask |= (1 << i)
            # Remove conflicts: iteratively remove vertices that conflict with already-selected
            # Greedy repair
            repaired = 0
            temp = init_mask
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                if conflicts[i] & repaired:
                    # Conflict, skip
                    pass
                else:
                    repaired |= (1 << i)
                temp ^= lsb
            init_mask = repaired
            
            # Local search
            size, mask = local_search_1swap(conflicts, n, init_mask, time_budget / max(1, num_starts - s))
            if size > best_size:
                best_size = size
                best_mask = mask
        
        # Also try greedy from sorted orderings
        for ordering in [sorted(range(n), key=lambda i: abs(values[i])), 
                         sorted(range(n), key=lambda i: abs(values[i]), reverse=True),
                         sorted(range(n), key=lambda i: values[i]),
                         sorted(range(n), key=lambda i: values[i], reverse=True)]:
            if time.time() > start_time + time_budget + 5:
                break
            init_mask = 0
            for i in ordering:
                if not (conflicts[i] & init_mask):
                    init_mask |= (1 << i)
            size, mask = local_search_1swap(conflicts, n, init_mask, 1.0)
            if size > best_size:
                best_size = size
                best_mask = mask
        
        result = [values[i] for i in range(n) if best_mask & (1 << i)]
        return result

    # Process instances
    instances = [(60, 801), (120, 802), (240, 803)]
    total_size = sum(s for s, _ in instances)
    
    results = []
    for size, seed in instances:
        if time.time() > DEADLINE - 5:
            results.append([])
            continue
        values = instance(size, seed)
        
        # Allocate time proportional to size
        time_budget = max(5, (size / total_size) * 120)
        subset = solve_one(values, time_budget=time_budget)
        
        results.append(subset)
    
    return results
