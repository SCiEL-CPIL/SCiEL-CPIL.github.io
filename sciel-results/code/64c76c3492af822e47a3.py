"""Sum-free subset maximizer with spectral graph partitioning and local search."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin under 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if (a + b) in Sset:
                    return False
        return True

    def build_schur_graph(vals):
        """Build adjacency matrix for Schur graph: edge (i,j) if vals[i]+vals[j] in vals."""
        n = len(vals)
        val_to_idx = {v: i for i, v in enumerate(vals)}
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # i and j conflict with k (i+j=k), also i and k conflict (j+k=i? no, j+k = j+i+k = s+k)
                    # Actually the constraint is: no a,b,c in S with a+b=c
                    # So if i+j=k, then {i,j,k} is a bad triple
                    # We want to model this as: we can't have all three
                    # For a 2-coloring approach, we can't directly model 3-cliques
                    # Instead, let's use a different approach: for each pair (i,j), 
                    # if i+j is in vals, then i and j are "compatible" only if i+j is not selected
                    # This is a 3-SAT-like problem, not a simple graph coloring
                    # Let's try a different approach: build a conflict graph where
                    # node i conflicts with node j if there exists k such that i+j=k, i+k=j, or j+k=i
                    # This means we can't have both i and j if k is also present
                    # But that's not quite right either
                    pass
        return adj

    def count_violations(S):
        Sset = set(S)
        cnt = 0
        for a in S:
            for b in S:
                if (a + b) in Sset:
                    cnt += 1
        return cnt

    def greedy_sum_free(vals):
        """Greedy: sort by |value| descending, add if no violation."""
        result = []
        rset = set()
        for x in sorted(vals, key=abs, reverse=True):
            ok = True
            for a in rset:
                if (x - a) in rset:
                    ok = False
                    break
            if ok:
                for b in rset:
                    if (x + b) in rset:
                        ok = False
                        break
            if ok and (2 * x) in rset:
                ok = False
            if ok:
                result.append(x)
                rset.add(x)
        return result

    def spectral_partition(vals):
        """Use spectral methods to partition vals into two sum-free-like sets."""
        n = len(vals)
        if n < 2:
            return list(vals)
        
        # Build a "compatibility" graph: edge (i,j) if vals[i] + vals[j] is NOT in vals
        # and vals[i] != -vals[j] (to avoid x + (-x) = 0 issues)
        # We want to find a large independent set in the conflict graph
        # Conflict: i conflicts with j if i+j in vals, or i+j = -k for some k in vals (then i+j+k=0, but that's not the constraint)
        # The constraint is purely: no a+b=c with a,b,c in S
        # So conflict between i and j exists if i+j is in vals
        # But also, if i is in S, then for any j in S, i+j must not be in S
        # This is a 3-uniform hypergraph independent set problem
        
        # Alternative: use the fact that if we take all numbers from a large interval 
        # (e.g., top n/2 by value), that's often sum-free or close to it
        # Let's try multiple spectral-like heuristics
        
        # Heuristic 1: Sort by value, take top half (large positives)
        sorted_vals = sorted(vals)
        half = n // 2
        top_half = sorted_vals[half:]
        
        # Check if top_half is sum-free
        if is_sum_free(top_half):
            return top_half
        
        # Heuristic 2: Use Fiedler vector on a proxy graph
        # Build graph where edge (i,j) weight = 1 if vals[i]+vals[j] NOT in vals
        # Then find a cut that maximizes edges within each part
        # This is max-cut on the compatibility graph
        val_set = set(vals)
        compat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j and (vals[i] + vals[j]) not in val_set:
                    compat[i, j] = 1.0
        
        # Compute Laplacian
        degree = compat.sum(axis=1)
        L = np.diag(degree) - compat
        
        # Get Fiedler vector (2nd smallest eigenvector)
        # For small n, use dense eigenvalue solver
        if n <= 200:
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
                # Fiedler vector is eigenvector for 2nd smallest eigenvalue
                fiedler = eigvecs[:, 1]
                # Partition by sign of Fiedler vector
                partition1 = [vals[i] for i in range(n) if fiedler[i] >= 0]
                partition2 = [vals[i] for i in range(n) if fiedler[i] < 0]
                
                # Check which partition is sum-free or closer
                p1_free = is_sum_free(partition1)
                p2_free = is_sum_free(partition2)
                
                if p1_free and p2_free:
                    return partition1 if len(partition1) >= len(partition2) else partition2
                elif p1_free:
                    return partition1
                elif p2_free:
                    return partition2
                else:
                    # Neither is sum-free, return larger one
                    return partition1 if len(partition1) >= len(partition2) else partition2
            except Exception:
                pass
        
        # Fallback: use value-based partition
        return top_half

    def local_search(S, vals, max_iters=5000, time_limit=15.0):
        """Try to improve S by adding elements or swapping. Returns improved set."""
        Sset = set(S)
        remaining = [x for x in vals if x not in Sset]
        best = list(S)
        best_size = len(best)
        rng = random.Random(42)
        t_start = time.time()

        # Phase 1: try to add each remaining element
        for x in remaining:
            if time.time() > t_start + time_limit:
                break
            ok = True
            for a in Sset:
                if (x - a) in Sset:
                    ok = False
                    break
            if ok:
                for a in Sset:
                    if (x + a) in Sset:
                        ok = False
                        break
            if ok and (2 * x) in Sset:
                ok = False
            if ok:
                Sset.add(x)
                best = list(Sset)
                best_size = len(best)

        # Phase 2: swap - remove one, try to add two
        for _ in range(max_iters):
            if time.time() > t_start + time_limit:
                break
            if len(best) < 1:
                break
            idx = rng.randrange(len(best))
            removed = best[idx]
            Sset.remove(removed)
            added = []
            for x in remaining:
                if x in Sset:
                    continue
                ok = True
                for a in Sset:
                    if (x - a) in Sset:
                        ok = False
                        break
                if ok:
                    for a in Sset:
                        if (x + a) in Sset:
                            ok = False
                            break
                if ok and (2 * x) in Sset:
                    ok = False
                if ok:
                    Sset.add(x)
                    added.append(x)
            if len(added) > 1:
                best = list(Sset)
                best_size = len(best)
                remaining = [x for x in vals if x not in Sset]
            else:
                Sset.add(removed)
                if added:
                    Sset.discard(added[0])
                    remaining = [x for x in vals if x not in Sset]

        return best

    def multi_start_local_search(vals, num_starts=5, time_budget=30.0):
        """Run local search from multiple starting points."""
        t_start = time.time()
        best_result = []
        best_size = 0
        
        # Generate diverse starting points
        starts = []
        
        # 1. Spectral partition
        spectral = spectral_partition(vals)
        starts.append(spectral)
        
        # 2. Greedy by abs value
        greedy = greedy_sum_free(vals)
        starts.append(greedy)
        
        # 3. Top half by value
        sorted_vals = sorted(vals)
        half = len(sorted_vals) // 2
        starts.append(sorted_vals[half:])
        
        # 4. Odd numbers
        odds = [x for x in vals if x % 2 != 0]
        starts.append(odds)
        
        # 5. Random subset of size n/2
        rng = random.Random(123)
        random_subset = rng.sample(vals, len(vals) // 2)
        starts.append(random_subset)
        
        # Run local search from each start
        for i, start in enumerate(starts):
            if time.time() - t_start > time_budget - 1.0:
                break
            remaining_time = time_budget - (time.time() - t_start) - 0.5
            if remaining_time < 1.0:
                break
            
            # Ensure start is sum-free by filtering
            filtered = []
            fset = set()
            for x in start:
                ok = True
                for a in fset:
                    if (x - a) in fset or (x + a) in fset:
                        ok = False
                        break
                if ok and (2 * x) in fset:
                    ok = False
                if ok:
                    filtered.append(x)
                    fset.add(x)
            
            improved = local_search(filtered, vals, time_limit=min(remaining_time, 10.0))
            if len(improved) > best_size:
                best_result = improved
                best_size = len(improved)
        
        return best_result

    def solve_one(vals, time_budget):
        """Find large sum-free subset of vals within time_budget seconds."""
        n = len(vals)
        t_start = time.time()
        
        # Use multi-start local search with spectral initialization
        result = multi_start_local_search(vals, num_starts=5, time_budget=time_budget)
        
        return result

    # Solve each instance
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]

    for size, seed in instances:
        vals = instance(size, seed)
        elapsed = time.time() - (deadline - 110.0)
        remaining = deadline - time.time()
        # Allocate time proportional to problem size, but at least some minimum
        if len(results) == 0:
            budget = min(remaining * 0.3, 30.0)
        elif len(results) == 1:
            budget = min(remaining * 0.35, 40.0)
        else:
            budget = min(remaining * 0.5, 50.0)
        budget = max(budget, 5.0)

        result = solve_one(vals, budget)
        results.append(result)

        if time.time() > deadline - 5.0:
            # Not enough time, use odd numbers for remaining
            for sz, sd in instances[len(results):]:
                v = instance(sz, sd)
                results.append([x for x in v if x % 2 != 0])
            break

    return results
# EVOLVE-BLOCK-END
