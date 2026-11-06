"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if adding x to S_set maintains sum-free property.
        Fast path: use set membership for O(1) lookups."""
        if 2 * x in S_set:
            return False
        # For each c in S_set: if c - x in S_set, violation (x + (c-x) = c)
        # For each a in S_set: if x - a in S_set, violation (a + (x-a) = x)
        # Combined: for each y in S_set, check if (x+y) in S_set or (x-y) in S_set
        for y in S_set:
            if (x + y) in S_set or (x - y) in S_set:
                return False
        return True

    def greedy_build(S, order=None):
        """Greedy construction: add elements in given order if they don't create conflicts."""
        if order is None:
            order = list(S)
        S_set = set()
        result = []
        for x in order:
            if can_add(x, S_set):
                result.append(x)
                S_set.add(x)
        return result

    def local_search_improve(S, initial_subset, max_rounds=50):
        """Local search: iteratively try to add elements that fit.
        Uses conflict-degree ordering to prioritize candidates likely to fit."""
        current = set(initial_subset)
        remaining = [x for x in S if x not in current]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            added = False
            # Sort remaining by how many conflicts they'd have with current set
            # (fewer conflicts = more likely to be addable)
            def conflict_count(x):
                cnt = 0
                if 2 * x in current:
                    cnt += 1
                for y in current:
                    if (x + y) in current:
                        cnt += 1
                    if (x - y) in current:
                        cnt += 1
                return cnt
            
            # Only re-sort if remaining is large enough to benefit
            if len(remaining) > 5:
                remaining.sort(key=conflict_count)
            
            for x in remaining[:]:
                if can_add(x, current):
                    current.add(x)
                    remaining.remove(x)
                    added = True
            if not added:
                break
        return list(current)

    def swap_improve(S, current, max_rounds=30):
        """Try to improve by removing one element and adding two others."""
        current_set = set(current)
        remaining = [x for x in S if x not in current_set]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for rem in list(current_set):
                if time.time() > deadline:
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
                    remaining = [x for x in S if x not in current_set]
                    improved = True
                    break
            if not improved:
                break
        return list(current_set)

    def two_opt_improve(S, current, max_rounds=30):
        """2-opt: remove 2 elements, try to add up to 3."""
        current_set = set(current)
        remaining = [x for x in S if x not in current_set]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            current_list = list(current_set)
            for i in range(len(current_list)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current_list)):
                    if time.time() > deadline:
                        break
                    rem_set = {current_list[i], current_list[j]}
                    temp_set = current_set - rem_set
                    added_count = 0
                    for x in remaining:
                        if added_count >= 3:
                            break
                        if can_add(x, temp_set):
                            temp_set.add(x)
                            added_count += 1
                    if added_count > 2:
                        current_set = temp_set
                        remaining = [x for x in S if x not in current_set]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return list(current_set)

    def build_conflict_graph(S):
        """Build a conflict graph where edges represent sum-free violations.
        Edge between i and j if S[i]+S[j] is in S (and the result is at index k,
        then we need edges i-j, i-k, j-k to form a triangle representing the violation).
        Actually, for independent set, we want: no three vertices a,b,c with a+b=c.
        This is a 3-uniform hypergraph constraint, not a simple graph.
        
        For spectral approach, we'll use a different strategy:
        Build a graph where edge (i,j) exists if S[i]+S[j] is in S.
        Then an independent set in this graph is NOT necessarily sum-free.
        
        Better approach: Use the fact that if we partition into two sets A and B,
        and ensure no a1+a2=c where all are in same set, we need:
        - For each pair (a,b) with a+b=c in S, at least one of a,b,c is in different partition.
        
        Let's use a simpler spectral heuristic:
        1. Build adjacency matrix where A[i,j]=1 if S[i]+S[j] is in S
        2. Compute Fiedler vector of the Laplacian
        3. Use sign of Fiedler vector to partition
        4. Take the larger independent set (verify and fix)
        """
        n = len(S)
        S_set = set(S)
        idx_map = {v: i for i, v in enumerate(S)}
        
        # Build conflict matrix: C[i,j] = 1 if S[i]+S[j] is in S
        # For sum-free, we need that for any a,b,c in our subset with a+b=c, this is forbidden.
        # So if we have a subset T, it's sum-free iff for all a,b in T, a+b not in T.
        
        # Spectral approach: Create a weighted graph where edges penalize sum-free violations.
        # If we assign each vertex a "color" (in/out), we want to minimize violations.
        
        # Simpler: Just use the greedy + local search which already works well.
        # Add spectral as an additional initialization strategy.
        
        # Build a graph where we connect i,j if S[i]+S[j] is in S.
        # The number of such "bad triples" involving vertices in our set should be 0.
        
        # For spectral bisection, let's create a graph and use Fiedler vector.
        # Edge weight between i and j: 1 if S[i]+S[j] in S, else 0.
        # But this doesn't directly give us independent sets for the 3-uniform constraint.
        
        # Alternative: Use a different spectral heuristic.
        # Sort by "conflict degree" and use that as initial ordering for greedy.
        
        conflict_degree = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if S[i] + S[j] in S_set:
                    conflict_degree[i] += 1
                    conflict_degree[j] += 1
        
        return conflict_degree

    def build_schur_graph(S):
        """Build the Schur graph for sum-free constraint.
        Vertices: elements of S.
        Edge (i,j) if S[i] + S[j] is in S (i.e., they form a bad pair).
        Also add self-loop consideration: if 2*S[i] is in S, mark as problematic.
        
        For 2-coloring: we want to partition S into A and B such that
        no a1 + a2 = c with all in same set. This is equivalent to finding
        large independent sets in a related graph structure.
        
        Actually, the proper formulation: create a graph G where edge (i,j) 
        exists if S[i]+S[j] ∈ S. Then a sum-free subset is NOT simply an
        independent set (that would only prevent pairs summing to something in S).
        
        Better approach: Use the fact that if we 2-color the elements, we want
        each color class to be sum-free. The constraint is: for any triple (a,b,c)
        with a+b=c, not all three can have the same color.
        
        This is a 3-uniform hypergraph coloring problem. We'll approximate it
        by building a pairwise conflict graph and using spectral methods.
        """
        n = len(S)
        S_set = set(S)
        idx_map = {v: i for i, v in enumerate(S)}
        
        # Build adjacency: edge (i,j) if S[i]+S[j] in S
        # This captures pairwise conflicts
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            si = S[i]
            # Self-conflict: if 2*si in S
            if 2 * si in S_set:
                rows.append(i)
                cols.append(i)
                data.append(2.0)  # Higher weight for self-conflicts
            
            for j in range(i + 1, n):
                sj = S[j]
                if si + sj in S_set:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([1.0, 1.0])
        
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        return A, S_set, idx_map
    
    def spectral_2coloring(S):
        """Use spectral bisection on the Schur graph to 2-color elements.
        Returns (color_A, color_B) where each is a list of elements.
        The goal is to find two large sum-free subsets."""
        n = len(S)
        if n < 4:
            return list(S), []
        
        A, S_set, idx_map = build_schur_graph(S)
        
        # Compute degree-weighted Laplacian
        L = laplacian(A, normed=True)
        
        try:
            if n <= 12:
                L_dense = L.toarray()
                eigvals, eigvecs = np.linalg.eigh(L_dense)
                # Fiedler vector: eigenvector for second smallest eigenvalue
                fiedler = eigvecs[:, 1]
            else:
                # Use sparse eigensolver for Fiedler vector
                k = min(3, n - 1)
                eigvals, eigvecs = eigsh(L, k=k, which='SM', tol=1e-8)
                idx = np.argsort(eigvals)
                fiedler = eigvecs[:, idx[1]]
            
            # Partition: positive side vs negative side
            # Try both assignments and pick the one with larger min-set
            median = np.median(fiedler)
            
            color_A = [S[i] for i in range(n) if fiedler[i] >= median]
            color_B = [S[i] for i in range(n) if fiedler[i] < median]
            
            # Also try threshold at 0
            color_A0 = [S[i] for i in range(n) if fiedler[i] >= 0]
            color_B0 = [S[i] for i in range(n) if fiedler[i] < 0]
            
            # Pick the partition that maximizes min(|A|, |B|)
            if min(len(color_A), len(color_B)) >= min(len(color_A0), len(color_B0)):
                return color_A, color_B
            else:
                return color_A0, color_B0
                
        except Exception:
            # Fallback: simple split by sign
            return [x for x in S if x > 0], [x for x in S if x <= 0]
    
    def iterative_2coloring(S, max_iters=20):
        """Iterative refinement of 2-coloring: start with spectral, then
        try to move elements to improve sum-freeness of each color class."""
        n = len(S)
        if n < 4:
            return list(S), []
        
        color_A, color_B = spectral_2coloring(S)
        A_set = set(color_A)
        B_set = set(color_B)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try moving elements from A to B and vice versa
            for x in list(color_A):
                if time.time() > deadline - 5:
                    break
                # Check if removing x from A and adding to B helps
                temp_A = A_set - {x}
                temp_B = B_set | {x}
                
                # Count violations in each set
                def count_violations(S_set):
                    cnt = 0
                    for a in S_set:
                        for b in S_set:
                            if a + b in S_set:
                                cnt += 1
                    return cnt
                
                # Only move if it reduces total violations
                old_viol = count_violations(A_set) + count_violations(B_set)
                new_viol = count_violations(temp_A) + count_violations(temp_B)
                
                if new_viol < old_viol:
                    A_set = temp_A
                    B_set = temp_B
                    color_A = list(A_set)
                    color_B = list(B_set)
                    improved = True
            
            if not improved:
                break
        
        return color_A, color_B
    
    def spectral_greedy_from_2color(S):
        """Use 2-coloring to get two candidate sum-free subsets,
        then greedily improve each one."""
        color_A, color_B = iterative_2coloring(S)
        
        # Clean up: remove elements that violate sum-freeness
        def clean_sum_free(subset):
            S_set = set(subset)
            result = []
            for x in subset:
                if can_add(x, S_set - set(result)):
                    result.append(x)
            return result
        
        cleaned_A = clean_sum_free(color_A)
        cleaned_B = clean_sum_free(color_B)
        
        # Local search improvement on each
        improved_A = local_search_improve(S, cleaned_A, max_rounds=30)
        improved_B = local_search_improve(S, cleaned_B, max_rounds=30)
        
        return improved_A if len(improved_A) >= len(improved_B) else improved_B

    def spectral_greedy(S):
        """Use spectral ordering to guide greedy construction."""
        if time.time() > deadline - 10:
            return greedy_build(S, sorted(S, key=abs))
        
        # Use the 2-coloring approach for ordering
        color_A, color_B = spectral_2coloring(S)
        
        # Try greedy on each color class
        result_A = greedy_build(S, color_A)
        result_B = greedy_build(S, color_B)
        
        # Also try combined orderings
        order1 = color_A + color_B
        order2 = color_B + color_A
        
        result1 = greedy_build(S, order1)
        result2 = greedy_build(S, order2)
        
        candidates = [result_A, result_B, result1, result2]
        return max(candidates, key=len)

    def solve_one(S, seed):
        """Find a large sum-free subset of S using multiple strategies."""
        n = len(S)
        if n == 0:
            return []
        
        best = []
        
        # Strategy 1: All odd numbers (always sum-free)
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > len(best):
            best = odds
        
        # Strategy 2: Greedy by absolute value
        greedy_abs = greedy_build(S, sorted(S, key=abs))
        if len(greedy_abs) > len(best):
            best = greedy_abs
        
        # Strategy 3: Greedy by value (ascending)
        greedy_asc = greedy_build(S, sorted(S))
        if len(greedy_asc) > len(best):
            best = greedy_asc
        
        # Strategy 4: Greedy by value (descending)
        greedy_desc = greedy_build(S, sorted(S, reverse=True))
        if len(greedy_desc) > len(best):
            best = greedy_desc
        
        # Strategy 4b: Greedy by conflict degree (fewer conflicts first)
        S_set = set(S)
        def conflict_deg(x):
            cnt = 0
            for y in S:
                if x != y and (x + y) in S_set:
                    cnt += 1
            return cnt
        greedy_conflict = greedy_build(S, sorted(S, key=conflict_deg))
        if len(greedy_conflict) > len(best):
            best = greedy_conflict
        
        # Strategy 4c: Greedy by conflict degree (more conflicts first - different heuristic)
        greedy_conflict_rev = greedy_build(S, sorted(S, key=conflict_deg, reverse=True))
        if len(greedy_conflict_rev) > len(best):
            best = greedy_conflict_rev
        
        # Strategy 5: Spectral 2-coloring approach (NEW)
        if time.time() < deadline - 20:
            spec_2color = spectral_greedy_from_2color(S)
            if len(spec_2color) > len(best):
                best = spec_2color
        
        # Strategy 5b: Spectral greedy (old approach)
        if time.time() < deadline - 15:
            spec_result = spectral_greedy(S)
            if len(spec_result) > len(best):
                best = spec_result
        
        # Strategy 6: Local search from odds
        if time.time() < deadline - 10:
            ls_from_odds = local_search_improve(S, odds, max_rounds=50)
            if len(ls_from_odds) > len(best):
                best = ls_from_odds
        
        # Strategy 7: Local search from greedy_abs
        if time.time() < deadline - 10:
            ls_from_greedy = local_search_improve(S, greedy_abs, max_rounds=50)
            if len(ls_from_greedy) > len(best):
                best = ls_from_greedy
        
        # Strategy 8: Local search from best
        if time.time() < deadline - 10:
            spec_ls = local_search_improve(S, best, max_rounds=50)
            if len(spec_ls) > len(best):
                best = spec_ls
        
        # Strategy 9: Random restarts with local search
        rng = Random(seed)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            p = 0.25 + rng.random() * 0.4
            random_subset = [x for x in S if rng.random() < p]
            order = sorted(random_subset, key=lambda x: (abs(x), rng.random()))
            cleaned = greedy_build(random_subset, order)
            improved = local_search_improve(S, cleaned, max_rounds=30)
            if len(improved) > len(best):
                best = improved
        
        # Strategy 10: Swap improvement on best
        if time.time() < deadline - 5:
            swapped = swap_improve(S, best, max_rounds=20)
            if len(swapped) > len(best):
                best = swapped
            if time.time() < deadline - 3:
                final_ls = local_search_improve(S, best, max_rounds=30)
                if len(final_ls) > len(best):
                    best = final_ls
        
        # Strategy 11: 2-opt improvement
        if time.time() < deadline - 3:
            two_opt = two_opt_improve(S, best, max_rounds=20)
            if len(two_opt) > len(best):
                best = two_opt
            if time.time() < deadline - 2:
                final_ls2 = local_search_improve(S, best, max_rounds=30)
                if len(final_ls2) > len(best):
                    best = final_ls2
        
        # Strategy 12: More diverse random restarts with perturbation
        if time.time() < deadline - 5:
            for _ in range(8):
                if time.time() > deadline - 3:
                    break
                best_set = set(best)
                perturbed = list(best_set)
                rng.shuffle(perturbed)
                remove_count = max(1, int(len(perturbed) * (0.1 + rng.random() * 0.15)))
                for _ in range(remove_count):
                    if perturbed:
                        perturbed.pop()
                improved = local_search_improve(S, perturbed, max_rounds=40)
                if len(improved) > len(best):
                    best = improved
        
        # Strategy 13: Simulated-annealing-style swaps
        if time.time() < deadline - 4:
            current_set = set(best)
            remaining = [x for x in S if x not in current_set]
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                improved = False
                for rem in list(current_set):
                    if time.time() > deadline - 3:
                        break
                    temp_set = current_set - {rem}
                    best_add = None
                    for x in remaining:
                        if can_add(x, temp_set):
                            best_add = x
                            break
                    if best_add is not None:
                        pass
                for rem in list(current_set):
                    if time.time() > deadline - 3:
                        break
                    temp_set = current_set - {rem}
                    added = []
                    for x in remaining:
                        if len(added) >= 2:
                            break
                        if can_add(x, temp_set):
                            temp_set.add(x)
                            added.append(x)
                    if len(added) >= 2:
                        current_set = temp_set
                        remaining = [x for x in S if x not in current_set]
                        improved = True
                        break
                if not improved:
                    break
            if len(current_set) > len(best):
                best = list(current_set)
        
        # Final polish: aggressive local search
        if time.time() < deadline - 2:
            final_polish = local_search_improve(S, best, max_rounds=100)
            if len(final_polish) > len(best):
                best = final_polish
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            result = solve_one(S, seed)
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
# EVOLVE-BLOCK-END
