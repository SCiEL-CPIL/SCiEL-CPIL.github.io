"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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

    def max_sum_free(elements, max_time=30.0):
        """
        Improved greedy + local search strategy:
        1. Sort elements by a heuristic score (e.g., centrality in conflict graph or random).
        2. Greedy construction: add elements if they don't violate sum-free property.
        3. Local search: 
           - Try to add any remaining element.
           - Try 1-for-1 swaps (remove one, add one) if it allows adding more later? 
             Actually, a simpler local search is: 
             - Try to add any element that fits.
             - If stuck, try removing an element and see if we can add two others.
        """
        n = len(elements)
        if n <= 3:
            return list(elements)
        
        # Heuristic ordering: 
        # Elements with high absolute value are "safer" because their sums/differences 
        # are less likely to be in the set if the set is sparse.
        # But a better heuristic might be based on the number of conflicts they create.
        # Let's use a dynamic greedy: at each step, pick the element that causes the 
        # fewest conflicts with the current set, or simply iterate through a sorted list.
        
        # Let's try sorting by absolute value descending first, as in the original, 
        # but then use a more aggressive local search.
        
        sorted_elems = sorted(elements, key=lambda x: abs(x), reverse=True)
        
        S = []
        S_set = set()
        
        # Precompute conflicts? No, dynamic check is O(|S|) per insertion.
        
        for x in sorted_elems:
            # Check if adding x violates sum-free property
            conflict = False
            for a in S:
                # Case 1: a + b = x => b = x - a
                if (x - a) in S_set:
                    conflict = True
                    break
                # Case 2: a + x = c => c = a + x
                if (a + x) in S_set:
                    conflict = True
                    break
            if (2 * x) in S_set:
                conflict = True
            
            if not conflict:
                S.append(x)
                S_set.add(x)
        
        # Local Search: Hill Climbing
        # 1. Try to add any remaining element.
        remaining = [x for x in elements if x not in S_set]
        rng_local = Random(42)
        rng_local.shuffle(remaining)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Pass 1: Add any element that fits
            for i in range(len(remaining)):
                if time.time() > max_time:
                    break
                x = remaining[i]
                conflict = False
                for a in S:
                    if (x - a) in S_set or (a + x) in S_set:
                        conflict = True
                        break
                if (2 * x) in S_set:
                    conflict = True
                
                if not conflict:
                    S.append(x)
                    S_set.add(x)
                    remaining.pop(i)
                    improved = True
                    # Restart the loop to check new possibilities
                    break
            
            # Pass 2: Try 1-for-1 swaps? 
            # If we remove x from S, can we add y from remaining?
            # This is tricky because removing x might allow adding y, but y might 
            # conflict with other elements in S.
            # A simpler improvement: try removing an element from S and see if we 
            # can add two elements from remaining that were blocked by it.
            # This is complex. Let's stick to the "add if fits" loop which is 
            # effective for many problems.
            
            # If no element could be added in the pass, try a random restart or 
            # accept the current solution. Given the time budget, we'll just stop.
            if not improved:
                # Try a few random swaps to escape local optima
                for _ in range(100):
                    if time.time() > max_time:
                        break
                    if not S or not remaining:
                        break
                    i = rng_local.randint(0, len(S) - 1)
                    j = rng_local.randint(0, len(remaining) - 1)
                    x_remove = S[i]
                    x_add = remaining[j]
                    
                    # Tentatively remove x_remove
                    S.remove(x_remove)
                    S_set.remove(x_remove)
                    
                    # Check if x_add can be added
                    conflict = False
                    for a in S:
                        if (x_add - a) in S_set or (a + x_add) in S_set:
                            conflict = True
                            break
                    if (2 * x_add) in S_set:
                        conflict = True
                    
                    if not conflict:
                        # Accept the swap
                        S.append(x_add)
                        S_set.add(x_add)
                        remaining.pop(j)
                        improved = True
                        break
                    else:
                        # Revert
                        S.insert(i, x_remove)
                        S_set.add(x_remove)
        
        return S

    def spectral_max_cut(elements, max_time=30.0):
        """
        Spectral method using the Laplacian of the "sum graph".
        Vertices: elements.
        Edges: (u, v) if u+v is in the set.
        We want to find a subset S that is sum-free.
        The constraint is: no a, b, c in S with a+b=c.
        This is equivalent to saying that for any a, b in S, a+b is NOT in S.
        
        Consider the graph G where (u, v) is an edge if u+v is in the set of all elements.
        If we choose S as an independent set in G, then for any u, v in S, u+v is not in the set of all elements, 
        hence not in S. So S is sum-free.
        However, independent set is a sufficient but not necessary condition.
        We can also have u, v in S such that u+v is in the set of all elements, but u+v is NOT in S.
        
        So the problem is not exactly Max Independent Set.
        But we can use the spectral bisection to find a "good" partition.
        
        Let's define the adjacency matrix A where A[i][j] = 1 if elements[i] + elements[j] is in the set.
        The Laplacian L = D - A.
        The Fiedler vector (eigenvector of the second smallest eigenvalue) gives a partition.
        We can take the elements with positive components in the Fiedler vector as S.
        Then we refine S using the local search.
        """
        n = len(elements)
        if n < 5:
            return max_sum_free(elements, max_time)
        
        elem_set = set(elements)
        
        # Build adjacency matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if elements[i] + elements[j] in elem_set:
                    A[i][j] = 1
        
        # Compute Laplacian
        D = np.sum(A, axis=1)
        L = np.diag(D) - A
        
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Find the Fiedler vector (second smallest eigenvalue)
            # If the graph is connected, eigvals[0] is 0.
            fiedler_idx = 1
            if abs(eigvals[0]) < 1e-8:
                if abs(eigvals[1]) < 1e-8:
                    # Disconnected graph, find the first non-zero
                    for k in range(len(eigvals)):
                        if abs(eigvals[k]) > 1e-8:
                            fiedler_idx = k
                            break
            
            v = eigvecs[:, fiedler_idx]
            
            # Partition based on sign of v
            S_indices = [i for i in range(n) if v[i] > 0]
            S = [elements[i] for i in S_indices]
            
            # Refine S using local search
            S_set = set(S)
            remaining = [x for x in elements if x not in S_set]
            
            # Check if S is sum-free, if not, repair it
            # We can just run the local search which will handle conflicts
            # But the local search assumes S is valid. 
            # So we need to repair S first.
            
            # Repair: Remove elements that cause conflicts
            # This is a greedy repair: iterate through S and remove any element 
            # that forms a triple with two other elements in S.
            
            # A simpler repair: start with an empty set and add elements from S 
            # using the greedy check.
            S_repaired = []
            S_repaired_set = set()
            for x in S:
                conflict = False
                for a in S_repaired:
                    if (x - a) in S_repaired_set or (a + x) in S_repaired_set:
                        conflict = True
                        break
                if (2 * x) in S_repaired_set:
                    conflict = True
                
                if not conflict:
                    S_repaired.append(x)
                    S_repaired_set.add(x)
            
            S = S_repaired
            S_set = S_repaired_set
            remaining = [x for x in elements if x not in S_set]
            
            # Run local search on S
            rng_local = Random(42)
            rng_local.shuffle(remaining)
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for i in range(len(remaining)):
                    if time.time() > max_time:
                        break
                    x = remaining[i]
                    conflict = False
                    for a in S:
                        if (x - a) in S_set or (a + x) in S_set:
                            conflict = True
                            break
                    if (2 * x) in S_set:
                        conflict = True
                    
                    if not conflict:
                        S.append(x)
                        S_set.add(x)
                        remaining.pop(i)
                        improved = True
                        break
                
                if not improved:
                    for _ in range(100):
                        if time.time() > max_time:
                            break
                        if not S or not remaining:
                            break
                        i = rng_local.randint(0, len(S) - 1)
                        j = rng_local.randint(0, len(remaining) - 1)
                        x_remove = S[i]
                        x_add = remaining[j]
                        
                        S.remove(x_remove)
                        S_set.remove(x_remove)
                        
                        conflict = False
                        for a in S:
                            if (x_add - a) in S_set or (a + x_add) in S_set:
                                conflict = True
                                break
                        if (2 * x_add) in S_set:
                            conflict = True
                        
                        if not conflict:
                            S.append(x_add)
                            S_set.add(x_add)
                            remaining.pop(j)
                            improved = True
                            break
                        else:
                            S.insert(i, x_remove)
                            S_set.add(x_remove)
            
            return S
            
        except Exception as e:
            return max_sum_free(elements, max_time)

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        time_budget = (size / 420.0) * 100.0
        
        remaining_time = deadline - time.time()
        if time_budget > remaining_time:
            time_budget = max(1.0, remaining_time - 1.0)
        
        # Use spectral method for larger sets, greedy for smaller
        if size > 100:
            subset = spectral_max_cut(elems, max_time=time_budget)
        else:
            subset = max_sum_free(elems, max_time=time_budget)
        
        # Verify sum-free property
        if not is_sum_free(subset):
            subset = max_sum_free(elems, max_time=time_budget)
        
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
