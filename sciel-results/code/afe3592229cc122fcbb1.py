"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if a subset is sum-free."""
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def count_conflicts(element, current_set):
        """Count how many conflicts element would create if added to current_set."""
        s = current_set
        count = 0
        for a in s:
            if a + element in s:
                count += 1
            if 2 * a == element:
                count += 1
        for a in s:
            for b in s:
                if a + b == element:
                    count += 1
        return count

    def greedy_add(elements, initial_subset):
        """Start with initial_subset, greedily add elements that don't create conflicts."""
        current = set(initial_subset)
        remaining = [x for x in elements if x not in current]
        
        # Sort by conflict count (fewest conflicts first)
        def conflict_score(x):
            return count_conflicts(x, current)
        
        remaining.sort(key=conflict_score)
        
        for x in remaining:
            if count_conflicts(x, current) == 0:
                current.add(x)
        
        return list(current)

    def local_search(elements, initial_subset, max_iters=5000):
        """Local search: start with initial_subset, try swaps to improve."""
        current = set(initial_subset)
        best = set(current)
        best_size = len(current)
        
        remaining = [x for x in elements if x not in current]
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding one element
            for x in remaining[:]:
                if count_conflicts(x, current) == 0:
                    current.add(x)
                    remaining.remove(x)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                    break
            
            if not improved:
                # Try swapping: remove one element, add another
                swapped = False
                elems_list = list(current)
                random.shuffle(elems_list)
                
                for rem in elems_list[:5]:
                    temp = set(current)
                    temp.discard(rem)
                    
                    for x in remaining[:20]:
                        if count_conflicts(x, temp) == 0:
                            temp.add(x)
                            if len(temp) > len(current):
                                current = temp
                                remaining.remove(x)
                                swapped = True
                                if len(current) > best_size:
                                    best_size = len(current)
                                    best = set(current)
                                break
                    if swapped:
                        break
            
            if not improved and not swapped:
                break
        
        return list(best)

    def spectral_bisection(elements):
        """
        Model as graph: edge between x,y if x+y is in elements.
        Use spectral bisection to partition, then take independent sets.
        """
        n = len(elements)
        if n <= 2:
            return list(elements)
        
        elem_set = set(elements)
        
        # Build adjacency: A[i][j] = 1 if elements[i] + elements[j] is in elements
        # This is a symmetric matrix
        A = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(i+1, n):
                s = elements[i] + elements[j]
                if s in elem_set:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Also add self-loops for 2x = y cases
        for i in range(n):
            for j in range(n):
                if i != j and 2 * elements[i] == elements[j]:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Compute eigenvector of smallest eigenvalue (Fiedler vector)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            fiedler = eigenvectors[:, 1]
        except:
            return list(elements)
        
        # Partition based on sign of Fiedler vector
        partition1 = [elements[i] for i in range(n) if fiedler[i] >= 0]
        partition2 = [elements[i] for i in range(n) if fiedler[i] < 0]
        
        # For each partition, find a large independent set (sum-free subset)
        def max_independent_set_in_partition(part):
            """Find a large sum-free subset within a partition."""
            if not part:
                return []
            
            # Start with all elements, remove conflicts
            current = set(part)
            part_set = set(part)
            
            improved = True
            while improved:
                improved = False
                elems_list = list(current)
                for i in range(len(elems_list)):
                    for j in range(i+1, len(elems_list)):
                        s = elems_list[i] + elems_list[j]
                        if s in current:
                            # Remove the sum (or the one with most conflicts)
                            current.discard(s)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try to add back removed elements
            removed = [x for x in part if x not in current]
            for x in removed:
                if count_conflicts(x, current) == 0:
                    current.add(x)
            
            return list(current)
        
        result1 = max_independent_set_in_partition(partition1)
        result2 = max_independent_set_in_partition(partition2)
        
        # Try both partitions and their combinations
        candidates = [result1, result2]
        
        # Also try the union if it's sum-free
        union = set(result1) | set(result2)
        if is_sum_free(union):
            candidates.append(list(union))
        
        # Also try starting with odds in each partition
        odd1 = [x for x in partition1 if x % 2 != 0]
        odd2 = [x for x in partition2 if x % 2 != 0]
        candidates.append(greedy_add(elements, odd1))
        candidates.append(greedy_add(elements, odd2))
        
        # Pick the best
        best = max(candidates, key=len)
        return best

    def find_best_sum_free(elements):
        """Try multiple strategies and return the best sum-free subset."""
        n = len(elements)
        if n == 0:
            return []
        
        # Strategy 1: All odds
        odds = [x for x in elements if x % 2 != 0]
        result1 = greedy_add(elements, odds)
        
        # Strategy 2: Spectral bisection
        result2 = spectral_bisection(elements)
        
        # Strategy 3: Local search from odds
        result3 = local_search(elements, odds, max_iters=3000)
        
        # Strategy 4: Local search from spectral result
        result4 = local_search(elements, result2, max_iters=3000)
        
        # Strategy 5: Random restarts
        best_random = []
        rng = random.Random(123)
        for _ in range(10):
            random_subset = rng.sample(elements, max(1, n // 2))
            # Clean up conflicts
            current = set(random_subset)
            improved = True
            while improved:
                improved = False
                elems_list = list(current)
                for i in range(len(elems_list)):
                    for j in range(i+1, len(elems_list)):
                        s = elems_list[i] + elems_list[j]
                        if s in current:
                            current.discard(s)
                            improved = True
                            break
                    if improved:
                        break
            # Add more
            remaining = [x for x in elements if x not in current]
            for x in remaining:
                if count_conflicts(x, current) == 0:
                    current.add(x)
            if len(current) > len(best_random):
                best_random = list(current)
        
        # Pick the best
        candidates = [result1, result2, result3, result4, best_random]
        best = max(candidates, key=len)
        
        # Verify and return
        if is_sum_free(best):
            return best
        else:
            # Fallback to odds
            return odds

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        subset = find_best_sum_free(elements)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
