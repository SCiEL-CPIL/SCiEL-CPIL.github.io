"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    from collections import defaultdict

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if a set is sum-free: no a+b=c for a,b,c in set."""
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def build_triple_map(elements):
        """For each element c, find all pairs (a,b) such that a+b=c, where a,b in elements."""
        val_to_idx = {v: i for i, v in enumerate(elements)}
        # conflicts[c_idx] = list of (a_idx, b_idx) pairs
        # Also for local search, we need: for each element, which other elements conflict
        n = len(elements)
        # conflict_matrix: if i and j are both selected, and i+j=k, then i,j,k cannot all be selected
        # For local search: if we try to add element k, we need to remove one element from each pair (a,b) where a+b=k and a,b in current set
        # If we try to remove element x, we need to check if any a+b=x with a,b in set
        
        # Precompute: for each pair (i,j), i<=j, if elements[i]+elements[j] is in elements
        pair_to_sum = {}
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    pair_to_sum[(i, j)] = k
        return pair_to_sum, val_to_idx

    def greedy_threshold(elements, threshold_fn):
        """Greedy: sort by threshold_fn, add if doesn't create triple with already selected."""
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        # Precompute conflicts for each element: if element x is added, which pairs (a,b) in current set would violate?
        # Actually simpler: for each element c, list of pairs (a,b) such that a+b=c
        sum_to_pairs = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    sum_to_pairs[k].append((i, j))
        
        order = sorted(range(n), key=lambda i: threshold_fn(elements[i]))
        selected = set()
        for idx in order:
            c = elements[idx]
            # Check if adding c creates a triple: need to check if there exist a,b in selected such that a+b=c
            # Also need to check: for any a in selected, is a+c in selected? (then a,c,a+c triple)
            # And: is 2*c in selected? (c,c,2c triple)
            valid = True
            # Check a+b=c for a,b in selected
            for a_idx, b_idx in sum_to_pairs[idx]:
                if a_idx in selected and b_idx in selected:
                    valid = False
                    break
            if valid:
                # Check a+c=d for a in selected, d in selected
                for a in selected:
                    s = elements[a] + c
                    if s in val_to_idx and val_to_idx[s] in selected:
                        valid = False
                        break
            if valid:
                # Check c+c=d
                if 2*c in val_to_idx and val_to_idx[2*c] in selected:
                    valid = False
            if valid:
                selected.add(idx)
        return [elements[i] for i in selected]

    def local_search_improve(elements, initial_subset, max_iters=5000, timeout=None):
        """Local search: try adding/removing elements to improve size while maintaining sum-free."""
        start_time = time.time()
        val_to_idx = {v: i for i, v in enumerate(elements)}
        n = len(elements)
        
        # Build conflict structure
        # For each element k, pairs (a,b) such that a+b=k
        sum_to_pairs = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    sum_to_pairs[k].append((i, j))
        
        # Convert initial_subset to set of indices
        current = set()
        for v in initial_subset:
            if v in val_to_idx:
                current.add(val_to_idx[v])
        
        def is_valid(idx):
            """Check if adding idx to current would maintain sum-free."""
            if idx in current:
                return True
            c = elements[idx]
            # a+b=c for a,b in current
            for a_idx, b_idx in sum_to_pairs[idx]:
                if a_idx in current and b_idx in current:
                    return False
            # a+c=d for a,d in current
            for a in current:
                s = elements[a] + c
                if s in val_to_idx and val_to_idx[s] in current:
                    return False
            # 2c in current
            if 2*c in val_to_idx and val_to_idx[2*c] in current:
                return False
            return True
        
        def try_add(idx):
            if idx in current:
                return False
            if is_valid(idx):
                current.add(idx)
                return True
            return False
        
        def try_remove(idx):
            if idx in current:
                current.discard(idx)
                return True
            return False
        
        best_size = len(current)
        no_improve_count = 0
        
        for iteration in range(max_iters):
            if timeout and time.time() - start_time > timeout:
                break
            improved = False
            
            # Try to add elements not in current
            candidates_to_add = [i for i in range(n) if i not in current]
            random.shuffle(candidates_to_add)
            for idx in candidates_to_add:
                if is_valid(idx):
                    current.add(idx)
                    if len(current) > best_size:
                        best_size = len(current)
                    improved = True
                    break
            
            if not improved:
                # Try single removals followed by multiple additions
                for remove_idx in list(current):
                    if len(current) <= best_size - 1:
                        continue  # Don't go below best_size - 1
                    current.discard(remove_idx)
                    # Try to add as many as possible
                    added = 0
                    candidates_to_add = [i for i in range(n) if i not in current]
                    random.shuffle(candidates_to_add)
                    for idx in candidates_to_add:
                        if is_valid(idx):
                            current.add(idx)
                            added += 1
                    if len(current) > best_size:
                        best_size = len(current)
                        improved = True
                        break
                    else:
                        # Revert
                        current.discard(remove_idx)
                        # Re-add what we added (we didn't actually add since we broke)
                        # Actually we need to revert properly
                        # Since we only added if valid, and we're reverting the removal,
                        # the set is back to original minus remove_idx, then we re-add remove_idx
                        current.add(remove_idx)
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > 50:
                        break
            else:
                no_improve_count = 0
        
        return [elements[i] for i in current]

    def solve_sum_free(elements):
        """Solve maximum sum-free subset using multiple strategies + local search."""
        if not elements:
            return []
        n = len(elements)
        if n <= 1:
            return elements[:]
        
        start_time = time.time()
        timeout = 35  # seconds per instance
        
        best_subset = None
        best_size = 0
        
        # Strategy 1: Threshold-based greedy (upper half)
        def threshold_upper(x):
            return -x  # Larger x first
        subset1 = greedy_threshold(elements, threshold_upper)
        if len(subset1) > best_size:
            best_size = len(subset1)
            best_subset = subset1[:]
        
        # Strategy 2: Threshold-based greedy (lower half)
        def threshold_lower(x):
            return x  # Smaller x first
        subset2 = greedy_threshold(elements, threshold_lower)
        if len(subset2) > best_size:
            best_size = len(subset2)
            best_subset = subset2[:]
        
        # Strategy 3: Absolute value based
        def threshold_abs(x):
            return -abs(x)  # Largest magnitude first
        subset3 = greedy_threshold(elements, threshold_abs)
        if len(subset3) > best_size:
            best_size = len(subset3)
            best_subset = subset3[:]
        
        # Strategy 4: All positive elements
        def threshold_positive(x):
            if x > 0:
                return -x  # Positives first, larger first
            else:
                return x  # Negatives after
        
        subset4 = greedy_threshold(elements, threshold_positive)
        if len(subset4) > best_size:
            best_size = len(subset4)
            best_subset = subset4[:]
        
        # Strategy 5: All negative elements
        def threshold_negative(x):
            if x < 0:
                return x  # Negatives first, smaller (more negative) first
            else:
                return -x  # Positives after
        
        subset5 = greedy_threshold(elements, threshold_negative)
        if len(subset5) > best_size:
            best_size = len(subset5)
            best_subset = subset5[:]
        
        # Local search on the best initial subset
        if best_subset is not None:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 1:
                improved = local_search_improve(elements, best_subset, max_iters=10000, timeout=remaining_time - 0.5)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_subset = improved
        
        # Also try local search from a different starting point
        if time.time() - start_time < timeout - 2:
            # Try from greedy with random order
            def threshold_random(x, rng):
                return rng.random()
            
            rng = random.Random(42)
            order = list(range(n))
            rng.shuffle(order)
            # Build sum_to_pairs again (reuse)
            val_to_idx = {v: i for i, v in enumerate(elements)}
            sum_to_pairs = defaultdict(list)
            for i in range(n):
                for j in range(i, n):
                    s = elements[i] + elements[j]
                    if s in val_to_idx:
                        k = val_to_idx[s]
                        sum_to_pairs[k].append((i, j))
            
            selected = set()
            for idx in order:
                c = elements[idx]
                valid = True
                for a_idx, b_idx in sum_to_pairs[idx]:
                    if a_idx in selected and b_idx in selected:
                        valid = False
                        break
                if valid:
                    for a in selected:
                        s = elements[a] + c
                        if s in val_to_idx and val_to_idx[s] in selected:
                            valid = False
                            break
                if valid:
                    if 2*c in val_to_idx and val_to_idx[2*c] in selected:
                        valid = False
                if valid:
                    selected.add(idx)
            
            rand_subset = [elements[i] for i in selected]
            if len(rand_subset) > best_size:
                best_size = len(rand_subset)
                best_subset = rand_subset[:]
            
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 1:
                improved = local_search_improve(elements, best_subset, max_iters=5000, timeout=remaining_time - 0.5)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_subset = improved
        
        return best_subset if best_subset is not None else elements[:]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = solve_sum_free(elems)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
