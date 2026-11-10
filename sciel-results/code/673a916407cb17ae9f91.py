import time
import random
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable(family):
    """Count unordered pairs where one set is a strict subset of the other."""
    n = len(family)
    # Represent each set as a frozenset for fast subset checks
    fsets = [frozenset(s) for s in family]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(fsets[i]) < len(fsets[j]):
                if fsets[i] <= fsets[j]:
                    count += 1
            elif len(fsets[j]) < len(fsets[i]):
                if fsets[j] <= fsets[i]:
                    count += 1
    return count

def _generate_initial_family():
    """Generate a good initial family: empty set + all singletons + best 27 size-2 sets."""
    n = P['n']
    m = P['m']
    family = [()]  # empty set
    # All singletons
    for i in range(n):
        family.append((i,))
    # Now add size-2 sets. We want to maximize comparability.
    # Each size-2 set {a,b} is comparable with {a} and {b} (2 pairs each)
    # All size-2 sets give the same 2 pairs, so any 27 work.
    # But to allow future swaps to size-3, let's pick size-2 sets that share elements.
    # Pick size-2 sets from a small subset of vertices to create density.
    # Use vertices 0..5 for size-2 sets: C(6,2)=15 sets, then need 12 more.
    # Actually, let's just pick the first 27 size-2 sets in lexicographic order.
    size2_sets = list(combinations(range(n), 2))
    # Shuffle for variety
    random.shuffle(size2_sets)
    for s in size2_sets[:m - len(family)]:
        family.append(s)
    return [list(s) for s in family]

def _local_search(family, time_limit=85.0):
    """Hill-climbing: try swapping one set for another to improve comparable count."""
    n = P['n']
    m = P['m']
    start_time = time.time()
    
    # Precompute all possible subsets as frozensets
    all_subsets = []
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            all_subsets.append(frozenset(combo))
    
    # For each possible subset, precompute which other subsets it's comparable with
    # comparable[i][j] = True if subset i is strict subset of subset j or vice versa
    num_subsets = len(all_subsets)
    # This is 4096 x 4096 = 16M entries, might be slow. Let's be smarter.
    
    # Alternative: for a given family, compute score. For swap, compute delta.
    def compute_score(fam):
        fsets = [frozenset(s) for s in fam]
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                if len(fsets[i]) < len(fsets[j]):
                    if fsets[i] <= fsets[j]:
                        cnt += 1
                elif len(fsets[j]) < len(fsets[i]):
                    if fsets[j] <= fsets[i]:
                        cnt += 1
        return cnt
    
    current_family = [list(s) for s in family]
    current_score = compute_score(current_family)
    
    # Precompute size of each set
    sizes = [len(s) for s in current_family]
    
    # For local search, try: pick one set to remove, try adding a different set
    # To make this efficient, precompute for each set in family, its comparability with others
    def get_comparable_mask(fset, fam):
        """Return list of indices j where fset is comparable with fam[j] (strict subset relation)."""
        result = []
        for j, fs in enumerate(fam):
            if len(fset) < len(fs):
                if fset <= fs:
                    result.append(j)
            elif len(fs) < len(fset):
                if fs <= fset:
                    result.append(j)
        return result
    
    # Greedy improvement loop
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        best_swap = None
        best_score = current_score
        
        # For each position i, try replacing family[i] with a new set
        # To limit computation, only try a subset of candidates
        for i in range(m):
            if time.time() - start_time > time_limit:
                break
            old_set = frozenset(current_family[i])
            old_size = sizes[i]
            
            # Find which other sets are comparable with old_set
            old_comparable = []
            for j in range(m):
                if j == i:
                    continue
                fs = frozenset(current_family[j])
                if old_size < len(fs):
                    if old_set <= fs:
                        old_comparable.append(j)
                elif len(fs) < old_size:
                    if fs <= old_set:
                        old_comparable.append(j)
            
            # Try candidate new sets: sets of size old_size-1, old_size, old_size+1
            # that might be comparable with more sets
            candidate_sizes = set()
            for sz in range(max(0, old_size - 2), min(n + 1, old_size + 3)):
                candidate_sizes.add(sz)
            
            # Generate candidates of those sizes
            candidates = []
            for sz in candidate_sizes:
                for combo in combinations(range(n), sz):
                    cfs = frozenset(combo)
                    if cfs == old_set:
                        continue
                    # Check if cfs is already in family
                    if cfs in set(frozenset(s) for s in current_family):
                        continue
                    candidates.append(cfs)
            
            # Limit candidates for speed
            if len(candidates) > 200:
                random.shuffle(candidates)
                candidates = candidates[:200]
            
            for cfs in candidates:
                csize = len(cfs)
                # Compute new comparable count
                new_comparable = 0
                for j in range(m):
                    if j == i:
                        continue
                    fs = frozenset(current_family[j])
                    if csize < len(fs):
                        if cfs <= fs:
                            new_comparable += 1
                    elif len(fs) < csize:
                        if fs <= cfs:
                            new_comparable += 1
                
                delta = new_comparable - len(old_comparable)
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_swap = (i, list(cfs))
        
        if best_swap is not None:
            i, new_set = best_swap
            current_family[i] = new_set
            sizes[i] = len(new_set)
            current_score = best_score
            improved = True
    
    return current_family, current_score

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin
    
    # Generate initial family
    random.seed(42)
    family = _generate_initial_family()
    
    # Local search
    family, score = _local_search(family, time_limit=deadline - (time.time() - start_time))
    
    # Check time
    if time.time() - start_time > deadline:
        # Return what we have
        pass
    
    # Ensure types are correct: list of lists of ints
    result = []
    for s in family:
        result.append([int(x) for x in s])
    
    return result
