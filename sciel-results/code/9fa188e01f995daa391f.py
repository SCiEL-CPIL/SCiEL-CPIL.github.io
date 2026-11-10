import time
import random
import numpy as np

def _count_collisions(arr, n=1000):
    """Count sum representations for a set. Returns array where counts[s] = number of pairs (a,b) with a<=b, a+b=s."""
    if len(arr) == 0:
        return np.zeros(2*n+1, dtype=np.int32)
    # Use convolution: indicator * indicator gives all ordered pairs
    # We need unordered with a<=b, so:
    # total_ordered[s] = sum over a+b=s of indicator[a]*indicator[b]
    # For a<b: counted twice in ordered, once in unordered
    # For a=b: counted once in ordered, once in unordered
    # So unordered[s] = (total_ordered[s] + diag[s]) / 2 where diag[s] = 1 if s/2 in set
    ind = np.zeros(2*n+1, dtype=np.int32)
    ind[arr] = 1
    # Convolution
    conv = np.convolve(ind, ind, mode='full')
    # conv[s] = sum_{a+b=s} ind[a]*ind[b] = number of ordered pairs
    # For unordered with a<=b:
    # If s is even and s//2 is in set, the diagonal term (s//2, s//2) is counted once in conv
    # Off-diagonal pairs (a,b) with a!=b are counted twice in conv, once in unordered
    # So unordered[s] = (conv[s] + diag[s]) // 2 where diag[s] = 1 if s%2==0 and s//2 in set
    diag = np.zeros(2*n+1, dtype=np.int32)
    even_s = np.arange(0, 2*n+1, 2)
    diag_vals = ind[even_s // 2]
    diag[even_s] = diag_vals
    counts = (conv + diag) // 2
    return counts

def _is_valid(arr, n=1000):
    """Check if all sum counts are <= 2."""
    if len(arr) == 0:
        return True
    counts = _count_collisions(arr, n)
    return np.all(counts <= 2)

def _get_violations(arr, n=1000):
    """Get elements involved in violations (sums with count > 2)."""
    if len(arr) == 0:
        return set(), np.zeros(2*n+1, dtype=np.int32)
    counts = _count_collisions(arr, n)
    violating_sums = set(np.where(counts > 2)[0])
    # Find elements involved in violating sums
    involved = set()
    arr_set = set(arr)
    for s in violating_sums:
        for a in arr:
            b = s - a
            if a <= b and b in arr_set:
                involved.add(a)
                involved.add(b)
    return involved, counts

def _remove_to_fix(arr, n=1000, max_removals=500):
    """Remove elements to fix violations. Returns (valid_arr, removed_list)."""
    arr = list(arr)
    removed = []
    for _ in range(max_removals):
        involved, counts = _get_violations(arr, n)
        if not involved:
            break
        # Remove the element involved in the most violating sums
        # Score each element by how many violating sums it's in
        arr_set = set(arr)
        best_elem = None
        best_score = -1
        for elem in involved:
            if elem not in arr_set:
                continue
            score = 0
            for s in range(2*elem, 2*n+1):
                if counts[s] > 2:
                    b = s - elem
                    if b >= elem and b in arr_set:
                        score += counts[s] - 2
            if score > best_score:
                best_score = score
                best_elem = elem
        if best_elem is None:
            best_elem = list(involved)[0]
        arr.remove(best_elem)
        removed.append(best_elem)
    return arr, removed

def _try_add_back(arr, removed, n=1000, deadline=None):
    """Try to add removed elements back in order."""
    arr = list(arr)
    arr_set = set(arr)
    added = []
    for x in removed:
        if deadline is not None and time.time() > deadline:
            break
        if x in arr_set:
            continue
        # Check if adding x keeps validity
        new_arr = arr + [x]
        if _is_valid(new_arr, n):
            arr.append(x)
            arr_set.add(x)
            added.append(x)
    return arr, added

def _local_search_swap(arr, n=1000, deadline=None, max_iter=500):
    """Local search: try swapping elements to improve size while maintaining validity."""
    arr = sorted(arr)
    arr_set = set(arr)
    best_arr = list(arr)
    best_size = len(arr)
    rng = random.Random()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a random element not in set
        candidates = [x for x in range(1, n+1) if x not in arr_set]
        if candidates:
            for _ in range(10):
                x = rng.choice(candidates)
                if x in arr_set:
                    continue
                new_arr = arr + [x]
                if _is_valid(new_arr, n):
                    arr.append(x)
                    arr_set.add(x)
                    best_size = len(arr)
                    best_arr = list(arr)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        if len(arr) > 0:
            idx = rng.randint(0, len(arr) - 1)
            old_elem = arr[idx]
            
            # Remove old_elem
            arr.pop(idx)
            arr_set.discard(old_elem)
            
            # Try to add a better element
            candidates = [x for x in range(1, n+1) if x not in arr_set]
            rng.shuffle(candidates)
            improved_swap = False
            for x in candidates[:20]:
                new_arr = arr + [x]
                if _is_valid(new_arr, n):
                    arr.append(x)
                    arr_set.add(x)
                    if len(arr) > best_size:
                        best_size = len(arr)
                        best_arr = list(arr)
                    improved_swap = True
                    break
            
            if not improved_swap:
                # Restore old_elem
                arr.append(old_elem)
                arr_set.add(old_elem)
                arr.sort()
    
    return best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_set = []
    best_size = 0
    n = 1000
    
    # Strategy 1: Dense random initialization + repair
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(seed)
        
        # Dense random initialization
        density = 0.7
        k = int(n * density)
        init = rng.sample(range(1, n+1), k)
        
        # Repair: remove to fix violations
        valid_arr, removed = _remove_to_fix(init, n, max_removals=400)
        
        # Try to add back
        valid_arr, _ = _try_add_back(valid_arr, removed, n, deadline)
        
        # Local search
        if time.time() < deadline - 3.0:
            improved = _local_search_swap(valid_arr, n, deadline, max_iter=300)
            if len(improved) > len(valid_arr):
                valid_arr = improved
        
        if len(valid_arr) > best_size:
            best_size = len(valid_arr)
            best_set = valid_arr
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 10.0:
        arr = []
        arr_set = set()
        for x in range(1, n+1):
            if time.time() > deadline - 5.0:
                break
            new_arr = arr + [x]
            if _is_valid(new_arr, n):
                arr.append(x)
                arr_set.add(x)
        if len(arr) > best_size:
            best_size = len(arr)
            best_set = arr
    
    # Strategy 3: Greedy descending
    if time.time() < deadline - 10.0:
        arr = []
        arr_set = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 5.0:
                break
            new_arr = arr + [x]
            if _is_valid(new_arr, n):
                arr.append(x)
                arr_set.add(x)
        if len(arr) > best_size:
            best_size = len(arr)
            best_set = arr
    
    # Strategy 4: More sparse random init + repair
    for seed in range(50, 100):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(seed)
        density = 0.5
        k = int(n * density)
        init = rng.sample(range(1, n+1), k)
        
        valid_arr, removed = _remove_to_fix(init, n, max_removals=300)
        valid_arr, _ = _try_add_back(valid_arr, removed, n, deadline)
        
        if time.time() < deadline - 3.0:
            improved = _local_search_swap(valid_arr, n, deadline, max_iter=200)
            if len(improved) > len(valid_arr):
                valid_arr = improved
        
        if len(valid_arr) > best_size:
            best_size = len(valid_arr)
            best_set = valid_arr
    
    return sorted(best_set)
