"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def build_greedy_vectorized():
        """Greedy with NumPy-vectorized conflict check, capped at MAX_SIZE."""
        selected = []
        triple_sums = set()
        
        for x in range(1, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 5:
                break
            
            if not selected:
                selected.append(x)
                triple_sums.add(3 * x)
                continue
            
            # Check all new triple sums involving x
            # New triples: (x,x,x), (x,x,y), (x,y,y), (x,y,z) for y,z in selected
            # We need to check if any of these sums are already in triple_sums
            # OR if any two new sums collide with each other
            
            arr = np.array(selected, dtype=np.int64)
            
            # (x,x,x)
            new_sums = [3 * x]
            
            # (x,x,y) for y in selected
            new_sums.extend(x * 2 + arr)
            
            # (x,y,y) for y in selected
            new_sums.extend(x + arr * 2)
            
            # (x,y,z) for y <= z in selected (unordered pairs with replacement)
            # To avoid duplicates within new_sums, we only consider y <= z
            for i in range(len(arr)):
                for j in range(i, len(arr)):
                    new_sums.append(x + arr[i] + arr[j])
            
            new_sums_arr = np.array(new_sums, dtype=np.int64)
            
            # Check for collisions within new_sums
            if len(new_sums_arr) != len(np.unique(new_sums_arr)):
                continue
            
            # Check for collisions with existing triple_sums
            # Convert triple_sums to numpy array for efficient intersection
            if triple_sums:
                existing_arr = np.array(list(triple_sums), dtype=np.int64)
                # Check if any new sum is in existing
                # Use sorting and binary search or set intersection
                # For efficiency with small sets, just use set
                new_set = set(new_sums_arr.tolist())
                if new_set & triple_sums:
                    continue
            
            # All good, add x
            selected.append(x)
            triple_sums.update(new_sums_arr.tolist())
        
        return selected

    def build_greedy_fast():
        """Faster greedy using set operations more efficiently."""
        selected = []
        triple_sums = set()
        
        for x in range(1, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 5:
                break
            
            if not selected:
                selected.append(x)
                triple_sums.add(3 * x)
                continue
            
            arr = selected
            
            # Generate all new triple sums involving x
            new_sums = set()
            
            # (x, x, x)
            new_sums.add(3 * x)
            
            # (x, x, y) for y in arr
            for y in arr:
                new_sums.add(2 * x + y)
            
            # (x, y, y) for y in arr
            for y in arr:
                new_sums.add(x + 2 * y)
            
            # (x, y, z) for y <= z in arr
            for i, y in enumerate(arr):
                for z in arr[i:]:
                    new_sums.add(x + y + z)
            
            # Check for internal collisions (shouldn't happen if we're careful, but just in case)
            # Actually, we need to check if any new sum is already in triple_sums
            if new_sums & triple_sums:
                continue
            
            # Also need to check that new_sums don't have duplicates among themselves
            # Since we used a set, duplicates are already handled. But wait - we need
            # to make sure the triples themselves produce unique sums. If two different
            # triples involving x produce the same sum, that's a collision.
            # Using a set for new_sums automatically handles this - if two triples
            # produce the same sum, the set will have fewer elements than expected.
            # Let's count expected number of new triples:
            n = len(arr)
            expected_count = 1 + n + n + n * (n + 1) // 2
            if len(new_sums) != expected_count:
                continue
            
            # All checks passed
            selected.append(x)
            triple_sums |= new_sums
        
        return selected

    # Try the fast greedy first
    result = build_greedy_fast()
    
    if len(result) >= MAX_SIZE:
        return result
    
    # If we didn't reach MAX_SIZE, try with different starting points or strategies
    # Try skipping some early numbers to see if we can do better
    best = result
    
    # Try greedy starting from different offsets
    for offset in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if time.time() > DEADLINE - 10:
            break
        alt = build_greedy_fast_with_offset(offset)
        if len(alt) > len(best):
            best = alt
    
    return best

def build_greedy_fast_with_offset(start_offset):
    """Greedy that skips some initial numbers."""
    N = 4096
    MAX_SIZE = 80
    selected = []
    triple_sums = set()
    
    # Start from start_offset
    for x in range(start_offset, N + 1):
        if len(selected) >= MAX_SIZE:
            break
        if time.time() > 80.0:  # rough deadline check
            break
        
        if not selected:
            selected.append(x)
            triple_sums.add(3 * x)
            continue
        
        arr = selected
        new_sums = set()
        
        new_sums.add(3 * x)
        for y in arr:
            new_sums.add(2 * x + y)
        for y in arr:
            new_sums.add(x + 2 * y)
        for i, y in enumerate(arr):
            for z in arr[i:]:
                new_sums.add(x + y + z)
        
        if new_sums & triple_sums:
            continue
        
        n = len(arr)
        expected_count = 1 + n + n + n * (n + 1) // 2
        if len(new_sums) != expected_count:
            continue
        
        selected.append(x)
        triple_sums |= new_sums
    
    return selected

# EVOLVE-BLOCK-END
