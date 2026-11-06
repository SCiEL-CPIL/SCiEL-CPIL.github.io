import time
import itertools

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    n = 12
    max_val = 1048576
    
    # Greedy construction
    x = []
    subset_sums = {0}
    
    for i in range(n):
        # Check deadline
        if time.time() - start_time > deadline - 5:
            # Fallback to powers of 2 if we run out of time
            # But for n=12, this is extremely unlikely
            return [1 << j for j in range(n)]
            
        # Find the smallest candidate x_i > max(x) such that
        # (subset_sums + x_i) is disjoint from subset_sums
        # We can start searching from max(x) + 1
        # Note: x_i must be distinct and positive.
        
        # Optimization: The new subset sums will be subset_sums and subset_sums + x_i.
        # They are disjoint if x_i is not in the difference set D = {a - b | a, b in subset_sums, a > b}.
        # Actually, we need x_i + s1 != s2 for all s1, s2 in subset_sums.
        # This means x_i != s2 - s1 for any s1, s2 in subset_sums.
        # So x_i must not be in the set of differences.
        
        # Let's compute the forbidden differences.
        # Since subset_sums size is 2^i, differences are O(4^i).
        # For i=11, size is 2048, differences ~ 4 million. This might be slow in Python.
        
        # Alternative: Just try candidates x and check intersection.
        # Current max sum is M. New sums are in [x, M+x].
        # We need x + s not in subset_sums.
        
        # Let's try a smarter search.
        # The condition is that the sets subset_sums and subset_sums + x are disjoint.
        
        # Heuristic: The smallest valid x is often close to the maximum difference or just above the max sum?
        # No, for powers of 2, x_{k} = 2^{k-1}.
        # For a "tight" set, the sums are dense.
        
        # Let's just iterate x upwards from max(x)+1.
        # For i=0, x=[1], sums={0,1}.
        # For i=1, try x=2. sums={0,1,2,3}.
        # For i=2, try x=4. sums={0..7}.
        
        # This greedy "smallest valid x" strategy is known as the "greedy Sidon" or similar.
        # Let's implement it carefully.
        
        current_max = max(x) if x else 0
        candidate = current_max + 1
        
        while True:
            # Check if candidate is valid
            # new_sums = {s + candidate for s in subset_sums}
            # if new_sums.isdisjoint(subset_sums): valid
            
            # Optimization: We only need to check if any s + candidate is in subset_sums.
            # i.e., candidate in {s2 - s1 for s1, s2 in subset_sums}
            
            # Let's use a set for subset_sums for O(1) lookups.
            valid = True
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    valid = False
                    break
            
            if valid:
                x.append(candidate)
                # Update subset_sums
                new_sums = set()
                for s in subset_sums:
                    new_sums.add(s + candidate)
                subset_sums.update(new_sums)
                break
            else:
                candidate += 1
                # Safety check: if candidate exceeds max_val, something is wrong or we hit the bound
                if candidate > max_val:
                    # Should not happen for n=12
                    return [1 << j for j in range(n)]
                
                # Deadline check inside loop if it takes too long
                if time.time() - start_time > deadline - 2:
                    return [1 << j for j in range(n)]
        
    return x

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Leave 5s margin for evaluation overhead
    
    n = 12
    max_val = 1048576
    
    x = []
    subset_sums = {0}
    
    for i in range(n):
        if time.time() - start_time > deadline:
            return [1 << j for j in range(n)]
            
        current_max = max(x) if x else 0
        candidate = current_max + 1
        
        while True:
            # Check validity: (subset_sums + candidate) disjoint from subset_sums
            # Equivalent to: no s1, s2 in subset_sums such that s2 - s1 == candidate
            valid = True
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    valid = False
                    break
            
            if valid:
                x.append(candidate)
                new_sums = {s + candidate for s in subset_sums}
                subset_sums.update(new_sums)
                break
            else:
                candidate += 1
                if candidate > max_val:
                    return [1 << j for j in range(n)]
                if time.time() - start_time > deadline:
                    return [1 << j for j in range(n)]
    return x
# EVOLVE-BLOCK-END
