import time
import random

def construct():
    n = 12
    deadline = time.time() + 55  # Safety margin
    
    best_max = float('inf')
    best_set = None
    
    # Initial greedy solution
    elements = []
    subset_sums = {0}
    used = set()
    for i in range(n):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            if time.time() > deadline:
                break
            if candidate not in used:
                valid = True
                for s in subset_sums:
                    if (s + candidate) in subset_sums:
                        valid = False
                        break
                if valid:
                    elements.append(candidate)
                    used.add(candidate)
                    new_sums = {s + candidate for s in subset_sums}
                    subset_sums = subset_sums | new_sums
                    break
            candidate += 1
    if len(elements) == n:
        best_set = elements
        best_max = max(elements)
        
    # Local search / Random restarts
    # We try to improve the max by replacing the largest element with a smaller one
    # or rearranging elements.
    
    while time.time() < deadline:
        # Start from best_set or a random valid set
        if best_set:
            current = list(best_set)
        else:
            # Generate a random valid set
            current = []
            subset_sums = {0}
            used = set()
            for i in range(n):
                if time.time() > deadline:
                    break
                # Randomly pick from a range
                # Upper bound for random pick: best_max * 2
                upper = best_max * 2 if best_max < float('inf') else 2048
                found = False
                while not found:
                    candidate = random.randint(1, upper)
                    if candidate not in used:
                        valid = True
                        for s in subset_sums:
                            if (s + candidate) in subset_sums:
                                valid = False
                                break
                        if valid:
                            current.append(candidate)
                            used.add(candidate)
                            new_sums = {s + candidate for s in subset_sums}
                            subset_sums = subset_sums | new_sums
                            found = True
            if len(current) != n:
                continue
            current.sort()
        
        # Try to improve current
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current.sort()
            # Try to reduce the largest element
            # Find the smallest valid number that can replace the largest
            # Or swap two elements
            
            # Simple move: replace largest with a smaller unused number
            largest_idx = n - 1
            largest_val = current[largest_idx]
            
            # Build sums of the first n-1 elements
            prefix = current[:n-1]
            prefix_sums = {0}
            for p in prefix:
                prefix_sums = prefix_sums | {s + p for s in prefix_sums}
            
            # Find smallest valid replacement for largest_val
            # It must be > prefix[-1] to maintain sorted order? No, just distinct.
            # But to minimize max, we want it as small as possible.
            # It must not be in prefix.
            # And adding it to prefix_sums must not create collisions.
            
            candidate = 1
            found_replacement = False
            while candidate < largest_val:
                if time.time() > deadline:
                    break
                if candidate not in prefix:
                    valid = True
                    for s in prefix_sums:
                        if (s + candidate) in prefix_sums:
                            valid = False
                            break
                    if valid:
                        # Check if the new set has unique sums
                        # We already checked collisions with prefix.
                        # Do we need to check collisions among new sums?
                        # No, because new sums are s+candidate.
                        # If s1+candidate == s2+candidate => s1==s2.
                        # Since prefix sums are distinct, new sums are distinct.
                        # We just need to ensure new sums don't intersect old sums.
                        # Which is what we checked.
                        
                        new_set = prefix + [candidate]
                        new_set.sort()
                        if max(new_set) < max(current):
                            current = new_set
                            improved = True
                            found_replacement = True
                            break
                candidate += 1
            
            # If no replacement found, try random swaps or other moves
            # For now, just stop if no improvement
            if not improved:
                break
        
        if best_set is None or max(current) < best_max:
            best_set = current
            best_max = max(current)
            
    return best_set if best_set else [2**i for i in range(n)]
