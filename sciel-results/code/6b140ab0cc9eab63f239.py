"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 1000
    MAX_REP = 2
    
    # Helper: compute sum counts using boolean array convolution-like approach
    # For a set S, the number of representations of sum s as a+b with a<=b
    # can be computed efficiently using numpy operations
    
    def compute_sum_counts(selected, N=N):
        """Compute count of representations for each sum using numpy."""
        if len(selected) == 0:
            return np.zeros(2*N+1, dtype=np.int32)
        
        # Use boolean array representation
        arr = np.zeros(N+1, dtype=bool)
        arr[selected] = True
        
        # Compute all pair sums using outer product approach
        # For efficiency, use the fact that we need count of (a,b) with a<=b
        # such that a+b = s
        
        # Method: for each element in selected, add it to all elements >= it
        # This is O(k^2) where k = len(selected), but we can vectorize per element
        
        counts = np.zeros(2*N+1, dtype=np.int32)
        sel = np.array(selected)
        
        # For each element, compute sums with all elements >= it
        for i, x in enumerate(sel):
            # Elements from i onwards (x + sel[j] for j >= i, which ensures x <= sel[j] when sorted)
            # Since sel is sorted, sel[i:] >= x
            sums = x + sel[i:]
            np.add.at(counts, sums, 1)
        
        return counts
    
    # Start with a dense random initialization
    np.random.seed(42)
    initial_size = 450  # Start reasonably dense
    initial_set = set(np.random.choice(N, size=initial_size, replace=False))
    
    # Ensure sorted for consistent processing
    selected = sorted(initial_set)
    
    # Iteratively fix violations
    max_iterations = 500
    for iteration in range(max_iterations):
        counts = compute_sum_counts(selected, N)
        
        # Find sums with more than MAX_REP representations
        if len(counts) == 0:
            break
        
        violation_sums = np.where(counts > MAX_REP)[0]
        
        if len(violation_sums) == 0:
            # No violations, try to add more elements
            # Find candidates that would not create violations
            current_counts = counts.copy()
            
            # Try adding elements from 1 to N that are not in selected
            added_any = False
            for x in range(1, N+1):
                if x in selected:
                    continue
                
                # Check if adding x would violate the constraint
                # New sums: x + y for y in selected where y >= x, plus 2*x
                new_sums = []
                for y in selected:
                    if y >= x:
                        new_sums.append(x + y)
                new_sums.append(2*x)
                
                # Check if any new sum would exceed MAX_REP
                ok = True
                for s in new_sums:
                    if s <= 2*N:
                        if current_counts[s] + 1 > MAX_REP:
                            ok = False
                            break
                
                if ok:
                    selected.append(x)
                    selected.sort()
                    added_any = True
                    break
            
            if not added_any:
                # Can't add any more, try a different strategy
                # Randomly swap: remove a random element, try to add a different one
                if len(selected) > 10:
                    remove_idx = np.random.randint(len(selected))
                    removed = selected.pop(remove_idx)
                    # Try to add a random element not in selected
                    candidates = [x for x in range(1, N+1) if x not in selected]
                    if candidates:
                        add_candidate = np.random.choice(candidates)
                        # Check if we can add it
                        current_counts = compute_sum_counts(selected, N)
                        new_sums = []
                        for y in selected:
                            if y >= add_candidate:
                                new_sums.append(add_candidate + y)
                        new_sums.append(2*add_candidate)
                        
                        ok = True
                        for s in new_sums:
                            if s <= 2*N:
                                if current_counts[s] + 1 > MAX_REP:
                                    ok = False
                                    break
                        
                        if ok:
                            selected.append(add_candidate)
                            selected.sort()
                        else:
                            # Put it back
                            selected.append(removed)
                            selected.sort()
                break
        else:
            # There are violations, remove an element involved in a violation
            # Find elements that are part of violating sums
            violation_sums_set = set(violation_sums.tolist())
            
            # For each violating sum, find which elements could form it
            # Remove one element that appears in many violations
            removal_candidates = {}
            for s in violation_sums_set:
                if s > 2*N:
                    continue
                # Find pairs (a,b) in selected with a<=b and a+b=s
                for i, a in enumerate(selected):
                    if a > s//2:
                        break
                    b = s - a
                    if b < a:
                        continue
                    if b in selected:
                        removal_candidates[a] = removal_candidates.get(a, 0) + 1
                        if b != a:
                            removal_candidates[b] = removal_candidates.get(b, 0) + 1
            
            if removal_candidates:
                # Remove the element with the most violations
                worst_elem = max(removal_candidates, key=lambda x: removal_candidates[x])
                selected.remove(worst_elem)
            else:
                # Fallback: remove a random element
                if selected:
                    selected.pop()
    
    return selected
# EVOLVE-BLOCK-END
