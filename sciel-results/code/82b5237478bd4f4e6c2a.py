import time

# EVOLVE-BLOCK-START
def construct():
    MAX_N = 1000
    MAX_SUM = 2 * MAX_N
    
    # Initialize with all numbers 1..1000
    A = list(range(1, MAX_N + 1))
    A_set = set(A)
    
    # counts[s] = number of pairs (a, b) in A with a <= b and a + b = s
    counts = [0] * (MAX_SUM + 1)
    
    # Precompute initial counts
    # O(N^2)
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            aj = A[j]
            s = ai + aj
            if s <= MAX_SUM:
                counts[s] += 1
            else:
                break # Since A is sorted, if ai+aj > MAX_SUM, further aj will be larger
    
    # Greedy Deletion
    # We need to efficiently find which element to remove.
    # conflict_score[x] = number of sums s such that count[s] > 2 and x is in some pair summing to s.
    
    # To make this efficient, we can just re-scan or maintain a list of "bad" sums.
    # Given N=1000, we can afford a simpler implementation.
    
    # We will iterate until no sum has count > 2.
    # In each step, we find the element x that contributes to the most "bad" sums.
    
    # Optimization: 
    # A sum s is "bad" if counts[s] > 2.
    # For each bad sum s, let P_s be the set of pairs (a,b) in A summing to s.
    # Any x in the union of P_s for all bad s is a candidate for removal.
    # We want to remove the x that is in the most bad sums? Or the one that removes the most pairs from bad sums?
    
    # Simpler heuristic: 
    # Pick a bad sum s. Pick one of the elements in a pair summing to s to remove.
    # Which one? The one that appears in the most bad sums.
    
    # Let's maintain a list of bad sums.
    # Initially, find all bad sums.
    
    bad_sums = [s for s in range(2, MAX_SUM + 1) if counts[s] > 2]
    
    while bad_sums:
        # Calculate conflict score for each element in A
        # score[x] = number of bad sums s such that x is part of a pair summing to s
        
        scores = {x: 0 for x in A}
        
        # For each bad sum, find pairs in A
        # This is expensive if done naively for every bad sum.
        # Number of bad sums can be large initially.
        
        # Alternative: 
        # Just pick the first bad sum, and remove the element in it that has the highest degree (in A) or something.
        # Let's try: Pick a random bad sum? No, must be deterministic.
        # Pick the smallest bad sum s.
        # Find all pairs (a,b) in A with a+b=s.
        # For each element in these pairs, count how many bad sums it is involved in.
        # Remove the one with the highest count.
        
        # To find pairs for a sum s in A:
        # a must be in A, b = s - a must be in A, a <= b.
        
        s = min(bad_sums)
        
        # Find pairs for s
        pairs = []
        for a in A:
            if a > s // 2:
                break
            b = s - a
            if b in A_set:
                pairs.append((a, b))
        
        if not pairs:
            # Should not happen if counts[s] > 0
            # But if counts[s] > 2, there must be pairs.
            # Maybe counts are stale? We update them.
            pass
            
        # Count involvement in bad sums for elements in pairs
        # We only need to compare elements in these pairs?
        # No, an element might be in a pair for s, but removing it might fix other sums too.
        # Let's calculate score for all elements in A? Too slow.
        # Let's calculate score for elements in the union of pairs for all bad sums?
        
        # Let's just pick the element in the current bad sum s that has the highest "degree" in the graph of bad sums.
        # Approximation: Count how many bad sums each element in `pairs` is part of.
        
        element_scores = {}
        for a, b in pairs:
            # Count bad sums involving a
            # This requires iterating all bad sums?
            # Let's just pick the smallest element in the pair for now? 
            # Or the one with smallest value?
            
            # Let's try a slightly better heuristic:
            # Remove the element that is part of the most pairs in A (highest degree).
            # This is static.
            
            # Let's just remove the smallest element in the first pair of the smallest bad sum.
            # This is very simple and deterministic.
            
            # But maybe we can do better.
            # Let's count for each x in the set of elements in `pairs` (and their partners), 
            # how many bad sums they are in.
            
            pass
        
        # Let's implement a simple version:
        # Remove the smallest element from the set of elements involved in the smallest bad sum.
        elements_involved = set()
        for a, b in pairs:
            elements_involved.add(a)
            elements_involved.add(b)
            
        x_to_remove = min(elements_involved)
        
        # Remove x_to_remove from A
        A.remove(x_to_remove)
        A_set.remove(x_to_remove)
        
        # Update counts
        # Decrement counts for sums x_to_remove + y for all y in A
        for y in A:
            s = x_to_remove + y
            if s <= MAX_SUM:
                counts[s] -= 1
            else:
                break # A is sorted
        
        # Also, we removed the pair (x, x) if x+x was a sum?
        # The loop `for y in A` covers y=x? 
        # If x is in A, and we remove x, the pair (x,x) is gone.
        # The loop iterates over the *new* A (after removal).
        # So we need to handle the pair (x,x) separately?
        # Original count included (x,x).
        # New count should not include (x,x).
        # The loop `for y in A` (where x is not in A) will NOT decrement counts[x+x].
        # So we must manually decrement counts[2*x].
        
        counts[2 * x_to_remove] -= 1
        
        # Update bad_sums list
        # Remove s if counts[s] <= 2
        # Add new bad sums? No, counts only decrease, so no new bad sums.
        bad_sums = [s for s in bad_sums if counts[s] > 2]
        
        # Optimization: If the removed x was not involved in s, s might still be bad.
        # We just filter the list.
        
    return A

# EVOLVE-BLOCK-END
