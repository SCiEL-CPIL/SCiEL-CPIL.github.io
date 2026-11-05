import random
import time

def construct():
    start_time = time.time()
    time_limit = 115
    
    best_set = []
    best_size = 0
    
    def prune_set(initial_set):
        # initial_set is a set of ints
        S = set(initial_set)
        # Calculate initial sum counts
        # sum_counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
        sum_counts = {}
        
        # Efficient initialization
        # Iterate over all pairs
        # To speed up, we can use the fact that we start with 1..1000
        # But for generality (if we restart with random subset), we just loop.
        # 1000 elements -> 500k pairs.
        for i, a in enumerate(S):
            for b in S:
                if a > b: continue
                s = a + b
                sum_counts[s] = sum_counts.get(s, 0) + 1
        
        # Pruning loop
        # We need a way to find "bad" sums quickly.
        # Maintain a list of bad sums? Or just scan?
        # Scanning 2000 sums is cheap.
        
        while True:
            # Find a bad sum
            bad_sum = None
            max_count = 2
            # Randomize selection among bad sums?
            # Let's collect all bad sums
            bad_sums = []
            for s, c in sum_counts.items():
                if c > 2:
                    bad_sums.append(s)
            
            if not bad_sums:
                break
            
            # Pick a random bad sum
            bad_sum = random.choice(bad_sums)
            
            # Find all pairs in S that sum to bad_sum
            # We need to identify which elements to remove.
            # We want to remove an element x that is part of the most "bad" sums.
            # Let's compute a score for each element in the pairs of bad_sum.
            
            pairs = []
            # Iterate through S to find pairs summing to bad_sum
            # This is O(N).
            # Since we do this for each removal, and we have ~1000 removals, total O(N^2) = 10^6. OK.
            for a in S:
                b = bad_sum - a
                if b in S and a <= b:
                    pairs.append((a, b))
            
            # If no pairs found? Should not happen if sum_counts is correct.
            if not pairs:
                # Error in logic or state, break to avoid infinite loop
                break
            
            # Calculate score for each element involved
            # Score = number of bad sums this element is part of.
            # We can approximate this by checking the counts of sums involving x.
            # But we don't have a direct "element to bad sums" map.
            # Simpler heuristic: Pick a random pair from the pairs of the current bad_sum,
            # and remove one of its elements randomly.
            # Or, pick the element that appears in the most pairs of *all* bad sums?
            # That requires more work.
            
            # Let's try: Pick a random pair (a,b) from pairs.
            # Then randomly pick a or b to remove.
            # This is simple and effective.
            
            a, b = random.choice(pairs)
            x_to_remove = random.choice([a, b])
            
            # Remove x_to_remove from S
            S.remove(x_to_remove)
            
            # Update sum_counts
            # Decrement counts for all sums involving x_to_remove
            for y in S:
                s = x_to_remove + y
                if s in sum_counts:
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
            
            # Also handle 2*x_to_remove
            s_double = 2 * x_to_remove
            if s_double in sum_counts:
                sum_counts[s_double] -= 1
                if sum_counts[s_double] == 0:
                    del sum_counts[s_double]
            
            # Note: The loop `for y in S` above iterates over the NEW S (x removed).
            # The pairs were (x, y) for y in old S, y != x.
            # In new S, y is still there.
            # So we correctly decrement x+y.
            # What about the pair (x, x)? Handled by s_double.
            
        return S

    # Run multiple iterations
    while time.time() - start_time < time_limit:
        # Start with full set 1..1000
        initial = set(range(1, 1001))
        result = prune_set(initial)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
            
        # To diversify, we can start with a slightly perturbed set?
        # Or just rely on random choices inside prune_set.
        # The random choices in prune_set (choice of bad_sum, choice of pair, choice of element)
        # should provide enough diversity.
        
    return best_set
