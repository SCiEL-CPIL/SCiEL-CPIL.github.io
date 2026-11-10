import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    def is_valid_addition(S, x, current_size):
        """Check if adding x to current set maintains the property."""
        # S[k] is the set of all subset sums of cardinality k
        # After adding x, new sums for cardinality k+1 are {s+x for s in S[k]}
        # We need to check that new S[k+1] doesn't intersect with any existing S[j] for j != k+1
        # Also need to check that existing S[k+1] (if any) union new sums doesn't intersect others
        
        # Compute new sums for each cardinality level
        # New cardinality k+1 sums: old S[k+1] ∪ {s+x for s in S[k]}
        # For all other levels j, S[j] stays the same
        
        # Check: for all i, j with i != j, new_S[i] ∩ new_S[j] = empty
        # Only S[k+1] changes (grows), so we only need to check new additions against all other levels
        
        new_sums = set(s + x for s in S[k])
        # The full new S[k+1] would be S[k+1] ∪ new_sums
        # But we only need to check that new_sums doesn't intersect with S[j] for j != k+1
        # AND that new_sums doesn't intersect with... wait, S[k+1] already exists and is fine with itself
        
        # Actually, the new elements are new_sums. They go into S[k+1].
        # We need: new_sums ∩ S[j] = empty for all j != k+1
        # Also: new_sums ∩ new_sums is fine (it's a set, so no duplicates)
        # And S[k+1] ∩ S[j] was already empty for j != k+1, so only new_sums needs checking
        
        for j in range(len(S)):
            if j == k + 1:
                continue
            if new_sums & S[j]:
                return False
        return True
    
    def try_greedy(candidates):
        """Try greedy addition with given candidate order."""
        # S[k] = set of all subset sums using exactly k elements from chosen set
        S = [set() for _ in range(61)]
        S[0].add(0)
        chosen = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            current_size = len(chosen)
            
            # Check if we can add x
            if current_size + 1 > 60:
                break
            
            # New sums for cardinality current_size+1 when adding x
            new_sums = set(s + x for s in S[current_size])
            
            # Check against all other cardinality levels
            valid = True
            for j in range(len(S)):
                if j == current_size + 1:
                    continue
                if new_sums & S[j]:
                    valid = False
                    break
            
            if valid:
                # Add x
                chosen.append(x)
                # Update S: new cardinality level is current_size+1
                # S[current_size+1] |= new_sums
                S[current_size + 1] = S[current_size + 1] | new_sums
        
        return chosen
    
    # Strategy: try different orderings of candidates
    # Key insight: larger numbers create larger sums, which may help avoid collisions
    # Try descending order first, then various heuristics
    
    # Generate candidate pool: all integers 1..256
    all_candidates = list(range(1, 257))
    
    best_result = []
    
    # Try descending order (large numbers first)
    result = try_greedy(sorted(all_candidates, reverse=True))
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return [int(x) for x in best_result]
    
    # Try ascending order
    result = try_greedy(sorted(all_candidates))
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return [int(x) for x in best_result]
    
    # Try a mixed strategy: alternate large and small
    # Or try powers of 2 based, then fill in
    # Let's try: start with large gaps, then fill
    
    # Try: 256, 255, 254, ... descending but skip some to create gaps
    # Actually, let's try a smarter approach:
    # Use the fact that if we have numbers a1 < a2 < ... < an, 
    # the subset sums of cardinality k range from sum of smallest k to sum of largest k
    
    # Try random restarts with different orderings
    import random
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Create a shuffled order with bias toward certain ranges
        candidates = all_candidates[:]
        rng.shuffle(candidates)
        result = try_greedy(candidates)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > deadline:
        return [int(x) for x in best_result]
    
    # Try more structured approaches
    # Approach: use numbers that are far apart to minimize collisions
    # Try: 1, 2, 4, 8, ... (powers of 2) then fill in
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
    remaining = [x for x in all_candidates if x not in powers_of_2]
    candidates = powers_of_2 + remaining
    result = try_greedy(candidates)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return [int(x) for x in best_result]
    
    # Try: large numbers spaced out
    # 256, 200, 150, 100, 50, ...
    spaced = list(range(256, 0, -5))
    result = try_greedy(spaced)
    if len(result) > len(best_result):
        best_result = result
    
    # Final: return best result found
    return [int(x) for x in best_result]
