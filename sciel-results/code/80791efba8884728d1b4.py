# EVOLVE-BLOCK-START
def construct():
    import time
    import random

    N = 256
    deadline = time.time() + 25.0

    # Greedy Set Cover Heuristic
    # S: selected elements
    # used_sums: set of sums a+b for a,b in S, a<=b
    # unblocked: set of elements not in S and not blocked
    
    def greedy_construct():
        S = []
        used_sums = set()
        unblocked = set(range(1, N + 1))
        
        while unblocked:
            if time.time() > deadline:
                break
            
            # If no unblocked elements, we are done (maximal)
            # Actually, the loop condition handles this.
            
            best_y = None
            max_blocked = -1
            best_blocked_set = None
            
            # Candidates are all unblocked elements
            # We need to pick the one that blocks the most other unblocked elements
            
            # Optimization: Precompute some properties of S?
            # For now, iterate through candidates
            
            candidates = list(unblocked)
            
            for y in candidates:
                # Calculate which unblocked elements x would be blocked if we add y
                # Blocked conditions for x (x != y, x in unblocked):
                # 1. 2x = y + a for some a in S
                # 2. x + y in used_sums
                # 3. x = 2y - a for some a in S
                # 4. x = y + a' - a for some a, a' in S
                
                blocked_x = set()
                
                # Precompute sets for faster lookup?
                # Let's just iterate S
                
                # Condition 1: 2x = y + a => x = (y + a) / 2
                # Condition 3: x = 2y - a
                # Condition 2: x = s - y for s in used_sums
                # Condition 4: x = y + a' - a
                
                # To speed up, we can generate the set of blocked x directly
                # and intersect with unblocked.
                
                new_blocked = set()
                
                # Cond 1: x = (y + a) / 2
                for a in S:
                    val = y + a
                    if val % 2 == 0:
                        x = val // 2
                        if x in unblocked and x != y:
                            new_blocked.add(x)
                
                # Cond 3: x = 2y - a
                for a in S:
                    x = 2 * y - a
                    if x in unblocked and x != y:
                        new_blocked.add(x)
                            
                # Cond 2: x + y in used_sums => x = s - y
                for s in used_sums:
                    x = s - y
                    if x in unblocked and x != y:
                        new_blocked.add(x)
                        
                # Cond 4: x = y + a' - a
                # This is x - y = a' - a
                # Let D = {a' - a | a, a' in S}
                # Then x = y + d for d in D
                # We can precompute D? D changes as S grows.
                # For now, iterate.
                # Optimization: If S is small, this is fine.
                
                for a in S:
                    for a_prime in S:
                        x = y + a_prime - a
                        if x in unblocked and x != y:
                            new_blocked.add(x)
                
                # The number of blocked elements is len(new_blocked)
                # Note: new_blocked might contain elements not in unblocked? 
                # We checked `if x in unblocked` so it's a subset of unblocked.
                
                if len(new_blocked) > max_blocked:
                    max_blocked = len(new_blocked)
                    best_y = y
                    best_blocked_set = new_blocked
                    
            if best_y is None:
                # No candidates? Should not happen if unblocked is not empty.
                break
                
            # Add best_y to S
            S.append(best_y)
            unblocked.remove(best_y)
            
            # Update used_sums
            used_sums.add(2 * best_y)
            for a in S:
                if a != best_y:
                    used_sums.add(best_y + a)
                    
            # Remove blocked elements from unblocked
            unblocked -= best_blocked_set
            
        return S

    # Run greedy
    best_set = greedy_construct()
    
    # If time remains, try to improve with local search?
    # Or just run multiple random restarts with different tie-breaking?
    # The greedy is deterministic. 
    # To get better results, we can randomize the order of candidates or use a different scoring.
    # But greedy max-block is usually strong.
    
    # Let's try a few variations with random tie-breaking or slight scoring noise
    # to see if we can find a smaller set.
    
    best_size = len(best_set)
    
    # Simple local search: try to swap elements?
    # Or just restart with different random seeds for tie-breaking?
    
    # Given the time limit, one greedy run is likely sufficient to beat 192.
    # Theoretical size is ~20-30.
    
    return best_set
# EVOLVE-BLOCK-END
