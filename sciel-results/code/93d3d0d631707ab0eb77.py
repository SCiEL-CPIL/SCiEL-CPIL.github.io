import random
import time

def construct():
        N = 1000
        max_sum = 2 * N
        counts = [0] * (max_sum + 1)
        S = set()
        
        # Greedy initialization
        # Try multiple random orders to find a good starting point
        best_S = set()
        best_size = 0
        
        start_time = time.time()
        time_limit = 110 # Leave 10s for cleanup/return
        
        # Phase 1: Find a good initial solution
        # We can run greedy a few times with different random seeds/orders
        # But standard greedy 1..1000 is deterministic.
        # Let's try greedy with random shuffle.
        
        for _ in range(50): # 50 restarts
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            S = set()
            counts = [0] * (max_sum + 1)
            
            for x in candidates:
                # Check if x can be added
                valid = True
                # Check 2x
                if counts[2 * x] >= 2:
                    valid = False
                else:
                    # Check x + y for y in S
                    # Optimization: iterate over S
                    for y in S:
                        if counts[x + y] >= 2:
                            valid = False
                            break
                    if not valid:
                        continue
                
                if valid:
                    # Add x
                    S.add(x)
                    counts[2 * x] += 1
                    for y in S:
                        if y != x: # y is already in S, x is new
                             # Wait, if we add x, we need to update counts for x+y
                             # The loop above checked counts[x+y] BEFORE adding x.
                             # Now we update.
                             pass 
                    # Correction: The check loop iterates over current S.
                    # When we add x, we must update counts for all pairs involving x.
                    # Pairs: (x,x) and (x,y) for y in S_old.
                    counts[2 * x] += 1
                    for y in S:
                         if y != x:
                             counts[x + y] += 1
                             
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                # Update best_counts? We need to recompute or store.
                # Storing counts is better.
                # Let's recompute counts for best_S if we update best_S?
                # Or just store the state.
                
        # Phase 2: Local Search / Simulated Annealing
        # Start from best_S
        S = set(best_S)
        counts = [0] * (max_sum + 1)
        for a in S:
            counts[2 * a] += 1
            for b in S:
                if a < b:
                    counts[a + b] += 1
        # Note: The above double loop is O(|S|^2). |S|~600, so 360k ops. Fine.
        
        # Helper to check validity of adding x to current S/counts
        def can_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                if counts[x + y] >= 2:
                    return False
            return True
            
        def add(x):
            S.add(x)
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
                    
        def remove(x):
            S.remove(x)
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1
                    
        # Local Search Loop
        while time.time() - start_time < time_limit:
            # Try to add a random element
            # Pick random x not in S
            # To pick random x not in S efficiently, we can maintain a list of availables
            # Or just pick random 1..N and check membership.
            
            x = random.randint(1, N)
            if x in S:
                continue
                
            if can_add(x):
                add(x)
            else:
                # Try to swap
                # Find conflicts
                conflicts = []
                if counts[2 * x] >= 2:
                    # The pairs summing to 2x are the conflicts
                    # We need to remove an element involved in a pair summing to 2x
                    # Pairs (a,b) in S such that a+b=2x
                    # Since a,b in S, we can find them by iterating S?
                    # Or just pick a random y in S and see if removing it helps?
                    # Simpler: Pick random y in S. Try swap.
                    pass
                
                # General swap strategy:
                # Pick a random y in S.
                # If S - {y} + {x} is valid, accept.
                # To check validity of S' = S - {y} + {x}:
                # 1. Remove y from counts.
                # 2. Check if x can be added to S - {y}.
                # 3. If yes, add x.
                # 4. If no, restore y.
                
                y = random.choice(list(S))
                
                # Temporarily remove y
                remove(y)
                
                if can_add(x):
                    add(x)
                    # Swap accepted
                else:
                    # Restore y
                    add(y) # Wait, add(y) adds to S and counts. 
                    # But S currently does NOT contain y. 
                    # So add(y) will put it back.
                    # Is this correct?
                    # remove(y) removed y from S and decremented counts.
                    # add(y) adds y to S and increments counts.
                    # Yes, this restores state.
        
        return list(S)
