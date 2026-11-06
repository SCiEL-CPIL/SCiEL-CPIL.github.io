"""Rewritten program for Large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 4096
    best_set = []
    best_size = 0
    
    # Helper to check if x can be added to S
    # S: list of elements
    # T: set of triple sums
    # P: set of pair sums (a+b for a<=b in S)
    def can_add(x, S, T, P):
        # 1. Check 3x
        if 3*x in T:
            return False
        
        # 2. Check 2x + a
        # Also need to check if 2x is in P (collision 3x = x+a+b -> 2x=a+b)
        if 2*x in P:
            return False
            
        for a in S:
            val = 2*x + a
            if val in T:
                return False
            
            # Check collision between new 2x+a and new x+c+d
            # 2x+a = x+c+d  => x+a = c+d
            # So check if x+a is in P
            if x + a in P:
                return False
        
        # 3. Check x + a + b
        # Also need to check if new pair sums x+a are in P (handled above)
        # And if new pair sums x+a collide with each other? x+a = x+b => a=b (ok)
        # And if new pair sums x+a collide with existing P? (handled above)
        
        # Iterate over pairs a <= b in S
        for i in range(len(S)):
            a = S[i]
            for j in range(i, len(S)):
                b = S[j]
                val = x + a + b
                if val in T:
                    return False
        
        return True

    def build_greedy(seed):
        random.seed(seed)
        S = []
        T = set()
        P = set()
        
        # Candidate pool
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, T, P):
                S.append(x)
                # Update T
                # New triples involving x:
                # (x, x, x)
                T.add(3*x)
                # (x, x, a)
                for a in S[:-1]: # S[:-1] is S before adding x
                    T.add(2*x + a)
                    # Update P for (x, a)
                    P.add(x + a)
                # (x, a, b)
                for i in range(len(S)-1):
                    a = S[i]
                    for j in range(i, len(S)-1):
                        b = S[j]
                        T.add(x + a + b)
                        # Note: P already updated for x+a, x+b?
                        # P needs to contain all pair sums.
                        # We added x+a in the loop above.
                        # We need to add x+b?
                        # The loop `for a in S[:-1]` adds x+a for all a in old S.
                        # So all pairs (x, a) are in P.
                        # Pairs (a, b) were already in P.
                
                # Optimization: If time is running out, break
                if time.time() > deadline - 5:
                    break
            
            # Optional: If we get stuck, we could backtrack, but simple greedy 
            # with random restarts is usually sufficient for this size.
            
        return S

    # Run multiple restarts
    num_restarts = 200
    seed = 42
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        current_set = build_greedy(seed + i)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we reach a high cardinality, we might stop early or try to improve
        # For now, just keep searching.
        
    return best_set
# EVOLVE-BLOCK-END
