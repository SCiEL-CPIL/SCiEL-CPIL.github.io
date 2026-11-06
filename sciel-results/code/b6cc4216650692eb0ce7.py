import random
import time
import math

N = 512

def construct():
    # Deadline guard: 85 seconds to be safe against 90s limit
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Greedy Initialization
    # Replicates the seed program's logic to get a valid starting set
    S = set()
    for x in range(1, N + 1):
        valid = True
        # Check if adding x creates a 4-AP
        # x can be 1st, 2nd, 3rd, or 4th term
        # 1st: x, x+d, x+2d, x+3d. d >= 1, x+3d <= N
        # 2nd: x-d, x, x+d, x+2d. d >= 1, x-d >= 1, x+2d <= N
        # 3rd: x-2d, x-d, x, x+d. d >= 1, x-2d >= 1, x+d <= N
        # 4th: x-3d, x-2d, x-d, x. d >= 1, x-3d >= 1
        
        # Max d for 1st: (N-x)//3
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                valid = False
                break
        if not valid:
            continue
            
        # Max d for 2nd: min((x-1)//1, (N-x)//2)
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                valid = False
                break
        if not valid:
            continue
            
        # Max d for 3rd: min((x-1)//2, (N-x)//1)
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                valid = False
                break
        if not valid:
            continue
            
        # Max d for 4th: (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - d) in S and (x - 2*d) in S and (x - 3*d) in S:
                valid = False
                break
        if not valid:
            continue
        
        if valid:
            S.add(x)
    
    best_set = set(S)
    best_size = len(S)
    
    # 2. Simulated Annealing
    # Parameters
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling = 0.9995
    kick_prob = 0.3
    add_prob = 0.6
    remove_prob = 0.1 # 1 - kick - add
    
    all_elements = list(range(1, N + 1))
    
    # Helper to check if adding x to current set S is valid
    def can_add(x, S):
        # 1st term
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        # 2nd term
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return False
        # 3rd term
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return False
        # 4th term
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - d) in S and (x - 2*d) in S and (x - 3*d) in S:
                return False
        return True

    while time.time() < deadline:
        if T < T_min:
            # Restart with higher temperature if stuck
            T = T_min * 2
            # Optional: small perturbation
            if len(S) > 0:
                for _ in range(5):
                    if S:
                        rem = random.choice(list(S))
                        S.remove(rem)
        
        r = random.random()
        
        if r < kick_prob:
            # Kick: remove k random elements
            k = random.randint(2, 5)
            if len(S) >= k:
                items_to_remove = random.sample(list(S), k)
                new_S = set(S)
                for item in items_to_remove:
                    new_S.discard(item)
                # Always accept kick? Or accept based on size drop?
                # Kicks are for exploration. We accept them to move out of local optima.
                # But we should only keep them if we can potentially add more later.
                # For simplicity in SA, we treat this as a move with delta = -k.
                delta = -k
                if delta >= 0 or random.random() < math.exp(delta / T):
                    S = new_S
        elif r < kick_prob + add_prob:
            # Try to add a random element
            # Pick a random element not in S
            # Optimization: pick from a list of candidates? 
            # For now, pick random from all_elements and check if in S
            # This is slow if S is large. 
            # Better: maintain a list of excluded elements? 
            # Given N=512, picking random and checking membership is O(1) avg.
            attempts = 0
            while attempts < 10:
                x = random.choice(all_elements)
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        delta = 1
                        # Always accept improvement
                        break
                    else:
                        # Could reject or try another
                        pass
                attempts += 1
            # If we added, delta is 1. If not, delta is 0.
            # We don't explicitly track delta for acceptance here because 
            # we only add if valid (improvement or neutral if we allowed it, but we only add if valid which is always size +1).
            # So this branch only increases size.
            
        else:
            # Remove a single random element
            if S:
                x = random.choice(list(S))
                S.remove(x)
                # Delta is -1. Acceptance logic is implicit in the next steps?
                # Actually, standard SA updates state and accepts/rejects.
                # Here we modified S directly. We should revert if rejected.
                # Let's restructure to propose a move, then accept/revert.
                
    # The loop above is a bit messy with direct modification. 
    # Let's refine the SA loop to be cleaner: Propose -> Evaluate -> Accept/Reject
    
    # Reset for cleaner SA implementation
    S = set(best_set) # Start from best found so far (greedy)
    
    T = 10.0
    T_min = 0.5
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        r = random.random()
        new_S = set(S)
        
        if r < 0.4:
            # Kick: Remove 2-4 elements
            k = random.randint(2, 4)
            if len(S) >= k:
                items = random.sample(list(S), k)
                for item in items:
                    new_S.discard(item)
                delta = -k
            else:
                delta = 0
        elif r < 0.9:
            # Add
            # Find a candidate to add
            # To speed up, we can try a few random candidates
            added = False
            for _ in range(5):
                x = random.choice(all_elements)
                if x not in S:
                    if can_add(x, S):
                        new_S.add(x)
                        delta = 1
                        added = True
                        break
            if not added:
                delta = 0
        else:
            # Remove single
            if S:
                x = random.choice(list(S))
                new_S.discard(x)
                delta = -1
            else:
                delta = 0
                
        # Acceptance
        if delta > 0:
            S = new_S
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        elif delta == 0:
            # Accept neutral moves to allow random walk
            if random.random() < 0.5:
                S = new_S
        else:
            # delta < 0
            if random.random() < math.exp(delta / T):
                S = new_S
                
        T *= cooling
        
    return sorted(best_set)
