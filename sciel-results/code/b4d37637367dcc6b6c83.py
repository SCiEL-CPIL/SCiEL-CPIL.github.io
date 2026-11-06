import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    N = 1000
    best_set = []
    best_size = 0
    
    def run_greedy_smart(random_tie_break=True):
        """
        Greedy with heuristic: pick valid candidate that minimizes 'tightness'.
        If tie, pick randomly if random_tie_break is True, else pick smallest.
        """
        chosen = []
        counts = [0] * (2 * N + 1)
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_score = float('inf')
            candidates_with_best_score = []
            
            for x in remaining:
                # Check validity
                valid = True
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                if valid:
                    if counts[2 * x] >= 2:
                        valid = False
                
                if valid:
                    # Calculate score: number of sums that would become count=2
                    score = 0
                    for y in chosen:
                        s = x + y
                        if counts[s] == 1:
                            score += 1
                    if counts[2 * x] == 1:
                        score += 1
                    
                    if score < best_score:
                        best_score = score
                        candidates_with_best_score = [x]
                    elif score == best_score:
                        candidates_with_best_score.append(x)
            
            if not candidates_with_best_score:
                break
            
            if random_tie_break and len(candidates_with_best_score) > 1:
                x = random.choice(candidates_with_best_score)
            else:
                x = candidates_with_best_score[0]
            
            # Remove x from remaining
            # Since remaining is a list, removal is O(N).
            # We can use a set for remaining for O(1) removal, but iteration order doesn't matter for smart greedy.
            # However, we iterate over remaining to find candidates.
            # Let's use a set for remaining to make removal fast, but we need to iterate.
            # Actually, iterating over a set is fine.
            
            remaining.remove(x)
            chosen.append(x)
            for y in chosen:
                if y == x:
                    continue
                s = x + y
                counts[s] += 1
            counts[2 * x] += 1
            
        return chosen

    def run_greedy_random():
        """
        Greedy with random order.
        """
        chosen = []
        counts = [0] * (2 * N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if valid:
                if counts[2 * x] >= 2:
                    valid = False
            
            if valid:
                chosen.append(x)
                for y in chosen:
                    if y == x:
                        continue
                    s = x + y
                    counts[s] += 1
                counts[2 * x] += 1
                
        return chosen

    # 1. Try Smart Greedy with random tie-breaking (20 times)
    for i in range(20):
        if time.time() > deadline - 10:
            break
        s = run_greedy_smart(random_tie_break=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 2. Try Random Greedy Restarts
    attempts = 0
    while time.time() < deadline:
        s = run_greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        attempts += 1
        
        # Check time every 10 attempts
        if attempts % 10 == 0 and time.time() > deadline - 2:
            break

    return best_set
