import random
import time

def construct():
    start_time = time.time()
    time_limit = 115
    
    N = 1000
    MAX_SUM = 2 * N
    
    def get_score(x, current_set, sum_counts):
        # Returns (is_valid, score)
        # Score: number of sums x+y that are currently count 1
        # We want to minimize this to keep sums "open"
        valid = True
        score = 0
        
        for y in current_set:
            s = x + y
            if sum_counts[s] == 2:
                valid = False
                break
            elif sum_counts[s] == 1:
                score += 1
        
        if not valid:
            return (False, 0)
            
        s_double = 2 * x
        if sum_counts[s_double] == 2:
            valid = False
        elif sum_counts[s_double] == 1:
            score += 1
            
        return (valid, score)

    def run_scored_greedy(strategy):
        sum_counts = [0] * (MAX_SUM + 1)
        current_set = []
        in_set = [False] * (N + 1)
        
        while True:
            best_candidate = -1
            best_score = float('inf')
            candidates_with_best_score = []
            
            # Iterate through all candidates
            for x in range(1, N + 1):
                if in_set[x]:
                    continue
                
                valid, score = get_score(x, current_set, sum_counts)
                
                if not valid:
                    continue
                
                if score < best_score:
                    best_score = score
                    candidates_with_best_score = [x]
                elif score == best_score:
                    candidates_with_best_score.append(x)

            if not candidates_with_best_score:
                break
            
            # Select based on strategy
            if strategy == "min_x":
                x = min(candidates_with_best_score)
            elif strategy == "max_x":
                x = max(candidates_with_best_score)
            elif strategy == "random":
                x = random.choice(candidates_with_best_score)
            else:
                x = candidates_with_best_score[0]

            # Add x
            current_set.append(x)
            in_set[x] = True
            
            # Update sums
            for y in current_set:
                s = x + y
                sum_counts[s] += 1
            
            s_double = 2 * x
            sum_counts[s_double] += 1
            
        return current_set

    best_set = []
    best_size = 0
    
    # Run multiple strategies
    strategies = ["min_x", "max_x", "random"]
    
    for strat in strategies:
        if time.time() - start_time > time_limit:
            break
        res = run_scored_greedy(strat)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    # Run random strategy multiple times to explore different local optima
    while time.time() - start_time < time_limit:
        res = run_scored_greedy("random")
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    return best_set
