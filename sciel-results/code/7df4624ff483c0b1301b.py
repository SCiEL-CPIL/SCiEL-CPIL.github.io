import random
import time
import itertools

def construct(q, n, radius):
    # Problem is fixed to q=3, n=7, radius=1 based on task description, 
    # but the interface passes these values. 
    # The logic below is generic for q, n, radius=1 but optimized for the specific case.
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    total_words = q ** n
    # Map word tuple to index and vice versa
    # For q=3, n=7, total_words=2187. 
    # Using integer representation for speed.
    
    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute balls (neighbors within radius 1)
    # balls[i] is a list of indices
    balls = [[] for _ in range(total_words)]
    
    # Generate all words
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    for i in range(total_words):
        w = all_words[i]
        balls[i].append(i) # distance 0
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = tuple_to_idx(new_w)
                balls[i].append(j)
                
    # Greedy Set Cover
    best_solution = None
    
    # We will run multiple greedy passes with randomization
    # and then minimize the result.
    
    def minimize_solution(solution):
        # solution is a list of indices
        # Remove redundant codewords
        # cover_count[w] = number of codewords in solution covering w
        cover_count = [0] * total_words
        for c in solution:
            for w in balls[c]:
                cover_count[w] += 1
                
        # Check for redundancy
        # A codeword c is redundant if for all w in balls[c], cover_count[w] > 1
        # If we remove c, words with cover_count 1 in balls[c] become uncovered.
        
        changed = True
        current_sol = list(solution)
        
        while changed:
            changed = False
            for c in list(current_sol):
                # Check if c is redundant
                redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    current_sol.remove(c)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    changed = True
        return current_sol

    def greedy_run(random_seed):
        random.seed(random_seed)
        uncovered = [True] * total_words
        gains = [len(balls[i]) for i in range(total_words)]
        selected = []
        
        remaining = total_words
        
        while remaining > 0:
            # Find max gain
            max_gain = -1
            best_idx = -1
            candidates = []
            
            # To add randomness in tie-breaking, we can collect all max gain indices
            # But scanning 2187 is fast.
            for i in range(total_words):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_idx = i
                    candidates = [i]
                elif gains[i] == max_gain:
                    candidates.append(i)
            
            # Pick one from candidates randomly
            if not candidates:
                break
            idx = random.choice(candidates)
            
            selected.append(idx)
            
            # Update gains and uncovered
            for w in balls[idx]:
                if uncovered[w]:
                    uncovered[w] = False
                    remaining -= 1
                    # For each codeword j that covers w, decrement gain
                    for j in balls[w]:
                        gains[j] -= 1
        
        return selected

    # Run several greedy instances
    num_runs = 50 
    run_count = 0
    
    while run_count < num_runs:
        if time.time() > deadline:
            break
        
        sol = greedy_run(run_count)
        sol = minimize_solution(sol)
        
        if best_solution is None or len(sol) < len(best_solution):
            best_solution = sol
            
        run_count += 1
        
    # If we have time, we can try a local search on the best solution
    # But the greedy + minimize is usually quite strong for this size.
    
    if best_solution is None:
        # Fallback to a trivial valid solution if something went wrong
        # e.g. all words
        return [idx_to_tuple(i) for i in range(total_words)]
        
    # Convert indices back to tuples
    return [idx_to_tuple(i) for i in best_solution]
