import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    total_words = q ** n
    
    # If problem is large, fall back to heuristic
    # For q=3, n=6, total_words=729.
    
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls
    balls = []
    for i, w in enumerate(words):
        ball = {i}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_w_list = list(w)
                        new_w_list[pos] = val
                        new_w = tuple(new_w_list)
                        ball.add(word_to_idx[new_w])
        balls.append(ball)
        
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = set(range(total_words))
        selected = []
        available = list(range(total_words))
        rng.shuffle(available)
        
        while uncovered:
            best_score = -1
            best_c = None
            best_ball = None
            
            for c in available:
                common = len(balls[c] & uncovered)
                if common > best_score:
                    best_score = common
                    best_c = c
                    best_ball = balls[c]
            
            if best_c is None:
                break
                
            selected.append(best_c)
            uncovered -= best_ball
            available.remove(best_c)
            
        return selected

    def local_search_reduce(selected):
        selected_set = set(selected)
        
        # Initialize coverage counts
        cover_count = [0] * total_words
        for c in selected_set:
            for w in balls[c]:
                cover_count[w] += 1
                
        changed = True
        while changed:
            changed = False
            for c in list(selected_set):
                # Check redundancy
                redundant = True
                # Decrement counts
                for w in balls[c]:
                    cover_count[w] -= 1
                    if cover_count[w] == 0:
                        redundant = False
                        break
                
                if not redundant:
                    # Restore counts
                    for w in balls[c]:
                        cover_count[w] += 1
                else:
                    # c is redundant, remove it
                    selected_set.remove(c)
                    changed = True
                    break # Restart loop
                    
        return list(selected_set)

    best_solution = None
    best_size = total_words
    
    # Initial greedy run
    sol = run_greedy(0)
    sol = local_search_reduce(sol)
    if len(sol) < best_size:
        best_size = len(sol)
        best_solution = sol
        
    # Randomized restarts
    seed = 1
    while time.time() < deadline:
        sol = run_greedy(seed)
        sol = local_search_reduce(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
        seed += 1
        
    if best_solution is None:
        # Fallback
        if q == 3 and n == 6:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # General fallback: greedy without local search
            sol = run_greedy(0)
            return [words[i] for i in sol]
            
    return [words[i] for i in best_solution]
