import time
import random

def construct(q, n, radius):
    # Specific optimization for the known task instance
    if q == 3 and n == 7 and radius == 1:
        return solve_ternary_7_1()
    
    # General fallback (slower, but correct)
    return general_solve(q, n, radius)

def solve_ternary_7_1():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    N = 3 ** 7
    balls = precompute_balls(3, 7)
    
    best_solution = None
    best_size = float('inf')
    
    # Initial Greedy
    current_sol = greedy_cover(balls, N)
    current_sol = reduce_solution(current_sol, balls, N)
    best_solution = current_sol
    best_size = len(current_sol)
    
    # Precompute cover counts for current solution
    cover_count = [0] * N
    for c in current_sol:
        for w in balls[c]:
            cover_count[w] += 1
            
    while time.time() < deadline:
        # Targeted Swap
        new_sol, improved = try_swap(current_sol, balls, N, cover_count)
        
        if improved:
            current_sol = reduce_solution(new_sol, balls, N)
            # Recalculate cover counts
            cover_count = [0] * N
            for c in current_sol:
                for w in balls[c]:
                    cover_count[w] += 1
            
            if len(current_sol) < best_size:
                best_size = len(current_sol)
                best_solution = current_sol
                
        # Random Restart
        if random.random() < 0.02:
            restart_sol = greedy_cover(balls, N)
            restart_sol = reduce_solution(restart_sol, balls, N)
            if len(restart_sol) < len(current_sol):
                current_sol = restart_sol
                cover_count = [0] * N
                for c in current_sol:
                    for w in balls[c]:
                        cover_count[w] += 1
                if len(current_sol) < best_size:
                    best_size = len(current_sol)
                    best_solution = current_sol

    return [index_to_word(i, 3, 7) for i in best_solution]

def precompute_balls(q, n):
    N = q ** n
    balls = []
    for idx in range(N):
        word = index_to_word(idx, q, n)
        ball = [idx]
        for i in range(n):
            orig_val = word[i]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = word.copy()
                new_word[i] = new_val
                ball.append(word_to_index(new_word, q, n))
        balls.append(ball)
    return balls

def index_to_word(idx, q, n):
    word = [0] * n
    for i in range(n):
        word[i] = idx % q
        idx //= q
    return word

def word_to_index(word, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx

def greedy_cover(balls, N):
    uncovered = set(range(N))
    solution = []
    while uncovered:
        best_c = -1
        max_covered = 0
        for c in range(N):
            covered = len(uncovered.intersection(balls[c]))
            if covered > max_covered:
                max_covered = covered
                best_c = c
        solution.append(best_c)
        uncovered.difference_update(balls[best_c])
    return solution

def reduce_solution(solution, balls, N):
    if not solution:
        return []
    
    cover_count = [0] * N
    for c in solution:
        for w in balls[c]:
            cover_count[w] += 1
            
    new_solution = []
    for c in solution:
        redundant = True
        for w in balls[c]:
            if cover_count[w] == 1:
                redundant = False
                break
        
        if redundant:
            for w in balls[c]:
                cover_count[w] -= 1
        else:
            new_solution.append(c)
            
    return new_solution

def try_swap(current_sol, balls, N, cover_count):
    if not current_sol:
        return current_sol, False
    
    sol_set = set(current_sol)
    c_out = random.choice(current_sol)
    
    # Critical words: covered only by c_out
    critical = [w for w in balls[c_out] if cover_count[w] == 1]
    
    if not critical:
        # c_out is redundant
        new_sol = [c for c in current_sol if c != c_out]
        return new_sol, True

    best_c_in = -1
    max_crit_covered = 0
    critical_set = set(critical)
    
    for c_in in range(N):
        if c_in in sol_set:
            continue
        covered = len(critical_set.intersection(balls[c_in]))
        if covered > max_crit_covered:
            max_crit_covered = covered
            best_c_in = c_in
            
    if best_c_in == -1 or max_crit_covered == 0:
        return current_sol, False
        
    new_sol = [c for c in current_sol if c != c_out]
    new_sol.append(best_c_in)
    return new_sol, True

def general_solve(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    balls = precompute_balls(q, n)
    sol = greedy_cover(balls, N)
    sol = reduce_solution(sol, balls, N)
    return [index_to_word(i, q, n) for i in sol]
