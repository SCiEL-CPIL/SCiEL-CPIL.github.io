"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Ternary tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Construct initial code
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
            
    if q != 3:
        pass

    # Map words to integers for fast indexing
    def word_to_int(w):
        val = 0
        for i in range(n):
            val += w[i] * (3 ** i)
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n):
            w[i] = val % 3
            val //= 3
        return w

    size_space = 3 ** n
    
    # Precompute balls for ALL possible words (there are only 3^6 = 729)
    # ball_cache[word_int] = list of word_ints in the ball of radius 1 around word_int
    ball_cache = {}
    for w_int in range(size_space):
        w = int_to_word(w_int)
        ball = [w_int]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_word = w[:]
                    new_word[pos] = val
                    ball.append(word_to_int(new_word))
        ball_cache[w_int] = ball
    
    def greedy_remove(code_list, coverage_arr, time_limit):
        """
        Greedy removal: repeatedly find the codeword whose ball has the highest
        minimum coverage (most redundant) and remove it if all coverages > 1.
        Returns (code_list, coverage_arr, num_removed).
        """
        code_ints = [word_to_int(cw) for cw in code_list]
        # Map from code_int to index in code_ints
        # We'll track active codewords
        
        active = list(range(len(code_ints)))
        # For each active codeword, track its ball's minimum coverage
        # We want to remove the one with max min-coverage (most redundant)
        
        # Precompute: for each codeword, its ball
        cw_balls = [ball_cache[code_ints[i]] for i in range(len(code_ints))]
        
        removed_count = 0
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            best_idx = -1
            best_min_cov = 0
            
            # Find the codeword with the highest minimum coverage in its ball
            for i in active:
                ball = cw_balls[i]
                min_cov = min(coverage_arr[w] for w in ball)
                if min_cov > best_min_cov:
                    best_min_cov = min_cov
                    best_idx = i
            
            if best_idx == -1 or best_min_cov <= 1:
                break
            
            # Remove codeword best_idx
            for w in cw_balls[best_idx]:
                coverage_arr[w] -= 1
            active.remove(best_idx)
            removed_count += 1
            changed = True
        
        # Rebuild code_list from active indices
        new_code = [code_list[i] for i in active]
        return new_code, coverage_arr, removed_count
    
    def try_random_removal(code_list, coverage_arr, time_limit):
        """
        Try random removals: pick random codewords and remove if possible.
        This can escape local optima of the greedy approach.
        """
        code_ints = [word_to_int(cw) for cw in code_list]
        cw_balls = [ball_cache[code_ints[i]] for i in range(len(code_ints))]
        active = list(range(len(code_ints)))
        
        removed_count = 0
        # Try random removals for a limited time
        attempts = 0
        max_attempts = 5000
        while time.time() < time_limit and attempts < max_attempts:
            attempts += 1
            if not active:
                break
            idx = random.randint(0, len(active) - 1)
            i = active[idx]
            ball = cw_balls[i]
            # Check if removable
            removable = True
            for w in ball:
                if coverage_arr[w] <= 1:
                    removable = False
                    break
            if removable:
                for w in ball:
                    coverage_arr[w] -= 1
                active.remove(idx)
                removed_count += 1
        
        new_code = [code_list[i] for i in active]
        return new_code, coverage_arr, removed_count
    
    # Main optimization loop with multiple strategies
    best_code = [list(c) for c in initial_code]
    best_size = len(best_code)
    
    # Strategy 1: Pure greedy removal
    code = [list(c) for c in initial_code]
    coverage = [0] * size_space
    for cw in code:
        ball = ball_cache[word_to_int(cw)]
        for w in ball:
            coverage[w] += 1
    
    time_limit_1 = deadline - 30.0
    code, coverage, _ = greedy_remove(code, coverage, time_limit_1)
    if len(code) < best_size:
        best_code = code
        best_size = len(code)
    
    # Strategy 2: Greedy + random restarts
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 10:
        num_restarts = min(5, int(remaining_time // 5))
        for restart in range(num_restarts):
            if time.time() > remaining_time + time_limit_1:
                break
            
            # Start from best_code found so far
            code = [list(c) for c in best_code]
            coverage = [0] * size_space
            for cw in code:
                ball = ball_cache[word_to_int(cw)]
                for w in ball:
                    coverage[w] += 1
            
            # Do some random removals first to perturb
            rand_time = min(2.0, remaining_time * 0.2)
            code, coverage, _ = try_random_removal(code, coverage, time.time() + rand_time)
            
            # Then greedy removal
            greedy_time = min(deadline - 5.0, time.time() + (remaining_time * 0.3))
            code, coverage, _ = greedy_remove(code, coverage, greedy_time)
            
            if len(code) < best_size:
                best_code = code
                best_size = len(code)
            
            remaining_time = deadline - 5.0 - time.time()
            if remaining_time < 5:
                break
    
    # Final: try one more greedy pass on best_code
    if time.time() < deadline - 2.0:
        code = [list(c) for c in best_code]
        coverage = [0] * size_space
        for cw in code:
            ball = ball_cache[word_to_int(cw)]
            for w in ball:
                coverage[w] += 1
        
        code, coverage, _ = greedy_remove(code, coverage, deadline - 1.0)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    return best_code

# EVOLVE-BLOCK-END
