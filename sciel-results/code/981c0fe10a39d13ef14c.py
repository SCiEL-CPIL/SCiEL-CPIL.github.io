import itertools
import random
import time

def construct(q, n, radius):
    # This task is specifically for q=3, n=6, radius=1
    # But we should handle general inputs if possible, or at least not crash.
    # Given the specific task contract, we assume q=3, n=6, radius=1.
    # If parameters differ, we might fall back to a simple construction.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy or direct product if possible
        # For now, just return a trivial valid set (all words) to ensure validity, 
        # though it will score poorly.
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific solution for K_3(6,1)
    
    # Precompute balls
    # Word is an integer 0..728. Convert to base-3 string of length 6 for neighbor generation.
    # Or use arithmetic.
    # Let's use a helper to get neighbors.
    
    def word_to_list(w):
        res = [0]*n
        for i in range(n):
            res[i] = w % q
            w //= q
        return res
    
    def list_to_word(lst):
        w = 0
        for i in range(n-1, -1, -1):
            w = w * q + lst[i]
        return w

    # Precompute all balls
    balls = []
    for w in range(q**n):
        lst = word_to_list(w)
        ball = {w}
        for i in range(n):
            orig = lst[i]
            for v in range(q):
                if v == orig: continue
                lst[i] = v
                ball.add(list_to_word(lst))
            lst[i] = orig
        balls.append(ball)
        
    # Greedy Initialization
    coverage_count = [0] * (q**n)
    codewords = set()
    
    # Helper to update coverage
    def add_codeword(c):
        codewords.add(c)
        for v in balls[c]:
            coverage_count[v] += 1
            
    def remove_codeword(c):
        codewords.remove(c)
        for v in balls[c]:
            coverage_count[v] -= 1
            
    def is_valid():
        return min(coverage_count) > 0
        
    # Greedy
    while not is_valid():
        best_c = -1
        best_gain = -1
        # Optimization: Only check words that might add new coverage?
        # For n=6, 729 words, 13 neighbors. 729*13 ~ 9k ops per step.
        # Max steps 73. 73*9k ~ 600k ops. Very fast.
        for c in range(q**n):
            if c in codewords:
                continue
            gain = 0
            for v in balls[c]:
                if coverage_count[v] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1:
            break # Should not happen if is_valid() is false
        add_codeword(best_c)
        
    # Local Search
    start_time = time.time()
    time_limit = 100 # Leave 20s margin for 120s limit
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant words
        removed = False
        for c in list(codewords):
            # Check if c is redundant
            redundant = True
            for v in balls[c]:
                if coverage_count[v] == 1:
                    redundant = False
                    break
            if redundant:
                remove_codeword(c)
                removed = True
                break # Restart search from clean state
        
        if removed:
            continue
            
        # 2. Try swaps to enable removal
        # Pick a random codeword to try swapping
        if not codewords:
            break
        c = random.choice(list(codewords))
        
        # Find words covered exclusively by c
        S_c = [v for v in balls[c] if coverage_count[v] == 1]
        
        if not S_c:
            # Should have been caught in removal step, but just in case
            remove_codeword(c)
            continue
            
        # Find a candidate c' such that S_c is subset of balls[c']
        # We can filter candidates: c' must cover all words in S_c.
        # Since |S_c| is small, we can just check all c' not in codewords.
        found_swap = False
        # To speed up, we can pre-filter candidates that cover the first word in S_c
        first_v = S_c[0]
        candidates = [c_prime for c_prime in balls[first_v] if c_prime not in codewords]
        
        for c_prime in random.sample(candidates, len(candidates)):
            # Check if c_prime covers all of S_c
            valid_swap = True
            for v in S_c:
                if v not in balls[c_prime]:
                    valid_swap = False
                    break
            if valid_swap:
                # Perform swap
                remove_codeword(c)
                add_codeword(c_prime)
                found_swap = True
                break
                
        if not found_swap:
            # If no swap found for this c, try another c? Or just loop.
            # To avoid getting stuck, we might want to try a random "add" if we are stuck for too long?
            # But usually, random selection of c will eventually find a valid swap or a redundant word.
            continue
            
    # Convert set to list of lists
    result = [word_to_list(c) for c in codewords]
    return result
