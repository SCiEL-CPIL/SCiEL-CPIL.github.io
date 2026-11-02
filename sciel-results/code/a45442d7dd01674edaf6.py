import itertools
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback: Return all words (valid, but large)
    # This ensures validity for any input
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1():
    q = 3
    n = 8
    total_words = 3**n
    
    # 1. Generate Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # 2. Helper functions
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * 3 + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = val % 3
            val //= 3
        return word
            
    # 3. Generate Initial Code (Direct Sum)
    initial_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n-4):
            w = h + list(tail)
            initial_words.append(to_int(w))
            
    # 4. Precompute Balls
    powers = [3**i for i in range(n)]
    balls = []
    for i in range(total_words):
        w = to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % 3
                new_i = i + (new_val - w[pos]) * powers[pos]
                ball.append(new_i)
        balls.append(ball)
        
    # 5. Greedy Removal Function
    def run_removal(seed):
        rng = random.Random(seed)
        code = initial_words[:]
        rng.shuffle(code)
        
        # Initialize multiplicity
        mult = [0] * total_words
        for c in code:
            for p in balls[c]:
                mult[p] += 1
                
        improved = True
        # Limit iterations to prevent infinite loops (though should converge)
        max_iters = 100
        iters = 0
        while improved and iters < max_iters:
            improved = False
            new_code = []
            rng.shuffle(code)
            for c in code:
                ball = balls[c]
                # Check if all points in ball have mult > 1
                redundant = True
                for p in ball:
                    if mult[p] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for p in ball:
                        mult[p] -= 1
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
            iters += 1
        return code
        
    # 6. Run multiple restarts
    start_time = time.time()
    best_code = None
    best_size = float('inf')
    
    # Run restarts
    # Estimate time per restart: ~0.5s
    # Target: use 90% of time (135s)
    # Let's try 100 restarts
    for i in range(100):
        if time.time() - start_time > 130:
            break
        current_code = run_removal(i)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code
            
    if best_code is None:
        best_code = initial_words
        
    return [to_word(c) for c in best_code]
