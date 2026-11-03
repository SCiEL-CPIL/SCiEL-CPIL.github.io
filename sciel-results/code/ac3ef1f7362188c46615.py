import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    alphabet = list(range(q))
    total_words = q ** n
    
    # Encode words as integers for fast operations
    def word_to_int(w):
        val = 0
        for digit in w:
            val = val * q + digit
        return val
    
    def int_to_word(v):
        w = []
        for _ in range(n):
            w.append(v % q)
            v //= q
        return list(reversed(w))
    
    all_words_int = list(range(total_words))
    
    # Precompute balls as sets of integers
    balls = {}
    for w_int in all_words_int:
        w = int_to_word(w_int)
        ball = {w_int}
        for i in range(n):
            for val in alphabet:
                if val == w[i]:
                    continue
                new_w = w[:]
                new_w[i] = val
                ball.add(word_to_int(tuple(new_w)))
        balls[w_int] = ball
    
    def fast_greedy(seed):
        """Randomized greedy set cover with multiple restarts"""
        rng = random.Random(seed)
        uncovered = set(all_words_int)
        code = []
        
        while uncovered:
            if time.time() > deadline:
                break
            # Pick a random uncovered word as pivot
            u = rng.choice(tuple(uncovered))
            candidates = balls[u]
            best_c = None
            best_gain = -1
            for c in candidates:
                gain = sum(1 for w in balls[c] if w in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is not None:
                code.append(best_c)
                for w in balls[best_c]:
                    uncovered.discard(w)
        
        return code
    
    def optimize(code, seed):
        """Local search optimization"""
        rng = random.Random(seed)
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        while time.time() < deadline:
            removed = False
            swapped = False
            
            # Try Removal
            shuffled_code = code[:]
            rng.shuffle(shuffled_code)
            for c in shuffled_code:
                safe = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        safe = False
                        break
                if safe:
                    code.remove(c)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try Swap
            swap_attempts = 0
            while swap_attempts < 50:
                swap_attempts += 1
                i = rng.randint(0, len(code) - 1)
                c = code[i]
                
                # Find words only covered by c
                U_c = [w for w in balls[c] if cover_count[w] == 1]
                
                if not U_c:
                    code.pop(i)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    removed = True
                    break
                
                # Compute intersection of balls of U_c
                S = set(balls[U_c[0]])
                for w in U_c[1:]:
                    S.intersection_update(balls[w])
                    if not S:
                        break
                
                if S:
                    valid_candidates = [x for x in S if x not in code and x != c]
                    if valid_candidates:
                        c_prime = rng.choice(valid_candidates)
                        for w in balls[c]:
                            cover_count[w] -= 1
                        for w in balls[c_prime]:
                            cover_count[w] += 1
                        code[i] = c_prime
                        swapped = True
                        break
                        
            if swapped:
                continue
            
            if not removed and not swapped:
                break
        
        return code
    
    # Multiple restarts with different seeds
    best_code = None
    best_len = float('inf')
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        seed = restart * 1000 + 42
        initial_code = fast_greedy(seed)
        
        if len(initial_code) < best_len:
            best_code = initial_code
            best_len = len(initial_code)
        
        if time.time() > deadline - 10:
            break
        
        optimized_code = optimize(best_code[:], seed)
        
        if len(optimized_code) < best_len:
            best_code = optimized_code
            best_len = len(optimized_code)
    
    # Convert back to list format
    return [int_to_word(c) for c in best_code]
