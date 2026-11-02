import numpy as np
import time
import itertools

def construct(q, n, radius):
    # We assume the task is specifically for q=3, n=8, radius=1
    # But the function signature is generic. 
    # If n != 8 or q != 3, fall back to simple construction or greedy.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy on direct sum
        # For n=4, perfect code exists.
        # For general, use tetracode if n>=4
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Pad or truncate? Just use tetracode for first 4, random for rest?
            # Better: Direct sum of tetracode and full space for n-4
            if n == 4:
                return [h[:] for h in tetracode]
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force or greedy for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, r=1
    
    def word_to_int(word):
        val = 0
        for i, d in enumerate(word):
            val += d * (q ** i)
        return val

    def int_to_word(val):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    def greedy_reduce(code_ints, n, q, r, balls, time_limit):
        # code_ints: list of ints
        # balls: precomputed balls for this n
        total_words = q ** n
        code_set = set(code_ints)
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for v in balls[c]:
                cover_count[v] += 1
        
        current_code = list(code_set)
        np.random.shuffle(current_code)
        
        removed_any = True
        while removed_any:
            if time.time() > time_limit:
                break
            removed_any = False
            new_code = []
            for c in current_code:
                idx = balls[c]
                min_cov = np.min(cover_count[idx])
                if min_cov > 1:
                    cover_count[idx] -= 1
                    removed_any = True
                else:
                    new_code.append(c)
            current_code = new_code
            
        return current_code

    # Precompute balls for n=5
    n5 = 5
    total_words_5 = q ** n5
    balls_5 = []
    for w in range(total_words_5):
        word = [0] * n5
        val = w
        for i in range(n5):
            word[i] = val % q
            val //= q
        b = [w]
        for i in range(n5):
            for delta in range(1, q):
                new_word = word.copy()
                new_word[i] = (word[i] + delta) % q
                # convert back to int
                new_val = 0
                for j, d in enumerate(new_word):
                    new_val += d * (q ** j)
                b.append(new_val)
        balls_5.append(b)
        
    # Initial code for n=5: Tetracode x Z_3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_5 = []
    for h in tetracode:
        for tail in range(3):
            w = h + [tail]
            init_code_5.append(word_to_int(w))
            
    # Reduce n=5 code
    # Try a few restarts
    best_code_5 = None
    best_size_5 = len(init_code_5)
    start_time = time.time()
    time_limit = 140.0
    
    for r in range(10):
        if time.time() > time_limit: break
        code = greedy_reduce(init_code_5, n5, q, radius, balls_5, time_limit)
        if len(code) < best_size_5:
            best_size_5 = len(code)
            best_code_5 = code
            # If we hit 12 or 13, it's likely optimal or near.
            if best_size_5 <= 13:
                break
                
    # Construct n=8 code from best_code_5
    # Direct sum with Z_3^3
    init_code_8 = []
    for c5 in best_code_5:
        # Convert c5 to word
        w5 = int_to_word(c5) # This uses n=8 logic, careful.
        # Let's make a local int_to_word for n=5
        def int_to_word_n(val, n):
            word = [0] * n
            for i in range(n):
                word[i] = val % q
                val //= q
            return word
        w5 = int_to_word_n(c5, n5)
        for tail in itertools.product(range(q), repeat=3):
            w8 = w5 + list(tail)
            init_code_8.append(word_to_int(w8))
            
    # Precompute balls for n=8
    total_words_8 = q ** 8
    balls_8 = []
    for w in range(total_words_8):
        word = int_to_word(w)
        b = [w]
        for i in range(8):
            for delta in range(1, q):
                new_word = word.copy()
                new_word[i] = (word[i] + delta) % q
                b.append(word_to_int(new_word))
        balls_8.append(b)
        
    # Reduce n=8 code
    best_code_8 = None
    best_size_8 = len(init_code_8)
    
    # We have less time left, but reduction is fast if we vectorize well.
    # The greedy loop is the bottleneck.
    
    for r in range(5):
        if time.time() > time_limit: break
        code = greedy_reduce(init_code_8, 8, q, radius, balls_8, time_limit)
        if len(code) < best_size_8:
            best_size_8 = len(code)
            best_code_8 = code
            
    # Convert back to list of lists
    if best_code_8:
        return [int_to_word(c) for c in best_code_8]
    else:
        return [int_to_word(c) for c in init_code_8]
