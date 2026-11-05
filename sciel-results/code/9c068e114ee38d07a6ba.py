import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # 10s margin for 120s limit? No, 150s limit. 10s is safe.
    
    # Fallback: Tetracode construction
    def tetracode_construction():
        if n < 4:
            # For small n, just return all words? Or a simple construction.
            # If n < 4, the tetracode doesn't fit.
            # For n=1, K_3(1,1) = 1.
            # For n=2, K_3(2,1) = 3? No, 1 word covers 1+2(2)=5 words. 3^2=9. 2 words enough?
            # Ball size 5. 9/5 = 1.8. So 2 words.
            # Let's just use greedy for small n too.
            pass
            
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        # This assumes q=3. If q != 3, this is wrong.
        # The task says q=3. But let's be safe.
        if q != 3:
            # Fallback to simple repetition or something?
            # The task is specifically for q=3.
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
            
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Check if we can use Greedy
    total_words = q ** n
    if total_words <= 5000:
        # Map word to int
        def word_to_int(w):
            val = 0
            for d in w:
                val = val * q + d
            return val

        def int_to_word(val):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = val % q
                val //= q
            return w

        # Precompute balls
        balls = []
        for i in range(total_words):
            w = int_to_word(i)
            ball = {i}
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    ball.add(word_to_int(w))
                w[pos] = orig
            balls.append(ball)

        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_c = -1
            best_count = 0
            # Optimization: only consider candidates that cover at least 1 uncovered word?
            # All candidates cover themselves. If a candidate is uncovered, it covers 1.
            # If a candidate is covered, it might cover 0.
            # So we can iterate all.
            for c in range(total_words):
                # If c is not in uncovered, it might still cover uncovered words.
                # But if c is in uncovered, it covers at least 1.
                count = len(balls[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == len(balls[c]):
                        # Can't get better than covering the whole ball
                        # But we want to maximize uncovered covered.
                        # If we cover the whole ball, that's max possible for this c.
                        # But another c might cover more uncovered words.
                        pass
                        
            if best_c == -1:
                break
                
            code.append(int_to_word(best_c))
            uncovered -= balls[best_c]
            
        if uncovered:
            # Greedy didn't finish in time or failed.
            # Return the tetracode construction to ensure validity.
            return tetracode_construction()
            
        return code

    else:
        return tetracode_construction()
