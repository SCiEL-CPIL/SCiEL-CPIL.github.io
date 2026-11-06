import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def get_smooth_part(x):
        y = 1
        for p in primes:
            while x % p == 0:
                y *= p
                x //= p
        return y
    
    def score(n):
        smooth_parts = set()
        for i in range(1, 25):
            smooth_parts.add(get_smooth_part(n + i))
        return len(smooth_parts)
    
    best_n = 1000
    best_score = score(best_n)
    
    # Random search
    # We can also try local improvements around random points
    attempts = 0
    while time.time() < deadline:
        # Random candidate
        n = random.randint(1000, 1000000)
        s = score(n)
        if s < best_score:
            best_score = s
            best_n = n
            # Optional: try to improve locally
            # for delta in range(-10, 11):
            #    if 1000 <= n+delta <= 1000000:
            #       s2 = score(n+delta)
            #       if s2 < best_score:
            #          best_score = s2
            #          best_n = n+delta
        attempts += 1
        # Throttle if too fast? No, just run until deadline.
        if attempts % 10000 == 0:
            # Could print progress if needed, but not required
            pass
            
    return best_n
