import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 for building M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421]
    
    best_N = 0
    best_pairs = []
    
    # Strategy:
    # 1. Use Q=1 to maximize M.
    # 2. Select largest set of distinct primes p_i = 1 mod 4 such that product P <= 10^18.
    # 3. Generate all representations of M as sum of two squares.
    
    def generate_reps(primes):
        base_reps = []
        for p in primes:
            found = False
            for u in range(1, int(math.isqrt(p)) + 1):
                v2 = p - u*u
                v = math.isqrt(v2)
                if v*v == v2 and v > 0:
                    base_reps.append((u, v))
                    found = True
                    break
            if not found:
                base_reps.append((1, 1))
        
        reps = [(1, 0)]
        for (u, v) in base_reps:
            new_reps = []
            for (x, y) in reps:
                x1 = x*u - y*v
                y1 = x*v + y*u
                new_reps.append((x1, y1))
                
                x2 = x*u + y*v
                y2 = -x*v + y*u
                new_reps.append((x2, y2))
            reps = new_reps
        
        pairs = []
        seen = set()
        for (x, y) in reps:
            if x < 0: x = -x
            if y < 0: y = -y
            if x == 0 or y == 0: continue
            if x > y:
                x, y = y, x
            if (x, y) not in seen:
                seen.add((x, y))
                pairs.append((x, y))
        
        return pairs

    M = 1
    used_primes = []
    for p in p_1mod4:
        if M * p > 10**18:
            break
        M *= p
        used_primes.append(p)
    
    N = M
    pairs_xy = generate_reps(used_primes)
    
    final_pairs = []
    seen_pairs = set()
    for (x, y) in pairs_xy:
        a = x * x
        b = y * y
        if a <= b:
            pair = (a, b)
        else:
            pair = (b, a)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            final_pairs.append(list(pair))
            
    # Try to find additional powerful pairs for N
    # We can't check all powerful numbers, but we can check some candidates
    # If time permits
    if time_left() > 10:
        # Generate small powerful numbers
        # Powerful numbers are of form k^2 * m where m is square-free and all prime factors of m have even exponent in n?
        # No, powerful means if p|n then p^2|n.
        # So n = \prod p_i^{e_i} with e_i >= 2.
        # We can generate powerful numbers up to 10^6
        powerful_small = set()
        # k^2 * s where s is powerful?
        # Simpler: iterate k up to 1000, k^3 <= 10^9 -> k<=1000
        # n = k^2 * m, where m is powerful?
        # Actually, just iterate i up to 10^4, i^2 is powerful.
        # i^2 * j^3?
        # Let's just use the squares we already have.
        # Non-square powerful numbers are rare.
        pass

    return {'N': N, 'pairs': final_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
