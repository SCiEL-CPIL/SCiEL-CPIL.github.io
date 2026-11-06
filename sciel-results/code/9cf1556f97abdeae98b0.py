import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 for building M
    # We need enough primes to reach the limit
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421]
    
    # Main logic
    # 1. Determine max k
    # 2. Construct M
    # 3. Generate pairs
    # 4. Scale by Q=1
    
    M = 1
    used_primes = []
    for p in p_1mod4:
        if M * p > 10**18:
            break
        M *= p
        used_primes.append(p)
    
    N = M
    
    def generate_reps(primes):
        """
        Generates all representations of M = product(primes) as sum of two squares.
        primes: list of distinct primes p = 1 mod 4.
        """
        base_reps = []
        for p in primes:
            found = False
            # p is small, brute force is fine
            for u in range(1, int(math.isqrt(p)) + 1):
                v2 = p - u*u
                v = math.isqrt(v2)
                if v*v == v2 and v > 0:
                    base_reps.append((u, v))
                    found = True
                    break
            if not found:
                # Should not happen for p=1 mod 4
                base_reps.append((1, 1)) # Fallback
        
        # Start with (1, 0) representing 1.
        reps = [(1, 0)]
        for (u, v) in base_reps:
            new_reps = []
            for (x, y) in reps:
                # Multiply (x+iy) by (u+iv)
                x1 = x*u - y*v
                y1 = x*v + y*u
                new_reps.append((x1, y1))
                
                # Multiply (x+iy) by (u-iv)
                x2 = x*u + y*v
                y2 = -x*v + y*u
                new_reps.append((x2, y2))
            reps = new_reps
        
        # Filter valid reps: x>0, y>0, x<=y
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

    pairs_xy = generate_reps(used_primes)
    
    final_pairs = []
    for (x, y) in pairs_xy:
        a = x * x
        b = y * y
        if a <= b:
            final_pairs.append([a, b])
        else:
            final_pairs.append([b, a])
    
    # Remove duplicates just in case
    unique_pairs = []
    seen_pairs = set()
    for p in final_pairs:
        if tuple(p) not in seen_pairs:
            seen_pairs.add(tuple(p))
            unique_pairs.append(p)
            
    return {'N': N, 'pairs': unique_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
