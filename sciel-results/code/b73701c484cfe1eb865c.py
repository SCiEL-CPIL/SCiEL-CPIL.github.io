import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 for building M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421]
    
    # Small primes for Q (square base)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_N = 0
    best_pairs = []
    
    # Try different k values for Q = (product of first k small primes)^2
    # Smaller k gives larger M_max, allowing more prime factors in M
    k_candidates = [2, 3, 4, 1, 5]
    
    for k in k_candidates:
        if time_left() < 10:
            break
            
        # Compute Q
        S = 1
        for i in range(k):
            S *= small_primes[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 100:
            continue
        
        # Find best M <= M_max to maximize representations as sum of two squares
        # M should be product of primes ≡ 1 (mod 4) to maximize reps
        # Number of reps ~ (1/4) * product(e_i + 1)
        
        best_M = 1
        best_rep_count = 0
        
        # Use first n primes 1 mod 4 - try to include as many as possible
        # For M_max ~ 2.5e17, we can fit many small primes
        # Product of first m primes 1mod4: 5*13*17*29*37*41*53*61*73*89*97*101*109*113*137*149*157*173*181*193
        # Let's use up to 25 primes
        n_primes = min(25, len(p_1mod4))
        search_primes = p_1mod4[:n_primes]
        
        def dfs(idx, current_M, current_prod, prev_exp):
            nonlocal best_M, best_rep_count
            
            if time_left() < 5:
                return
            
            # current_prod = product of (e_i + 1) so far
            if current_M > 1 and current_prod > best_rep_count:
                best_rep_count = current_prod
                best_M = current_M
            
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            # Try exponent 0, 1, 2, ... up to prev_exp
            val = 1
            for e in range(prev_exp + 1):
                next_M = current_M * val
                if next_M > M_max:
                    break
                next_prod = current_prod * (e + 1)
                dfs(idx + 1, next_M, next_prod, e)
                val *= p
                if val > M_max // max(current_M, 1):
                    break
        
        dfs(0, 1, 1, 50)
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Enumerate all (x, y) with x^2 + y^2 = M, x >= 1, x <= y
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # Be conservative on iteration limit
        max_iters = min(limit_x, 12000000)  # 12 million max
        
        if limit_x > max_iters and time_left() < 15:
            continue
        
        pairs = []
        M_val = M
        Q_val = Q
        isqrt = math.isqrt
        
        for x in range(1, max_iters + 1):
            x2 = x * x
            y2 = M_val - x2
            if y2 < 0:
                break
            y = isqrt(y2)
            if y * y == y2 and x <= y:
                a = Q_val * x2
                b = Q_val * y2
                pairs.append([a, b])
            
            if x % 500000 == 0 and time_left() < 8:
                break
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
    
    # Fallback: guarantee at least 1 valid pair
    if not best_pairs:
        # N=5: 1+4, both powerful (1 is powerful, 4=2^2 is powerful)
        best_N = 5
        best_pairs = [[1, 4]]
    
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
