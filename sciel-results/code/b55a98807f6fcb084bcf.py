import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4 for M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]

    # Primes for Q (primorial base)
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    best_N = 0
    best_pairs = []

    # Try different Q values
    for k in range(1, 13):
        if time_left() < 10:
            break
            
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find best M <= M_max using DFS
        # M = prod p_i^e_i, p_i in p_1mod4
        # Maximize prod (e_i + 1)
        
        best_M = 1
        best_score = 0
        
        # Use more primes in search
        search_primes = p_1mod4[:30]
        
        def dfs(idx, current_M, current_score, prev_exp):
            nonlocal best_M, best_score
            
            if time_left() < 5:
                return
                
            if current_score > best_score:
                best_score = current_score
                best_M = current_M
                
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            val = 1
            e = 0
            while current_M * val <= M_max and e <= prev_exp:
                next_M = current_M * val
                next_score = current_score * (e + 1)
                dfs(idx + 1, next_M, next_score, e)
                val *= p
                e += 1
                if val > M_max // max(current_M, 1):
                    break

        dfs(0, 1, 1, 200)
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Enumerate pairs: x^2 + y^2 = M
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # Estimate time: ~0.1us per iteration in Python, so 10^7 ~ 1s
        # Allow up to 8*10^7 iterations (~8s) if we have time
        max_iters = min(limit_x, 8 * 10**7)
        
        if limit_x > max_iters:
            # If too large, we might still try if we have lots of time
            if time_left() < 12:
                continue
        
        pairs = []
        M_val = M
        Q_val = Q
        isqrt = math.isqrt
        
        for x in range(1, max_iters):
            x2 = x * x
            y2 = M_val - x2
            y = isqrt(y2)
            if y * y == y2 and x <= y:
                a = Q_val * x2
                b = Q_val * y2
                pairs.append([a, b])
            
            if x % 2000000 == 0 and time_left() < 8:
                break

        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
