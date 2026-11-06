import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4 for M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]

    # Primes for Q (primorial base)
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    best_N = 0
    best_pairs = []

    # Try different Q values
    # Q = (product of first k primes)^2
    # We want to balance M size.
    # Larger k -> Smaller M -> Faster enumeration, but potentially fewer pairs.
    # Smaller k -> Larger M -> Slower enumeration, but potentially more pairs.
    
    # We'll try k from 1 to 10.
    # For each k, we find the best M.
    # Then we enumerate pairs.
    
    for k in range(1, 11):
        if time_left() < 15:
            break
            
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find best M <= M_max
        # M = prod p_i^e_i, p_i in p_1mod4
        # Maximize prod (e_i + 1)
        # Use DFS with monotonic exponents e_0 >= e_1 >= ...
        
        best_M = 1
        best_score = 0
        best_exponents = []
        
        # Limit the depth of DFS to first 20 primes to keep it fast
        search_primes = p_1mod4[:20]
        
        def dfs(idx, current_M, current_score, prev_exp, exponents):
            nonlocal best_M, best_score, best_exponents
            
            if time_left() < 10:
                return
                
            if current_score > best_score:
                best_score = current_score
                best_M = current_M
                best_exponents = exponents[:]
                
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            # Max exponent for this prime
            # current_M * p^e <= M_max
            # e <= log_p(M_max / current_M)
            
            max_e = prev_exp
            val = 1
            e = 0
            while current_M * val <= M_max:
                if e > max_e:
                    break
                next_M = current_M * val
                next_score = current_score * (e + 1)
                
                exponents.append(e)
                dfs(idx + 1, next_M, next_score, e, exponents)
                exponents.pop()
                
                val *= p
                e += 1
                if val > M_max:
                    break

        dfs(0, 1, 1, 100, [])
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Enumerate pairs
        # x^2 + y^2 = M
        # x <= y => x^2 <= M/2 => x <= sqrt(M/2)
        # x >= 1 (assuming a,b >= 1)
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # Check if enumeration is feasible
        # 10^7 iterations ~ 1-2s
        # 5*10^7 iterations ~ 5-10s
        # We have ~100s left.
        # Let's allow up to 2*10^7 iterations to be safe and allow multiple k tries.
        # If limit_x is too large, we might skip or use a different strategy.
        # However, if M is large, the score is likely high.
        # Let's try to enumerate if limit_x <= 3*10^7.
        
        if limit_x > 3 * 10**7:
            # Too slow. Skip this M?
            # Or we can try to increase k (larger Q) to reduce M.
            # But we are already in the loop for this k.
            # We'll just skip this specific M for this k.
            # The next k (larger Q) will have smaller M.
            continue
        
        pairs = []
        M_val = M
        Q_val = Q
        isqrt = math.isqrt
        
        # Optimized loop
        # Start x from 1 to avoid a=0
        # If x=0 is allowed, we should include it.
        # Let's include x=0. If a=0 is invalid, the evaluator will reject.
        # But "1 counts as powerful" suggests positive integers.
        # I will start from 1 to be safe.
        # Actually, if x=0, a=0.
        # Let's check if 0 is powerful.
        # If I include x=0, I might get an extra pair.
        # If 0 is not powerful, the pair is invalid.
        # The evaluator checks "checked by exact factorization".
        # Factorization of 0 is undefined.
        # So 0 is likely not considered powerful.
        # I will start x from 1.
        
        for x in range(1, limit_x):
            x2 = x * x
            y2 = M_val - x2
            y = isqrt(y2)
            if y * y == y2:
                if x <= y:
                    a = Q_val * x2
                    b = Q_val * y2
                    pairs.append([a, b])
                    
            # Time check every 10^6 iterations
            if x % 1000000 == 0 and time_left() < 10:
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
