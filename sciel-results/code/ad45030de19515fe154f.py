import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    # We will sieve up to a limit L
    # L = 10^7 is a good start. 
    # If time allows, we could go higher, but 10^7 is fast.
    # Let's try 2 * 10^7 to be safe on coverage, as it's still fast.
    L = 20_000_000
    
    # Check if we have enough time for a larger sieve
    # Sieve time is roughly proportional to L.
    # 10^7 takes ~0.5-1s in Python with optimization?
    # Let's stick to 10^7 first to be safe on time, or 5*10^6?
    # Actually, pure Python linear sieve for 10^7 might take 5-10s.
    # Let's use 5 * 10^6 to be safe, or optimize.
    # 5 * 10^6 is 5 million.
    # Let's try 10^7 and see. If it's too slow, we can reduce.
    # But we can't test interactively.
    # Let's use 5 * 10^6 as a safe bet for speed, which should find many collisions.
    # Actually, the number of collisions grows with L.
    # Let's aim for 10^7.
    
    # To speed up, we can use arrays from `array` module or just lists.
    # Lists of ints are fine.
    
    is_prime = [True] * (L + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    # Arrays for linear sieve
    sigma = [0] * (L + 1)
    sigma[1] = 1
    sigma_coprime = [0] * (L + 1)
    sigma_coprime[1] = 1
    p_pow = [1] * (L + 1)
    p_exp = [0] * (L + 1)
    p_val = [0] * (L + 1)
    
    for i in range(2, L + 1):
        if is_prime[i]:
            primes.append(i)
            sigma[i] = 1 + i
            sigma_coprime[i] = 1
            p_pow[i] = i
            p_exp[i] = 1
            p_val[i] = i
            
        for p in primes:
            if i * p > L:
                break
            is_prime[i * p] = False
            if i % p == 0:
                p_val[i * p] = p
                p_exp[i * p] = p_exp[i] + 1
                p_pow[i * p] = p_pow[i] * p
                sigma_coprime[i * p] = sigma_coprime[i]
                sigma[i * p] = sigma[i] + sigma_coprime[i] * p_pow[i * p]
                break
            else:
                p_val[i * p] = p
                p_exp[i * p] = 1
                p_pow[i * p] = p
                sigma_coprime[i * p] = sigma[i]
                sigma[i * p] = sigma[i] * (1 + p)
        
        if time.time() > deadline:
            break

    # Now count solutions
    # M = k * sigma(k)
    # We need M <= 10^18
    max_m = 10**18
    
    solutions = {}
    
    # We processed up to some N <= L
    N = len(is_prime) - 1
    # Actually, the loop might have broken early.
    # But `is_prime` is initialized to L+1.
    # We should only iterate up to the last fully processed number.
    # However, if we break early, the arrays are partially filled.
    # To be safe, we can track the last processed `i`.
    # But `break` inside the inner loop doesn't stop the outer loop.
    # The `if time.time() > deadline: break` is inside the outer loop.
    # So we processed up to `i` where time ran out.
    # Let's track `last_processed`.
    
    # Re-run logic to track last processed?
    # No, just check the time again or assume we finished if no break.
    # If we break, we might have incomplete data for `i+1`.
    # But we only use `sigma[k]` for `k` that were fully processed.
    # Since `sigma[k]` is computed when `k` is the outer loop variable,
    # and the inner loop fills `sigma[i*p]` for future `i`,
    # if we break at `i`, then `sigma[i]` is valid.
    # `sigma[i+1]` might be invalid if it depends on `i`.
    # So we should only use `k` up to `last_i`.
    
    # Let's modify the loop to track `last_i`.
    # But I already wrote the code above.
    # Let's assume we finish or check time at the end.
    # To be robust, let's just use a smaller L if we are worried.
    # L = 5 * 10^6 is very safe.
    
    # Let's restart with L = 5 * 10^6 to ensure speed.
    L = 5 * 10**6
    
    is_prime = [True] * (L + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    sigma = [0] * (L + 1)
    sigma[1] = 1
    sigma_coprime = [0] * (L + 1)
    sigma_coprime[1] = 1
    p_pow = [1] * (L + 1)
    p_exp = [0] * (L + 1)
    p_val = [0] * (L + 1)
    
    last_processed = 1
    for i in range(2, L + 1):
        if is_prime[i]:
            primes.append(i)
            sigma[i] = 1 + i
            sigma_coprime[i] = 1
            p_pow[i] = i
            p_exp[i] = 1
            p_val[i] = i
            
        for p in primes:
            if i * p > L:
                break
            is_prime[i * p] = False
            if i % p == 0:
                p_val[i * p] = p
                p_exp[i * p] = p_exp[i] + 1
                p_pow[i * p] = p_pow[i] * p
                sigma_coprime[i * p] = sigma_coprime[i]
                sigma[i * p] = sigma[i] + sigma_coprime[i] * p_pow[i * p]
                break
            else:
                p_val[i * p] = p
                p_exp[i * p] = 1
                p_pow[i * p] = p
                sigma_coprime[i * p] = sigma[i]
                sigma[i * p] = sigma[i] * (1 + p)
        
        last_processed = i
        if time.time() > deadline:
            break

    solutions = {}
    for k in range(2, last_processed + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            if m_val not in solutions:
                solutions[m_val] = []
            solutions[m_val].append(k)
            
    # Also check k=1
    if last_processed >= 1:
        m_val = 1 * sigma[1]
        if m_val not in solutions:
            solutions[m_val] = []
        solutions[m_val].append(1)

    best_m = 0
    best_k_list = []
    
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
    
    if not best_k_list:
        return {'M': 1, 'k': [1]}
        
    # Filter k <= sqrt(M) just in case, though it should be true
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in best_k_list if k <= sqrt_m]
    
    # Ensure types are correct
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
