import time
import random

def construct():
    start_time = time.time()
    deadline = 75.0
    
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    candidates = set()
    for i in range(1, 10001, 30):
        candidates.add(i)
    for i in range(1, 10001, 6):
        candidates.add(i)
    for i in range(1, 10001, 10):
        candidates.add(i)
    for i in range(1, 10001, 15):
        candidates.add(i)
    for i in range(2, 10001, 4):
        candidates.add(i)
        
    candidates = list(candidates)
    random.shuffle(candidates)
    pool = candidates[:400]
    
    current_set = list(set(random.sample(pool, 12)))
    while len(current_set) < 12:
        for c in pool:
            if c not in current_set:
                current_set.append(c)
            if len(current_set) == 12:
                break

    def evaluate(s):
        primes = set()
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                sum_val = s[i] + s[j]
                primes.update(get_prime_factors(sum_val))
        return len(primes)

    best_score = evaluate(current_set)
    best_set = list(current_set)
    
    attempts = 0
    while time.time() - start_time < deadline:
        attempts += 1
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        new_val = random.choice(pool)
        if new_val == old_val or new_val in current_set:
            continue
            
        current_set[idx] = new_val
        new_score = evaluate(current_set)
        
        if new_score <= best_score:
            best_score = new_score
            best_set = list(current_set)
            if best_score == 3:
                break
        else:
            current_set[idx] = old_val
            
        if attempts % 1000 == 0:
             current_set = list(set(random.sample(pool, 12)))
             while len(current_set) < 12:
                 for c in pool:
                     if c not in current_set:
                         current_set.append(c)
                     if len(current_set) == 12:
                         break

    return best_set
