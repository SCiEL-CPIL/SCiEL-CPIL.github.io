import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    limit = P['n']
    deadline = start_time + 85.0
    
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    S = set(primes)
    best_S = set(S)
    best_size = len(S)
    
    def can_add(S, x):
        # S is a set
        # Check if x can be added to S
        # Triples: (x, b, c), (b, x, c), (b, c, x)
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            b = s_list[i]
            for j in range(i+1, n):
                c = s_list[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True

    def greedy_add(S):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            candidates = [x for x in range(2, limit + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S, x):
                    S.add(x)
                    changed = True
                    break
        return S

    S = greedy_add(S)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)

    current_S = set(S)
    
    while time.time() < deadline:
        # Try to add
        improved = False
        candidates = [x for x in range(2, limit + 1) if x not in current_S]
        if candidates:
            random.shuffle(candidates)
            for x in candidates:
                if can_add(current_S, x):
                    current_S.add(x)
                    improved = True
                    break
        
        if improved:
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
            continue
        
        # Try swaps
        # Attempt a few random swaps
        swap_made = False
        y_candidates = list(current_S)
        x_candidates = [x for x in range(2, limit + 1) if x not in current_S]
        
        if not x_candidates:
            break
            
        random.shuffle(y_candidates)
        random.shuffle(x_candidates)
        
        # Try up to 10 random swaps
        for _ in range(10):
            if not y_candidates or not x_candidates:
                break
            y = y_candidates.pop()
            x = x_candidates.pop()
            
            S_minus_y = current_S - {y}
            if can_add(S_minus_y, x):
                current_S = S_minus_y
                current_S.add(x)
                swap_made = True
                break
        
        if not swap_made:
            # If no swap worked, we might be stuck.
            # To avoid infinite loop of failing swaps, we could break or restart.
            # But maybe a different random selection works.
            # We just loop again.
            pass

    return list(best_S)
