import time
import random

def _is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0: return False
    return True

def _can_add(S, x):
    """Check if x can be added to S without violating constraints."""
    s_list = list(S)
    n = len(s_list)
    
    # For every pair (a, b) in S with a != b:
    #   Check if x divides a*b -> invalid
    #   Check if a divides x*b -> invalid
    #   Check if b divides x*a -> invalid (symmetric to above)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            # x | a*b ?
            if (a * b) % x == 0:
                return False
            # a | x*b ?
            if (x * b) % a == 0:
                return False
            # b | x*a ?
            if (x * a) % b == 0:
                return False
    return True

def _is_valid(S):
    """Full validation of set S."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if i == j: continue
            b = s_list[j]
            for k in range(n):
                if i == k or j == k: continue
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    S = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, all_nums, deadline, max_iter=20000):
    current = set(S)
    best = set(S)
    best_size = len(S)
    non_members = [x for x in all_nums if x not in current]
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if not non_members:
            break
        
        r = random.random()
        if r < 0.4:
            # Try add
            idx = random.randint(0, len(non_members) - 1)
            x = non_members[idx]
            if _can_add(current, x):
                current.add(x)
                non_members.pop(idx)
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
        else:
            # Try swap
            if current:
                y = random.choice(list(current))
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                temp = current - {y}
                if _can_add(temp, x):
                    current = temp | {x}
                    non_members.remove(x)
                    non_members.append(y)
    return best

def construct():
    deadline = time.time() + 55.0
    all_nums = list(range(2, 257))
    
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]
    
    # Sort composites by number of distinct prime factors (fewer = easier to add)
    def num_distinct_prime_factors(n):
        count = 0
        d = 2
        m = n
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    composites_sorted = sorted(composites, key=num_distinct_prime_factors)
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        primes + composites_sorted,
        composites_sorted + primes,
    ]
    
    rng = random.Random(42)
    for _ in range(50):
        shuffled = all_nums.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    best_set = set()
    best_size = 0
    
    for order in orders:
        if time.time() > deadline - 15.0:
            break
        S = _greedy(order, deadline - 15.0)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search
    if time.time() < deadline - 3.0:
        improved = _local_search(best_set, all_nums, deadline - 3.0, max_iter=30000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # More restarts
    if time.time() < deadline - 1.0:
        for _ in range(10):
            if time.time() > deadline - 1.0:
                break
            shuffled = all_nums.copy()
            random.shuffle(shuffled)
            S = _greedy(shuffled, deadline - 1.0)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            elif len(S) == best_size:
                improved = _local_search(S, all_nums, deadline - 0.5, max_iter=5000)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved
    
    if not _is_valid(best_set):
        best_set = set(primes)
    
    return list(best_set)
