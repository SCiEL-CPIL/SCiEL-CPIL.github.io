"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

START_TIME = time.time()
DEADLINE = START_TIME + 85  # safety margin before 90s kill

def _check_time():
    return time.time() > DEADLINE

def _build_gcd_table(n=128):
    """Precompute gcd table for 1..n."""
    table = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        table[i][i] = i
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table

def _check_triple(gcd_table, a, b, c):
    """Check if three elements have equal pairwise gcds."""
    g_ab = gcd_table[a][b]
    g_ac = gcd_table[a][c]
    if g_ab == g_ac:
        g_bc = gcd_table[b][c]
        if g_bc == g_ab:
            return False
    return True

def _is_valid_addition(gcd_table, S_set, S_list, x):
    """Check if x can be added to S without creating an invalid triple."""
    for i in range(len(S_list)):
        a = S_list[i]
        g_ax = gcd_table[a][x]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                return False
    return True

def _greedy_with_priority(gcd_table, order):
    """Greedy construction with given priority order."""
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        if _is_valid_addition(gcd_table, S_set, S, x):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _local_search_add(gcd_table, S, S_set, rng, max_iters=1000):
    """Try to add elements not in S."""
    S = list(S)
    S_set = set(S)
    not_in = [x for x in range(1, 129) if x not in S_set]
    rng.shuffle(not_in)
    
    added = 0
    for x in not_in:
        if _check_time():
            break
        if _is_valid_addition(gcd_table, S_set, S, x):
            S.append(x)
            S_set.add(x)
            added += 1
    return S, S_set, added

def _swap_improve(gcd_table, S, S_set, rng, max_iters=500):
    """Try to improve by swapping elements: remove one, add another."""
    S = list(S)
    S_set = set(S)
    not_in = [x for x in range(1, 129) if x not in S_set]
    if not not_in:
        return S, S_set
    
    improved = 0
    for _ in range(max_iters):
        if _check_time():
            break
        if not S or not not_in:
            break
        
        # Try random swap
        idx = rng.randint(0, len(S) - 1)
        old = S[idx]
        new = rng.choice(not_in)
        
        # Check if removing old and adding new is valid
        S_test = [s for s in S if s != old]
        S_set_test = set(S_test)
        
        valid = True
        for i in range(len(S_test)):
            a = S_test[i]
            g_an = gcd_table[a][new]
            for j in range(i + 1, len(S_test)):
                b = S_test[j]
                if gcd_table[a][b] == g_an and gcd_table[b][new] == g_an:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S[idx] = new
            S_set.remove(old)
            S_set.add(new)
            not_in.remove(new)
            improved += 1
    
    return S, S_set, improved

def construct():
    if _check_time():
        return [1]
    
    gcd_table = _build_gcd_table(128)
    rng = random.Random(42)
    
    best_S = None
    best_len = 0
    
    # Strategy 1: Greedy ascending
    S, _ = _greedy_with_priority(gcd_table, range(1, 129))
    if len(S) > best_len:
        best_S = S
        best_len = len(S)
    
    # Strategy 2: Greedy descending
    S, _ = _greedy_with_priority(gcd_table, range(128, 0, -1))
    if len(S) > best_len:
        best_S = S
        best_len = len(S)
    
    # Strategy 3: Greedy by number of prime factors (fewer first)
    def count_prime_factors(x):
        cnt = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                cnt += 1
                n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt
    
    order3 = sorted(range(1, 129), key=lambda x: count_prime_factors(x))
    S, _ = _greedy_with_priority(gcd_table, order3)
    if len(S) > best_len:
        best_S = S
        best_len = len(S)
    
    # Strategy 4: Many random greedy attempts
    for i in range(500):
        if _check_time():
            break
        order = list(range(1, 129))
        rng.shuffle(order)
        S, S_set = _greedy_with_priority(gcd_table, order)
        S, S_set, _ = _local_search_add(gcd_table, S, S_set, rng, max_iters=500)
        if len(S) > best_len:
            best_S = S
            best_len = len(S)
    
    # Strategy 5: Swap improvement on best
    if best_S:
        best_S_set = set(best_S)
        for i in range(200):
            if _check_time():
                break
            S, S_set, improved = _swap_improve(gcd_table, list(best_S), set(best_S), rng, max_iters=200)
            if len(S) > best_len:
                best_S = S
                best_S_set = S_set
                best_len = len(S)
    
    # Strategy 6: Structural approaches
    # Odd numbers
    S, _ = _greedy_with_priority(gcd_table, [x for x in range(1, 129) if x % 2 == 1])
    if len(S) > best_len:
        best_S = S
        best_len = len(S)
    
    # Numbers coprime to 6
    S, _ = _greedy_with_priority(gcd_table, [x for x in range(1, 129) if x % 2 != 0 and x % 3 != 0])
    if len(S) > best_len:
        best_S = S
        best_len = len(S)
    
    # Numbers with no small prime factors (largest smallest prime factor first)
    def smallest_prime_factor(x):
        if x < 2:
            return 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                return d
            d += 1
        return x
    
    order6 = sorted(range(1, 129), key=lambda x: -smallest_prime_factor(x))
    S, _ = _greedy_with_priority(gcd_table, order6)
    if len(S) > best_len:
        best_S = S
        best_len = len(S)
    
    # Strategy 7: More diverse random with local search
    for seed_val in [1, 2, 3, 7, 13, 100, 999, 12345, 54321, 77777]:
        if _check_time():
            break
        rng2 = random.Random(seed_val)
        order = list(range(1, 129))
        rng2.shuffle(order)
        S, S_set = _greedy_with_priority(gcd_table, order)
        S, S_set, _ = _local_search_add(gcd_table, S, S_set, rng2, max_iters=1000)
        S, S_set, _ = _swap_improve(gcd_table, S, S_set, rng2, max_iters=300)
        if len(S) > best_len:
            best_S = S
            best_len = len(S)
    
    # Strategy 8: Start from best and try to add more
    if best_S:
        best_S_set = set(best_S)
        for i in range(100):
            if _check_time():
                break
            S, S_set, added = _local_search_add(gcd_table, list(best_S), set(best_S), rng, max_iters=500)
            if len(S) > best_len:
                best_S = S
                best_S_set = S_set
                best_len = len(S)
    
    if best_S is None or len(best_S) == 0:
        return [1]
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
