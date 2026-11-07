import time
import random
from math import isqrt

N_MAX = 256

def _get_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _can_add_fast(S_list, x):
    """Check if x can be added to S_list (list of ints)."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            # Check x | a*b
            if (a * b) % x == 0:
                return False
            # Check a | x*b
            if (x * b) % a == 0:
                return False
            # Check b | x*a
            if (x * a) % b == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = N_MAX
    primes = _get_primes(n)
    S = set(primes)
    S_list = list(S)
    
    # Greedy Add
    for x in range(2, n + 1):
        if time.time() > deadline:
            break
        if x not in S:
            if _can_add_fast(S_list, x):
                S.add(x)
                S_list.append(x)
                
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        added = False
        for x in range(2, n + 1):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add_fast(S_list, x):
                    S.add(x)
                    S_list.append(x)
                    added = True
        if added:
            continue
        
        # Try 1-swap
        c_list = [x for x in range(2, n + 1) if x not in S]
        
        if not S_list or not c_list:
            break
            
        s_list = S_list[:]
        random.shuffle(s_list)
        random.shuffle(c_list)
        
        swap_found = False
        for y in s_list:
            if swap_found or time.time() > deadline:
                break
            # Remove y from S_list temporarily
            S_list.remove(y)
            for x in c_list:
                if swap_found or time.time() > deadline:
                    break
                if _can_add_fast(S_list, x):
                    S_list.append(x)
                    S = set(S_list)
                    improved = True
                    swap_found = True
                    break
            if not swap_found:
                S_list.append(y)
        
        if improved:
            continue
            
        # Try 2-swap (limited)
        if len(S_list) >= 2 and len(c_list) >= 2:
            # Pick random pairs
            for _ in range(20):
                if time.time() > deadline:
                    break
                y1_idx = random.randint(0, len(S_list) - 1)
                y2_idx = random.randint(0, len(S_list) - 1)
                if y1_idx == y2_idx:
                    continue
                y1 = S_list[y1_idx]
                y2 = S_list[y2_idx]
                
                x1_idx = random.randint(0, len(c_list) - 1)
                x2_idx = random.randint(0, len(c_list) - 1)
                if x1_idx == x2_idx:
                    continue
                x1 = c_list[x1_idx]
                x2 = c_list[x2_idx]
                
                # Remove y1 and y2
                S_list.remove(y1)
                S_list.remove(y2)
                
                # Check if x1 can be added
                if _can_add_fast(S_list, x1):
                    S_list.append(x1)
                    # Check if x2 can be added
                    if _can_add_fast(S_list, x2):
                        S_list.append(x2)
                        S = set(S_list)
                        improved = True
                        break
                    else:
                        S_list.remove(x1)
                
                # Restore y1 and y2 if not improved
                if not improved:
                    S_list.append(y1)
                    S_list.append(y2)

    return [int(x) for x in S]
