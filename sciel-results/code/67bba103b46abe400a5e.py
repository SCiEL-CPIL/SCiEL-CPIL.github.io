import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    
    N = 256
    max_val = N
    
    # Precompute req[a][b] = a // gcd(a,b)
    req = [[0]*(max_val+1) for _ in range(max_val+1)]
    for a in range(2, max_val+1):
        for b in range(2, max_val+1):
            req[a][b] = a // gcd(a,b)
    
    def is_valid(S):
        """S is a set of ints. Check validity."""
        S_list = list(S)
        n = len(S_list)
        if n <= 2:
            return True
        
        # Precompute div_members[r] = list of x in S where r|x
        div_members = [None]*(max_val+1)
        for r in range(1, max_val+1):
            div_members[r] = [x for x in S_list if x % r == 0]
        
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                r = req[a][b]
                if r == 1:
                    if n > 2:
                        return False
                    continue
                dm = div_members[r]
                cnt = len(dm)
                if b % r == 0:
                    if cnt >= 3:
                        return False
                else:
                    if cnt >= 2:
                        return False
        return True
    
    # Initial: primes + prime squares
    def is_prime(x):
        if x < 2: return False
        if x == 2: return True
        if x % 2 == 0: return False
        for i in range(3, int(x**0.5)+1, 2):
            if x % i == 0: return False
        return True
    
    primes = [x for x in range(2, max_val+1) if is_prime(x)]
    prime_squares = [p*p for p in primes if p*p <= max_val]
    initial_set = set(primes + prime_squares)
    
    assert is_valid(initial_set), f"Initial invalid, size={len(initial_set)}"
    
    best_set = initial_set.copy()
    best_size = len(best_set)
    current_set = initial_set.copy()
    
    # Greedy add
    changed = True
    while changed:
        changed = False
        for elem in range(2, max_val+1):
            if elem not in current_set:
                trial = current_set | {elem}
                if is_valid(trial):
                    current_set = trial
                    changed = True
                    if len(current_set) > best_size:
                        best_set = current_set.copy()
                        best_size = len(best_set)
                    break
    
    # Local search
    random.seed(42)
    no_improve = 0
    
    while time.time() < deadline - 2 and no_improve < 500:
        # Try random swap
        elems = list(current_set)
        if len(elems) >= 1:
            rem = random.choice(elems)
            cands = [x for x in range(2, max_val+1) if x not in current_set]
            if cands:
                add = random.choice(cands)
                trial = (current_set - {rem}) | {add}
                if is_valid(trial):
                    if len(trial) > len(current_set):
                        current_set = trial
                        no_improve = 0
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(best_set)
                    elif random.random() < 0.2:
                        current_set = trial
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        # Try to add
        for elem in range(2, max_val+1):
            if elem not in current_set:
                trial = current_set | {elem}
                if is_valid(trial):
                    current_set = trial
                    no_improve = 0
                    if len(current_set) > best_size:
                        best_set = current_set.copy()
                        best_size = len(best_set)
                    break
    
    assert is_valid(best_set), "Final invalid!"
    return list(best_set)
