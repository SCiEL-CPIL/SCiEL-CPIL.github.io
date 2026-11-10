import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute conflicts
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            ga_x = gcds[a][x]
            for b in range(a + 1, N + 1):
                if b == x: continue
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    conflicts[x].add((a, b))
                    
    def is_valid_addition(S_set, x):
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True

    # Build a strong initial set using structural properties
    def build_structural_set():
        # Include 1 (gcd(1,x)=1 for all x, so it's safe with most)
        S = {1}
        
        # Add prime powers: for each prime p, add p^k where p^k <= 128
        # This ensures that numbers sharing a prime factor have GCD that's a power of that prime
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
        
        for p in primes:
            power = p
            while power <= N:
                if is_valid_addition(S, power):
                    S.add(power)
                power *= p
        
        # Add numbers with unique prime factors (squarefree with distinct primes)
        # These have GCD 1 with most other numbers
        for x in range(2, N + 1):
            if x in S:
                continue
            # Check if x is squarefree
            is_squarefree = True
            temp = x
            for p in primes:
                if p * p > temp:
                    break
                if temp % (p * p) == 0:
                    is_squarefree = False
                    break
            if is_squarefree and is_valid_addition(S, x):
                S.add(x)
        
        return S

    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    best_set = build_structural_set()
    best_size = len(best_set)
    
    def greedy_init():
        S = set()
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def local_search(S, max_iter=10000):
        S = set(S)
        current_size = len(S)
        no_improve = 0
        local_best_size = current_size
        local_best_set = set(S)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add an element
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                random.shuffle(not_in_S)
                added = False
                for x in not_in_S:
                    if is_valid_addition(S, x):
                        S.add(x)
                        current_size += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_set = set(S)
                        no_improve = 0
                        added = True
                        break
                
                if added:
                    continue
            
            # If no add, try a swap (remove u, add v)
            if S and not_in_S:
                # Try random swaps
                for _ in range(3):
                    u = random.choice(list(S))
                    v = random.choice(not_in_S)
                    if u == v:
                        continue
                    
                    # Check if removing u and adding v is valid
                    S.remove(u)
                    if is_valid_addition(S, v):
                        S.add(v)
                        if len(S) > local_best_size:
                            local_best_size = len(S)
                            local_best_set = set(S)
                        no_improve = 0
                        break
                    else:
                        S.add(u)  # Restore
                
                if no_improve > 30:
                    # Kick: remove a random element
                    if S:
                        u = random.choice(list(S))
                        S.remove(u)
                        current_size -= 1
                        no_improve = 0
            else:
                no_improve += 1
        
        return local_best_set

    # Run local search from the structural set
    improved = local_search(best_set, max_iter=50000)
    if len(improved) > len(best_set):
        best_set = improved
    
    # Run multiple restarts with different strategies
    for run in range(100):
        if time.time() > deadline - 5:
            break
            
        # Alternate between structural and random init
        if run % 3 == 0:
            S = build_structural_set()
        elif run % 3 == 1:
            S = greedy_init()
        else:
            # Hybrid: start with structural, then random additions
            S = build_structural_set()
            elems = [x for x in elements if x not in S]
            random.shuffle(elems)
            for x in elems:
                if is_valid_addition(S, x):
                    S.add(x)
        
        S = local_search(S, max_iter=20000)
        
        if len(S) > len(best_set):
            best_set = set(S)
            best_size = len(best_set)
    
    # Final polish
    not_in_best = [x for x in elements if x not in best_set]
    random.shuffle(not_in_best)
    for x in not_in_best:
        if is_valid_addition(best_set, x):
            best_set.add(x)

    return list(best_set)
