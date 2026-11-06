"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 192
    MAX_K = 3
    
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    # Precompute primes
    primes = set(x for x in range(1, N + 1) if is_prime(x))
    
    def can_add_fast(x, S):
        """
        Check if x can be added to set S.
        S is a list of selected elements.
        Constraint: no element a in S ∪ {x} divides the sum of any 3 distinct OTHER elements in S ∪ {x}.
        """
        n = len(S)
        
        # Check 1: x does not divide sum of any 3 distinct elements in S
        if n >= 3:
            for i in range(n):
                a = S[i]
                for j in range(i + 1, n):
                    b = S[j]
                    for k in range(j + 1, n):
                        c = S[k]
                        if (a + b + c) % x == 0:
                            return False
        
        # Check 2: For each a in S, a does not divide (x + u + v) for distinct u,v in S\{a}
        # (x + u + v) % a == 0  <=>  (u + v) % a == (-x) % a
        for idx_a, a in enumerate(S):
            target = (-x) % a
            # Check all pairs (u,v) from S\{a}
            for i in range(n):
                if i == idx_a:
                    continue
                u = S[i]
                for j in range(i + 1, n):
                    if j == idx_a:
                        continue
                    v = S[j]
                    if (u + v) % a == target:
                        return False
        
        return True
    
    def greedy(order, deadline):
        """Greedy construction with fast validity checks."""
        S = []
        S_set = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            if can_add_fast(x, S):
                S.append(x)
                S_set.add(x)
        
        return S
    
    def local_search(S, deadline):
        """Local search with 1-in additions and 1-out-1-in swaps."""
        S = list(S)
        S_set = set(S)
        candidates = [x for x in range(1, N + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try additions
            for x in candidates:
                if time.time() > deadline:
                    return S
                if can_add_fast(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-out-1-in swaps
            S_copy = S[:]
            random.shuffle(S_copy)
            for a in S_copy:
                if time.time() > deadline:
                    return S
                S_temp = [s for s in S if s != a]
                S_temp_set = set(S_temp)
                
                for x in candidates:
                    if time.time() > deadline:
                        return S
                    if x in S_temp_set:
                        continue
                    if can_add_fast(x, S_temp):
                        S = S_temp + [x]
                        S_set = S_temp_set | {x}
                        candidates = [c for c in candidates if c != a]
                        candidates.append(a)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    best_set = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Strategy 1: Descending order
    if time.time() < deadline:
        order = list(range(N, 0, -1))
        S = greedy(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy 2: Primes first (descending)
    if time.time() < deadline:
        prime_list = sorted([x for x in range(1, N + 1) if x in primes], reverse=True)
        non_prime_list = sorted([x for x in range(1, N + 1) if x not in primes], reverse=True)
        rng.shuffle(non_prime_list)
        order = prime_list + non_prime_list
        S = greedy(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy 3: Multiple random restarts with varied orderings
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if i % 6 == 0:
            # Large numbers with random perturbation
            order = list(range(N, 0, -1))
            rng.shuffle(order)
            order.sort(key=lambda x: -x + rng.random() * 30)
        elif i % 6 == 1:
            # Primes first, then composites by size
            prime_list = sorted([x for x in range(1, N + 1) if x in primes], reverse=True)
            non_prime_list = [x for x in range(1, N + 1) if x not in primes]
            rng.shuffle(non_prime_list)
            order = prime_list + non_prime_list
        elif i % 6 == 2:
            # Random
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif i % 6 == 3:
            # Ascending
            order = list(range(1, N + 1))
        elif i % 6 == 4:
            # Large composites first (they're "safe" because they divide few things)
            large_composites = [x for x in range(N, 0, -1) if x not in primes]
            small_primes = sorted([x for x in range(1, N + 1) if x in primes])
            rng.shuffle(small_primes)
            order = large_composites + small_primes
        else:
            # Weighted random: bias toward large numbers
            weights = [x for x in range(1, N + 1)]
            order = list(range(1, N + 1))
            rng.shuffle(order)
            order.sort(key=lambda x: -x**0.5 + rng.random() * 10)
        
        S = greedy(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            # Try local search
            if best_size < 48 and time.time() < deadline - 3.0:
                local_deadline = min(deadline, time.time() + 1.5)
                improved_S = local_search(best_set, local_deadline)
                if len(improved_S) > best_size:
                    best_size = len(improved_S)
                    best_set = improved_S[:]
    
    # Final local search
    if best_size < 48 and time.time() < deadline:
        improved_S = local_search(best_set, deadline)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_set = improved_S[:]
    
    return best_set
# EVOLVE-BLOCK-END
