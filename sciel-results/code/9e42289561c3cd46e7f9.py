import time
import random
import math
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict pairs: for each x, store set of (a,b) pairs that form bad triple with x
    conflict_pairs = {i: set() for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    # Convert to lists for faster iteration
    conflict_lists = {i: list(conflict_pairs[i]) for i in range(1, N + 1)}
    
    # Fast validity check: can we add x to set S (represented as boolean array)?
    def can_add(x, S_bool):
        for (a, b) in conflict_lists[x]:
            if S_bool[a] and S_bool[b]:
                return False
        return True

    def greedy_construct(ordering='random'):
        S = set()
        S_bool = [False] * (N + 1)
        
        if ordering == 'random':
            candidates = numbers[:]
            random.shuffle(candidates)
        elif ordering == 'ascending':
            candidates = numbers[:]
        elif ordering == 'descending':
            candidates = list(reversed(numbers))
        elif ordering == 'prime_first':
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in numbers if is_prime(x)]
            composites = [x for x in numbers if not is_prime(x)]
            random.shuffle(primes)
            random.shuffle(composites)
            candidates = primes + composites
        else:
            candidates = numbers[:]
            random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_bool):
                S.add(x)
                S_bool[x] = True
        
        return S

    def local_search(S, max_iters=5000):
        S = set(S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # 1. Try to add any element
            added = False
            not_in_S = [x for x in numbers if not S_bool[x]]
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    added = True
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                    break
            
            if not added:
                no_improve += 1
                
                # 2. Try swaps to escape local optimum
                if S and not_in_S:
                    improved = False
                    # Try a few random swaps
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        u = random.choice(list(S))
                        v = random.choice(not_in_S)
                        
                        # Check if v can be added if u is removed
                        S_bool[u] = False
                        S.remove(u)
                        
                        if can_add(v, S_bool):
                            S.add(v)
                            S_bool[v] = True
                            improved = True
                        else:
                            S.add(u)
                            S_bool[u] = True
                    
                    if not improved:
                        if no_improve > 20:
                            break
        
        return best_S, best_size

    best_S = set()
    best_size = 0
    
    # ILS Loop
    # Phase 1: Initial Greedy + LS
    for ordering in ['random', 'ascending', 'descending', 'prime_first']:
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        S, size = local_search(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 2: ILS Restart
    # We perturb the best solution or start fresh
    while time.time() < deadline - 2:
        # Perturb current best or start new
        if best_S:
            # Create a perturbed version of best_S
            S = set(best_S)
            # Remove 5-10 random elements
            k = random.randint(5, 10)
            if len(S) > k:
                to_remove = random.sample(list(S), k)
                for x in to_remove:
                    S.remove(x)
            # Add some random elements not in S
            not_in_S = [x for x in numbers if x not in S]
            if not_in_S:
                to_add = random.sample(not_in_S, min(5, len(not_in_S)))
                for x in to_add:
                    S.add(x)
                    # Note: S might be invalid now!
                    # We need to repair it.
            
            # Repair: Remove elements that cause conflicts?
            # Simpler: Just run local_search on the perturbed set.
            # But local_search assumes valid input?
            # The current local_search adds valid elements and swaps.
            # It does NOT remove elements to fix validity.
            # If we start with an invalid set, `can_add` might behave weirdly?
            # Actually, `can_add` checks if adding x creates a bad triple with existing elements.
            # If the existing set is already invalid, we have bad triples inside S.
            # The problem requires the final set to be valid.
            # So we must ensure S is valid at the end.
            
            # Strategy:
            # 1. Perturb by removing elements (keeps validity).
            # 2. Run Local Search to add elements back.
            
            S = set(best_S)
            k = random.randint(3, 8)
            if len(S) > k:
                to_remove = random.sample(list(S), k)
                for x in to_remove:
                    S.remove(x)
            
            S, size = local_search(S)
            
            if size > best_size:
                best_size = size
                best_S = S.copy()
        
        # Occasionally start fresh
        if random.random() < 0.1:
            S = greedy_construct('random')
            S, size = local_search(S)
            if size > best_size:
                best_size = size
                best_S = S.copy()

    return list(best_S)

# EVOLVE-BLOCK-END
