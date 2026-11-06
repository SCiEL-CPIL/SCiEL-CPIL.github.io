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
    
    def is_valid(S_bool):
        """Check if the current set is valid (no bad triples)"""
        for x in range(1, N + 1):
            if S_bool[x]:
                for (a, b) in conflict_lists[x]:
                    if S_bool[a] and S_bool[b] and a != b and a != x and b != x:
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
        elif ordering == 'divisor_count':
            # Sort by number of divisors (fewer divisors = more likely to be coprime to others)
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 1
                        if i != n // i:
                            count += 1
                return count
            candidates = sorted(numbers, key=lambda x: num_divisors(x))
            random.shuffle(candidates)
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
                    # Try more random swaps with better neighborhood
                    S_list = list(S)
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        u = random.choice(S_list)
                        v = random.choice(not_in_S)
                        
                        # Check if v can be added if u is removed
                        S_bool[u] = False
                        S.remove(u)
                        
                        if can_add(v, S_bool):
                            S.add(v)
                            S_bool[v] = True
                            improved = True
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
                        else:
                            S.add(u)
                            S_bool[u] = True
                        
                        if improved:
                            no_improve = 0
                            break
                    
                    if not improved:
                        if no_improve > 30:
                            break
        
        return best_S, best_size

    def simulated_annealing(S, T0=1.0, cooling=0.999, max_iters=20000):
        """Simulated annealing with swap moves"""
        S = set(S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        
        T = T0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate a random move: swap u in S with v not in S
            if not S:
                break
                
            S_list = list(S)
            not_in_S = [x for x in numbers if not S_bool[x]]
            if not not_in_S:
                break
            
            u = random.choice(S_list)
            v = random.choice(not_in_S)
            
            # Try the swap
            S_bool[u] = False
            S.remove(u)
            
            # Check if v can be added
            if can_add(v, S_bool):
                S.add(v)
                S_bool[v] = True
                new_size = len(S)
                
                # Accept if better or with probability exp(-delta/T)
                delta = current_size - new_size  # negative if improvement
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
                    # Reject: undo
                    S.remove(v)
                    S_bool[v] = False
                    S.add(u)
                    S_bool[u] = True
            else:
                # v can't be added, undo
                S.add(u)
                S_bool[u] = True
            
            T *= cooling
        
        return best_S, best_size

    best_S = set()
    best_size = 0
    
    # Phase 1: Initial Greedy + LS with multiple orderings
    for ordering in ['random', 'ascending', 'descending', 'prime_first', 'divisor_count']:
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        S, size = local_search(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 2: Simulated Annealing on best solution
    if best_S and time.time() < deadline - 5:
        S, size = simulated_annealing(best_S)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 3: ILS with perturbation
    while time.time() < deadline - 3:
        if best_S:
            # Create a perturbed version of best_S
            S = set(best_S)
            k = random.randint(3, 8)
            if len(S) > k:
                to_remove = random.sample(list(S), k)
                for x in to_remove:
                    S.remove(x)
            
            # Run local search on perturbed set
            S, size = local_search(S)
            
            # Optionally run SA on the result
            if size >= best_size - 2 and time.time() < deadline - 5:
                S, size = simulated_annealing(S, T0=0.5, cooling=0.9995, max_iters=10000)
            
            if size > best_size:
                best_size = size
                best_S = S.copy()
        
        # Occasionally start fresh
        if random.random() < 0.15:
            ordering = random.choice(['random', 'prime_first', 'divisor_count'])
            S = greedy_construct(ordering)
            S, size = local_search(S)
            if size > best_size:
                best_size = size
                best_S = S.copy()

    # Final polish: try to add any remaining elements
    if best_S:
        S_bool = [False] * (N + 1)
        for x in best_S:
            S_bool[x] = True
        for x in numbers:
            if not S_bool[x] and can_add(x, S_bool):
                best_S.add(x)
                S_bool[x] = True

    return list(best_S)

# EVOLVE-BLOCK-END
