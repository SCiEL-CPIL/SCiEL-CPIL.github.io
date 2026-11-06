import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute bad triples and conflicts
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a, b, c))

    # conflicts[x] = list of (u, v) such that (u, v, x) is a bad triple
    conflicts = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # Precompute conflict sets for faster membership testing
    conflict_sets = {x: set() for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for (u, v) in conflicts[x]:
            conflict_sets[x].add((u, v))
            conflict_sets[x].add((v, u))

    def is_valid_add(S_set, x):
        for (u, v) in conflicts[x]:
            if u in S_set and v in S_set:
                return False
        return True

    # Initial Solution: Greedy with multiple strategies
    random.seed(42)
    best_set = set()
    best_size = 0
    
    # Compute degree (number of conflicts) for each element
    degree = {x: len(conflicts[x]) for x in range(1, N + 1)}
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: degree[x]),
        sorted(range(1, N + 1), key=lambda x: -degree[x]),
        sorted(range(1, N + 1), key=lambda x: (degree[x], -x)),
        sorted(range(1, N + 1), key=lambda x: (-degree[x], x)),
    ]
    
    # Add random orders
    for _ in range(15):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    # Also try prime-heavy and composite-heavy orders
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(1, N + 1) if is_prime(x)]
    composites = [x for x in range(1, N + 1) if not is_prime(x)]
    orders.append(primes + composites)
    orders.append(composites + primes)

    for order in orders:
        S = set()
        for x in order:
            if is_valid_add(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    S = best_set.copy()
    current_size = len(S)

    # Optimized local search functions
    def try_add_one(S):
        """Try to add one element. Return new S if successful, else None."""
        for x in range(1, N + 1):
            if x not in S:
                if is_valid_add(S, x):
                    new_S = S.copy()
                    new_S.add(x)
                    return new_S
        return None

    def try_swap_1_for_2(S):
        """Try to remove 1 element and add 2. Optimized version."""
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        for u in S_list:
            if time.time() > deadline:
                return None
            temp_S = S - {u}
            # Find all candidates that can be added to temp_S
            valid_candidates = []
            for x in candidates:
                if is_valid_add(temp_S, x):
                    valid_candidates.append(x)
            
            # Try all pairs from valid_candidates
            if len(valid_candidates) >= 2:
                for i in range(len(valid_candidates)):
                    x = valid_candidates[i]
                    temp_S_with_x = temp_S | {x}
                    for j in range(i + 1, len(valid_candidates)):
                        y = valid_candidates[j]
                        if is_valid_add(temp_S_with_x, y):
                            new_S = temp_S | {x, y}
                            return new_S
        return None

    def try_swap_2_for_3(S):
        """Try to remove 2 elements and add 3. Limited version for time efficiency."""
        if time.time() > deadline - 2.0:
            return None
        
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        # Limit the search to avoid timeout
        max_pairs_to_check = 20
        
        for pair_idx, (u, v) in enumerate(combinations(S_list, 2)):
            if pair_idx >= max_pairs_to_check:
                break
            if time.time() > deadline - 1.0:
                break
            
            temp_S = S - {u, v}
            # Find candidates that can be added to temp_S
            valid_candidates = []
            for x in candidates:
                if is_valid_add(temp_S, x):
                    valid_candidates.append(x)
            
            # Try all triples from valid_candidates (limited)
            if len(valid_candidates) >= 3:
                # Limit triple search
                max_triples = 50
                triple_count = 0
                for i in range(len(valid_candidates)):
                    if triple_count >= max_triples:
                        break
                    x = valid_candidates[i]
                    temp_S_with_x = temp_S | {x}
                    for j in range(i + 1, len(valid_candidates)):
                        if triple_count >= max_triples:
                            break
                        y = valid_candidates[j]
                        temp_S_with_xy = temp_S_with_x | {y}
                        for k in range(j + 1, len(valid_candidates)):
                            if triple_count >= max_triples:
                                break
                            z = valid_candidates[k]
                            if is_valid_add(temp_S_with_xy, z):
                                new_S = temp_S | {x, y, z}
                                return new_S
                            triple_count += 1
        return None

    # VNS Loop with multiple neighborhoods
    stuck_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # N0: Add 1
        new_S = try_add_one(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N1: 1-for-2
        new_S = try_swap_1_for_2(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N2: 2-for-3
        new_S = try_swap_2_for_3(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # If stuck, try a 1-for-1 swap to change the structure (size remains same)
        improved_structure = False
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        for u in S_list:
            if time.time() > deadline:
                break
            temp_S = S - {u}
            for v in candidates:
                if is_valid_add(temp_S, v):
                    # This swap is valid
                    S = temp_S | {v}
                    improved_structure = True
                    break
            if improved_structure:
                break
        
        if improved_structure:
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 15 or no_improve_count > 50:
                # Random restart from best with perturbation
                S = best_set.copy()
                stuck_count = 0
                no_improve_count = 0
                
                # Apply 1-3 random 1-for-1 swaps to perturb
                S_list = list(S)
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    if not S_list:
                        break
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if v not in S:
                        temp_S = S - {u}
                        if is_valid_add(temp_S, v):
                            S = temp_S | {v}
                            S_list = list(S)

    return list(best_set)
