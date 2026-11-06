import math
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # Safety margin
    N = 511
    C = 512
    
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Weights: w[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
    
    # Precompute prime factors for each number
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)
    
    # Build conflict masks using bitsets (use Python ints as arbitrary-precision bitsets)
    # conflict_mask[a] has bit b set if a and b share a prime factor
    conflict_mask = [0] * (N + 1)
    for a in range(2, N + 1):
        if not pf[a]:
            continue
        for b in range(a + 1, N + 1):
            if not pf[b]:
                continue
            common = False
            for p in pf[a]:
                if p in pf[b]:
                    common = True
                    break
            if common:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def get_score(selected):
        """selected is a set of vertices"""
        return sum(w[a] for a in selected)
    
    def can_add_to_set(S_set, a):
        """Check if a can be added to S_set"""
        for b in S_set:
            if (conflict_mask[b] >> a) & 1:
                return False
        return True
    
    # Strategy 1: Greedy prime-matching heuristic
    def greedy_prime_matching():
        S = set()
        S.add(1)
        
        prime_max_weight = {}
        for p in primes:
            max_w = 0.0
            for multiple in range(p, N + 1, p):
                if w[multiple] > max_w:
                    max_w = w[multiple]
            prime_max_weight[p] = max_w
        
        sorted_primes = sorted(primes, key=lambda p: -prime_max_weight[p])
        
        for p in sorted_primes:
            best_a = None
            best_w = -1
            for a in range(N, p - 1, -1):
                if a % p != 0:
                    continue
                if a in S:
                    continue
                if can_add_to_set(S, a):
                    if w[a] > best_w:
                        best_w = w[a]
                        best_a = a
            if best_a is not None:
                S.add(best_a)
        
        return S
    
    # Strategy 2: Standard greedy by weight
    def greedy_by_weight():
        S = set()
        S.add(1)
        for a in range(N, 1, -1):
            if a in S:
                continue
            if can_add_to_set(S, a):
                S.add(a)
        return S
    
    # Get initial solutions from both strategies
    S_pm = greedy_prime_matching()
    S_gw = greedy_by_weight()
    
    best_score = max(get_score(S_pm), get_score(S_gw))
    best_S = S_pm.copy() if get_score(S_pm) >= get_score(S_gw) else S_gw.copy()
    
    # Branch and bound for Maximum Weight Independent Set
    # Vertices are 1..511, edges connect non-coprime pairs
    # Use bitset representation for candidate sets
    
    def branch_and_bound(initial_set, time_limit):
        """
        Branch and bound to find maximum weight independent set.
        Uses the initial_set as a lower bound.
        Returns the best set found.
        """
        best_set = initial_set.copy()
        best_weight = get_score(initial_set)
        
        # Order vertices by weight descending for branching
        vertices_by_weight = sorted(range(1, N + 1), key=lambda a: -w[a])
        
        # Precompute for each vertex, the set of vertices it conflicts with (as bitmask)
        # conflict_mask[a] already has this info
        
        # We'll use a recursive approach with pruning
        # State: (available_vertices_bitmask, current_set, current_weight)
        # available_vertices_bitmask: bit a is set if a is available (not excluded by conflicts with current_set)
        
        def compute_available(current_set):
            """Compute bitmask of vertices that can be added to current_set"""
            avail = 0
            for a in range(1, N + 1):
                if a in current_set:
                    continue
                if can_add_to_set(current_set, a):
                    avail |= (1 << a)
            return avail
        
        # Upper bound: sum of weights of all available vertices
        # (This is a loose bound but easy to compute)
        def upper_bound(avail_mask):
            ub = 0.0
            for a in range(1, N + 1):
                if (avail_mask >> a) & 1:
                    ub += w[a]
            return ub
        
        # Better upper bound: for each prime, only one vertex divisible by it can be chosen
        # Group available vertices by their "dominant" prime factor
        def tighter_upper_bound(avail_mask):
            # For each prime, sum the max weight of available vertices divisible by it
            # But a vertex might be divisible by multiple primes, so this overcounts
            # Actually, we need: for each prime p, at most one vertex divisible by p
            # So UB = sum over primes p of (max weight of available vertex divisible by p)
            # This is still an overestimate because a vertex counted for p1 might also be divisible by p2
            
            # Simpler: just use the sum of all available weights (loose but fast)
            return upper_bound(avail_mask)
        
        nodes_explored = 0
        
        def bnb(avail_mask, current_set, current_weight, depth):
            nonlocal best_set, best_weight, nodes_explored, deadline
            
            if time.time() > deadline:
                return
            
            nodes_explored += 1
            
            # If no available vertices, update best if improved
            if avail_mask == 0:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_set = current_set.copy()
                return
            
            # Upper bound check
            ub = upper_bound(avail_mask)
            if ub <= best_weight:
                return
            
            # Find the vertex with highest weight in avail_mask
            # Branch on whether to include it or not
            
            # Find highest-weight available vertex
            best_v = -1
            best_v_w = -1
            for a in vertices_by_weight:
                if (avail_mask >> a) & 1:
                    best_v = a
                    best_v_w = w[a]
                    break
            
            if best_v == -1:
                return
            
            # Option 1: Include best_v
            new_set = current_set.copy()
            new_set.add(best_v)
            new_weight = current_weight + best_v_w
            
            # Compute new available mask (remove best_v and all its conflicts)
            new_avail = avail_mask & ~(1 << best_v)
            # Remove all vertices that conflict with best_v
            conflicts = conflict_mask[best_v]
            new_avail &= ~conflicts
            
            bnb(new_avail, new_set, new_weight, depth + 1)
            
            # Option 2: Exclude best_v
            new_avail2 = avail_mask & ~(1 << best_v)
            bnb(new_avail2, current_set, current_weight, depth + 1)
        
        # Start with empty set, all vertices available
        initial_avail = 0
        for a in range(1, N + 1):
            initial_avail |= (1 << a)
        
        bnb(initial_avail, set(), 0.0, 0)
        
        return best_set, best_weight
    
    # Run branch and bound with time limit
    bnb_time_limit = deadline - 30.0  # Leave 30s for other methods
    if time.time() < bnb_time_limit:
        S_bnb, score_bnb = branch_and_bound(best_S, bnb_time_limit)
        if score_bnb > best_score:
            best_score = score_bnb
            best_S = S_bnb
    
    # Multi-swap hill climbing with simulated annealing
    rng = random.Random(42)
    all_vertices = list(range(1, N + 1))
    
    def multi_swap_hill_climb(S_init, time_limit):
        S = S_init.copy()
        current_score = get_score(S)
        best_S_local = S.copy()
        best_score_local = current_score
        
        T = 0.05
        T_min = 1e-9
        cooling = 0.999
        
        step = 0
        while time.time() < time_limit and T > T_min:
            step += 1
            
            r = rng.random()
            if r < 0.6:
                swap_size = 1
            elif r < 0.9:
                swap_size = 2
            else:
                swap_size = 3
            
            if len(S) < swap_size:
                T *= cooling
                continue
            
            to_remove = rng.sample(list(S), swap_size)
            new_S = S.copy()
            for x in to_remove:
                new_S.discard(x)
            
            candidates = []
            for a in all_vertices:
                if a in new_S:
                    continue
                if can_add_to_set(new_S, a):
                    candidates.append(a)
            
            if len(candidates) < swap_size:
                T *= cooling
                continue
            
            candidates.sort(key=lambda a: -w[a])
            
            best_addition = None
            best_addition_score = -1
            
            top_k = min(15, len(candidates))
            if swap_size == 1:
                if candidates:
                    best_addition = [candidates[0]]
                    best_addition_score = w[candidates[0]]
            elif swap_size == 2:
                for i in range(top_k):
                    a1 = candidates[i]
                    for j in range(i + 1, top_k):
                        a2 = candidates[j]
                        if (conflict_mask[a1] >> a2) & 1:
                            continue
                        if can_add_to_set(new_S, a1) and can_add_to_set(new_S | {a1}, a2):
                            score = w[a1] + w[a2]
                            if score > best_addition_score:
                                best_addition_score = score
                                best_addition = [a1, a2]
            elif swap_size == 3:
                top_k3 = min(8, len(candidates))
                for i in range(top_k3):
                    a1 = candidates[i]
                    for j in range(i + 1, top_k3):
                        a2 = candidates[j]
                        if (conflict_mask[a1] >> a2) & 1:
                            continue
                        for k in range(j + 1, top_k3):
                            a3 = candidates[k]
                            if (conflict_mask[a1] >> a3) & 1 or (conflict_mask[a2] >> a3) & 1:
                                continue
                            if can_add_to_set(new_S, a1) and can_add_to_set(new_S | {a1}, a2) and can_add_to_set(new_S | {a1, a2}, a3):
                                score = w[a1] + w[a2] + w[a3]
                                if score > best_addition_score:
                                    best_addition_score = score
                                    best_addition = [a1, a2, a3]
            
            if best_addition is None:
                T *= cooling
                continue
            
            removed_score = sum(w[x] for x in to_remove)
            added_score = best_addition_score
            delta = added_score - removed_score
            
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                for a in best_addition:
                    new_S.add(a)
                S = new_S
                current_score += delta
                
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_S_local = S.copy()
            
            T *= cooling
        
        return best_S_local, best_score_local
    
    # Run multi-swap hill climbing from current best
    remaining_time = deadline - time.time() - 10.0
    if remaining_time > 5.0:
        S1, score1 = multi_swap_hill_climb(best_S, time.time() + remaining_time)
        if score1 > best_score:
            best_score = score1
            best_S = S1
    
    # Final polish: try single-add improvements
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for a in all_vertices:
            if a in best_S:
                continue
            if can_add_to_set(best_S, a):
                best_S.add(a)
                improved = True
    
    return list(best_S)
