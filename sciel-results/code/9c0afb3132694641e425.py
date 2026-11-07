"""Combined program: constraint-aware greedy with local search refinement."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from itertools import combinations

def _precompute_conflicts():
    """
    Precompute for each element a, which pairs (b,c) with b!=c cause a | b*c.
    conflicts[a] = set of frozensets {b,c} where b!=c and a | b*c
    """
    N = 256
    conflicts = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(2, N + 1):
            for c in range(b + 1, N + 1):
                if (b * c) % a == 0:
                    conflicts[a].add(frozenset((b, c)))
    return conflicts

def _can_add_fast(a, in_set, conflicts):
    """
    Check if a can be added to set represented by in_set (bool list).
    Uses precomputed conflicts for speed.
    """
    # Check: for each b in S, is there c in S (c != b) such that a | b*c?
    # Equivalently: for each b in S, check if any c in S (c != b) is in the conflict set
    for b in range(2, 257):
        if not in_set[b]:
            continue
        # Check if a divides b*c for some c in S, c != b
        g = gcd(a, b)
        k1 = a // g
        # Need c in S, c != b, such that k1 | c
        # Also need to check b | a*c for some c in S, c != b
        k2 = b // g
        # For k1 | c: check multiples of k1 in S
        for c in range(k1, 257, k1):
            if c == b:
                continue
            if in_set[c]:
                return False
        # For k2 | c: check multiples of k2 in S
        for c in range(k2, 257, k2):
            if c == b:
                continue
            if in_set[c]:
                return False
    return True

def _greedy_construct(order, in_set, deadline):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S_list = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(x, in_set, None):
            in_set[x] = True
            S_list.append(x)
    return S_list

def _local_search(S_list, in_set, deadline, max_iters=50000):
    """
    Local search to improve the set.
    Tries to add elements and swap elements to increase size.
    """
    current_size = len(S_list)
    non_members = [x for x in range(2, 257) if not in_set[x]]
    
    no_improve = 0
    max_no_improve = 200
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        if not non_members:
            break
            
        improved = False
        
        # Try random add (70% of the time)
        if random.random() < 0.7:
            if non_members:
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                if _can_add_fast(x, in_set, None):
                    in_set[x] = True
                    S_list.append(x)
                    current_size += 1
                    non_members.pop(idx)
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1
        else:
            # Try swap: remove y, add x
            if S_list and non_members:
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                x_idx = random.randint(0, len(non_members) - 1)
                x = non_members[x_idx]
                
                # Temporarily remove y
                in_set[y] = False
                S_list_temp = S_list[:y_idx] + S_list[y_idx+1:]
                
                if _can_add_fast(x, in_set, None):
                    # Accept swap
                    in_set[x] = True
                    S_list = S_list_temp + [x]
                    # Update non_members
                    non_members[x_idx] = y
                    no_improve = 0
                    improved = True
                else:
                    # Revert
                    in_set[y] = True
                    no_improve += 1
        
        if no_improve > max_no_improve:
            no_improve = 0
            # Occasionally restart from a fresh random order
            if random.random() < 0.1:
                break
    
    return S_list, in_set, current_size

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    all_nums = list(range(2, N + 1))
    
    # Precompute primes for ordering
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0: return False
        return True
    
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    best_set = set()
    best_size = 0
    best_in_set = [False] * (N + 1)
    
    # Phase 1: Multiple greedy constructions with different orderings
    orders = [
        list(range(2, N + 1)),
        list(range(N, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    # Add random orderings
    for _ in range(20):
        random.shuffle(all_nums)
        orders.append(all_nums.copy())
    
    for order in orders:
        if time.time() > deadline:
            break
        
        in_set = [False] * (N + 1)
        S_list = _greedy_construct(order, in_set, deadline)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = set(S_list)
            best_in_set = in_set.copy()
    
    # Phase 2: Local search on the best set
    if best_set:
        S_list = list(best_set)
        in_set = best_in_set.copy()
        S_list, in_set, current_size = _local_search(S_list, in_set, deadline)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(S_list)
            best_in_set = in_set.copy()
    
    # Phase 3: Try a few more restarts if time permits
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        
        random.shuffle(all_nums)
        in_set = [False] * (N + 1)
        S_list = _greedy_construct(all_nums, in_set, deadline)
        
        if len(S_list) > best_size:
            S_list, in_set, current_size = _local_search(S_list, in_set, deadline)
            if current_size > best_size:
                best_size = current_size
                best_set = set(S_list)
                best_in_set = in_set.copy()
    
    # Final validation
    result = list(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
