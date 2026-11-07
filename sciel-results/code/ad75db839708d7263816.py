"""Combined program: greedy construction with local search refinement."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _can_add(S_list, S_set, x):
    """
    Check if x can be added to S.
    S_list is list(S), S_set is S as a set. x is an int not in S.
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
                
    return True

def _is_valid(S_list):
    """
    Check if S is a valid set.
    For every three distinct members a, b, c in S, a does not divide b*c.
    """
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_construct(order, deadline):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    S_list = []
    
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S_list, S, x):
            S.add(x)
            S_list.append(x)
            
    return S, S_list

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    start_time = time.time()
    
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple Greedy Constructions
    # Try specific orders first
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return False
        return True
            
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    # Add random orders
    for _ in range(30):
        random.shuffle(all_nums)
        orders.append(all_nums.copy())
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S, S_list = _greedy_construct(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 2: Local Search on the best set
    # We try to improve best_set using adds and swaps.
    current_set = best_set.copy()
    current_list = list(current_set)
    current_size = best_size
    
    # Precompute non-members for faster selection
    non_members = [x for x in all_nums if x not in current_set]
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0:
        if not non_members:
            break
            
        improved = False
        
        # Try random Add
        if random.random() < 0.7:
            # Try to add a random non-member
            idx = random.randint(0, len(non_members) - 1)
            x = non_members[idx]
            if _can_add(current_list, current_set, x):
                current_set.add(x)
                current_list.append(x)
                current_size += 1
                non_members.pop(idx)
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Try a Swap: remove random y from current_set, add random x from non_members
            if len(current_set) > 0:
                y = random.choice(current_list)
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                
                # Check if x can be added to S_temp = current_set - {y}
                # We need to check if x is compatible with current_list excluding y
                temp_list = [v for v in current_list if v != y]
                temp_set = set(temp_list)
                
                if _can_add(temp_list, temp_set, x):
                    # Swap accepted
                    current_set.remove(y)
                    current_set.add(x)
                    current_list = temp_list + [x]
                    # Update non_members
                    non_members[idx] = y
                    
                    # Size is unchanged, but structure changed.
                    # This might allow future adds.
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart with a different random seed or order
            no_improve_count = 0
            # Try to reset to best and start fresh local search
            if random.random() < 0.5:
                current_set = best_set.copy()
                current_list = list(current_set)
                current_size = best_size
                non_members = [x for x in all_nums if x not in current_set]
    
    # Final validation and cleanup
    result = list(best_set)
    # Ensure types are correct
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
