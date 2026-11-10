import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies Sidon and Maximality.
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
        
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        if 2*a in sums:
            return False
        sums.add(2*a)
        for j in range(i+1, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # If x can be added, it's not maximal.
        # x can be added if 2x not in sums AND no x+a in sums for a in arr.
        
        if 2*x in sums:
            continue
            
        can_add = True
        for a in arr:
            if x + a in sums:
                can_add = False
                break
                
        if can_add:
            return False
            
    return True

def _greedy_bitmask(order):
    """
    Greedily constructs a maximal Sidon set using bitmask operations.
    'order' is a list of candidate elements in the order they should be considered.
    Returns (set_list, size).
    """
    # Bitmask for elements in the set (1 to N)
    # We use 1-indexed bits for simplicity: bit i represents element i
    elem_mask = 0 
    # Bitmask for sums. Sums range from 2 to 2N.
    # Max sum is 512. So we need bits up to 512.
    sum_mask = 0
    
    selected = []
    
    # Precompute shift amounts? No, dynamic shift is fast enough in Python for integers.
    
    for x in order:
        # Check if x is already in set
        if (elem_mask >> x) & 1:
            continue
            
        # Check Sidon property:
        # 1. 2x must not be in sum_mask
        # 2. x + a must not be in sum_mask for any a in selected.
        #    This is equivalent to checking if (sum_mask << x) & sum_mask == 0 ?
        #    Let S be the set of sums. We want to ensure (S + x) is disjoint from S.
        #    If we shift sum_mask left by x, bit i becomes bit i+x.
        #    If (sum_mask << x) & sum_mask is non-zero, there exists s in S such that s+x is also in S.
        #    This implies x+a = s and a = s-x? No.
        #    If s1, s2 in S and s2 = s1 + x, then s2 - s1 = x.
        #    Since s1, s2 are sums of elements in the set, this difference x implies a conflict.
        #    Specifically, if s1 = a+b and s2 = c+d, and s2-s1=x, it doesn't immediately mean x+a is a sum.
        #    Wait, the condition for adding x is:
        #    - 2x not in S
        #    - {x+a | a in A} disjoint from S
    
    # Let's refine the bitmask check.
    # Let M be the bitmask of sums.
    # When adding x, new sums are {2x} U {x+a | a in A}.
    # We need:
    # 1. 2x not in M.
    # 2. For all a in A, x+a not in M.
    #    This is equivalent to: (M << x) & M == 0 ? 
    #    If M has bit s set, M<<x has bit s+x set.
    #    If (M<<x) & M has bit s set, then M has bit s AND M has bit s-x.
    #    So s and s-x are both in M.
    #    This means there is a sum s and a sum s-x.
    #    Does this prevent adding x?
    #    If we add x, we add sums x+a.
    #    If x+a = s (where s in M), then conflict.
    #    So we need {x+a | a in A} disjoint from M.
    #    Let K = bitmask of elements A.
    #    The set of new sums is (K << x) plus bit 2x.
    #    We need ( (K << x) | (1 << 2x) ) & M == 0.
    
    # So the check is:
    # new_sums_mask = (elem_mask << x) | (1 << (2*x))
    # if new_sums_mask & sum_mask: continue
    
    # If valid:
    # sum_mask |= new_sums_mask
    # elem_mask |= (1 << x)
    # selected.append(x)
    
    # We also need to ensure that we don't add x if x itself is a sum? 
    # No, x being a sum doesn't prevent x from being in the set. 
    # E.g., {1, 2, 4} -> sums {2,3,4,5,6,8}. 2 is a sum. 2 is in set.
    # But 2+1=3 (in sums), 2+2=4 (in sums). So 2 cannot be added to {1,4}?
    # {1,4} sums {2,5,8}. 
    # Add 2: new sums {4, 3, 6}. 
    # 4 is in sums? No. 3 in sums? No. 6 in sums? No.
    # So 2 can be added to {1,4}. 
    # Check: (elem_mask << 2) | (1 << 4).
    # elem_mask for {1,4} is (1<<1) | (1<<4).
    # (elem_mask << 2) = (1<<3) | (1<<6).
    # sum_mask for {1,4} is (1<<2) | (1<<5) | (1<<8).
    # Intersection: 0. So valid.
    
    # Wait, is it sufficient?
    # We need 2x not in M. (1 << 2x) & M == 0.
    # We need x+a not in M for all a in A.
    # This is ( (elem_mask << x) & M ) == 0.
    
    # So total check: ( (elem_mask << x) | (1 << (2*x)) ) & sum_mask == 0.
    
    # This is O(1) bitwise ops on big ints. Very fast.
    
    for x in order:
        if (elem_mask >> x) & 1:
            continue
            
        new_sums = (elem_mask << x) | (1 << (2*x))
        if new_sums & sum_mask:
            continue
            
        # Valid
        sum_mask |= new_sums
        elem_mask |= (1 << x)
        selected.append(x)
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    rng = random.Random(42)
    
    best_set = None
    best_size = N + 1
    
    # We will perform random restarts.
    # Each restart generates a random order and runs greedy.
    # To find *small* sets, we want orders that lead to efficient blocking.
    # Random order is a baseline.
    # Heuristic: Sort by "potential to block". 
    # Elements with many multiples or sums with other likely elements?
    # Actually, just random shuffles with different seeds often find different local optima.
    
    # Also, we can try "greedy from small" and "greedy from large".
    
    # Strategy:
    # 1. Try a few fixed heuristic orders (ascending, descending, random).
    # 2. Then do random restarts with random shuffles.
    
    # Fixed heuristics
    orders = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        list(range(1, N+1, 2)), # Odds first
        list(range(2, N+1, 2)), # Evens first
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_bitmask(order)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Random restarts
    # How many can we do?
    # Bitmask ops are very fast. 
    # Let's estimate. 256 elements. 
    # Loop 256 times. Bitwise shift and AND on ~512 bit int.
    # Should be microseconds. 
    # We can do thousands of restarts.
    
    # To diversify, we can also bias the order.
    # E.g., shuffle, but keep some structure?
    # Or just pure random.
    
    while time.time() < deadline:
        # Generate random order
        order = list(range(1, N+1))
        rng.shuffle(order)
        
        # Optional: Sort by a random key to create different "greedy pressures"
        # For now, pure random shuffle is a good exploration of the search space.
        
        s = _greedy_bitmask(order)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
            # If we find a very small set (e.g., < 20), we might want to verify more aggressively
            # or stop if we hit a theoretical lower bound? 
            # Lower bound for maximal Sidon set in [1,N] is roughly sqrt(N) * constant?
            # Actually, maximal Sidon sets are quite small. 
            # For N=256, typical maximal Sidon sets are size ~15-25.
            # If we find 15, it's likely optimal or near-optimal.
            if best_size <= 15:
                # We can continue to see if we get smaller, but it's rare.
                # Let's continue for a bit longer to be safe, but maybe slow down?
                # No, just keep going.
                pass

    # Final verification of the best set found
    if best_set and not _is_valid_solution(best_set):
        # Fallback to a known valid construction if verification fails (shouldn't happen)
        # Re-run a simple greedy
        order = list(range(1, N+1))
        best_set = _greedy_bitmask(order)
        
    return best_set

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
