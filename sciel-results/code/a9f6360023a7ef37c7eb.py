"""Improved program for Large sets with Property P."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_valid(S):
    """Check if set S satisfies: for all a<b<=c in S, a does NOT divide b+c."""
    S_sorted = sorted(S)
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i + 1, n):
            b = S_sorted[j]
            for k in range(j, n):
                c = S_sorted[k]
                if (b + c) % a == 0:
                    return False
    return True

def _check_add(S, x):
    """Check if adding x to S keeps validity. S is a set, x not in S."""
    if x in S:
        return True
    S_list = sorted(S)
    # x can be a, b, or c
    # Case 1: x is the smallest (a), so x < b <= c for all b,c in S
    # Need: x does not divide b+c for all b<c in S with b>=x... but x < b so b >= x+1
    # Actually x < b <= c, so b >= x (but x < b means b >= x+1)
    # For all b,c in S with b < c and b >= x: x does not divide b+c
    # But also b could be x? No, x < b so b > x
    for i in range(len(S_list)):
        b = S_list[i]
        if b <= x:
            continue
        for j in range(i, len(S_list)):
            c = S_list[j]
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is in the middle or largest, so a < x <= c or a < b <= x
    # For a in S with a < x: need a does not divide x + c for all c in S with c >= x
    # and a does not divide b + x for all b in S with a < b <= x (but b in S, b > a, b <= x)
    # Actually: a < b <= c. If x is b: a < x <= c, need a not divide x+c
    # If x is c: a < b <= x, need a not divide b+x
    for a in S_list:
        if a >= x:
            break
        # x as b: a < x <= c, c in S, c >= x
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
        # x as c: a < b <= x, b in S, b > a, b <= x
        for b in S_list:
            if b <= a:
                continue
            if b > x:
                break
            if (b + x) % a == 0:
                return False
    
    return True

def _greedy_construct():
    """Greedy construction: start from upper tail, try to add elements from below."""
    # Upper tail [171, 256] is a good starting point
    S = set(range(171, 257))
    
    # Try adding elements from 1 to 170
    for x in range(1, 171):
        if _check_add(S, x):
            S.add(x)
    
    return S

def _local_search(S, max_iterations=50000, timeout=50.0):
    """Local search to improve the set."""
    start_time = time.time()
    best_S = set(S)
    best_size = len(best_S)
    
    all_elements = list(range(1, 257))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Try adding a random element not in S
        if len(best_S) < 256:
            not_in_S = [x for x in all_elements if x not in best_S]
            if not_in_S:
                x = random.choice(not_in_S)
                if _check_add(best_S, x):
                    best_S.add(x)
                    best_size = len(best_S)
        
        # Try swapping: remove one, add another
        if len(best_S) > 1:
            elements_in_S = list(best_S)
            x_to_remove = random.choice(elements_in_S)
            not_in_S = [x for x in all_elements if x not in best_S]
            if not_in_S:
                x_to_add = random.choice(not_in_S)
                candidate = best_S - {x_to_remove}
                if _check_add(candidate, x_to_add):
                    candidate.add(x_to_add)
                    if len(candidate) > best_size:
                        best_S = candidate
                        best_size = len(best_S)
                    elif random.random() < 0.1:  # occasional equal-size move
                        best_S = candidate
                        best_size = len(best_S)
        
        # Try removing one element and adding two
        if len(best_S) > 1:
            elements_in_S = list(best_S)
            x_to_remove = random.choice(elements_in_S)
            not_in_S = [x for x in all_elements if x not in best_S]
            if len(not_in_S) >= 2:
                x1, x2 = random.sample(not_in_S, 2)
                candidate = best_S - {x_to_remove}
                if _check_add(candidate, x1):
                    candidate.add(x1)
                    if _check_add(candidate, x2):
                        candidate.add(x2)
                        if len(candidate) > best_size:
                            best_S = candidate
                            best_size = len(best_S)
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # Start with greedy construction
    S = _greedy_construct()
    size_greedy = len(S)
    
    best_S = set(S)
    best_size = size_greedy
    
    random.seed(42)
    
    # Local search
    remaining = deadline - time.time()
    if remaining > 5:
        result = _local_search(S, max_iterations=200000, timeout=remaining - 3)
        if len(result) > best_size:
            best_S = result
            best_size = len(result)
    
    # Additional restarts if time permits
    remaining = deadline - time.time()
    restart_count = 0
    while remaining > 15 and restart_count < 8:
        # Perturb the best solution
        perturbed = set(best_S)
        for _ in range(random.randint(2, 5)):
            if perturbed:
                x_to_remove = random.choice(list(perturbed))
                perturbed.remove(x_to_remove)
            not_in_S = [x for x in range(1, 257) if x not in perturbed]
            if not_in_S:
                x_to_add = random.choice(not_in_S)
                perturbed.add(x_to_add)
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = _local_search(perturbed, max_iterations=100000, timeout=remaining - 5)
        if len(result) > best_size:
            best_S = result
            best_size = len(result)
        restart_count += 1
    
    # Final safety check
    if not _is_valid(best_S) and len(best_S) > 0:
        return list(range(171, 257))
    
    return list(best_S)
# EVOLVE-BLOCK-END
