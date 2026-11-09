import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # safety margin before 90s kill

    # Precompute conflict structure
    # For each candidate a (as the divisor), and each pair (b,c) with a<b<=c,
    # a conflicts if a divides b+c.
    # We'll precompute for each a, the set of (b,c) pairs that would violate.
    # Actually, for fast checking, let's precompute:
    # conflict[a] = set of values v such that there exists b<=c in set with a<b<=c and a|(b+c)
    # This is complex. Instead, let's use a different approach.

    # Precompute: for each a in 1..n, for each b in a+1..n, for each c in b..n:
    #   if (b+c) % a == 0, then (a,b,c) is a forbidden triple.
    # Store as: for each a, a set of "conflict pairs" (b,c) where a<b<=c and a|(b+c).
    # Then checking if a can be added to set S: for each b in S with b>a, for each c in S with c>=b,
    #   check if (b,c) in conflict_pairs[a].

    # Actually, let me think differently. For a given set S, to check if adding x is valid:
    # We need to check all triples involving x.
    # Let's precompute for each x, the list of (b,c) pairs (b<=c, b>x or x in {b,c}) that conflict.

    # Precompute conflict_pairs[x] = set of (b,c) with x<b<=c and x|(b+c), 
    #   plus (a,c) pairs where a<x<=c and a|(x+c) [x is b],
    #   plus (a,b) pairs where a<b<=x and a|(b+x) [x is c].

    # This is getting complex. Let me just precompute the full conflict matrix differently.

    # For each pair (i,j) with i<j, define: does i divide 2*j? (case b=c=j, a=i)
    # For each triple, it's O(n^3) to precompute which is 256^3 ~ 16M, too much memory.

    # Alternative: for each a, precompute the set of sums s = b+c (a<b<=c<=n) that are divisible by a.
    # Then for a given set S, when checking if a is compatible:
    #   For each b in S with b>a, for each c in S with c>=b, check if (b+c)%a==0.
    # This is O(|S|^2) per a, and we do it for each a we consider adding.

    # Let me just write a fast check function and use it.

    def is_valid_set(S):
        """Full validity check for a set."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def check_add(S, x, S_set):
        """Check if adding x to S keeps it valid. S is sorted list, S_set is set for O(1) lookup."""
        # x is not in S
        # Check triples where x is a, b, or c.
        
        # Case 1: x is the smallest (a=x). Need b,c in S with x<b<=c and x | (b+c).
        # For each b in S with b>x:
        #   For each c in S with c>=b:
        #     if (b+c) % x == 0: return False
        # Optimize: for each b>x, we need c such that c>=b and c ≡ -b (mod x).
        # Precompute for each x, for each residue r mod x, the sorted list of elements in S with that residue.
        # But S changes, so let's just do it directly but efficiently.
        
        # For case 1: x is a
        for b in S:
            if b <= x:
                continue
            # Need c in S, c >= b, and (b+c) % x == 0
            # c ≡ (-b) % x (mod x)
            target_res = (-b) % x
            # Find c in S with c >= b and c % x == target_res
            # We'll just iterate
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b (middle). Need a in S with a<x, c in S with c>=x, and a | (x+c).
        for a in S:
            if a >= x:
                continue
            # Need c in S, c >= x, and (x+c) % a == 0
            # c ≡ (-x) % a (mod a)
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c (largest). Need a<b in S, b<=x, and a | (b+x).
        for i, a in enumerate(S):
            for b in S[i+1:]:
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True

    # Initial seeds
    seeds = [
        list(range(171, 257)),  # upper tail, size 86
        list(range(160, 257)),  # slightly larger tail
        list(range(150, 257)),
    ]
    
    best_set = None
    best_size = 0
    
    # Try each seed
    for seed in seeds:
        if time.time() - start_time > time_limit - 10:
            break
        if not is_valid_set(seed):
            continue
        if len(seed) > best_size:
            best_size = len(seed)
            best_set = list(seed)
    
    if best_set is None:
        # Fallback
        best_set = list(range(171, 257))
        best_size = len(best_set)
    
    current_set = list(best_set)
    current_size = best_size
    
    # Greedy addition phase
    for candidate in range(1, 171):
        if time.time() - start_time > time_limit - 5:
            break
        if candidate not in current_set:
            S_sorted = sorted(current_set)
            S_set = set(current_set)
            if check_add(S_sorted, candidate, S_set):
                current_set.append(candidate)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
    
    # Simulated annealing / local search with swaps
    random.seed(42)
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    in_set = set(current_set)
    not_in_set = [i for i in range(1, n+1) if i not in in_set]
    
    while temp > min_temp and (time.time() - start_time < time_limit - 5):
        S_sorted = sorted(current_set)
        S_set = set(current_set)
        
        # Try a random move: add one, remove one (swap), or add one, or remove one
        move_type = random.random()
        
        if move_type < 0.6 and not_in_set:
            # Swap: remove y, add x
            x = random.choice(not_in_set)
            y = random.choice(current_set)
            if x == y:
                continue
            # Check if current_set - {y} + {x} is valid
            temp_set = [v for v in current_set if v != y]
            temp_sorted = sorted(temp_set)
            temp_set_set = set(temp_set)
            if check_add(temp_sorted, x, temp_set_set):
                # Accept swap (size same)
                current_set = temp_set
                current_set.append(x)
                in_set.remove(y)
                in_set.add(x)
                not_in_set.remove(x)
                not_in_set.append(y)
        elif move_type < 0.8 and not_in_set:
            # Add only
            x = random.choice(not_in_set)
            if check_add(S_sorted, x, S_set):
                current_set.append(x)
                current_size += 1
                in_set.add(x)
                not_in_set.remove(x)
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
        else:
            # Remove only (to escape local minima)
            if len(current_set) > 50:
                y = random.choice(current_set)
                current_set.remove(y)
                current_size -= 1
                in_set.remove(y)
                not_in_set.append(y)
        
        temp *= cooling
    
    # Final greedy addition pass
    for candidate in range(1, n+1):
        if time.time() - start_time > time_limit - 2:
            break
        if candidate not in current_set:
            S_sorted = sorted(current_set)
            S_set = set(current_set)
            if check_add(S_sorted, candidate, S_set):
                current_set.append(candidate)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
    
    # Return the best found
    return best_set if best_set else current_set
