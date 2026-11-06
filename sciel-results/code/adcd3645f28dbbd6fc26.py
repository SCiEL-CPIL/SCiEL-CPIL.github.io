import time
import random
from itertools import combinations

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    S is a set of integers. x is an integer not in S.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    """
    S_list = list(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    #    We need: NOT (exists a,b,c in S, a<b<c, (a+b+c) % x == 0)
    if m >= 3:
        # Map residue to list of elements having that residue mod x
        res_to_elems = {}
        for v in S_list:
            r = v % x
            if r not in res_to_elems:
                res_to_elems[r] = []
            res_to_elems[r].append(v)
        
        # Iterate pairs (a, b) from S
        for i in range(m - 1):
            a = S_list[i]
            ra = a % x
            for j in range(i + 1, m):
                b = S_list[j]
                rb = b % x
                # We need c such that (a + b + c) % x == 0 => c % x == (-a - b) % x
                target_r = (-ra - rb) % x
                
                # Check if there is an element c in S with residue target_r
                # such that c != a and c != b
                if target_r in res_to_elems:
                    candidates_c = res_to_elems[target_r]
                    # We need to find if there is a c in candidates_c distinct from a and b
                    # Optimization: if len(candidates_c) > 2, it's almost certainly yes unless a,b are the only ones?
                    # Actually, if len > 2, and a,b are in it, there is a 3rd.
                    # If len == 2, we need to check if one is not a or b.
                    # If len == 1, we need to check if it is not a or b.
                    
                    found = False
                    if len(candidates_c) >= 3:
                        # If there are at least 3 elements with this residue, 
                        # even if a and b are among them, there is a third one c.
                        # Wait, if a and b are in candidates_c, and len is 3, the third one is distinct.
                        # If a is in candidates_c and b is not, and len is 3, there are 2 others, one is distinct from a.
                        # So if len >= 3, it's always True?
                        # Yes, because we need ONE c distinct from a and b.
                        # The set of elements with residue target_r has size >= 3.
                        # The set {a, b} has size <= 2.
                        # So there is always an element in candidates_c not in {a, b}.
                        found = True
                    elif len(candidates_c) == 2:
                        # Check if at least one is not a or b
                        c1, c2 = candidates_c
                        if (c1 != a and c1 != b) or (c2 != a and c2 != b):
                            found = True
                    elif len(candidates_c) == 1:
                        c1 = candidates_c[0]
                        if c1 != a and c1 != b:
                            found = True
                    
                    if found:
                        return False
                        
    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y}.
    #    Equivalent: (u + v) % y == (-x) % y
    for y in S_list:
        target = (-x) % y
        # We need a pair u, v in S \ {y} such that (u + v) % y == target
        # Use a set of residues of S \ {y}
        # To handle distinctness correctly, we iterate and check.
        
        # Optimization: Precompute residues of S \ {y} mod y?
        # Doing this inside the loop is O(m^2) per y, total O(m^3).
        # Can we do better?
        # For each y, we do a 2-sum check on S \ {y} mod y.
        # 2-sum check is O(m).
        # So total O(m^2).
        
        seen_residues = set()
        # We iterate u in S \ {y}
        # We need to know if there exists v in S \ {y} \ {u} such that (u+v) % y == target.
        # i.e., v % y == (target - u) % y.
        
        # To ensure v != u, we can check if the residue (target - u) % y exists in the multiset of residues of S \ {y} \ {u}.
        # A simple way:
        # 1. Count residues of all elements in S \ {y} mod y.
        # 2. For each u, decrement count of u % y, check if count of (target - u) % y > 0.
        # 3. Increment count back.
        
        res_counts = {}
        for v in S_list:
            if v == y:
                continue
            r = v % y
            res_counts[r] = res_counts.get(r, 0) + 1
        
        for u in S_list:
            if u == y:
                continue
            ru = u % y
            needed_r = (target - ru) % y
            
            # Temporarily remove u from counts
            res_counts[ru] -= 1
            
            if res_counts.get(needed_r, 0) > 0:
                return False
            
            # Restore
            res_counts[ru] += 1
            
    return True

def _is_valid(S):
    """Full validation."""
    S_list = list(S)
    m = len(S_list)
    for idx_a, a in enumerate(S_list):
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        
        # Check if a divides sum of any 3 in others
        # Use the same logic as Part 1 of _can_add but with x=a and S=others
        # Note: _can_add checks if adding x to S is valid.
        # Here we check if a is valid given S=others.
        # But _can_add(x, S) checks:
        # 1. x divides sum of 3 in S?
        # 2. For each y in S, y divides x + u + v?
        # This is exactly the condition for a to be in S U {others}.
        
        if not _can_add(set(others), a):
            return False
    return True

def _greedy_with_order(order, deadline):
    """Greedy selection following a specific order."""
    current_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(current_set, x):
            current_set.add(x)
    return current_set

def _local_search_improve(S, deadline):
    """Try to improve set S by adding elements or doing swaps."""
    improved = True
    # Limit iterations to avoid infinite loops if something goes wrong
    max_iters = 50
    iter_count = 0
    
    while improved and time.time() < deadline and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. Try adding any unselected element
        for x in range(1, 193):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    improved = True
                    break
        
        # 2. If no direct add, try 1-out 2-in swap
        if not improved:
            best_swap_gain = 0
            best_swap = None
            
            # Try removing one element and adding two
            for a in list(S):
                if time.time() > deadline:
                    break
                trial_set = S - {a}
                
                # Find all elements that can be added to trial_set
                addable = []
                for x in range(1, 193):
                    if x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        addable.append(x)
                
                # Try pairs from addable
                # Optimization: if len(addable) is large, just try a few?
                # But we want max cardinality, so any valid swap is good.
                for x, y in combinations(addable, 2):
                    # Check if adding y to trial_set | {x} is valid
                    # _can_add(trial_set | {x}, y)
                    if _can_add(trial_set | {x}, y):
                        # Valid swap: remove a, add x and y
                        if best_swap is None:
                            best_swap = (a, x, y)
                            best_swap_gain = 1
                        break
                if best_swap is not None:
                    break
            
            if best_swap is not None:
                a, x, y = best_swap
                S.remove(a)
                S.add(x)
                S.add(y)
                improved = True
                
        # 3. Try 2-out 3-in swap? (Expensive, maybe only if time permits)
        if not improved and time.time() < deadline - 5:
             # This is O(|S|^2 * 192^2) which is too slow.
             # Skip for now.
             pass

    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 192
    best_set = set()
    
    # Strategy 1: Descending Greedy
    order = list(range(n, 0, -1))
    current_set = _greedy_with_order(order, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
    
    # Strategy 2: Randomized Greedy with multiple restarts
    rng = random.Random(42)
    
    # Try different bias strategies
    # We can afford more restarts now due to faster _can_add
    num_restarts = 200
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 10: # Reserve time for local search
            break
        
        if attempt % 5 == 0:
            # Pure random
            order = list(range(1, n + 1))
            rng.shuffle(order)
        elif attempt % 5 == 1:
            # Bias towards larger numbers
            order = list(range(n, 0, -1))
            # Randomize slightly
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif attempt % 5 == 2:
            # Bias towards numbers with fewer divisors (primes are safer)
            # Precompute divisor counts? Or just use a heuristic.
            # For now, just randomize a sorted list by "complexity"
            # Simpler: just random
            order = list(range(1, n + 1))
            rng.shuffle(order)
        elif attempt % 5 == 3:
            # Mix: interleave large and small
            large = list(range(n, n//2, -1))
            small = list(range(n//2, 0, -1))
            rng.shuffle(large)
            rng.shuffle(small)
            order = []
            for i in range(max(len(large), len(small))):
                if i < len(large):
                    order.append(large[i])
                if i < len(small):
                    order.append(small[i])
        else:
            # Random
            order = list(range(1, n + 1))
            rng.shuffle(order)
        
        current_set = _greedy_with_order(order, deadline)
        
        # Local search improvement
        current_set = _local_search_improve(current_set, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
    
    # Final local search on best_set
    if time.time() < deadline - 5:
        best_set = _local_search_improve(best_set, deadline)
    
    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback
    return [192]
