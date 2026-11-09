import time
from random import Random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(vals, time_budget):
        n = len(vals)
        if n == 0:
            return []
        
        # Precompute parity
        odds = [i for i in range(n) if vals[i] % 2 != 0]
        evens = [i for i in range(n) if vals[i] % 2 == 0]
        
        def can_add(val, current_vals):
            # Check if val can be added to current_vals (set of values)
            # Conditions:
            # 1. No a, b in current_vals such that a + b == val
            # 2. No a in current_vals such that val + a in current_vals
            # Note: If a+b=val, then val is the sum.
            # If val+a=c, then val is part of a sum.
            # The condition "no a+b=c with a,b,c in S" is equivalent to:
            # For any x in S, and any y in S:
            #   x+y not in S
            #   x-y not in S (if x-y != 0 and x-y in S, then (x-y)+y=x)
            # Actually, the problem states "no a + b = c".
            # This implies:
            # 1. If we add v, we must ensure v is not the sum of two existing elements.
            #    i.e., for all a, b in S: a + b != v
            # 2. We must ensure v is not part of a sum with existing elements.
            #    i.e., for all a in S: v + a not in S
            #    AND v - a not in S (since (v-a) + a = v, so if v-a in S and a in S, then v is the sum)
            #    Wait, if v-a in S, let u = v-a. Then u + a = v. So v is the sum of u and a.
            #    So we need to check:
            #    - For all a, b in S: a + b != v
            #    - For all a in S: v + a not in S
            #    - For all a in S: v - a not in S
            
            # Optimization: 
            # Check 1: v - a in S for any a in S? (Covers v = a + (v-a))
            # Check 2: v + a in S for any a in S? (Covers v+a = c)
            # Does Check 1 cover "a+b=v"?
            # If a+b=v, then b = v-a. So if b is in S, then v-a is in S.
            # So checking "v - a in S" for all a in S covers both "v is sum of two elements" 
            # and "v + a is in S" is NOT covered by v-a.
            # Wait.
            # Case 1: a + b = v. Then b = v - a. If b in S, then v - a in S.
            # Case 2: v + a = c. Then c - a = v. This is not directly "v - a in S".
            # But if v + a = c, then c is in S. We are adding v. c is already in S.
            # So we need to check if v + a is in S.
            
            # So the check is:
            # for a in current_vals:
            #   if v - a in current_vals: return False
            #   if v + a in current_vals: return False
            
            # This is O(K).
            for a in current_vals:
                if v - a in current_vals:
                    return False
                if v + a in current_vals:
                    return False
            return True

        def greedy_fill(order, start_set=None, start_vals=None):
            if start_set is None:
                current_set = set()
                current_vals = set()
            else:
                current_set = set(start_set)
                current_vals = set(start_vals)
            
            for idx in order:
                if idx in current_set:
                    continue
                val = vals[idx]
                if can_add(val, current_vals):
                    current_set.add(idx)
                    current_vals.add(val)
            return current_set, current_vals

        best_set = set()
        best_vals = set()
        best_size = 0
        rng = Random(12345)
        
        # Strategy 1: Start with all odds (guaranteed sum-free)
        # Then try to add evens
        init_set = set(odds)
        init_vals = {vals[i] for i in odds}
        
        # Try adding evens in different orders
        # Order 1: Random
        evens_list = list(evens)
        rng.shuffle(evens_list)
        s, v = greedy_fill(evens_list, init_set, init_vals)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v
            
        # Order 2: Sorted by abs value
        evens_list = sorted(evens, key=lambda i: abs(vals[i]), reverse=True)
        s, v = greedy_fill(evens_list, init_set, init_vals)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v
            
        # Order 3: Sorted by value
        evens_list = sorted(evens, key=lambda i: vals[i])
        s, v = greedy_fill(evens_list, init_set, init_vals)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 2: Pure Greedy from scratch with different orders
        # Sometimes starting from scratch finds a better mix
        for _ in range(5):
            order = list(range(n))
            rng.shuffle(order)
            s, v = greedy_fill(order)
            if len(s) > best_size:
                best_size = len(s)
                best_set, best_vals = s, v

        # Local Search: Remove-and-Fill
        # We iterate, removing one element and greedily filling the rest.
        # We accept if the new set is larger.
        # To avoid getting stuck, we randomize the fill order.
        
        current_set = set(best_set)
        current_vals = set(best_vals)
        current_size = best_size
        
        sa_deadline = start_time + time_budget
        
        # We will use a simple hill-climbing with random restarts of the fill order
        # If we get stuck, we might try removing 2 elements.
        
        steps = 0
        last_improve = 0
        
        while time.time() < sa_deadline:
            if not current_set:
                break
                
            # Pick a random element to remove
            # Bias towards removing elements that "conflict" with many others?
            # For now, random is fine.
            remove_idx = rng.choice(list(current_set))
            
            temp_set = current_set - {remove_idx}
            temp_vals = current_vals - {vals[remove_idx]}
            
            # Greedy fill with random order
            fill_order = list(range(n))
            rng.shuffle(fill_order)
            
            new_set, new_vals = greedy_fill(fill_order, temp_set, temp_vals)
            new_size = len(new_set)
            
            if new_size > current_size:
                current_set = new_set
                current_vals = new_vals
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_vals = set(current_vals)
                last_improve = steps
            else:
                # No improvement. 
                # Occasionally, try a more aggressive perturbation: remove 2 random elements
                if steps - last_improve > 100:
                    if len(current_set) > 1:
                        # Remove 2 random elements
                        rem1 = rng.choice(list(current_set))
                        rem2_candidates = list(current_set - {rem1})
                        if rem2_candidates:
                            rem2 = rng.choice(rem2_candidates)
                            temp_set = current_set - {rem1, rem2}
                            temp_vals = current_vals - {vals[rem1], vals[rem2]}
                            
                            fill_order = list(range(n))
                            rng.shuffle(fill_order)
                            new_set, new_vals = greedy_fill(fill_order, temp_set, temp_vals)
                            new_size = len(new_set)
                            
                            if new_size > current_size:
                                current_set = new_set
                                current_vals = new_vals
                                current_size = new_size
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = set(current_set)
                                    best_vals = set(current_vals)
                                last_improve = steps
                            else:
                                # Reset current to best if stuck for too long?
                                # Or just continue
                                pass
                    else:
                        # Reset to best
                        current_set = set(best_set)
                        current_vals = set(best_vals)
                        current_size = best_size
                        last_improve = steps
            
            steps += 1
            
            # Safety check for time
            if time.time() > sa_deadline:
                break
        
        # Final Polish: Try to add any missing elements
        final_set = set(best_set)
        final_vals = set(best_vals)
        
        missing = [i for i in range(n) if i not in final_set]
        # Sort missing by heuristic: maybe those with fewer conflicts?
        # For now, just try adding them in random order a few times
        for _ in range(20):
            rng.shuffle(missing)
            for idx in missing:
                if idx in final_set:
                    continue
                if can_add(vals[idx], final_vals):
                    final_set.add(idx)
                    final_vals.add(vals[idx])
            # Update missing list
            missing = [i for i in range(n) if i not in final_set]
            if not missing:
                break

        return [vals[i] for i in sorted(final_set)]

    results = []
    time_alloc = {60: 20.0, 120: 30.0, 240: 50.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            # Fallback: All odds
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
            
        budget = min(time_alloc.get(size, 30.0), remaining - 5.0)
        if budget < 5.0:
            budget = max(5.0, remaining - 5.0)
        
        vals = instance(size, seed)
        results.append(solve_instance(vals, budget))

    return results
