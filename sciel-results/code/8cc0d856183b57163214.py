"""Improved program using Simulated Annealing and Greedy Refinement for sum-free subsets."""

import random
import time
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free(vals, time_limit=30.0):
        n = len(vals)
        if n == 0:
            return []
        
        # 1. Precompute triples
        # val_to_idx
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        triples_list = []
        elem_to_triples = [[] for _ in range(n)]
        
        # Find all triples (i, j, k) such that vals[i] + vals[j] == vals[k]
        # i, j can be same.
        for i in range(n):
            for j in range(n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    triple = (i, j, k)
                    triples_list.append(triple)
                    # Add to elem_to_triples for i, j, k
                    elem_to_triples[i].append(len(triples_list) - 1)
                    elem_to_triples[j].append(len(triples_list) - 1)
                    elem_to_triples[k].append(len(triples_list) - 1)
        
        # 2. Simulated Annealing to minimize violations
        # State: active (list of bools)
        # We start with all active? That has many violations.
        # Better to start with a random sparse set or use the greedy result as a seed?
        # Let's start with all False, and add randomly?
        # Or start with all True and anneal to remove?
        # Starting with all True and removing is good for finding a valid core.
        
        best_size = 0
        best_set = set()
        
        start_time = time.time()
        rng = random.Random(12345) # Fixed seed for SA
        
        # Parameters
        T_start = 1.0
        T_end = 0.001
        cooling = 0.99
        penalty = 10.0 # Penalty for each violation
        
        # We will run multiple SA restarts
        for restart in range(20):
            # Initialize
            # Strategy: Start with all True, then let SA remove things.
            # Or start with a random subset of size n/2.
            # Let's try starting with all True.
            active = [True] * n
            
            # Calculate initial violations
            curr_violations = 0
            for t_idx in range(len(triples_list)):
                i, j, k = triples_list[t_idx]
                if active[i] and active[j] and active[k]:
                    curr_violations += 1
            
            curr_size = n
            T = T_start
            
            # SA Loop
            while T > T_end:
                # Pick a random element to flip
                idx = rng.randint(0, n - 1)
                
                # Calculate delta violations
                delta_viol = 0
                if active[idx]:
                    # Turning OFF
                    # Count how many triples involving idx are currently active
                    for t_idx in elem_to_triples[idx]:
                        i, j, k = triples_list[t_idx]
                        # Check if the other two are active
                        # Since idx is one of i,j,k, we need to check the other two.
                        # To avoid checking idx itself (which is being turned off), 
                        # we check the status of the other two.
                        # Note: if i==j, we just check i.
                        # But in our triple storage, i,j,k are indices.
                        # If idx == i, we check j and k.
                        # If idx == j, we check i and k.
                        # If idx == k, we check i and j.
                        
                        # Simple check:
                        # The triple was active if all 3 were True.
                        # Now idx is False. So it becomes inactive.
                        # It was active ONLY IF the other two are True.
                        # So we increment delta_viol by 1 if the other two are True.
                        
                        # Identify the other two
                        if i == idx:
                            other1, other2 = j, k
                        elif j == idx:
                            other1, other2 = i, k
                        else: # k == idx
                            other1, other2 = i, j
                        
                        if active[other1] and active[other2]:
                            delta_viol += 1
                            
                    # curr_violations decreases by delta_viol
                    curr_violations -= delta_viol
                    curr_size -= 1
                    active[idx] = False
                else:
                    # Turning ON
                    for t_idx in elem_to_triples[idx]:
                        i, j, k = triples_list[t_idx]
                        if i == idx:
                            other1, other2 = j, k
                        elif j == idx:
                            other1, other2 = i, k
                        else:
                            other1, other2 = i, j
                        
                        if active[other1] and active[other2]:
                            delta_viol += 1
                            
                    # curr_violations increases by delta_viol
                    curr_violations += delta_viol
                    curr_size += 1
                    active[idx] = True
                
                # Calculate Score Delta
                # Score = size - penalty * violations
                # Delta Score = Delta Size - penalty * Delta Violations
                # If turning OFF: Delta Size = -1, Delta Violations = -delta_viol
                #   Delta Score = -1 - penalty * (-delta_viol) = -1 + penalty * delta_viol
                # If turning ON: Delta Size = 1, Delta Violations = delta_viol
                #   Delta Score = 1 - penalty * delta_viol
                
                if active[idx]: # Just turned ON
                    delta_score = 1.0 - penalty * delta_viol
                else: # Just turned OFF
                    delta_score = -1.0 + penalty * delta_viol
                
                # Acceptance
                if delta_score > 0:
                    accept = True
                else:
                    # Metropolis criterion
                    # P = exp(delta_score / T)
                    # If delta_score is very negative, exp is 0.
                    if T > 0:
                        try:
                            p = math.exp(delta_score / T)
                        except OverflowError:
                            p = 0.0
                    else:
                        p = 0.0
                    accept = (rng.random() < p)
                
                if not accept:
                    # Revert
                    if active[idx]: # Was turned ON, so revert to OFF
                        active[idx] = False
                        curr_violations -= delta_viol
                        curr_size -= 1
                    else: # Was turned OFF, so revert to ON
                        active[idx] = True
                        curr_violations += delta_viol
                        curr_size += 1
                
                T *= cooling
                
                # Check time
                if time.time() - start_time > time_limit / 2.0: # Use half time for SA, half for refinement?
                    break
            
            # Refinement: Greedy Add
            # Now we have a valid (or nearly valid) set.
            # If curr_violations == 0, we can try to add more.
            # If not, we might need to clean up.
            # Let's assume we found a good core.
            # Convert active to set
            current_set = {vals[i] for i in range(n) if active[i]}
            
            # Greedy Add:
            # Try to add elements from vals not in current_set
            # We need to check if adding x creates a violation.
            # A violation is created if x + y = z or x + z = y or y + z = x for y,z in set.
            # Or 2x = y or 2y = x.
            
            # To do this efficiently, we can just check against the set.
            # Since n is small, O(n^2) check per addition is fine.
            
            added = True
            while added:
                added = False
                for x in vals:
                    if x in current_set:
                        continue
                    # Check if x can be added
                    can_add = True
                    for y in current_set:
                        # Check x + y = z
                        if (x + y) in current_set:
                            can_add = False
                            break
                        # Check x + x = y (if 2x in set)
                        if (2 * x) == y:
                            can_add = False
                            break
                        # Check y + y = x (if 2y == x)
                        if (2 * y) == x:
                            can_add = False
                            break
                        # Check y + x = z is same as x+y=z
                        # Check x - y = z? No, sum-free is a+b=c.
                        # If x - y = z, then x = y + z. So if x is in set, and y,z are in set, violation.
                        # So we must check if x - y is in current_set OR if x + y is in current_set.
                        # Wait.
                        # If we add x, we need to ensure no a,b in (current_set U {x}) sum to c in (current_set U {x}).
                        # Pairs involving x:
                        # 1. x + x = 2x. If 2x in current_set, violation.
                        # 2. x + y = z. If z in current_set, violation.
                        # 3. y + x = z. Same as 2.
                        # 4. y + z = x. If y,z in current_set, and x is the new element, violation.
                        
                        # So we need to check:
                        # - 2x in current_set
                        # - x + y in current_set
                        # - x - y in current_set (since y + (x-y) = x)
                        
                        if (x - y) in current_set:
                            can_add = False
                            break
                        
                    if can_add:
                        current_set.add(x)
                        added = True
                        # If we added one, we can break and restart the loop to be safe, 
                        # or continue. Continuing is faster but might miss better combos? 
                        # Greedy is greedy.
                        break 
                
                if time.time() - start_time > time_limit:
                    break

            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            
            if time.time() - start_time > time_limit:
                break

        return list(best_set)

    results = []
    time_alloc = {60: 15.0, 120: 25.0, 240: 70.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        t_limit = time_alloc.get(size, 10.0)
        subset = solve_sum_free(vals, t_limit)
        results.append(subset)
        
    return results
