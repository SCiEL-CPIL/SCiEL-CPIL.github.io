"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from collections import Counter, defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n+1):
        if time.time() > deadline - 10:
            break
        for b in range(a+1, n+1):
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    bad_triples.append((a,b,c))
    
    if time.time() > deadline - 8:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    
    if not bad_triples:
        return list(range(1, n+1))
    
    # Precompute triple indices for each element
    # elem_triples[x] = list of indices of triples containing x
    elem_triples = defaultdict(list)
    for idx, t in enumerate(bad_triples):
        for x in t:
            elem_triples[x].append(idx)
            
    # Map triple index to the tuple
    triple_map = bad_triples
    
    best_result = None
    best_size = 0
    
    num_restarts = 50 # Can be increased if fast enough
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset state
        in_set = [True] * (n + 1)
        triple_counts = [3] * len(bad_triples)
        
        # Greedy Removal with Randomization
        # We want to reduce all triple_counts to < 3.
        # Heuristic: Remove element with most active (count==3) triples.
        
        active_count = [0] * (n + 1)
        for idx, t in enumerate(bad_triples):
            if triple_counts[idx] == 3:
                for x in t:
                    active_count[x] += 1
        
        while True:
            if time.time() > deadline - 2:
                break
            
            # Find max active_count
            max_count = 0
            candidates = []
            for x in range(1, n+1):
                if not in_set[x]:
                    continue
                if active_count[x] > max_count:
                    max_count = active_count[x]
                    candidates = [x]
                elif active_count[x] == max_count and max_count > 0:
                    candidates.append(x)
            
            if max_count == 0:
                break
                
            # Pick random candidate
            x = random.choice(candidates)
            
            # Remove x
            in_set[x] = False
            
            # Update triple_counts and active_count
            for idx in elem_triples[x]:
                old_count = triple_counts[idx]
                triple_counts[idx] -= 1
                new_count = triple_counts[idx]
                
                if old_count == 3 and new_count < 3:
                    # This triple is no longer active
                    for y in triple_map[idx]:
                        active_count[y] -= 1
        
        # Now we have a valid set (no triple with count 3).
        current_set = [x for x in range(1, n+1) if in_set[x]]
        
        # Phase 2: Greedy Add-back
        # Try to add back removed elements.
        # A removed element y can be added if for all triples t containing y, count[t] < 2.
        
        removed_elems = [x for x in range(1, n+1) if not in_set[x]]
        random.shuffle(removed_elems)
        
        for y in removed_elems:
            if time.time() > deadline - 2:
                break
            
            can_add = True
            for idx in elem_triples[y]:
                if triple_counts[idx] >= 2:
                    can_add = False
                    break
            
            if can_add:
                in_set[y] = True
                for idx in elem_triples[y]:
                    triple_counts[idx] += 1
        
        current_set = [x for x in range(1, n+1) if in_set[x]]
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set[:]
            
        # Phase 3: Simulated Annealing with penalty for invalid states
        # Use a penalty function to allow temporary violations for escape from local optima
        sa_iterations = 500
        temperature = 10.0
        cooling_rate = 0.995
        penalty_weight = 5.0
        
        # Precompute for faster penalty calculation
        # penalty = number of active triples (count >= 3)
        
        for sa_iter in range(sa_iterations):
            if time.time() > deadline - 2:
                break
            
            temperature *= cooling_rate
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add an element
                not_in_set = [z for z in range(1, n+1) if not in_set[z]]
                if not not_in_set:
                    continue
                y = random.choice(not_in_set)
                
                # Calculate penalty change
                penalty_change = 0
                for idx in elem_triples[y]:
                    if triple_counts[idx] >= 2:
                        penalty_change += 1
                
                # Acceptance probability
                if penalty_change == 0:
                    accept = True
                else:
                    delta_e = -penalty_change * penalty_weight
                    accept = random.random() < (1.0 / (1.0 + __import__('math').exp(-delta_e / temperature)))
                
                if accept:
                    in_set[y] = True
                    current_set.append(y)
                    for idx in elem_triples[y]:
                        triple_counts[idx] += 1
                    
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_result = current_set[:]
                        
            elif move_type < 0.5:
                # Try to remove an element
                if not current_set:
                    continue
                x = random.choice(current_set)
                
                # Calculate penalty change
                penalty_change = 0
                for idx in elem_triples[x]:
                    if triple_counts[idx] >= 3:
                        penalty_change -= 1  # Removing helps
                
                # Acceptance probability (removal usually accepted if it helps)
                if penalty_change <= 0:
                    accept = True
                else:
                    delta_e = -penalty_change * penalty_weight
                    accept = random.random() < (1.0 / (1.0 + __import__('math').exp(-delta_e / temperature)))
                
                if accept:
                    in_set[x] = False
                    current_set.remove(x)
                    for idx in elem_triples[x]:
                        triple_counts[idx] -= 1
                    
            else:
                # Try a swap
                if not current_set:
                    continue
                x = random.choice(current_set)
                not_in_set = [z for z in range(1, n+1) if not in_set[z]]
                if not not_in_set:
                    continue
                y = random.choice(not_in_set)
                
                # Calculate penalty change for swap
                penalty_change = 0
                # Adding y
                for idx in elem_triples[y]:
                    t = triple_map[idx]
                    if x in t:
                        continue  # count unchanged
                    else:
                        if triple_counts[idx] >= 2:
                            penalty_change += 1
                # Removing x
                for idx in elem_triples[x]:
                    t = triple_map[idx]
                    if y in t:
                        continue  # count unchanged
                    else:
                        if triple_counts[idx] >= 3:
                            penalty_change -= 1
                
                # Acceptance probability
                if penalty_change == 0:
                    accept = True
                else:
                    delta_e = -penalty_change * penalty_weight
                    accept = random.random() < (1.0 / (1.0 + __import__('math').exp(-delta_e / temperature)))
                
                if accept:
                    in_set[x] = False
                    in_set[y] = True
                    current_set.remove(x)
                    current_set.append(y)
                    
                    # Update counts
                    for idx in elem_triples[x]:
                        triple_counts[idx] -= 1
                    for idx in elem_triples[y]:
                        triple_counts[idx] += 1
                    
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_result = current_set[:]
                        
            # Kick move: occasionally do a 2-swap to escape local optima
            if sa_iter % 50 == 0 and sa_iter > 0 and temperature > 1.0:
                if len(current_set) >= 2:
                    not_in_set = [z for z in range(1, n+1) if not in_set[z]]
                    if len(not_in_set) >= 2:
                        # Pick two elements to remove and two to add
                        x1, x2 = random.sample(current_set, 2)
                        y1, y2 = random.sample(not_in_set, 2)
                        
                        # Calculate penalty change
                        penalty_change = 0
                        # Adding y1, y2
                        for y in [y1, y2]:
                            for idx in elem_triples[y]:
                                t = triple_map[idx]
                                if x1 in t or x2 in t:
                                    continue
                                else:
                                    if triple_counts[idx] >= 2:
                                        penalty_change += 1
                        # Removing x1, x2
                        for x in [x1, x2]:
                            for idx in elem_triples[x]:
                                t = triple_map[idx]
                                if y1 in t or y2 in t:
                                    continue
                                else:
                                    if triple_counts[idx] >= 3:
                                        penalty_change -= 1
                        
                        # Acceptance probability
                        if penalty_change == 0:
                            accept = True
                        else:
                            delta_e = -penalty_change * penalty_weight * 2  # Higher penalty for kicks
                            accept = random.random() < (1.0 / (1.0 + __import__('math').exp(-delta_e / temperature)))
                        
                        if accept:
                            in_set[x1] = False
                            in_set[x2] = False
                            in_set[y1] = True
                            in_set[y2] = True
                            current_set.remove(x1)
                            current_set.remove(x2)
                            current_set.extend([y1, y2])
                            
                            # Update counts
                            for x in [x1, x2]:
                                for idx in elem_triples[x]:
                                    triple_counts[idx] -= 1
                            for y in [y1, y2]:
                                for idx in elem_triples[y]:
                                    triple_counts[idx] += 1
                            
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_result = current_set[:]
        
        # After SA, try to add more elements greedily
        for y in range(1, n+1):
            if time.time() > deadline - 2:
                break
            if in_set[y]:
                continue
            
            can_add = True
            for idx in elem_triples[y]:
                if triple_counts[idx] >= 2:
                    can_add = False
                    break
            
            if can_add:
                in_set[y] = True
                for idx in elem_triples[y]:
                    triple_counts[idx] += 1
        
        current_set = [x for x in range(1, n+1) if in_set[x]]
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set[:]
            
        # Final repair: remove elements from active triples to make set valid
        # This ensures we return a valid set even if SA left it invalid
        while True:
            if time.time() > deadline - 1:
                break
            
            # Find an active triple
            active_triple_idx = None
            for idx in range(len(bad_triples)):
                if triple_counts[idx] >= 3:
                    active_triple_idx = idx
                    break
            
            if active_triple_idx is None:
                break
            
            # Remove one element from this triple
            t = triple_map[active_triple_idx]
            x_to_remove = random.choice(t)
            in_set[x_to_remove] = False
            if x_to_remove in current_set:
                current_set.remove(x_to_remove)
            
            for idx in elem_triples[x_to_remove]:
                triple_counts[idx] -= 1
        
        # Update best if we have a valid set
        current_set = [x for x in range(1, n+1) if in_set[x]]
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set[:]
        
    if best_result is None:
        best_result = [1]
        
    # Final validation: ensure no bad triples remain
    # If validation fails, fall back to a safe set
    final_set = set(best_result)
    valid = True
    for a in range(1, n+1):
        if a not in final_set:
            continue
        for b in range(a+1, n+1):
            if b not in final_set:
                continue
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if c not in final_set:
                    continue
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid set
        best_result = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return best_result
# EVOLVE-BLOCK-END
