"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _score(chosen):
    """Compute the score sum 1/(512-a) for chosen numbers."""
    total = 0.0
    for a in chosen:
        total += 1.0 / (512 - a)
    return total

def _is_coprime_with_all(a, chosen_set):
    """Check if a is coprime with all elements in chosen_set."""
    for b in chosen_set:
        if gcd(a, b) != 1:
            return False
    return True

def _fill_greedy(chosen_set, used_set):
    """Fill remaining slots greedily from 511 down."""
    for a in range(511, 0, -1):
        if a not in used_set and _is_coprime_with_all(a, chosen_set):
            chosen_set.add(a)
            used_set.add(a)

def _local_search(chosen_list, time_limit=110.0):
    """Try to improve by swapping elements for better ones."""
    chosen_set = set(chosen_list)
    used_set = set(chosen_list)
    best_score = _score(chosen_list)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        for i, a in enumerate(chosen_list):
            # Try replacing a with a larger number
            for new_a in range(511, a, -1):
                if time.time() - start_time > time_limit:
                    break
                if new_a in used_set:
                    continue
                # Check if new_a is coprime with all others (excluding a)
                others = [b for j, b in enumerate(chosen_list) if j != i]
                if all(gcd(new_a, b) == 1 for b in others):
                    # Check if this improves score
                    old_contrib = 1.0 / (512 - a)
                    new_contrib = 1.0 / (512 - new_a)
                    if new_contrib > old_contrib + 1e-12:
                        # Do the swap
                        chosen_list[i] = new_a
                        chosen_set.remove(a)
                        chosen_set.add(new_a)
                        used_set.remove(a)
                        used_set.add(new_a)
                        improved = True
                        break  # Move to next i
    
    # Try adding more numbers
    for a in range(511, 0, -1):
        if time.time() - start_time > time_limit:
            break
        if a not in used_set and _is_coprime_with_all(a, chosen_set):
            chosen_list.append(a)
            chosen_set.add(a)
            used_set.add(a)
    
    return chosen_list

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s
    
    # Phase 1: Greedy from 511 down
    chosen = []
    for a in range(511, 0, -1):
        if time.time() - start_time > time_limit:
            break
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    # Phase 2: Local search to improve
    if time.time() - start_time < time_limit:
        chosen = _local_search(chosen, time_limit - (time.time() - start_time))
    
    return chosen
# EVOLVE-BLOCK-END
