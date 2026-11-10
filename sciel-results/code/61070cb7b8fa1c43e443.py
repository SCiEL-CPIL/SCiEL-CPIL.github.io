"""Combined approach: greedy baseline + local search refinement."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _score(chosen):
    """Compute sum 1/(512-a) for the chosen set."""
    total = 0.0
    for a in chosen:
        total += 1.0 / (512 - a)
    return total

def _is_coprime_to_all(a, chosen_set, chosen_list):
    """Check if a is coprime to all elements in chosen."""
    for b in chosen_list:
        if gcd(a, b) != 1:
            return False
    return True

def _greedy_descending():
    """Seed approach: greedy descending."""
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    return chosen, chosen_set

def _local_search(chosen, chosen_set, deadline, max_iterations=5000):
    """
    Local search: try to improve by swapping out one or two selected numbers
    for one or two unselected numbers that yield a better score.
    """
    start_time = time.time()
    best_score = _score(chosen)
    best_chosen = list(chosen)
    best_set = set(chosen)
    
    # Precompute scores for all candidates
    candidate_scores = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Build list of unselected numbers
    unselected = [a for a in range(1, 512) if a not in chosen_set]
    
    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time.time() > deadline:
            break
        
        # Try single swaps: remove one selected, add one unselected
        for i, a_out in enumerate(chosen):
            if time.time() > deadline:
                break
            # Try adding each unselected number in place of a_out
            for a_in in unselected:
                # Quick check: is a_in coprime to all chosen except a_out?
                ok = True
                for j, b in enumerate(chosen):
                    if j == i:
                        continue
                    if gcd(a_in, b) != 1:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check score improvement
                old_score = best_score
                new_score = old_score - candidate_scores[a_out] + candidate_scores[a_in]
                if new_score > best_score + 1e-12:
                    # Accept the swap
                    chosen[i] = a_in
                    chosen_set.remove(a_out)
                    chosen_set.add(a_in)
                    unselected.remove(a_in)
                    unselected.append(a_out)
                    best_score = new_score
                    best_chosen = list(chosen)
                    best_set = set(chosen)
                    improved = True
                    break  # Restart scan after a swap
        
        # Try double swaps: remove two selected, add two unselected
        if not improved and len(chosen) >= 2:
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                a_out1 = chosen[i]
                for j in range(i + 1, len(chosen)):
                    if time.time() > deadline:
                        break
                    a_out2 = chosen[j]
                    
                    # Find unselected pairs that are coprime to each other
                    # and to all chosen except a_out1, a_out2
                    for idx1 in range(len(unselected)):
                        if time.time() > deadline:
                            break
                        a_in1 = unselected[idx1]
                        
                        # Check a_in1 coprime to chosen except a_out1, a_out2
                        ok1 = True
                        for k, b in enumerate(chosen):
                            if k == i or k == j:
                                continue
                            if gcd(a_in1, b) != 1:
                                ok1 = False
                                break
                        if not ok1:
                            continue
                        
                        for idx2 in range(idx1 + 1, len(unselected)):
                            if time.time() > deadline:
                                break
                            a_in2 = unselected[idx2]
                            
                            # Check a_in2 coprime to a_in1
                            if gcd(a_in1, a_in2) != 1:
                                continue
                            
                            # Check a_in2 coprime to chosen except a_out1, a_out2
                            ok2 = True
                            for k, b in enumerate(chosen):
                                if k == i or k == j:
                                    continue
                                if gcd(a_in2, b) != 1:
                                    ok2 = False
                                    break
                            if not ok2:
                                continue
                            
                            # Check score improvement
                            old_score = best_score
                            new_score = (old_score - candidate_scores[a_out1] 
                                       - candidate_scores[a_out2]
                                       + candidate_scores[a_in1]
                                       + candidate_scores[a_in2])
                            if new_score > best_score + 1e-12:
                                # Accept the double swap
                                chosen[i] = a_in1
                                chosen[j] = a_in2
                                chosen_set.remove(a_out1)
                                chosen_set.remove(a_out2)
                                chosen_set.add(a_in1)
                                chosen_set.add(a_in2)
                                unselected.remove(a_in1)
                                unselected.remove(a_in2)
                                unselected.append(a_out1)
                                unselected.append(a_out2)
                                best_score = new_score
                                best_chosen = list(chosen)
                                best_set = set(chosen)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_chosen, best_set, best_score

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Phase 1: Greedy descending baseline
    chosen, chosen_set = _greedy_descending()
    
    # Phase 2: Local search refinement
    if time.time() < deadline:
        chosen, chosen_set, score = _local_search(chosen, chosen_set, deadline)
    
    # Verify validity
    if len(chosen) != len(set(chosen)):
        chosen = list(set(chosen))
    for a in chosen:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if gcd(chosen[i], chosen[j]) != 1:
                raise ValueError(f"Not coprime: {chosen[i]}, {chosen[j]}")
    
    return chosen
# EVOLVE-BLOCK-END
