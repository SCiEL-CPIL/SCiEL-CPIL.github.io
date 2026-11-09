"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, log
from fractions import Fraction

def _score_fraction(chosen):
    """Compute exact rational score sum 1/(512-a) for a list of chosen integers."""
    if not chosen:
        return Fraction(0, 1)
    s = Fraction(0, 1)
    for a in chosen:
        s += Fraction(1, 512 - a)
    return s

def _score_float(chosen):
    """Compute float score for quick comparison."""
    return sum(1.0 / (512.0 - a) for a in chosen)

def _build_coprime_bitmasks():
    """Build bitmask for each number 1..511 indicating which other numbers are coprime to it."""
    coprime_mask = [0] * 512
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if a != b and gcd(a, b) == 1:
                mask |= (1 << (b - 1))
        coprime_mask[a] = mask
    return coprime_mask

def _is_coprime_to_all_bitmask(a, chosen_set, coprime_mask):
    """Check if a is coprime to all elements in chosen_set using bitmask."""
    chosen_mask = 0
    for b in chosen_set:
        chosen_mask |= (1 << (b - 1))
    # Check if any element in chosen_set is NOT coprime to a
    # i.e., chosen_mask & ~coprime_mask[a] should be 0
    return (chosen_mask & ~copime_mask[a]) == 0 if False else (chosen_mask & ~coprime_mask[a]) == 0

def _greedy_construct(ordering, coprime_mask, deadline):
    """Greedy construction with given ordering."""
    chosen = []
    chosen_set = set()
    for a in ordering:
        if time.time() > deadline:
            break
        # Check if a is coprime to all chosen
        ok = True
        for b in chosen_set:
            if gcd(a, b) != 1:
                ok = False
                break
        if ok:
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _local_search(chosen, coprime_mask, deadline, max_iterations=100000):
    """Fast local search: try additions, 1-for-1 swaps, and 2-for-2 swaps."""
    best_score = _score_float(chosen)
    best_chosen = list(chosen)
    best_set = set(best_chosen)
    
    # Precompute 1/(512-a) for all a
    val = [0.0] * 512
    for a in range(1, 512):
        val[a] = 1.0 / (512.0 - a)
    
    def _can_add(a, cur_set):
        """Check if a can be added to cur_set."""
        if a in cur_set:
            return False
        for b in cur_set:
            if gcd(a, b) != 1:
                return False
        return True
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Try adding elements (most valuable first)
        for a in range(511, 0, -1):
            if a in best_set:
                continue
            if _can_add(a, best_set):
                new_score = best_score + val[a]
                if new_score > best_score + 1e-15:
                    best_score = new_score
                    best_chosen.append(a)
                    best_set.add(a)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove one, add one
        swapped = False
        for a_out in list(best_chosen):
            for a_in in range(511, 0, -1):
                if a_in == a_out or a_in in best_set:
                    continue
                # Check if a_in is coprime to all except a_out
                ok = True
                for b in best_set:
                    if b != a_out and gcd(a_in, b) != 1:
                        ok = False
                        break
                if ok:
                    delta = val[a_in] - val[a_out]
                    if delta > 1e-15:
                        # Perform swap
                        best_chosen.remove(a_out)
                        best_chosen.append(a_in)
                        best_set.remove(a_out)
                        best_set.add(a_in)
                        best_score += delta
                        improved = True
                        swapped = True
                        break
            if swapped:
                break
        
        if improved:
            continue
        
        # Try 2-for-2 swaps: remove two, add two
        found_22 = False
        n = len(best_chosen)
        for i in range(n):
            for j in range(i + 1, n):
                a_out1 = best_chosen[i]
                a_out2 = best_chosen[j]
                # Try all pairs (a_in1, a_in2) with a_in1 > a_in2
                for a_in1 in range(511, 0, -1):
                    if a_in1 in best_set or a_in1 == a_out1 or a_in1 == a_out2:
                        continue
                    # Check a_in1 coprime to all except a_out1, a_out2
                    ok1 = True
                    for b in best_set:
                        if b != a_out1 and b != a_out2 and gcd(a_in1, b) != 1:
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    for a_in2 in range(a_in1 - 1, 0, -1):
                        if a_in2 in best_set or a_in2 == a_out1 or a_in2 == a_out2:
                            continue
                        # Check a_in2 coprime to all except a_out1, a_out2, and to a_in1
                        ok2 = True
                        for b in best_set:
                            if b != a_out1 and b != a_out2 and gcd(a_in2, b) != 1:
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        if gcd(a_in1, a_in2) != 1:
                            continue
                        delta = val[a_in1] + val[a_in2] - val[a_out1] - val[a_out2]
                        if delta > 1e-15:
                            # Perform 2-for-2 swap
                            best_chosen.remove(a_out1)
                            best_chosen.remove(a_out2)
                            best_chosen.append(a_in1)
                            best_chosen.append(a_in2)
                            best_set.remove(a_out1)
                            best_set.remove(a_out2)
                            best_set.add(a_in1)
                            best_set.add(a_in2)
                            best_score += delta
                            improved = True
                            found_22 = True
                            break
                    if found_22:
                        break
                if found_22:
                    break
            if found_22:
                break
        
        if not improved:
            break
    
    return best_chosen

def construct():
    start_time = time.time()
    # Use 140s as deadline to stay well within 150s limit
    deadline = start_time + 140.0
    
    # Build coprime masks
    coprime_mask = _build_coprime_bitmasks()
    
    # Phase 1: Multi-start greedy with different orderings
    orderings = [
        list(range(511, 0, -1)),  # descending
        list(range(1, 512)),      # ascending
    ]
    
    # Add some randomized orderings
    random.seed(42)
    for _ in range(5):
        orderings.append(list(range(1, 512)))
        random.shuffle(orderings[-1])
    
    best_overall = []
    best_overall_score = 0.0
    
    for ordering in orderings:
        if time.time() > deadline - 30:  # Reserve 30s for local search
            break
        chosen = _greedy_construct(ordering, coprime_mask, deadline - 30)
        score = _score_float(chosen)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = chosen
    
    if time.time() > deadline - 10:
        return best_overall
    
    # Phase 2: Local search on best greedy result
    result = _local_search(best_overall, coprime_mask, deadline - 5)
    
    return result
# EVOLVE-BLOCK-END
