import time
import random
from math import gcd

def construct():
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcds
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            gcd_table[(m1, m2)] = gcd(m1, m2)
            
    # Identify "free" primes
    free_primes = []
    hard_moduli = []
    for m in all_moduli:
        is_prime = True
        for d in range(2, int(m**0.5) + 1):
            if m % d == 0:
                is_prime = False
                break
        if is_prime and m > MAX_MOD // 2:
            free_primes.append(m)
        else:
            hard_moduli.append(m)
            
    base_classes = [[0, p] for p in free_primes]
    
    # Heuristic sort for hard moduli
    conflict_counts = {}
    for m in hard_moduli:
        count = 0
        for m2 in hard_moduli:
            if m != m2 and gcd_table[(m, m2)] > 1:
                count += 1
        conflict_counts[m] = count
        
    hard_moduli.sort(key=lambda x: conflict_counts[x], reverse=True)

    # 1. Greedy Warm-up to get a good lower bound
    def greedy_solve(order):
        current = []
        for m in order:
            # Check against current
            found_r = -1
            for r in range(m):
                ok = True
                for r_old, m_old in current:
                    g = gcd_table[(m, m_old)]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            if found_r != -1:
                current.append([found_r, m])
        return current + base_classes

    best_greedy = []
    # Try a few random orders
    start_time = time.time()
    for _ in range(50):
        if time.time() - start_time > 2: # 2s budget for greedy
            break
        rand_order = hard_moduli[:]
        random.shuffle(rand_order)
        res = greedy_solve(rand_order)
        if len(res) > len(best_greedy):
            best_greedy = res
            
    # Also try sorted order
    res = greedy_solve(hard_moduli)
    if len(res) > len(best_greedy):
        best_greedy = res
        
    best_size = len(best_greedy)
    best_set = best_greedy
    
    # 2. Backtracking Search
    def get_valid_residues(m, current_hard_classes):
        valid = [True] * m
        for r_old, m_old in current_hard_classes:
            g = gcd_table[(m, m_old)]
            if g == 1:
                continue
            target_res = r_old % g
            for r in range(target_res, m, g):
                valid[r] = False
        return [r for r, v in enumerate(valid) if v]

    def backtrack(idx, current_hard_classes):
        nonlocal best_set, best_size
        
        # Pruning
        remaining = len(hard_moduli) - idx
        if len(current_hard_classes) + remaining + len(base_classes) <= best_size:
            return
            
        if idx == len(hard_moduli):
            if len(current_hard_classes) + len(base_classes) > best_size:
                best_size = len(current_hard_classes) + len(base_classes)
                best_set = [list(c) for c in (current_hard_classes + base_classes)]
            return

        m = hard_moduli[idx]
        
        # Option 1: Include m
        valid_residues = get_valid_residues(m, current_hard_classes)
        # Shuffle residues to avoid bias? Or keep deterministic.
        # Deterministic is fine for correctness.
        for r in valid_residues:
            backtrack(idx + 1, current_hard_classes + [[r, m]])
            
        # Option 2: Exclude m
        backtrack(idx + 1, current_hard_classes)

    # Set deadline for backtracking
    deadline = time.time() + 80 # 80s limit, leaving 10s margin
    
    # We need a time check inside backtrack. 
    # Python recursion is slow. 114 depth is okay.
    # But branching might be high.
    
    # To prevent hanging, we add a global time check.
    search_state = {"aborted": False}
    
    def timed_backtrack(idx, current_hard_classes):
        nonlocal best_set, best_size
        if search_state["aborted"]:
            return
        
        if time.time() > deadline:
            search_state["aborted"] = True
            return

        remaining = len(hard_moduli) - idx
        if len(current_hard_classes) + remaining + len(base_classes) <= best_size:
            return
            
        if idx == len(hard_moduli):
            if len(current_hard_classes) + len(base_classes) > best_size:
                best_size = len(current_hard_classes) + len(base_classes)
                best_set = [list(c) for c in (current_hard_classes + base_classes)]
            return

        m = hard_moduli[idx]
        
        valid_residues = get_valid_residues(m, current_hard_classes)
        
        # Try Include
        for r in valid_residues:
            if search_state["aborted"]:
                return
            timed_backtrack(idx + 1, current_hard_classes + [[r, m]])
            
        # Try Exclude
        if not search_state["aborted"]:
            timed_backtrack(idx + 1, current_hard_classes)

    timed_backtrack(0, [])
    
    if not best_set:
        return [[0, 2]]
        
    return best_set
