"""Rewritten program: CSP/backtracking + local search to maximize disjoint residue classes."""
import math
import random
import time

MAX_MOD = 128
DEADLINE = 85.0  # safety margin before 90s wall-clock

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _compatible(r1, m1, r2, m2):
    """Two classes {x: x ≡ r1 mod m1} and {x: x ≡ r2 mod m2} are disjoint iff r1 != r2 mod gcd(m1,m2)."""
    g = _gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid_solution(pairs):
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _compatible(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _greedy_insertion(moduli_order, existing):
    """
    Try to insert each modulus in moduli_order into existing list.
    Returns a new list (copy) with as many as possible inserted.
    existing: list of [r, m]
    """
    result = [list(p) for p in existing]
    for m in moduli_order:
        # Check if m is already used
        if any(p[1] == m for p in result):
            continue
        # Find a residue r in [0, m) that is compatible with all existing
        found = False
        for r in range(m):
            ok = True
            for (er, em) in result:
                if not _compatible(r, m, er, em):
                    ok = False
                    break
            if ok:
                result.append([r, m])
                found = True
                break
    return result

def _local_search_1_for_2(pairs, max_iters=20000):
    """
    Try to replace one class with two classes (using unused moduli) to increase count by 1.
    Returns improved list or original if no improvement.
    """
    used_moduli = {p[1] for p in pairs}
    all_moduli = list(range(2, MAX_MOD + 1))
    unused_moduli = [m for m in all_moduli if m not in used_moduli]
    
    best = [list(p) for p in pairs]
    best_len = len(best)
    
    for _ in range(max_iters):
        # Pick a random existing class to remove
        idx = random.randrange(len(best))
        removed = best[idx]
        remaining = [list(p) for i, p in enumerate(best) if i != idx]
        
        # Try to add two unused moduli
        if len(unused_moduli) < 2:
            break
        
        # Shuffle unused moduli for diversity
        candidates = unused_moduli[:]
        random.shuffle(candidates)
        
        # Try first few pairs of candidates
        tried = 0
        for i in range(min(len(candidates), 10)):
            for j in range(i + 1, min(len(candidates), 10)):
                tried += 1
                if tried > 50:
                    break
                m1, m2 = candidates[i], candidates[j]
                
                # Find residues for m1
                found1 = False
                for r1 in range(m1):
                    ok1 = True
                    for (er, em) in remaining:
                        if not _compatible(r1, m1, er, em):
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    
                    # Find residue for m2 compatible with remaining and (r1, m1)
                    for r2 in range(m2):
                        ok2 = True
                        for (er, em) in remaining:
                            if not _compatible(r2, m2, er, em):
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        if not _compatible(r2, m2, r1, m1):
                            continue
                        
                        # Valid replacement
                        new_pairs = remaining + [[r1, m1], [r2, m2]]
                        if len(new_pairs) > best_len:
                            best = new_pairs
                            best_len = len(best)
                            # Update unused_moduli
                            used_moduli = {p[1] for p in best}
                            unused_moduli = [m for m in all_moduli if m not in used_moduli]
                        break
                    break
    return best

def _backtracking_search(moduli_set, existing, time_limit):
    """
    Recursive backtracking to add more classes.
    moduli_set: set of available moduli
    existing: list of [r, m]
    Returns best list found within time_limit.
    """
    start_time = time.time()
    best = [list(p) for p in existing]
    best_len = len(best)
    
    # Sort moduli for deterministic exploration
    avail = sorted(moduli_set)
    
    def backtrack(remaining_moduli, current):
        nonlocal best, best_len
        if time.time() - start_time > time_limit:
            return
        if len(current) > best_len:
            best = [list(p) for p in current]
            best_len = len(current)
        
        if not remaining_moduli:
            return
        
        # MRV: pick modulus with fewest valid residues
        best_idx = -1
        best_count = float('inf')
        best_residues = None
        
        for i, m in enumerate(remaining_moduli):
            valid_residues = []
            for r in range(m):
                ok = True
                for (er, em) in current:
                    if not _compatible(r, m, er, em):
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            if len(valid_residues) < best_count:
                best_count = len(valid_residues)
                best_idx = i
                best_residues = valid_residues
            if best_count == 0:
                break
        
        if best_idx == -1 or best_count == 0:
            return
        
        m = remaining_moduli[best_idx]
        new_remaining = remaining_moduli[:best_idx] + remaining_moduli[best_idx + 1:]
        
        # Try residues in order
        for r in best_residues:
            backtrack(new_remaining, current + [[r, m]])
            if time.time() - start_time > time_limit:
                return
    
    backtrack(avail, [list(p) for p in existing])
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Multi-ordering greedy to get a good initial solution
    all_moduli = list(range(2, MAX_MOD + 1))
    
    best_solution = []
    best_len = 0
    
    # Try different orderings
    orderings = []
    orderings.append(sorted(all_moduli))  # ascending
    orderings.append(sorted(all_moduli, reverse=True))  # descending
    
    # Try random orderings
    for _ in range(20):
        orderings.append(all_moduli[:])
        random.shuffle(orderings[-1])
    
    for order in orderings:
        sol = _greedy_insertion(order, [])
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = [list(p) for p in sol]
    
    # Phase 2: Local search 1-for-2
    if time.time() - start_time < DEADLINE - 5:
        best_solution = _local_search_1_for_2(best_solution, max_iters=50000)
        best_len = len(best_solution)
    
    # Phase 3: Backtracking on unused moduli
    if time.time() - start_time < DEADLINE - 10:
        used_moduli = {p[1] for p in best_solution}
        unused = set(all_moduli) - used_moduli
        if unused:
            time_left = DEADLINE - 10 - (time.time() - start_time)
            if time_left > 1:
                improved = _backtracking_search(unused, best_solution, time_left)
                if len(improved) > best_len:
                    best_solution = improved
                    best_len = len(best_solution)
    
    # Phase 4: Another round of local search
    if time.time() - start_time < DEADLINE - 2:
        best_solution = _local_search_1_for_2(best_solution, max_iters=30000)
    
    # Final validation
    if not _is_valid_solution(best_solution):
        # Fallback to a known valid solution
        best_solution = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    # Ensure we have a nonempty list
    if not best_solution:
        best_solution = [[0, 2]]
    
    return best_solution
