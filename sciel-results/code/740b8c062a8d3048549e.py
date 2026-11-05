"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from functools import lru_cache
import random

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Precompute reciprocals
    reciprocals = [Fraction(0) for _ in range(n+1)]
    for i in range(1, n+1):
        reciprocals[i] = Fraction(1, i)
    
    # Precompute for each x, the set of y such that x and y form a "conflict" 
    # in the sense that there exists z such that {x, y, z} is a 3-AP.
    # Actually, let's precompute for each x, the list of (a, c) where x is the midpoint.
    midpoint_pairs = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        for a in range(1, x):
            c = 2*x - a
            if c > x and c <= n:
                midpoint_pairs[x].append((a, c))
    
    def is_3ap_fast(selected_set):
        for x in selected_set:
            for (a, c) in midpoint_pairs[x]:
                if a in selected_set and c in selected_set:
                    return False
        return True
    
    def compute_sum(selected):
        return sum(reciprocals[x] for x in selected)
    
    # Greedy with backtracking on first K elements
    def greedy_with_backtrack(first_k=8):
        best = None
        best_sum = Fraction(0)
        
        def backtrack(pos, selected, forbidden):
            nonlocal best, best_sum
            if pos > n:
                s = compute_sum(selected)
                if s > best_sum:
                    best_sum = s
                    best = set(selected)
                return
            
            # Upper bound
            ub = compute_sum(selected)
            for y in range(pos, n+1):
                if y not in forbidden:
                    ub += reciprocals[y]
            if ub <= best_sum:
                return
            
            # Try not selecting pos
            backtrack(pos+1, selected, forbidden)
            
            # Try selecting pos
            if pos in forbidden:
                return
            
            # Check if pos can be added
            can_add = True
            for (a, c) in midpoint_pairs[pos]:
                if a in selected and c in selected:
                    can_add = False
                    break
            if not can_add:
                return
            
            # Add pos, update forbidden
            new_forbidden = set(forbidden)
            for (a, c) in midpoint_pairs[pos]:
                if a in selected:
                    new_forbidden.add(c)
                if c in selected:
                    new_forbidden.add(a)
            # Also, for any a in selected, c = 2*pos - a; if c > pos and c <= n, forbid c
            for a in selected:
                c = 2*pos - a
                if c > pos and c <= n:
                    new_forbidden.add(c)
                a2 = 2*pos - a
                if 1 <= a2 < pos:
                    # This would mean (a2, pos, a) is a 3-AP, but a2 < pos, already processed
                    pass
            
            backtrack(pos+1, selected | {pos}, new_forbidden)
        
        backtrack(1, set(), set())
        return best if best else set()
    
    # Use greedy for a good initial solution
    def greedy_solution():
        selected = set()
        for x in range(1, n+1):
            can_add = True
            for (a, c) in midpoint_pairs[x]:
                if a in selected and c in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected
    
    initial = greedy_solution()
    best_set = set(initial)
    best_sum = compute_sum(best_set)
    
    # Try to improve with local search
    def local_search(selected, max_iter=2000):
        current_sum = compute_sum(selected)
        for _ in range(max_iter):
            improved = False
            # Try to add
            for x in range(1, n+1):
                if x in selected:
                    continue
                can_add = True
                for (a, c) in midpoint_pairs[x]:
                    if a in selected and c in selected:
                        can_add = False
                        break
                if can_add:
                    selected.add(x)
                    current_sum += reciprocals[x]
                    improved = True
                    break
            if improved:
                continue
            # Try to swap
            for a in list(selected):
                for b in range(1, n+1):
                    if b in selected:
                        continue
                    new_sel = (selected - {a}) | {b}
                    if is_3ap_fast(new_sel):
                        new_sum = current_sum - reciprocals[a] + reciprocals[b]
                        if new_sum > current_sum:
                            selected = new_sel
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return selected, current_sum
    
    best_set, best_sum = local_search(best_set)
    
    # Random restarts
    for _ in range(20):
        size = random.randint(25, 45)
        weights = [1.0/i for i in range(1, n+1)]
        random_set = set(random.choices(range(1, n+1), weights=weights, k=size))
        if is_3ap_fast(random_set):
            improved_set, improved_sum = local_search(random_set, max_iter=1000)
            if improved_sum > best_sum:
                best_set, best_sum = improved_set, improved_sum
    
    return list(best_set)
# EVOLVE-BLOCK-END
