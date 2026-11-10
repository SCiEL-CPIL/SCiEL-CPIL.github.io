"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    
    deadline = time.time() + 85.0
    random.seed(42)
    
    N = 256
    universe = list(range(2, N + 1))
    
    def conflicts(a, b):
        return math.lcm(a, b) <= N
    
    def is_valid(subset):
        s = list(subset)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if conflicts(s[i], s[j]):
                    return False
        return True
    
    def reciprocal_sum(subset):
        return sum(1.0 / x for x in subset)
    
    def greedy_by_weight(order):
        """Greedy: process in given order, add if no conflict."""
        selected = []
        for x in order:
            if all(not conflicts(x, y) for y in selected):
                selected.append(x)
        return selected
    
    # Multiple greedy passes with different orderings
    best = []
    best_score = 0.0
    
    # Pass 1: ascending (original)
    sol = greedy_by_weight(universe)
    if reciprocal_sum(sol) > best_score:
        best_score = reciprocal_sum(sol)
        best = sol[:]
    
    # Pass 2: descending (prefer large numbers, but they have small reciprocal)
    sol = greedy_by_weight(sorted(universe, reverse=True))
    if reciprocal_sum(sol) > best_score:
        best_score = reciprocal_sum(sol)
        best = sol[:]
    
    # Pass 3: random orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        order = universe[:]
        random.shuffle(order)
        sol = greedy_by_weight(order)
        s = reciprocal_sum(sol)
        if s > best_score:
            best_score = s
            best = sol[:]
    
    # Local search: try to add, remove, or swap elements
    best_set = set(best)
    current_score = best_score
    
    def try_improve(subset_list, score):
        """Local search with add/remove/swap moves."""
        nonlocal deadline
        improved = True
        cur_set = set(subset_list)
        cur_score = score
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding each unselected element
            for x in universe:
                if x in cur_set:
                    continue
                if all(not conflicts(x, y) for y in cur_set):
                    new_score = cur_score + 1.0 / x
                    if new_score > cur_score:
                        cur_set.add(x)
                        cur_score = new_score
                        improved = True
                        break  # restart after improvement
            
            if not improved:
                # Try swapping: remove one, add another
                for x in list(cur_set):
                    if time.time() > deadline:
                        break
                    cur_set.remove(x)
                    base_score = cur_score - 1.0 / x
                    
                    # Try adding any compatible element
                    for y in universe:
                        if y in cur_set:
                            continue
                        if all(not conflicts(y, z) for z in cur_set):
                            new_score = base_score + 1.0 / y
                            if new_score > cur_score:
                                cur_set.add(y)
                                cur_score = new_score
                                improved = True
                                break
                        if time.time() > deadline:
                            break
                    
                    if not improved:
                        cur_set.add(x)  # put x back
                    
                    if improved:
                        break
            
            if not improved:
                # Try removing one element (if it allows adding two better ones)
                for x in list(cur_set):
                    if time.time() > deadline:
                        break
                    cur_set.remove(x)
                    base_score = cur_score - 1.0 / x
                    
                    # Find best single addition
                    best_add = None
                    best_add_score = base_score
                    for y in universe:
                        if y in cur_set:
                            continue
                        if all(not conflicts(y, z) for z in cur_set):
                            s = base_score + 1.0 / y
                            if s > best_add_score:
                                best_add_score = s
                                best_add = y
                        if time.time() > deadline:
                            break
                    
                    if best_add is not None and best_add_score > cur_score:
                        cur_set.add(best_add)
                        cur_score = best_add_score
                        improved = True
                    else:
                        cur_set.add(x)  # put x back
                    
                    if improved:
                        break
        
        return list(cur_set), cur_score
    
    best, current_score = try_improve(best, current_score)
    
    # More random restarts with local search
    for _ in range(20):
        if time.time() > deadline:
            break
        order = universe[:]
        random.shuffle(order)
        sol = greedy_by_weight(order)
        s = reciprocal_sum(sol)
        if s > current_score:
            best, current_score = try_improve(sol, s)
    
    return best
# EVOLVE-BLOCK-END
