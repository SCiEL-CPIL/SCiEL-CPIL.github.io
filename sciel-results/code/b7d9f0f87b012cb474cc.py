# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 600  # safety margin before 630s kill

    # Compute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)

    forbidden_set = forbidden  # alias

    def is_valid_addition(candidate, selected_set):
        for s in selected_set:
            d1 = (candidate - s) % m
            if d1 in forbidden_set:
                return False
        return True

    def greedy_init(order):
        selected = []
        selected_set = set()
        for c in order:
            if is_valid_addition(c, selected_set):
                selected.append(c)
                selected_set.add(c)
        return selected

    # Multiple greedy initializations with different orderings
    best = []
    
    # Natural order
    s1 = greedy_init(range(m))
    if len(s1) > len(best):
        best = s1[:]
    
    # Reverse order
    s2 = greedy_init(range(m - 1, -1, -1))
    if len(s2) > len(best):
        best = s2[:]
    
    # Random shuffles
    random.seed(42)
    for _ in range(20):
        order = list(range(m))
        random.shuffle(order)
        s = greedy_init(order)
        if len(s) > len(best):
            best = s[:]

    # Local search: try to add elements not in set, try swaps
    best_set = set(best)
    
    def try_add_to_set(selected_list, selected_set):
        """Try to add any candidate not in set."""
        added = False
        for c in range(m):
            if c not in selected_set:
                if is_valid_addition(c, selected_set):
                    selected_list.append(c)
                    selected_set.add(c)
                    added = True
        return added

    # Repeatedly try to add
    for _ in range(10):
        if not try_add_to_set(best, best_set):
            break
        if time.time() > deadline - 10:
            break

    # Local search with swaps: remove one, try to add two
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_list = best[:]
        best_set_local = set(best_list)
        
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            removed = best_list[i]
            new_list = best_list[:i] + best_list[i+1:]
            new_set = set(new_list)
            
            # Try to add as many as possible
            for c in range(m):
                if c not in new_set and c != removed:
                    if is_valid_addition(c, new_set):
                        new_list.append(c)
                        new_set.add(c)
            
            if len(new_list) > len(best_list):
                best = new_list
                best_set = set(best)
                improved = True
                break

    # More aggressive: random restarts with local search
    if time.time() < deadline - 60:
        random.seed(123)
        for restart in range(50):
            if time.time() > deadline - 30:
                break
            order = list(range(m))
            random.shuffle(order)
            current = greedy_init(order)
            current_set = set(current)
            
            # Local improvement
            for _ in range(5):
                if not try_add_to_set(current, current_set):
                    break
            
            # Swap-based local search
            for _ in range(3):
                if time.time() > deadline - 30:
                    break
                found_improvement = False
                for i in range(len(current)):
                    if time.time() > deadline - 30:
                        break
                    removed = current[i]
                    new_list = current[:i] + current[i+1:]
                    new_set = set(new_list)
                    for c in range(m):
                        if c not in new_set and c != removed:
                            if is_valid_addition(c, new_set):
                                new_list.append(c)
                                new_set.add(c)
                    if len(new_list) > len(current):
                        current = new_list
                        current_set = new_set
                        found_improvement = True
                        break
                if not found_improvement:
                    break
            
            if len(current) > len(best):
                best = current
                best_set = set(best)

    # Final verification
    best_set_final = set(best)
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            d1 = (best[i] - best[j]) % m
            d2 = (best[j] - best[i]) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                # Should not happen, but safety
                pass

    return np.array(sorted(best), dtype=np.int64)


# EVOLVE-BLOCK-END
