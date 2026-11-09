"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

DEADLINE = 325.0  # wall-clock safety margin before 330s kill


def _is_isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    for i in range(len(chosen)):
        a = chosen[i]
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            if _is_isosceles(p, a, b):
                return False
    return True


def _greedy_maximal(start_order, chosen):
    """Continue greedy: for each point in start_order not in chosen, add if possible."""
    chosen_set = set(chosen)
    for p in start_order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _try_remove_and_refill(chosen, rng, max_iterations=5000):
    """Try removing random points and re-filling greedily to get a smaller maximal set."""
    best = list(chosen)
    best_len = len(best)
    for _ in range(max_iterations):
        if len(chosen) <= best_len:
            continue
        # Pick a random point to remove
        idx = rng.randint(0, len(chosen) - 1)
        removed = chosen[idx]
        new_chosen = chosen[:idx] + chosen[idx + 1:]
        # Re-fill greedily in random order
        remaining = [p for p in itertools.product(range(N), repeat=2) if p not in set(new_chosen)]
        rng.shuffle(remaining)
        new_chosen = _greedy_maximal(remaining, list(new_chosen))
        if len(new_chosen) < best_len:
            best = list(new_chosen)
            best_len = len(best)
            chosen = list(new_chosen)
        else:
            # Keep original if no improvement
            pass
    return best


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_len = float('inf')
    
    # Multiple randomized greedy restarts
    max_restarts = 50
    restarts_done = 0
    while restarts_done < max_restarts:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Randomize the order
        order = list(all_points)
        rng.shuffle(order)
        
        # Greedy construction
        chosen = []
        for p in order:
            if _can_add(p, chosen):
                chosen.append(p)
        
        # Ensure maximality by filling in remaining points
        chosen = _greedy_maximal(all_points, chosen)
        
        if len(chosen) < best_len:
            best = list(chosen)
            best_len = len(chosen)
        
        restarts_done += 1
    
    # Local search: remove-and-refill on the best solution
    if best is not None and time.time() - start_time < DEADLINE - 10:
        # Try local search with time limit
        local_start = time.time()
        max_iter = 20000
        iter_count = 0
        current = list(best)
        current_len = len(current)
        
        while iter_count < max_iter and time.time() - local_start < 20:
            # Pick a random point to remove
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            new_current = current[:idx] + current[idx + 1:]
            
            # Re-fill greedily in random order
            remaining = [p for p in all_points if p not in set(new_current)]
            rng.shuffle(remaining)
            new_current = _greedy_maximal(remaining, list(new_current))
            
            if len(new_current) < current_len:
                current = list(new_current)
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
            
            iter_count += 1
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 5:
        return best if best is not None else []
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
