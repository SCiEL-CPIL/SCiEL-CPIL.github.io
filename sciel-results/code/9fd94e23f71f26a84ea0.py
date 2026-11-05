import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _greedy_build(order, cap, counts_buf=None):
    """Greedy build from a given order. Returns (chosen_list, counts_array)."""
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen_np = np.empty(0, dtype=np.int64)
    
    for x in order:
        if chosen_np.size == 0:
            new_sums = np.array([2 * x], dtype=np.int64)
        else:
            new_sums = np.concatenate((chosen_np + x, [2 * x]))
        
        if np.all(counts[new_sums] < 2):
            chosen.append(x)
            chosen_np = np.concatenate((chosen_np, [x]))
            counts[new_sums] += 1
    
    return chosen, counts

def _local_improve(chosen, counts, cap, rng, max_steps=200):
    """Local search: try remove-one-add-two or remove-one-add-one to grow the set."""
    chosen_set = set(chosen)
    chosen_list = list(chosen)
    n = len(chosen_list)
    
    # Precompute which sums are at capacity (count == 2)
    # For each element in chosen, find sums it participates in
    # that are at capacity
    
    improved = True
    steps = 0
    while improved and steps < max_steps and _time_left() > 3.0:
        improved = False
        steps += 1
        
        # Try removing one element and adding one or two others
        # Sort chosen by some heuristic - try removing elements that block the most additions
        # For simplicity, try random removals
        
        for _ in range(50):  # try 50 random removals per step
            if _time_left() < 3.0:
                break
            idx = rng.randint(0, n - 1)
            x_remove = chosen_list[idx]
            
            # Temporarily remove x_remove
            new_counts = counts.copy()
            new_chosen = [v for v in chosen_list if v != x_remove]
            new_chosen_np = np.array(new_chosen, dtype=np.int64)
            
            # Subtract x_remove's contributions from counts
            if len(new_chosen_np) > 0:
                for y in new_chosen_np:
                    new_counts[x_remove + y] -= 1
            new_counts[2 * x_remove] -= 1
            
            # Now try to add two new elements
            candidates = [v for v in range(1, cap + 1) if v not in chosen_set and v != x_remove]
            rng.shuffle(candidates)
            
            added = 0
            added_elems = []
            for v in candidates:
                if added >= 2:
                    break
                # Check if v can be added
                if len(added_elems) == 0 and len(new_chosen) == 0:
                    sums_to_check = np.array([2 * v], dtype=np.int64)
                elif len(added_elems) == 0:
                    sums_to_check = np.concatenate((np.array(new_chosen, dtype=np.int64) + v, [2 * v]))
                else:
                    # Already added one, check against existing + the first added
                    first_added = added_elems[0]
                    sums_to_check = np.concatenate((
                        np.array(new_chosen, dtype=np.int64) + v,
                        [first_added + v, 2 * v]
                    ))
                
                if np.all(new_counts[sums_to_check] < 2):
                    # Add v
                    new_counts[sums_to_check] += 1
                    new_chosen.append(v)
                    added_elems.append(v)
                    added += 1
            
            # If we added at least 1 element, it's an improvement
            if added >= 1:
                # Commit the change
                chosen_list = new_chosen
                chosen_set = set(chosen_list)
                counts = new_counts
                n = len(chosen_list)
                improved = True
                break
    
    return chosen_list, counts

def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0
    
    cap = 1000
    best = []
    best_len = 0
    rng = random.Random(42)
    
    while _time_left() > 5.0:
        # Random order
        order = list(range(1, cap + 1))
        rng.shuffle(order)
        
        # Greedy Build
        chosen, counts = _greedy_build(order, cap)
        
        # Local improvement
        chosen, counts = _local_improve(chosen, counts, cap, rng, max_steps=100)
        
        if len(chosen) > best_len:
            best_len = len(chosen)
            best = chosen
    
    return best
