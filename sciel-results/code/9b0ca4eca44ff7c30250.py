import random
import time

def construct():
    deadline = time.time() + 85
    
    def is_valid(s):
        reachable = 1  # bit 0 set
        for x in s:
            reachable |= (reachable << x)
        return (reachable >> 200) & 1 == 0
    
    def greedy(order):
        selected = []
        reachable = 1
        for x in order:
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1 == 0:
                selected.append(x)
                reachable = new_reachable
        return selected
    
    # Try multiple orderings
    best_set = set()
    best_size = 0
    
    orderings = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
    ]
    
    # Add several random orderings
    for _ in range(20):
        random_order = list(range(1, 101))
        random.shuffle(random_order)
        orderings.append(random_order)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Local search: try to add elements not in best_set
    unselected = [x for x in range(1, 101) if x not in best_set]
    random.shuffle(unselected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in unselected:
            if time.time() > deadline:
                break
            new_set = best_set | {x}
            if is_valid(new_set):
                best_set = new_set
                best_size += 1
                unselected.remove(x)
                improved = True
                break
    
    # Local search: try swaps (remove one, add another)
    for _ in range(50):
        if time.time() > deadline:
            break
        unselected = [x for x in range(1, 101) if x not in best_set]
        if not unselected:
            break
        random.shuffle(unselected)
        found_swap = False
        for x in unselected:
            if time.time() > deadline:
                break
            for y in list(best_set):
                if time.time() > deadline:
                    break
                candidate = (best_set - {y}) | {x}
                if is_valid(candidate):
                    best_set = candidate
                    found_swap = True
                    break
            if found_swap:
                break
        if not found_swap:
            break
    
    return list(best_set)
