import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 1000
    MAX_REP = 2

    def is_valid_addition(S, counts, x):
        for y in S:
            s = x + y
            if counts.get(s, 0) + 1 > MAX_REP:
                return False
        s = 2 * x
        if counts.get(s, 0) + 1 > MAX_REP:
            return False
        return True

    def add_to_set(S, counts, x):
        S.append(x)
        for y in S[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1

    def remove_from_set(S, counts, x):
        S.remove(x)
        for y in S:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]

    def greedy_construction(order):
        S = []
        counts = {}
        for x in order:
            if is_valid_addition(S, counts, x):
                add_to_set(S, counts, x)
        return S, counts

    # 1. Multiple Greedy Initializations
    best_set = []
    best_counts = {}
    best_size = 0
    
    # Standard ascending
    S, counts = greedy_construction(range(1, N + 1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        best_counts = counts

    # Randomized Greedy (a few tries)
    for _ in range(10):
        if time.time() > deadline - 5: break
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        S, counts = greedy_construction(random_order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            best_counts = counts

    # 2. Local Search on the best initial solution
    current_S = list(best_set)
    current_counts = dict(best_counts)
    current_size = len(current_S)
    S_set = set(current_S)
    all_numbers = list(range(1, N + 1))
    
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset to best found
            current_S = list(best_set)
            current_counts = dict(best_counts)
            current_size = len(best_set)
            S_set = set(current_S)
            no_improve_count = 0
            # Shuffle candidates to break symmetry
            random.shuffle(all_numbers)
            continue

        # Strategy: Try to add. If fail, try swap.
        # We prioritize additions as they increase the score.
        
        # 1. Try Additions
        # Pick random candidates
        added = False
        for _ in range(20):
            y = random.choice(all_numbers)
            if y not in S_set:
                if is_valid_addition(current_S, current_counts, y):
                    add_to_set(current_S, current_counts, y)
                    S_set.add(y)
                    current_size += 1
                    added = True
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_set = list(current_S)
                        best_counts = dict(current_counts)
                    break
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break
        
        if added:
            continue

        # 2. Try Swaps
        if current_size > 0:
            # Pick random x to remove
            x_idx = random.randint(0, len(current_S) - 1)
            x = current_S[x_idx]
            
            swapped = False
            for _ in range(10):
                y = random.choice(all_numbers)
                if y in S_set or y == x:
                    continue
                
                # Try Swap: Remove x, Add y
                remove_from_set(current_S, current_counts, x)
                S_set.remove(x)
                
                if is_valid_addition(current_S, current_counts, y):
                    add_to_set(current_S, current_counts, y)
                    S_set.add(y)
                    swapped = True
                    # We don't update best_size here as size is same
                    # But we reset no_improve to allow further exploration
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    add_to_set(current_S, current_counts, x)
                    S_set.add(x)
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break
            
            if swapped:
                # After a successful swap, immediately try to add again?
                # The loop will handle it in the next iteration.
                continue

    return best_set
