import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    universe = list(range(1, 23))
    BASE = 232792560
    scaled = {i: BASE // i for i in range(1, 23)}

    best_set = []
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)

    def is_dissociated(S):
        if not S:
            return True
        sums = {0}
        for x in S:
            val = scaled[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s + val)
            if len(sums | new_sums) < len(sums) + len(new_sums):
                return False
            sums = sums | new_sums
        return True

    # Initial Greedy to get a good lower bound
    # Sort by "difficulty" or just reverse
    current_S = []
    for x in sorted(universe, reverse=True):
        if is_dissociated(current_S + [x]):
            current_S.append(x)
    update_best(current_S)
    
    # Try a few specific good sets
    test_sets = [
        [1, 2, 4, 8, 16, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [1, 2, 4, 8, 16, 3, 6, 12, 24], # 24 not in universe
        [1, 2, 4, 8, 16, 3, 5, 10, 20]
    ]
    for ts in test_sets:
        valid_ts = [x for x in ts if x in universe]
        if is_dissociated(valid_ts):
            update_best(valid_ts)

    # Optimized DFS
    def dfs_search(order, current_idx, current_set, current_sums_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return

        # Pruning
        if len(current_set) + (22 - current_idx) <= best_size:
            return

        if current_idx == 22:
            if len(current_set) > best_size:
                update_best(current_set)
            return

        x = order[current_idx]
        
        # Option 1: Exclude x
        dfs_search(order, current_idx + 1, current_set, current_sums_set)

        # Option 2: Include x
        val = scaled[x]
        
        # Check collision: is val in differences of current_sums_set?
        # i.e., exists s1, s2 in current_sums_set such that s1 - s2 = val
        can_add = True
        # Optimization: if current_sums_set is large, this loop is slow.
        # But current_sums_set size is 2^k.
        for s in current_sums_set:
            if s - val in current_sums_set:
                can_add = False
                break
        
        if can_add:
            # Create new sums set
            # new_sums = current_sums_set U {s + val for s in current_sums_set}
            new_sums_set = set(current_sums_set)
            for s in current_sums_set:
                new_sums_set.add(s + val)
            
            dfs_search(order, current_idx + 1, current_set + [x], new_sums_set)

    # Run searches
    random.seed(42)
    num_searches = 5000
    
    for i in range(num_searches):
        if time.time() > deadline - 5:
            break
        
        order = universe.copy()
        random.shuffle(order)
        
        # Start DFS
        dfs_search(order, 0, [], {0})
        
        # If we find a very large set, we can reduce the effort or break
        if best_size >= 18:
             # 18 is likely max. Run a few more to confirm, then break if time is low
             if time.time() > deadline - 10:
                 break

    if not best_set:
        best_set = [22]

    return best_set

if __name__ == "__main__":
    print(construct())
