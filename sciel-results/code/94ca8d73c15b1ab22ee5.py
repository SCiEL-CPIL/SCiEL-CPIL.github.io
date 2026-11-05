import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            # Optimization: if reachable is large, this set comprehension might be slow
            # But max sum is 200, so reachable size is at most 201
            new = set()
            for s in reachable:
                val = s + x
                if val <= 200:
                    new.add(val)
            if 200 not in new:
                selected.append(x)
                reachable.update(new)
        return selected

    best_set = []
    
    # 1. Descending order (Seed strategy)
    s1 = greedy(range(100, 0, -1))
    if len(s1) > len(best_set):
        best_set = s1

    # 2. Ascending order
    s2 = greedy(range(1, 101))
    if len(s2) > len(best_set):
        best_set = s2

    # 3. Randomized orders
    # We can run many iterations within the time limit
    # Each iteration is O(100 * |reachable|). |reachable| <= 201.
    # So ~20k ops per iteration. Very fast.
    # 10,000 iterations is ~200M ops. Might take a few seconds in Python.
    # Let's run as many as possible.
    
    base_list = list(range(1, 101))
    
    while time.time() < deadline:
        random.shuffle(base_list)
        s_rand = greedy(base_list)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            # If we hit 100, we can't do better
            if len(best_set) == 100:
                break

    # Ensure non-empty
    if not best_set:
        best_set = [1]
        
    return best_set
