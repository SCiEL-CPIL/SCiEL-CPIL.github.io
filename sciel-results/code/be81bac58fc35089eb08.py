import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0
    best_set = []
    best_size = 0
    
    def try_greedy(order):
        chosen = []
        counts = [0] * 2001
        for x in order:
            valid = True
            for y in chosen:
                if counts[x + y] >= 2:
                    valid = False
                    break
            if valid:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
        return chosen

    # Initial deterministic greedy
    best_set = try_greedy(range(1, 1001))
    best_size = len(best_set)
    
    # Randomized greedy
    while time.time() - start_time < time_limit:
        order = list(range(1, 1001))
        random.shuffle(order)
        current_set = try_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set
