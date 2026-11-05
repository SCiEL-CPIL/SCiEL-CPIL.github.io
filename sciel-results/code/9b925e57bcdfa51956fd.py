import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def greedy(order):
        values = []
        sums = set()
        for x in order:
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    # Natural order
    best = greedy(range(1, 257))
    best_len = len(best)
    
    # Randomized restarts
    rng = random.Random(12345)
    candidates = list(range(1, 257))
    
    while time.time() < deadline:
        rng.shuffle(candidates)
        vals = greedy(candidates)
        if len(vals) > best_len:
            best_len = len(vals)
            best = vals
    
    return best
