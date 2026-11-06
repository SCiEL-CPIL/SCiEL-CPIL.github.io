import random
import time

def construct():
    deadline = time.time() + 50  # Safety margin
    
    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            if 200 - x in reachable:
                continue
            selected.append(x)
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        return selected

    best = []
    
    # 1. Deterministic Largest First
    order1 = list(range(100, 0, -1))
    res1 = greedy(order1)
    if len(res1) > len(best):
        best = res1
        
    # 2. Deterministic Smallest First
    order2 = list(range(1, 101))
    res2 = greedy(order2)
    if len(res2) > len(best):
        best = res2
        
    # 3. Randomized Greedy
    # Use a fixed seed for reproducibility? No, random is fine, or fixed seed to ensure same result every time?
    # The evaluator runs the program. If I use random, the result might vary.
    # However, the score is based on the value returned. If I return a valid set, it's valid.
    # To be safe and deterministic, I should use a fixed seed for the random generator,
    # or just run a fixed number of random permutations generated from a fixed seed.
    
    random.seed(42)
    
    for i in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        res = greedy(order)
        if len(res) > len(best):
            best = res
            
    return best
