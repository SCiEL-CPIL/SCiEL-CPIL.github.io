import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82
    n = 512
    
    def can_add(sel, x):
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in sel and (x + 2*d) in sel and (x + 3*d) in sel:
                return False
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in sel and (x + d) in sel and (x + 2*d) in sel:
                return False
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d) in sel and (x - d) in sel and (x + d) in sel:
                return False
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d) in sel and (x - 2*d) in sel and (x - d) in sel:
                return False
        return True
    
    best_set = set()
    best_size = 0
    rng = random.Random(999)
    all_elements = list(range(1, n+1))
    
    def run_greedy(order):
        sel = set()
        for x in order:
            if can_add(sel, x):
                sel.add(x)
        # Fill in any remaining
        for x in all_elements:
            if x not in sel and can_add(sel, x):
                sel.add(x)
        return sel
    
    # Deterministic forward
    sel = run_greedy(all_elements)
    best_set = set(sel)
    best_size = len(sel)
    
    # Deterministic reverse
    sel = run_greedy(list(reversed(all_elements)))
    if len(sel) > best_size:
        best_set = set(sel)
        best_size = len(sel)
    
    # Randomized restarts
    while time.time() < deadline:
        order = all_elements[:]
        rng.shuffle(order)
        sel = run_greedy(order)
        if len(sel) > best_size:
            best_set = set(sel)
            best_size = len(sel)
    
    return sorted(best_set)
