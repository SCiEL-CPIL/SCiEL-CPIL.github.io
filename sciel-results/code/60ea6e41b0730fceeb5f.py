import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    n = 512
    numbers = list(range(1, n + 1))
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3 * d <= n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element, list of AP indices it belongs to
    elem_aps = {x: [] for x in range(1, n + 1)}
    for i, ap in enumerate(aps):
        for x in ap:
            elem_aps[x].append(i)
    
    def greedy(order):
        selected = set()
        ap_count = [0] * len(aps)
        
        for x in order:
            can_add = True
            for ap_idx in elem_aps[x]:
                if ap_count[ap_idx] == 3:
                    can_add = False
                    break
            if can_add:
                selected.add(x)
                for ap_idx in elem_aps[x]:
                    ap_count[ap_idx] += 1
        return selected
    
    best_set = set()
    best_size = 0
    
    # Deterministic orderings
    for order in [numbers, numbers[::-1], 
                  sorted(numbers, key=lambda x: len(elem_aps[x])),
                  sorted(numbers, key=lambda x: -len(elem_aps[x]))]:
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Random orderings
    random.seed(42)
    while time.time() < deadline:
        order = numbers[:]
        random.shuffle(order)
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
