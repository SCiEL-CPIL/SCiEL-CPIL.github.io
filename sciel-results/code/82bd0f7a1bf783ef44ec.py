import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Precompute APs
    # AP is (a, a+d, a+2d, a+3d)
    # We store for each element x, the indices of APs it belongs to.
    # And for each AP, the list of elements.
    
    aps = []
    element_aps = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                continue
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            element_aps[a].append(ap_idx)
            element_aps[b].append(ap_idx)
            element_aps[c].append(ap_idx)
            element_aps[e].append(ap_idx)
            
    num_aps = len(aps)
    ap_count = [0] * num_aps
    in_set = [False] * (N + 1)
    current_size = 0
    
    def update_count(x, delta):
        nonlocal current_size
        for ap_idx in element_aps[x]:
            ap_count[ap_idx] += delta

    def can_add(x):
        for ap_idx in element_aps[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True

    def add(x):
        nonlocal current_size
        in_set[x] = True
        current_size += 1
        update_count(x, 1)

    def remove(x):
        nonlocal current_size
        in_set[x] = False
        current_size -= 1
        update_count(x, -1)
        
    def get_set():
        return [i for i in range(1, N + 1) if in_set[i]]

    def try_greedy_add():
        nonlocal current_size
        changed = True
        while changed:
            changed = False
            # Randomize order to avoid bias
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x):
                    add(x)
                    changed = True
                    break # Restart loop with new state

    # 2. Initialization: Greedy Add
    try_greedy_add()
    best_set = get_set()
    best_size = current_size
    
    # 3. Local Search with Kicks
    # We try to remove an element and then greedily add back.
    # If the size increases, we keep it.
    
    while time.time() < deadline:
        improved = False
        # Try a few kicks in each iteration
        for _ in range(10):
            if time.time() > deadline:
                break
            if current_size == 0:
                break
                
            # Pick a random element to remove
            x = random.choice(best_set)
            
            # Snapshot
            old_in_set = in_set[:]
            old_ap_count = ap_count[:]
            old_size = current_size
            
            # Remove x
            remove(x)
            
            # Try to add as many as possible
            try_greedy_add()
            
            if current_size > best_size:
                best_size = current_size
                best_set = get_set()
                improved = True
                # We can break here or continue to see if we can get even larger
                break 
            else:
                # Restore
                in_set = old_in_set
                ap_count = old_ap_count
                current_size = old_size
        
        if not improved:
            # If no improvement in this batch, we might be stuck.
            # We can try a "double kick" or just continue.
            # For simplicity, we just continue the loop.
            # To escape local optima, randomization in try_greedy_add helps.
            pass

    return sorted(best_set)
