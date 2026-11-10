import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 12
    max_val_limit = 1048576

    def _valid(arr):
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums |= new_sums
        return len(sums) == (1 << n)

    def _greedy_construct(start_val, max_val_limit):
        arr = []
        sums = {0}
        current_min = start_val
        
        # Add start_val
        if start_val <= max_val_limit:
            arr.append(start_val)
            sums |= {start_val}
            current_min = start_val + 1
        else:
            return None

        for _ in range(1, n):
            chosen = None
            # Search for next element
            # We can search from current_min upwards
            # But we can also search from 1 if we don't enforce sorted order?
            # The task requires distinct positive integers. Order doesn't matter for validity.
            # However, for greedy, sorted is standard.
            # Let's search from 1 to max_val_limit to find the smallest valid one.
            # This is O(max_val) which is too slow if max_val is large.
            # But we can prune.
            # If we enforce sorted, we search from current_min.
            
            found = False
            # Optimization: check collisions quickly
            for val in range(current_min, max_val_limit + 1):
                if val in sums:
                    continue
                new_sums = {s + val for s in sums}
                if not (new_sums & sums):
                    arr.append(val)
                    sums |= new_sums
                    current_min = val + 1
                    found = True
                    break
            
            if not found:
                return None
        return arr

    # Generate candidates
    candidates = []
    
    # Standard greedy (starts with 1)
    g1 = _greedy_construct(1, max_val_limit)
    if g1:
        candidates.append(g1)

    # Greedy starting with 2
    g2 = _greedy_construct(2, max_val_limit)
    if g2:
        candidates.append(g2)
        
    # Greedy starting with 3
    g3 = _greedy_construct(3, max_val_limit)
    if g3:
        candidates.append(g3)

    if not candidates:
        return [1 << i for i in range(n)]

    # Pick the best candidate (lowest max)
    best = min(candidates, key=lambda x: max(x))
    best_max = max(best)

    # Refine
    def _refine(arr, budget):
        arr = arr[:]
        if not _valid(arr):
            return None
        cur_max = max(arr)
        best = arr[:]
        best_max = cur_max
        T = 50.0
        T_min = 0.1
        alpha = 0.99

        while time.time() < budget and T > T_min:
            neighbor = arr[:]
            r = random.random()
            if r < 0.6:
                idx = random.randrange(n)
                old = neighbor[idx]
                # Try smaller
                new = max(1, old - random.randint(1, max(1, old // 4)))
                if new != old:
                    neighbor[idx] = new
            elif r < 0.8:
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                idx = random.randrange(n)
                old = neighbor[idx]
                new = old + random.randint(1, max(1, old // 4))
                if new <= max_val_limit:
                    neighbor[idx] = new

            sums = {0}
            valid = True
            for x in neighbor:
                new_sums = {s + x for s in sums}
                if new_sums & sums:
                    valid = False
                    break
                sums |= new_sums
            
            if valid and len(sums) == (1 << n):
                cost = max(neighbor)
            else:
                cost = max(neighbor) + ((1 << n) - len(sums)) * 1000

            delta = cost - cur_max
            if delta < 0:
                arr = neighbor
                cur_max = cost
                if valid and cost < best_max:
                    best_max = cost
                    best = neighbor[:]
            else:
                if T > T_min:
                    if random.random() < 2.71828 ** (-delta / T):
                        arr = neighbor
                        cur_max = cost
            T *= alpha
        return best

    if time.time() < deadline:
        refined = _refine(best, deadline)
        if refined is not None and max(refined) < best_max:
            best = refined
            best_max = max(best)

    return best
