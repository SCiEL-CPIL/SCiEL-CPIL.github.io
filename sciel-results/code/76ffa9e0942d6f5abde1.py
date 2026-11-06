# EVOLVE-BLOCK-START
import time
import random
from array import array

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def construct():
    _t0 = time.time()
    _deadline = _t0 + 140  # 10s safety margin

    N = 1000
    MAX_SUM = 2000

    # Precompute all pairs (i,j) with i<=j, i,j in 1..N, grouped by sum
    # pair_sums[s] = list of (i, j) with i+j=s
    pair_sums = [[] for _ in range(MAX_SUM + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            pair_sums[i + j].append((i, j))

    def make_counts():
        return array('B', [0]) * (MAX_SUM + 1)

    def try_add(chosen_set, counts, x):
        """Try to add x. Returns True if valid, updates counts in place."""
        ok = True
        for y in chosen_set:
            if y > x:
                break
            s = x + y
            if counts[s] >= 2:
                ok = False
                break
        if ok:
            if counts[2 * x] >= 2:
                ok = False
        if ok:
            for y in chosen_set:
                if y > x:
                    break
                counts[x + y] += 1
            counts[2 * x] += 1
            return True
        return False

    def remove(chosen_set, counts, x):
        """Remove x from chosen_set, update counts in place."""
        for y in chosen_set:
            if y == x:
                continue
            if y > x:
                break
            counts[x + y] -= 1
        counts[2 * x] -= 1

    def greedy_construct(order, deadline):
        """Build a set by iterating through `order`, adding if valid."""
        chosen = []
        chosen_set = set()
        counts = make_counts()
        for x in order:
            if time.time() > deadline:
                break
            if try_add(chosen_set, counts, x):
                chosen.append(x)
                chosen_set.add(x)
        return chosen, counts

    def local_search(chosen, counts, deadline, max_rounds=50):
        """Improve via removal + re-addition moves."""
        chosen_set = set(chosen)
        best_size = len(chosen)
        
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            
            # Try removing a random element and re-adding a different one
            if chosen:
                idx = random.randrange(len(chosen))
                x = chosen[idx]
                remove(chosen_set, counts, x)
                chosen.pop(idx)
                
                # Try to re-add x (revert) or try adding other elements
                # First, try to re-add x
                if try_add(chosen_set, counts, x):
                    chosen.append(x)
                    chosen_set.add(x)
                    continue
                
                # Try adding other elements not in chosen
                candidates = []
                for y in range(1, N + 1):
                    if y not in chosen_set:
                        candidates.append(y)
                
                random.shuffle(candidates)
                added = False
                for y in candidates:
                    if time.time() > deadline:
                        break
                    if try_add(chosen_set, counts, y):
                        chosen.append(y)
                        chosen_set.add(y)
                        added = True
                        if len(chosen) > best_size:
                            best_size = len(chosen)
                        break
                
                if not added:
                    # Revert: add x back
                    if try_add(chosen_set, counts, x):
                        chosen.append(x)
                        chosen_set.add(x)
            
            # Try to add any missing element
            for y in range(1, N + 1):
                if time.time() > deadline:
                    break
                if y not in chosen_set:
                    if try_add(chosen_set, counts, y):
                        chosen.append(y)
                        chosen_set.add(y)
                        if len(chosen) > best_size:
                            best_size = len(chosen)
                        improved = True
            
            if not improved and time.time() > deadline - 20:
                break
        
        return chosen, counts, best_size

    # Generate diverse orderings
    all_orders = []
    
    # Ascending
    all_orders.append(list(range(1, N + 1)))
    # Descending
    all_orders.append(list(range(N, 0, -1)))
    # Random shuffles
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        all_orders.append(order)
    # Gap-based: prioritize numbers that create fewer collisions
    # (heuristic: numbers with fewer pair partners in range)
    for _ in range(5):
        order = list(range(1, N + 1))
        random.shuffle(order)
        all_orders.append(order)
    # Reverse gap
    for _ in range(5):
        order = list(range(1, N + 1))
        random.shuffle(order)
        order.reverse()
        all_orders.append(order)

    best_chosen = []
    best_counts = make_counts()
    best_size = 0

    for order in all_orders:
        if time.time() > _deadline:
            break
        
        chosen, counts = greedy_construct(order, _deadline)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen
            best_counts = counts
        
        # Local search to improve
        if time.time() > _deadline - 30:
            break
        
        chosen_ls, counts_ls, size_ls = local_search(chosen, counts, _deadline, max_rounds=30)
        
        if size_ls > best_size:
            best_size = size_ls
            best_chosen = chosen_ls
            best_counts = counts_ls

    # Final attempt: try to add any remaining elements
    if time.time() < _deadline - 5:
        chosen_set = set(best_chosen)
        for y in range(1, N + 1):
            if time.time() > _deadline:
                break
            if y not in chosen_set:
                if try_add(chosen_set, best_counts, y):
                    best_chosen.append(y)
                    chosen_set.add(y)
                    best_size = len(best_chosen)

    return best_chosen
# EVOLVE-BLOCK-END
