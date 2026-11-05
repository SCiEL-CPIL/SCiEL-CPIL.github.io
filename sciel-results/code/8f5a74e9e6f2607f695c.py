"""Rewritten program with CSP-style bitset forbidden-value tracking and heuristic backtracking."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n_max = 4096
    best = []
    best_len = 0

    def try_add(selected, sums_set, x):
        """Try adding x to selected. Returns True if successful, updates sums_set."""
        s = 3 * x
        if s in sums_set:
            return False
        for a in selected:
            s = 2 * x + a
            if s in sums_set:
                return False
        n = len(selected)
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                s = x + ai + selected[j]
                if s in sums_set:
                    return False
        sums_set.add(3 * x)
        for a in selected:
            sums_set.add(2 * x + a)
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                sums_set.add(x + ai + selected[j])
        return True

    def remove_element_sums(selected, sums_set, idx):
        """Remove all triple sums involving selected[idx]."""
        x = selected[idx]
        n = len(selected)
        removed = set()
        removed.add(3 * x)
        for i in range(n):
            if i == idx:
                continue
            a = selected[i]
            removed.add(2 * x + a)
        for i in range(n):
            if i == idx:
                continue
            ai = selected[i]
            for j in range(i, n):
                if j == idx:
                    continue
                removed.add(x + ai + selected[j])
        for s in removed:
            sums_set.discard(s)
        return removed

    def greedy_build(start_val, step_fn=None, max_iter=50000):
        """Greedy build with optional step function."""
        selected = []
        sums_set = set()
        count = 0
        if step_fn is None:
            x = start_val
            while x <= n_max:
                if time.time() > deadline:
                    break
                if try_add(selected, sums_set, x):
                    selected.append(x)
                x += 1
        else:
            x = start_val
            while x <= n_max and count < max_iter:
                if time.time() > deadline:
                    break
                if try_add(selected, sums_set, x):
                    selected.append(x)
                x = step_fn(x)
                count += 1
        return selected, sums_set

    def compute_forbidden(selected, sums_set):
        """For each candidate y not in selected, compute the number of conflicts if y were added.
        Returns a dict mapping y -> conflict_count (0 means valid)."""
        forbidden = {}
        n = len(selected)
        # For each y in range, check if adding y would cause a collision
        # We only need to check y values that could potentially be added
        # For efficiency, we'll check a window around the current max
        max_val = selected[-1] if selected else 0
        min_val = max_val + 1
        max_check = min(n_max, max_val + 200)
        
        for y in range(min_val, max_check + 1):
            conflicts = 0
            s = 3 * y
            if s in sums_set:
                conflicts += 1
            for a in selected:
                s = 2 * y + a
                if s in sums_set:
                    conflicts += 1
            for i in range(n):
                ai = selected[i]
                for j in range(i, n):
                    s = y + ai + selected[j]
                    if s in sums_set:
                        conflicts += 1
            forbidden[y] = conflicts
        return forbidden

    def backtrack_search(selected, sums_set, max_depth, budget, window_size=12):
        """Backtracking search with heuristic ordering by conflict count."""
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
        
        if len(selected) >= max_depth:
            return
        
        # Compute forbidden/conflict counts for candidates
        max_val = selected[-1] if selected else 0
        min_val = max_val + 1
        max_check = min(n_max, max_val + window_size)
        
        # Build list of (conflict_count, y) for candidates
        candidates = []
        for y in range(min_val, max_check + 1):
            conflicts = 0
            s = 3 * y
            if s in sums_set:
                conflicts += 1
            for a in selected:
                s = 2 * y + a
                if s in sums_set:
                    conflicts += 1
            n = len(selected)
            for i in range(n):
                ai = selected[i]
                for j in range(i, n):
                    s = y + ai + selected[j]
                    if s in sums_set:
                        conflicts += 1
            candidates.append((conflicts, y))
        
        # Sort by conflict count (fewest conflicts first - MRV heuristic)
        candidates.sort()
        
        for conflicts, x in candidates:
            if time.time() > deadline:
                return
            if budget[0] <= 0:
                return
            if conflicts == 0:
                if try_add(selected, sums_set, x):
                    selected.append(x)
                    backtrack_search(selected, sums_set, max_depth, budget, window_size)
                    selected.pop()
                    # Remove sums of x
                    n = len(selected)
                    sums_set.discard(3 * x)
                    for a in selected:
                        sums_set.discard(2 * x + a)
                    for i in range(n):
                        ai = selected[i]
                        for j in range(i, n):
                            sums_set.discard(x + ai + selected[j])
            budget[0] -= 1
    
    def local_search(selected, sums_set, max_iters=200):
        """Try to improve by swapping elements: remove one, try to add a larger one."""
        nonlocal best, best_len
        if time.time() > deadline:
            return selected, sums_set
        
        improved = True
        iteration = 0
        while improved and iteration < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            n = len(selected)
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    break
                x = selected[i]
                remove_element_sums(selected, sums_set, i)
                selected.pop(i)
                
                found = False
                for new_x in range(x + 1, min(x + 200, n_max + 1)):
                    if try_add(selected, sums_set, new_x):
                        selected.append(new_x)
                        selected.sort()
                        found = True
                        improved = True
                        break
                
                if not found:
                    selected.insert(i, x)
                    n2 = len(selected)
                    sums_set.add(3 * x)
                    for j in range(n2):
                        if j == i:
                            continue
                        a = selected[j]
                        sums_set.add(2 * x + a)
                    for j in range(n2):
                        if j == i:
                            continue
                        aj = selected[j]
                        for k in range(j, n2):
                            if k == i:
                                continue
                            sums_set.add(x + aj + selected[k])
        
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
        
        return selected, sums_set

    # === Strategy 1: Linear greedy from 1 ===
    selected, sums_set = greedy_build(1)
    if len(selected) > best_len:
        best = list(selected)
        best_len = len(selected)
    
    # === Strategy 2: Greedy from various offsets ===
    random.seed(42)
    for restart in range(80):
        if time.time() > deadline:
            break
        offset = random.randint(1, 300)
        selected, sums_set = greedy_build(offset)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    # === Strategy 3: Backtracking with MRV heuristic ===
    for restart in range(20):
        if time.time() > deadline:
            break
        offset = random.randint(1, 100)
        selected, sums_set = greedy_build(offset)
        budget = [1000]
        backtrack_search(selected, sums_set, len(selected) + 8, budget, window_size=15)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    # === Strategy 4: Exponential spacing ===
    for restart in range(25):
        if time.time() > deadline:
            break
        base = random.uniform(1.3, 3.0)
        def step_fn(x, base=base):
            return int(x * base) + 1
        selected, sums_set = greedy_build(1, step_fn)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    # === Strategy 5: Local search polish ===
    if best:
        sums_set = set()
        n = len(best)
        for i in range(n):
            ai = best[i]
            for j in range(i, n):
                aj = best[j]
                for k in range(j, n):
                    ak = best[k]
                    sums_set.add(ai + aj + ak)
        local_search(list(best), sums_set, max_iters=400)
    
    # === Strategy 6: Random skip greedy ===
    for restart in range(40):
        if time.time() > deadline:
            break
        selected = []
        sums_set = set()
        current = random.randint(1, 50)
        while current <= n_max:
            if time.time() > deadline:
                break
            if try_add(selected, sums_set, current):
                selected.append(current)
            skip = random.randint(1, 15)
            current += skip
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
    
    return best
# EVOLVE-BLOCK-END
