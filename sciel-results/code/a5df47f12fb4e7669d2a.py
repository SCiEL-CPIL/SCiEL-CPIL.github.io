"""Extended Sidon set search with simulated annealing and backtracking."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _check_sidon(a):
    """Check if a sorted list is a valid Sidon set (pairwise sums distinct)."""
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            s = a[i] + a[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _conflicts(a, x, sums_set):
    """Check if adding x to set a (with existing sums) creates conflicts.
    Returns True if x can be added without conflict."""
    for y in a:
        s = x + y
        if s in sums_set:
            return False
    if 2 * x in sums_set:
        return False
    return True

def _rebuild_sums(a):
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            sums.add(a[i] + a[j])
    return sums

def _greedy_extend(a, sums):
    """Greedily extend the Sidon set by scanning 12..284."""
    a = list(a)
    sums = set(sums)
    for x in range(12, 285):
        if x in a:
            continue
        ok = True
        for y in a:
            if (x + y) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            a.append(x)
            for y in a:
                if y != x:
                    sums.add(x + y)
            sums.add(2 * x)
    a.sort()
    return a, sums

def _sa_improve(a, max_time=30):
    """Simulated annealing to improve the Sidon set size.
    Moves: remove an element, add a different element.
    Objective: maximize size, subject to Sidon constraint."""
    a = sorted(a)
    n = len(a)
    sums = _rebuild_sums(a)
    best = list(a)
    best_n = n
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.01
    iterations = 50000
    deadline = time.time() + max_time
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / iterations)
        
        # Random move: either remove or add
        if random.random() < 0.4 and len(a) > 4:
            # Remove a random element (not one of the prescribed {1,2,5,11})
            removable = [i for i, x in enumerate(a) if x not in (1, 2, 5, 11)]
            if not removable:
                continue
            idx = random.choice(removable)
            x = a[idx]
            new_a = a[:idx] + a[idx+1:]
            new_sums = _rebuild_sums(new_a)
            new_n = len(new_a)
            
            # Energy: we want to maximize size, so "cost" = -size
            # But we need validity. If valid, accept based on size change.
            delta_cost = -(new_n - n)  # negative if size increased
            if delta_cost <= 0 or random.random() < _exp_accept(delta_cost, T):
                a = new_a
                sums = new_sums
                n = new_n
                if n > best_n:
                    best = list(a)
                    best_n = n
        else:
            # Try to add a random element
            candidates = [x for x in range(1, 285) if x not in a]
            if not candidates:
                continue
            x = random.choice(candidates)
            if not _conflicts(a, x, sums):
                new_a = sorted(a + [x])
                new_sums = _rebuild_sums(new_a)
                new_n = len(new_a)
                delta_cost = -(new_n - n)
                if delta_cost <= 0 or random.random() < _exp_accept(delta_cost, T):
                    a = new_a
                    sums = new_sums
                    n = new_n
                    if n > best_n:
                        best = list(a)
                        best_n = n
            else:
                # Could try swap: remove one, add another
                if len(a) > 4:
                    removable = [i for i, x2 in enumerate(a) if x2 not in (1, 2, 5, 11)]
                    if removable:
                        idx = random.choice(removable)
                        x2 = a[idx]
                        new_a = sorted([x3 for i3, x3 in enumerate(a) if i3 != idx] + [x])
                        if _check_sidon(new_a):
                            new_sums = _rebuild_sums(new_a)
                            new_n = len(new_a)
                            delta_cost = -(new_n - n)
                            if delta_cost <= 0 or random.random() < _exp_accept(delta_cost, T):
                                a = new_a
                                sums = new_sums
                                n = new_n
                                if n > best_n:
                                    best = list(a)
                                    best_n = n
    return best

def _exp_accept(delta_cost, T):
    """Metropolis acceptance probability."""
    if T <= 0:
        return 0.0
    import math
    if delta_cost < 0:
        return 1.0
    if delta_cost > 50 * T:
        return 0.0
    return math.exp(-delta_cost / T)

def _backtrack_construct():
    """Backtracking search to find a large Sidon set containing {1,2,5,11}."""
    prescribed = {1, 2, 5, 11}
    a = [1, 2, 5, 11]
    sums = _rebuild_sums(a)
    best = list(a)
    best_n = 4
    
    def bt(current_a, current_sums, start_x):
        nonlocal best, best_n
        if len(current_a) > best_n:
            best = list(current_a)
            best_n = len(current_a)
        
        for x in range(start_x, 285):
            if x in current_a:
                continue
            if _conflicts(current_a, x, current_sums):
                new_a = current_a + [x]
                new_sums = set(current_sums)
                for y in current_a:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                bt(new_a, new_sums, x + 1)
    
    bt(a, sums, 12)
    return best

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with prescribed elements
    a = [1, 2, 5, 11]
    sums = _rebuild_sums(a)
    
    # Phase 1: Greedy extension
    a, sums = _greedy_extend(a, sums)
    best = list(a)
    best_n = len(a)
    
    if time.time() > deadline - 30:
        return best
    
    # Phase 2: Multiple SA restarts with different random seeds
    random.seed(42)
    sa_budget = 15
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        # Start from current best, perturb slightly
        start_a = list(best)
        # Randomly remove a few non-prescribed elements to diversify
        if len(start_a) > 4:
            removable = [i for i, x in enumerate(start_a) if x not in (1, 2, 5, 11)]
            if removable:
                k = min(3, len(removable))
                to_remove = random.sample(removable, k)
                to_remove.sort(reverse=True)
                for idx in to_remove:
                    start_a.pop(idx)
        
        improved = _sa_improve(start_a, max_time=sa_budget)
        if len(improved) > best_n:
            best = improved
            best_n = len(improved)
    
    if time.time() > deadline - 15:
        return best
    
    # Phase 3: Try backtracking with limited depth for final improvement
    # Only if we have time
    if time.time() < deadline - 5 and best_n < 20:
        # Use a time-limited backtracking
        bt_deadline = time.time() + min(10, max(0, deadline - time.time() - 2))
        # Simple limited backtracking
        a2 = list(best)
        sums2 = _rebuild_sums(a2)
        
        def bt_limited(current_a, current_sums, start_x, depth):
            nonlocal best, best_n
            if time.time() > bt_deadline:
                return
            if len(current_a) > best_n:
                best = list(current_a)
                best_n = len(current_a)
            if depth <= 0:
                return
            for x in range(start_x, 285):
                if x in current_a:
                    continue
                if _conflicts(current_a, x, current_sums):
                    new_a = current_a + [x]
                    new_sums = set(current_sums)
                    for y in current_a:
                        new_sums.add(x + y)
                    new_sums.add(2 * x)
                    bt_limited(new_a, new_sums, x + 1, depth - 1)
                    if time.time() > bt_deadline:
                        return
        
        bt_limited(a2, sums2, 12, 3)
    
    return best

# EVOLVE-BLOCK-END
