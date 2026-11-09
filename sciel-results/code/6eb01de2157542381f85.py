import time
import random
from collections import defaultdict

def _greedy_seeded(n=1000, seed=42):
    """Seeded greedy: pick smallest available x that keeps B2[2]."""
    random.seed(seed)
    chosen = []
    counts = defaultdict(int)
    for x in range(1, n + 1):
        new_sums = [x + y for y in chosen]
        new_sums.append(2 * x)
        ok = True
        for s in new_sums:
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
    return chosen

def _greedy_random(n=1000, seed=42):
    """Random-order greedy."""
    random.seed(seed)
    pool = list(range(1, n + 1))
    random.shuffle(pool)
    chosen = []
    counts = defaultdict(int)
    for x in pool:
        ok = True
        for y in chosen:
            if counts[x + y] >= 2:
                ok = False
                break
        if ok and counts[2 * x] >= 2:
            ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
    return chosen

def _local_search(base_set, n=1000, seed=42, time_budget=2.0):
    """Local search: try to add or swap elements to grow the set."""
    random.seed(seed)
    chosen = sorted(base_set)
    counts = defaultdict(int)
    for i, a in enumerate(chosen):
        for b in chosen[i:]:
            counts[a + b] += 1
    chosen_set = set(chosen)
    
    deadline = time.time() + time_budget
    best = len(chosen)
    best_set = list(chosen)
    
    # Repeatedly try to add a random candidate
    while time.time() < deadline:
        # Try adding
        candidates = [x for x in range(1, n + 1) if x not in chosen_set]
        random.shuffle(candidates)
        added = False
        for x in candidates[:20]:
            ok = True
            for y in chosen:
                if counts[x + y] >= 2:
                    ok = False
                    break
            if ok and counts[2 * x] >= 2:
                ok = False
            if ok:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_set.add(x)
                added = True
                if len(chosen) > best:
                    best = len(chosen)
                    best_set = list(chosen)
                break
        
        if not added:
            # Try swap: remove one, add another
            if not chosen:
                break
            x_remove = random.choice(chosen)
            # Temporarily remove
            for y in chosen:
                if y != x_remove:
                    counts[x_remove + y] -= 1
            counts[2 * x_remove] -= 1
            chosen.remove(x_remove)
            chosen_set.remove(x_remove)
            
            swapped = False
            for x_add in candidates[:20]:
                if x_add in chosen_set:
                    continue
                ok = True
                for y in chosen:
                    if counts[x_add + y] >= 2:
                        ok = False
                        break
                if ok and counts[2 * x_add] >= 2:
                    ok = False
                if ok:
                    for y in chosen:
                        counts[x_add + y] += 1
                    counts[2 * x_add] += 1
                    chosen.append(x_add)
                    chosen_set.add(x_add)
                    swapped = True
                    break
            
            if not swapped:
                # Revert
                for y in chosen:
                    counts[x_remove + y] += 1
                counts[2 * x_remove] += 1
                chosen.append(x_remove)
                chosen_set.add(x_remove)
                # No progress; perturb
                if len(chosen) > 10:
                    x = random.choice(chosen)
                    for y in chosen:
                        if y != x:
                            counts[x + y] -= 1
                    counts[2 * x] -= 1
                    chosen.remove(x)
                    chosen_set.remove(x)
                    # Try to add a random element
                    for x2 in candidates[:10]:
                        if x2 in chosen_set:
                            continue
                        ok = True
                        for y in chosen:
                            if counts[x2 + y] >= 2:
                                ok = False
                                break
                        if ok and counts[2 * x2] >= 2:
                            ok = False
                        if ok:
                            for y in chosen:
                                counts[x2 + y] += 1
                            counts[2 * x2] += 1
                            chosen.append(x2)
                            chosen_set.add(x2)
                            break
    
    return best_set

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    best_size = 0
    best_result = []
    
    # Phase 1: Greedy baselines with multiple seeds
    for seed in [42, 137, 256, 777, 1234]:
        if time.time() > deadline - 60:
            break
        g = _greedy_seeded(1000, seed)
        if len(g) > best_size:
            best_size = len(g)
            best_result = list(g)
    
    # Phase 2: Random-order greedy with multiple seeds
    for seed in [42, 137, 256, 777, 1234, 9999, 5555, 8888]:
        if time.time() > deadline - 40:
            break
        g = _greedy_random(1000, seed)
        if len(g) > best_size:
            best_size = len(g)
            best_result = list(g)
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Local search from best
    remaining = deadline - time.time() - 10
    if remaining > 5:
        # Split time among a few restarts
        n_restarts = min(5, max(1, int(remaining / 8)))
        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            budget = min(remaining / (n_restarts - i), 12.0)
            seed = 100 + i * 7
            # Start from a slightly perturbed version
            start = list(best_result)
            random.seed(seed)
            # Randomly swap a few elements
            for _ in range(3):
                if len(start) > 2:
                    i1 = random.randint(0, len(start) - 1)
                    i2 = random.randint(0, len(start) - 1)
                    if i1 != i2:
                        start[i1], start[i2] = start[i2], start[i1]
            ls_result = _local_search(start, 1000, seed, budget)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_result = list(ls_result)
    
    return best_result
