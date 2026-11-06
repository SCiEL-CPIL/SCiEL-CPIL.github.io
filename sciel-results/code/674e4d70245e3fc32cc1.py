import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_sidon(a):
    """Check if sorted list a is a Sidon set. Returns (bool, set_of_diffs)."""
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False, diffs
            diffs.add(d)
    return True, diffs

def _covers_1_to_30(diffs):
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _known_constructions():
    """Generate candidate sets from known perfect difference set constructions."""
    results = []
    # Known perfect difference sets of size 9
    results.append([0, 1, 4, 10, 12, 28, 32, 45, 63])
    return results

def _greedy_set_cover(deadline):
    """Greedy set-covering: iteratively select element maximizing new diffs covered per diameter cost."""
    best_set = None
    best_diam = float('inf')
    
    for seed in range(5):
        if time.time() > deadline:
            break
        rng = random.Random(seed * 1000 + 42)
        a = [0]
        used_diffs = set()
        covered = set()
        
        for _ in range(30):
            if time.time() > deadline:
                break
            best_x = None
            best_score = -1
            best_new_covered = set()
            
            for x in range(1, 100):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    d = x - v
                    if d <= 0:
                        continue
                    if d in used_diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                
                new_covered = new_diffs & set(range(1, 31)) - covered
                # Score: new required diffs covered, penalize large x
                score = len(new_covered) * 100 - x
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_covered = new_covered
            
            if best_x is None:
                break
            
            a.append(best_x)
            a.sort()
            used_diffs |= best_new_covered
            covered |= best_new_covered
            
            if _covers_1_to_30(used_diffs):
                break
        
        if _covers_1_to_30(used_diffs) and len(a) <= 32:
            diam = a[-1]
            if diam < best_diam:
                best_diam = diam
                best_set = list(a)
    
    return best_set, best_diam

def _backtrack_optimal(deadline):
    """Find minimal diameter Sidon set covering diffs 1..30 using bitset."""
    best_diam = [float('inf')]
    best_set = [None]
    target_mask = 0
    for i in range(1, 31):
        target_mask |= (1 << i)
    
    def rec(S, used_mask, covered_mask, current_max):
        if time.time() > deadline:
            return
        if current_max >= best_diam[0]:
            return
        if (covered_mask & target_mask) == target_mask:
            if current_max < best_diam[0]:
                best_diam[0] = current_max
                best_set[0] = list(S)
            return
        n = len(S)
        if n >= 32:
            return
        # Prune: max new diffs from remaining slots
        remaining = 32 - n
        max_new = remaining * n
        missing_count = 0
        for d in range(1, 31):
            if not (covered_mask & (1 << d)):
                missing_count += 1
        if max_new < missing_count:
            return
        
        start = S[-1] + 1 if S else 1
        upper = min(best_diam[0] - 1, 80)
        if upper < start:
            return
        
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
            ok = True
            new_mask = 0
            for v in S:
                d = x - v
                bit = 1 << d
                if (used_mask & bit) or (new_mask & bit):
                    ok = False
                    break
                new_mask |= bit
            if not ok:
                continue
            new_used = used_mask | new_mask
            new_covered = covered_mask | new_mask
            S.append(x)
            rec(S, new_used, new_covered, x)
            S.pop()
    
    rec([0], 0, 0, 0)
    return best_set[0], best_diam[0]

def _local_search(a, deadline, max_iter=5000):
    """Local search to minimize diameter while maintaining Sidon property and coverage."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    rng = random.Random(42)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(a)
        if n < 2:
            break
        
        found = False
        # Try to reduce the largest element first
        last_idx = n - 1
        old_val = a[last_idx]
        
        for new_val in range(1, old_val):
            if new_val in a:
                continue
            trial = a[:last_idx] + [new_val]
            trial = sorted(trial)
            ok, diffs = _is_sidon(trial)
            if ok and _covers_1_to_30(diffs):
                a = trial
                best = list(a)
                best_diam = a[-1]
                found = True
                break
        
        if not found:
            # Try replacing any non-zero element
            for i in range(n - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = a[i]
                for new_val in range(1, old_val):
                    if new_val in a:
                        continue
                    trial = a[:i] + [new_val] + a[i+1:]
                    trial = sorted(trial)
                    ok, diffs = _is_sidon(trial)
                    if ok and _covers_1_to_30(diffs):
                        diam = trial[-1]
                        if diam < best_diam:
                            a = trial
                            best = list(a)
                            best_diam = diam
                            found = True
                            break
                if found:
                    break
        
        if not found:
            # Random perturbation
            if rng.random() < 0.05:
                new_a = list(a)
                idx = rng.randint(1, n - 1)
                new_val = rng.randint(1, best_diam)
                if new_val not in new_a:
                    new_a[idx] = new_val
                    new_a = sorted(new_a)
                    ok, diffs = _is_sidon(new_a)
                    if ok and _covers_1_to_30(diffs):
                        if new_a[-1] < best_diam:
                            a = new_a
                            best = list(a)
                            best_diam = a[-1]
    
    return best, best_diam

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Try known constructions
    for candidate in _known_constructions():
        if time.time() > deadline:
            break
        ok, diffs = _is_sidon(candidate)
        if ok and _covers_1_to_30(diffs):
            diam = candidate[-1]
            if diam < best_diam:
                best_diam = diam
                best_set = list(candidate)
    
    # Phase 2: Greedy set-covering
    if time.time() < deadline - 30:
        gset, gdiam = _greedy_set_cover(deadline - 30)
        if gset is not None and gdiam < best_diam:
            best_set = gset
            best_diam = gdiam
    
    # Phase 3: Backtracking search
    if time.time() < deadline - 20:
        result, diam = _backtrack_optimal(deadline - 20)
        if result is not None and diam < best_diam:
            best_set = result
            best_diam = diam
    
    # Phase 4: Local search to improve
    if best_set is not None and time.time() < deadline - 5:
        improved, improved_diam = _local_search(best_set, deadline - 5)
        if improved_diam < best_diam:
            best_set = improved
            best_diam = improved_diam
    
    # Fallback: greedy construction
    if best_set is None:
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            found = False
            for x in range(1, 200):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    diff = abs(x - v)
                    if diff in used or diff in new_diffs:
                        ok = False
                        break
                    new_diffs.add(diff)
                if ok:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                x = max(a) + d + 1
                while True:
                    new_diffs = set()
                    ok = True
                    for v in a:
                        diff = x - v
                        if diff in used or diff in new_diffs:
                            ok = False
                            break
                        new_diffs.add(diff)
                    if ok:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                        break
                    x += 1
        best_set = a
        best_diam = max(a)
    
    return best_set

# EVOLVE-BLOCK-END
