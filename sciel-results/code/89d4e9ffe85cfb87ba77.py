import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls using positional weights
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse mapping: for each word, which codewords cover it
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    
    inv_arrays = [np.array(lst, dtype=np.int32) for lst in inv_lists]
    
    # Precompute overlap: overlap[i][j] = number of words in ball[i] ∩ ball[j]
    # This is expensive (6561^2 * 17), so we'll compute lazily or use a different approach
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    def greedy_init(seed=42):
        """Fast greedy initialization."""
        np.random.seed(seed)
        random.seed(seed)
        
        selected = []
        selected_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0 and time.time() < deadline - 15.0:
            # Compute gains vectorized
            gains = np.zeros(total, dtype=np.int32)
            for c in range(total):
                if not active[c]:
                    continue
                gains[c] = np.sum(cover_count[balls[c]] == 0)
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            candidates = np.where((gains == max_gain) & active)[0]
            if len(candidates) == 0:
                break
            
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            active[best_idx] = False
            
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                cover_count[w] += 1
        
        return selected, selected_set, cover_count, uncovered_count
    
    def prune_redundant(selected, selected_set, cover_count):
        """Remove all redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected_set.discard(c)
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, selected_set, cover_count
    
    def local_search(selected, selected_set, cover_count, max_iters=20000):
        """Advanced local search with multiple move types."""
        T = 3.0
        T_min = 0.01
        alpha = 0.9999
        stall_count = 0
        best_local = len(selected)
        best_local_set = selected.copy()
        best_local_cc = cover_count.copy()
        tabu = {}
        tabu_len = 50
        
        def get_critical(cover_count):
            return int(np.sum(cover_count == 1))
        
        current_critical = get_critical(cover_count)
        iter_count = 0
        
        while time.time() < deadline - 5.0 and T > T_min and iter_count < max_iters:
            iter_count += 1
            if len(selected) == 0:
                break
            
            c_old = selected[random.randint(0, len(selected) - 1)]
            ball_old = balls[c_old]
            cc_old = cover_count[ball_old]
            critical_mask = cc_old == 1
            n_critical = int(np.sum(critical_mask))
            
            if n_critical == 0:
                # Can remove
                cover_count[ball_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                current_critical = get_critical(cover_count)
                stall_count = 0
                if len(selected) < best_local:
                    best_local = len(selected)
                    best_local_set = selected.copy()
                    best_local_cc = cover_count.copy()
            elif n_critical <= 3:
                # Try to find replacement
                critical_words = ball_old[critical_mask]
                candidates = None
                
                if n_critical == 1:
                    w = critical_words[0]
                    cands = inv_arrays[w]
                    candidates = [c for c in cands if c != c_old and c not in selected_set]
                elif n_critical == 2:
                    w1, w2 = critical_words[0], critical_words[1]
                    set1 = set(inv_arrays[w1].tolist())
                    set2 = set(inv_arrays[w2].tolist())
                    common = set1.intersection(set2)
                    candidates = [c for c in common if c != c_old and c not in selected_set]
                else:
                    common = None
                    for w in critical_words:
                        if common is None:
                            common = set(inv_arrays[w].tolist())
                        else:
                            common.intersection_update(inv_arrays[w].tolist())
                    if common:
                        candidates = [c for c in common if c != c_old and c not in selected_set]
                    else:
                        candidates = []
                
                if not candidates:
                    stall_count += 1
                    if stall_count > 100:
                        break
                    T *= alpha
                    continue
                
                c_new = candidates[random.randint(0, len(candidates) - 1)]
                
                # Simulate swap
                old_cc_old = cc_old.copy()
                old_cc_new = cover_count[balls[c_new]].copy()
                
                cover_count[ball_old] -= 1
                if np.min(cover_count[ball_old]) < 0:
                    cover_count[ball_old] = old_cc_old
                    stall_count += 1
                    T *= alpha
                    continue
                
                cover_count[balls[c_new]] += 1
                new_critical = get_critical(cover_count)
                delta = new_critical - current_critical
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    selected.remove(c_old)
                    selected_set.discard(c_old)
                    selected.append(c_new)
                    selected_set.add(c_new)
                    current_critical = new_critical
                    stall_count = 0
                    if len(selected) < best_local:
                        best_local = len(selected)
                        best_local_set = selected.copy()
                        best_local_cc = cover_count.copy()
                else:
                    cover_count[balls[c_new]] -= 1
                    cover_count[ball_old] = old_cc_old
                    stall_count += 1
                    if stall_count > 100:
                        break
            
            T *= alpha
        
        # Return best found
        if len(best_local_set) < len(selected):
            return best_local_set, set(best_local_set), best_local_cc
        return selected, selected_set, cover_count
    
    # Main search loop
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    seed_idx = 0
    
    while time.time() < deadline - 10.0:
        seed = seeds[seed_idx % len(seeds)]
        seed_idx += 1
        
        selected, selected_set, cover_count, uncovered_count = greedy_init(seed)
        
        if uncovered_count > 0:
            continue
        
        # Prune
        selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Local search
        selected, selected_set, cover_count = local_search(selected, selected_set, cover_count)
        selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If close to best, do another round
        if len(selected) <= best_size + 3 and time.time() < deadline - 15.0:
            selected, selected_set, cover_count = local_search(selected, selected_set, cover_count, max_iters=10000)
            selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected.copy()
    
    if best_set is None:
        best_set = list(range(total))
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
