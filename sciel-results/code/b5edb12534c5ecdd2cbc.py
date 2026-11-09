import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0
    N = 4096
    MAX_SIZE = 80
    random.seed(123)

    def can_add(selected, x, triple_sum_set):
        new_sums = set()
        new_sums.add(3 * x)
        for a in selected:
            new_sums.add(2 * x + a)
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                new_sums.add(x + selected[i] + selected[j])
        expected = 3 + len(selected) + len(selected)*(len(selected)+1)//2
        if len(new_sums) != expected:
            return False, None
        if new_sums & triple_sum_set:
            return False, None
        return True, new_sums

    def build_greedy(order):
        selected = []
        triple_sum_set = set()
        for x in order:
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 15:
                break
            ok, new_sums = can_add(selected, x, triple_sum_set)
            if ok:
                selected.append(x)
                triple_sum_set.update(new_sums)
        return selected, triple_sum_set

    def local_search(selected, triple_sum_set, max_iters=500):
        for _ in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            if len(selected) >= MAX_SIZE:
                break
            if not selected:
                break
            
            idx = random.randint(0, len(selected)-1)
            trial = selected[:idx] + selected[idx+1:]
            
            trial_sums = set()
            for i in range(len(trial)):
                for j in range(i, len(trial)):
                    for k in range(j, len(trial)):
                        trial_sums.add(trial[i]+trial[j]+trial[k])
            
            candidates = [x for x in range(1, N+1) if x not in set(trial)]
            random.shuffle(candidates)
            
            for x in candidates[:300]:
                if time.time() > DEADLINE - 10:
                    break
                ok, new_sums = can_add(trial, x, trial_sums)
                if ok:
                    return trial + [x], trial_sums | new_sums
        
        return selected, triple_sum_set

    # Multiple greedy strategies
    best_size = 0
    best_sel = []
    best_tset = set()
    
    # Strategy 1: Sequential
    sel, tset = build_greedy(range(1, N+1))
    if len(sel) > best_size:
        best_size, best_sel, best_tset = len(sel), sel, tset
    
    # Strategy 2: Reverse
    if time.time() < DEADLINE - 20:
        sel, tset = build_greedy(range(N, 0, -1))
        if len(sel) > best_size:
            best_size, best_sel, best_tset = len(sel), sel, tset
    
    # Strategy 3: Power of 3
    if time.time() < DEADLINE - 20:
        p3 = []
        v = 1
        while v <= N:
            p3.append(v)
            v *= 3
        rest = [x for x in range(1, N+1) if x not in set(p3)]
        sel, tset = build_greedy(p3 + rest)
        if len(sel) > best_size:
            best_size, best_sel, best_tset = len(sel), sel, tset
    
    # Strategy 4-6: Random
    for _ in range(3):
        if time.time() > DEADLINE - 25:
            break
        o = list(range(1, N+1))
        random.shuffle(o)
        sel, tset = build_greedy(o)
        if len(sel) > best_size:
            best_size, best_sel, best_tset = len(sel), sel, tset
    
    # Local search on best
    if time.time() < DEADLINE - 15:
        best_sel, best_tset = local_search(best_sel, best_tset)
        best_size = len(best_sel)
    
    return best_sel

