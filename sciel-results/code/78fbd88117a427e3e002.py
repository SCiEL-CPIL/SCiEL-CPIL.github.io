import time
import random
import sys

N = 256
DEADLINE = 58.0

def _sidon_sums(S):
    sums = set()
    for i, a in enumerate(S):
        for b in S[i:]:
            sums.add(a + b)
    return sums

def _is_sidon(S):
    sums = set()
    for i, a in enumerate(S):
        for b in S[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _addable_elements(S, sums):
    S_set = set(S)
    addable = []
    for y in range(1, N + 1):
        if y in S_set:
            continue
        new_sums = [y + a for a in S] + [2 * y]
        if len(set(new_sums)) == len(new_sums) and not (sums & set(new_sums)):
            addable.append(y)
    return addable

def _is_maximal(S):
    sums = _sidon_sums(S)
    return len(_addable_elements(S, sums)) == 0

def _make_maximal(S):
    S = list(S)
    sums = _sidon_sums(S)
    S_set = set(S)
    changed = True
    while changed:
        changed = False
        # Iterate in increasing order for deterministic greedy, or random?
        # Increasing is fine for a baseline.
        for y in range(1, N + 1):
            if y in S_set:
                continue
            new_sums = [y + a for a in S] + [2 * y]
            if len(set(new_sums)) == len(new_sums) and not (sums & set(new_sums)):
                S.append(y)
                S_set.add(y)
                sums.update(new_sums)
                changed = True
    return S

def _local_search_minimize(S, t_start):
    S = list(S)
    
    while time.time() - t_start < DEADLINE - 2.0:
        improved = False
        
        # 1. Try single removals
        for i in range(len(S)):
            if time.time() - t_start > DEADLINE - 2.0:
                break
            u = S[i]
            S_trial = [x for j, x in enumerate(S) if x != u]
            if _is_maximal(S_trial):
                S = S_trial
                improved = True
                break
        
        if improved:
            continue

        # 2. If no removal works, try a swap
        # Find u in S, v not in S such that (S \ {u}) U {v} is Sidon
        S_set = set(S)
        sums = _sidon_sums(S)
        
        # Precompute sums for S \ {u} is expensive if done for all u.
        # Just try a few random swaps.
        swap_found = False
        # Try a limited number of random swaps to avoid O(N^2) per step
        for _ in range(20):
            if time.time() - t_start > DEADLINE - 2.0:
                break
            if len(S) == 0:
                break
            u = random.choice(S)
            v = random.randint(1, N)
            if v in S_set:
                continue
            
            S_trial = [x for x in S if x != u]
            # Check if v can be added to S_trial
            new_sums = [v + a for a in S_trial] + [2 * v]
            if len(set(new_sums)) == len(new_sums):
                sums_trial = _sidon_sums(S_trial)
                if not (sums_trial & set(new_sums)):
                    # Valid swap
                    S = S_trial + [v]
                    S.sort()
                    # Make it maximal
                    S = _make_maximal(S)
                    swap_found = True
                    break
        
        if not swap_found:
            # No valid swap found in random sample, break
            break
            
    return S

def construct():
    t_start = time.time()
    best_set = None
    best_size = N + 1
    
    while time.time() - t_start < DEADLINE - 1.0:
        # Random restart
        S = []
        sums = set()
        S_set = set()
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            new_sums = [x + a for a in S] + [2 * x]
            if len(set(new_sums)) == len(new_sums) and not (sums & set(new_sums)):
                S.append(x)
                S_set.add(x)
                sums.update(new_sums)
                
        S_max = _make_maximal(S)
        S_min = _local_search_minimize(S_max, t_start)
        
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_set = S_min
            print(f"New best: {best_size}", file=sys.stderr)
            
        if best_size <= 17:
            break

    if best_set is None:
        S = _make_maximal([1, 2, 4])
        best_set = _local_search_minimize(S, t_start)
        
    return best_set
