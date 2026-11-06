import time
import random
from fractions import Fraction

DEADLINE = 75.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _find_one_subset_sum(vals, target):
    n = len(vals)
    if n == 0:
        return None if target != 0 else []
    
    vals_sorted = sorted(vals, reverse=True)
    
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + vals_sorted[i]
        
    if suffix_sums[0] < target:
        return None

    def dfs(idx, remaining, current_subset):
        if remaining == 0:
            return current_subset[:]
        if idx >= n:
            return None
        if remaining < 0:
            return None
        if suffix_sums[idx] < remaining:
            return None
            
        current_subset.append(vals_sorted[idx])
        res = dfs(idx + 1, remaining - vals_sorted[idx], current_subset)
        if res is not None:
            return res
        current_subset.pop()
        
        res = dfs(idx + 1, remaining, current_subset)
        if res is not None:
            return res
            
        return None

    return dfs(0, target, [])

def get_conflicts(S):
    conflicts = []
    S_list = list(S)
    recips = {v: Fraction(1, v) for v in S_list}
    
    for a in S_list:
        others = [recips[v] for v in S_list if v != a]
        frac_to_elem = {Fraction(1, v): v for v in S_list if v != a}
        
        subset_fracs = _find_one_subset_sum(others, recips[a])
        if subset_fracs is not None:
            conflict_elems = {a}
            for f in subset_fracs:
                conflict_elems.add(frac_to_elem[f])
            conflicts.append(conflict_elems)
            
    return conflicts

def greedy_hitting_set(conflicts, universe):
    if not conflicts:
        return set()
    
    to_remove = set()
    remaining_conflicts = [set(c) for c in conflicts]
    
    while remaining_conflicts and not _deadline_hit():
        active_conflicts = []
        for c in remaining_conflicts:
            if c & to_remove:
                continue
            active_conflicts.append(c)
            
        if not active_conflicts:
            break
            
        freq = {}
        for c in active_conflicts:
            for e in c:
                freq[e] = freq.get(e, 0) + 1
        
        if not freq:
            break
            
        max_freq = max(freq.values())
        candidates = [e for e, f in freq.items() if f == max_freq]
        
        random.shuffle(candidates)
        chosen = candidates[0]
        
        to_remove.add(chosen)
        remaining_conflicts = [c for c in active_conflicts if chosen not in c]
        
    return to_remove

def construct():
    global _START
    _START = time.time()
    
    best_set = set(range(13, 25))
    best_size = len(best_set)
    
    rng = random.Random(42)
    universe = set(range(1, 25))
    
    for iteration in range(2000):
        if _deadline_hit():
            break
            
        current_set = set(universe)
        
        conflicts = get_conflicts(current_set)
        
        if not conflicts:
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            continue
            
        to_remove = greedy_hitting_set(conflicts, current_set)
        
        if not to_remove:
            continue
            
        current_set -= to_remove
        
        candidates_to_add = list(to_remove)
        rng.shuffle(candidates_to_add)
        
        for elem in candidates_to_add:
            if _deadline_hit():
                break
            trial_set = current_set | {elem}
            if get_conflicts(trial_set) == []:
                current_set = trial_set
                
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    if not best_set:
        best_set = set(range(13, 25))
        
    return list(best_set)

if __name__ == "__main__":
    print(construct())
