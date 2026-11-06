import time
import sys
from functools import lru_cache

def _compute_table():
    """Pre-compute minimal antichain representations for targets 1..128."""
    start_time = time.time()
    deadline = 80.0  # Leave 10s safety margin before 90s kill
    
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    reps = [None] * 129
    
    # k=1: smooth numbers
    for t in range(1, 129):
        if t in smooth:
            reps[t] = (t,)
    
    # k=2: pairs
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for i in range(len(smooth_list)):
            a = smooth_list[i]
            if a >= t:
                break
            b = t - a
            if b < a:
                continue
            if b in smooth and a % b != 0 and b % a != 0:
                reps[t] = (a, b)
                break
    
    # k=3: triples
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for i in range(len(smooth_list)):
            a = smooth_list[i]
            if a >= t // 3:
                break
            for j in range(i + 1, len(smooth_list)):
                b = smooth_list[j]
                if a + b >= t:
                    break
                c = t - a - b
                if c <= b:
                    continue
                if c in smooth:
                    if (b % a != 0) and (c % a != 0) and (c % b != 0):
                        reps[t] = (a, b, c)
                        break
            if reps[t] is not None:
                break
    
    # Check if we need to do DFS
    unresolved = [t for t in range(1, 129) if reps[t] is None]
    
    if unresolved:
        # k=4 and above: DFS with time guard and better pruning
        def dfs(rem, start_idx, current_terms, k_remaining):
            if time.time() > deadline:
                return None
            if k_remaining == 0:
                return current_terms if rem == 0 else None
            if rem <= 0:
                return None
            # Pruning: if remaining sum can't be achieved with k_remaining terms
            min_possible = smooth_list[start_idx] if start_idx < len(smooth_list) else float('inf')
            if min_possible * k_remaining > rem:
                return None
            for idx in range(start_idx, len(smooth_list)):
                val = smooth_list[idx]
                if val > rem:
                    break
                # Check antichain property
                valid = True
                for existing in current_terms:
                    if val % existing == 0 or existing % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                res = dfs(rem - val, idx + 1, current_terms + [val], k_remaining - 1)
                if res is not None:
                    return res
            return None
        
        for t in unresolved:
            if time.time() > deadline:
                break
            for k in range(4, 11):
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(res)
                    break
    
    # Fallback: if any targets still unresolved, use a simple greedy approach
    for t in range(1, 129):
        if reps[t] is None:
            # Try to find any valid representation
            found = False
            for k in range(1, 11):
                if found:
                    break
                from itertools import combinations_with_replacement
                for combo in combinations_with_replacement(smooth_list, k):
                    if sum(combo) == t:
                        valid = True
                        for i in range(len(combo)):
                            for j in range(i+1, len(combo)):
                                if combo[i] % combo[j] == 0 or combo[j] % combo[i] == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            reps[t] = tuple(combo)
                            found = True
                            break
    
    return reps

_TABLE = _compute_table()

def construct():
    return [list(_TABLE[t]) for t in range(1, 129)]
