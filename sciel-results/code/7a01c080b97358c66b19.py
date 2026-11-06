import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin (90s limit)
    
    n = 192
    max_size = 48
    
    def is_valid_set(S):
        """Verify the entire set S is valid."""
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    s = S_list[i] + S_list[j] + S_list[k]
                    for x in S_list:
                        if x != S_list[i] and x != S_list[j] and x != S_list[k]:
                            if s % x == 0:
                                return False
        return True

    def try_add(selected, triple_sums, pair_sums_excl, candidate):
        m = len(selected)
        
        # Check 1: candidate divides sum of any 3 distinct elements in selected
        if m >= 3:
            for s in triple_sums:
                if s % candidate == 0:
                    return False, None, None
            
        # Check 2: for each a in selected, a divides (b+c+candidate) for distinct b,c in selected, b,c != a
        # Equivalent to: (-candidate) % a is in pair_sums_excl[a]
        for a in selected:
            target_res = (-candidate) % a
            if target_res in pair_sums_excl[a]:
                return False, None, None
        
        # If valid, compute new state
        new_triple_sums = list(triple_sums)
        
        # Add new triple sums involving candidate
        for i in range(m):
            u = selected[i]
            for j in range(i + 1, m):
                v = selected[j]
                new_triple_sums.append(u + v + candidate)
        
        new_pair_sums_excl = {}
        
        # Update pair_sums_excl for existing elements
        for a in selected:
            old_set = pair_sums_excl[a]
            new_set = set(old_set)
            for u in selected:
                if u != a:
                    new_set.add(candidate + u)
            new_pair_sums_excl[a] = new_set
            
        # Create pair_sums_excl for the new candidate
        new_candidate_pairs = set()
        for i in range(m):
            u = selected[i]
            for j in range(i + 1, m):
                v = selected[j]
                new_candidate_pairs.add(u + v)
        new_pair_sums_excl[candidate] = new_candidate_pairs
        
        return True, new_triple_sums, new_pair_sums_excl

    best_set = []
    current_best = 0
    
    # Initial greedy: Descending order
    selected = []
    triple_sums = []
    pair_sums_excl = {}
    
    for x in range(n, 0, -1):
        if time.time() > deadline - 3.0:
            break
        valid, new_ts, new_ps = try_add(selected, triple_sums, pair_sums_excl, x)
        if valid:
            selected.append(x)
            triple_sums = new_ts
            pair_sums_excl = new_ps
        if len(selected) >= max_size:
            break
            
    if len(selected) > current_best:
        current_best = len(selected)
        best_set = list(selected)
        
    # Random restarts
    candidates_base = list(range(1, n + 1))
    
    # Increase restart count as can_add is faster
    max_restarts = 300 
    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
        
        candidates = candidates_base[:]
        random.shuffle(candidates)
        
        selected = []
        triple_sums = []
        pair_sums_excl = {}
        
        for x in candidates:
            if len(selected) >= max_size:
                break
            valid, new_ts, new_ps = try_add(selected, triple_sums, pair_sums_excl, x)
            if valid:
                selected.append(x)
                triple_sums = new_ts
                pair_sums_excl = new_ps
                
        if len(selected) > current_best:
            current_best = len(selected)
            best_set = list(selected)
            
    # Final validity check
    if best_set and not is_valid_set(best_set):
        # Fallback
        selected = []
        triple_sums = []
        pair_sums_excl = {}
        for x in range(n, 0, -1):
            valid, new_ts, new_ps = try_add(selected, triple_sums, pair_sums_excl, x)
            if valid:
                selected.append(x)
                triple_sums = new_ts
                pair_sums_excl = new_ps
            if len(selected) >= max_size:
                break
        best_set = selected

    return best_set[:max_size]
