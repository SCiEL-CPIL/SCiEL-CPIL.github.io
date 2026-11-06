"""Search for 12-element ambient set in [0,128] minimizing largest Sidon subset."""
import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def is_sidon(subset):
        """Check if subset has all unordered pair sums (with repetition) unique."""
        s = sorted(subset)
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True

    def max_sidon_size_fast(ambient):
        """Find largest Sidon subset via branch and bound."""
        ambient = sorted(ambient)
        n = len(ambient)
        
        # Precompute all pair sums and their frequencies
        # Use backtracking with pruning
        
        def backtrack(idx, chosen, sums_set, current_size):
            """Recursively build Sidon subset, tracking best found."""
            nonlocal best_found
            if current_size >= best_found:
                return
            if idx == n:
                return
            
            # Option 1: skip ambient[idx]
            backtrack(idx + 1, chosen, sums_set, current_size)
            
            # Option 2: include ambient[idx] if it doesn't create conflicts
            val = ambient[idx]
            # Check if adding val creates any duplicate sum with existing chosen elements
            can_add = True
            new_sums = []
            for existing in chosen:
                sm = val + existing
                if sm in sums_set:
                    can_add = False
                    break
                new_sums.append(sm)
            if can_add:
                sm_diag = val + val
                if sm_diag in sums_set:
                    can_add = False
                else:
                    new_sums.append(sm_diag)
            
            if can_add:
                # Add val and its sums
                sums_set.update(new_sums)
                chosen.append(val)
                backtrack(idx + 1, chosen, sums_set, current_size + 1)
                chosen.pop()
                for s in new_sums:
                    sums_set.discard(s)
        
        best_found = 0
        backtrack(0, [], set(), 0)
        return best_found

    def count_conflicts(ambient):
        """Count all pair-sum conflicts (including diagonals)."""
        ambient = sorted(ambient)
        n = len(ambient)
        sum_count = {}
        for i in range(n):
            for j in range(i, n):
                sm = ambient[i] + ambient[j]
                sum_count[sm] = sum_count.get(sm, 0) + 1
        conflicts = 0
        for sm, cnt in sum_count.items():
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts

    # Generate structured candidates with many APs
    candidates = []
    
    # Pure arithmetic progressions with various steps
    for d in range(1, 11):
        for start in range(0, 129 - 11*d):
            cand = [start + i*d for i in range(12)]
            if all(0 <= x <= 128 for x in cand):
                candidates.append(cand)
    
    # Two interleaved APs
    for d1 in range(1, 8):
        for d2 in range(d1+1, 12):
            for start in range(0, 129):
                cand = []
                for i in range(6):
                    cand.append(start + i*d1)
                for i in range(6):
                    cand.append(start + 1 + i*d2)
                if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                    candidates.append(cand)
    
    # Dense small sets: use numbers 0-20 with many overlaps
    for base in range(0, 10):
        cand = [base + i for i in range(12)]
        if all(0 <= x <= 128 for x in cand):
            candidates.append(cand)
    
    # Mix of two APs with different starts
    for d in range(2, 8):
        for s1 in range(0, 20):
            for s2 in range(s1+1, min(s1+d*3, 129)):
                cand1 = [s1 + i*d for i in range(6)]
                cand2 = [s2 + i*d for i in range(6)]
                cand = cand1 + cand2
                if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                    candidates.append(cand)
    
    # Three interleaved APs of length 4
    for d in range(2, 10):
        for start in range(0, 129 - 3*d):
            cand = []
            for offset in range(3):
                for i in range(4):
                    cand.append(start + offset + i*d)
            if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                candidates.append(cand)
    
    # Quadratic-like sequences: i*i mod something, or i*(i+1)//2 style
    for shift in range(0, 50):
        cand = [shift + i*i for i in range(12)]
        if all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
            candidates.append(cand)
    
    # Evaluate candidates, prioritizing those with more conflicts
    scored_candidates = []
    for cand in candidates:
        if len(cand) == 12 and len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            conf = count_conflicts(cand)
            scored_candidates.append((conf, cand))
    
    # Sort by conflict count descending (more conflicts = likely smaller max Sidon)
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_set = None
    best_size = 12
    
    for conf, cand in scored_candidates:
        if time.time() > deadline - 10:
            break
        sz = max_sidon_size_fast(cand)
        if sz < best_size:
            best_size = sz
            best_set = list(cand)
        if best_size <= 5:  # Good enough, stop early
            break
    
    if best_set is None:
        best_set = list(range(12))
        best_size = max_sidon_size_fast(best_set)
    
    # Local search to improve
    current = list(best_set)
    current_size = best_size
    random.seed(42)
    
    # Single-element replacement random search
    for iteration in range(8000):
        if time.time() > deadline - 15:
            break
        if current_size <= 4:
            break
        
        # Try replacing one element with a random value
        i = random.randint(0, 11)
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
        trial = list(current)
        trial[i] = new_val
        if len(set(trial)) != 12:
            continue
        
        trial_size = max_sidon_size_fast(trial)
        if trial_size < current_size:
            current = trial
            current_size = trial_size
            if current_size < best_size:
                best_size = current_size
                best_set = list(current)
    
    # Two-element swap local search
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
        if current_size <= 4:
            break
        
        i, j = random.sample(range(12), 2)
        new_i = random.randint(0, 128)
        new_j = random.randint(0, 128)
        if new_i in current or new_j in current:
            continue
        if new_i == new_j:
            continue
        trial = list(current)
        trial[i] = new_i
        trial[j] = new_j
        if len(set(trial)) != 12:
            continue
        
        trial_size = max_sidon_size_fast(trial)
        if trial_size < current_size:
            current = trial
            current_size = trial_size
            if current_size < best_size:
                best_size = current_size
                best_set = list(current)
    
    # Systematic local search: try all values for each position
    for round_num in range(3):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(12):
            if time.time() > deadline - 5:
                break
            for new_val in range(0, 129):
                if new_val == current[i] or new_val in current:
                    continue
                trial = list(current)
                trial[i] = new_val
                trial_size = max_sidon_size_fast(trial)
                if trial_size < current_size:
                    current = trial
                    current_size = trial_size
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(current)
                    break
        if not improved:
            break
    
    if best_set is None:
        best_set = current
    
    # Verify
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
