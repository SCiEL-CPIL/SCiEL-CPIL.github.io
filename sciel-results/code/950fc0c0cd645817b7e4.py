"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    def _check_sidon(sets):
        """Check if each set is a Sidon set."""
        for s in sets:
            if len(s) == 0:
                continue
            sums = set()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    sm = s[i] + s[j]
                    if sm in sums:
                        return False
                    sums.add(sm)
        return True
    
    def _check_disjoint_diffs(sets):
        """Check if nonzero difference sets are disjoint."""
        diffs = [set() for _ in range(2)]
        for side in range(2):
            s = sets[side]
            for i in range(len(s)):
                for j in range(len(s)):
                    if i != j:
                        d = s[i] - s[j]
                        if d != 0:
                            diffs[side].add(d)
        return len(diffs[0] & diffs[1]) == 0
    
    def _is_valid(sets):
        return _check_sidon(sets) and _check_disjoint_diffs(sets)
    
    def _score(sets):
        return len(sets[0]) + len(sets[1])
    
    def _greedy_random(seed_val, balance=True):
        """Randomized greedy construction with optional balancing."""
        random.seed(seed_val)
        sets = [[], []]
        used_elements = set()
        
        # Precompute all candidate elements
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Determine which side to try
            if balance:
                # Alternate or pick shorter side
                if len(sets[0]) <= len(sets[1]):
                    sides = [0, 1]
                else:
                    sides = [1, 0]
            else:
                sides = random.sample([0, 1], 2)
            
            added = False
            for side in sides:
                candidate_set = sets[side] + [x]
                if len(candidate_set) > max_size:
                    continue
                
                # Check Sidon property
                valid = True
                sums = set()
                for i in range(len(candidate_set)):
                    for j in range(i, len(candidate_set)):
                        sm = candidate_set[i] + candidate_set[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                if not valid:
                    continue
                
                # Check disjoint differences
                new_diffs = set()
                for i in range(len(candidate_set)):
                    for j in range(len(candidate_set)):
                        if i != j:
                            d = candidate_set[i] - candidate_set[j]
                            if d != 0:
                                new_diffs.add(d)
                
                other_side = 1 - side
                other_diffs = set()
                for i in range(len(sets[other_side])):
                    for j in range(len(sets[other_side])):
                        if i != j:
                            d = sets[other_side][i] - sets[other_side][j]
                            if d != 0:
                                other_diffs.add(d)
                
                if new_diffs & other_diffs:
                    continue
                
                # Valid addition
                sets[side] = candidate_set
                added = True
                break
        
        return sets
    
    def _local_search(sets, max_iters=100):
        """Try to improve by swapping elements."""
        random.seed(int(time.time() * 1000) % 10000)
        best = [list(sets[0]), list(sets[1])]
        best_score = _score(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding a random element to either set
            for side in random.sample([0, 1], 2):
                if len(best[side]) >= max_size:
                    continue
                # Pick a random candidate
                candidates = [x for x in range(1, n + 1) if x not in best[0] and x not in best[1]]
                if not candidates:
                    continue
                x = random.choice(candidates)
                candidate_set = best[side] + [x]
                
                # Quick check
                valid = True
                sums = set()
                for i in range(len(candidate_set)):
                    for j in range(i, len(candidate_set)):
                        sm = candidate_set[i] + candidate_set[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                if not valid:
                    continue
                
                new_diffs = set()
                for i in range(len(candidate_set)):
                    for j in range(len(candidate_set)):
                        if i != j:
                            d = candidate_set[i] - candidate_set[j]
                            if d != 0:
                                new_diffs.add(d)
                
                other_side = 1 - side
                other_diffs = set()
                for i in range(len(best[other_side])):
                    for j in range(len(best[other_side])):
                        if i != j:
                            d = best[other_side][i] - best[other_side][j]
                            if d != 0:
                                other_diffs.add(d)
                
                if new_diffs & other_diffs:
                    continue
                
                # Valid - accept
                best[side] = candidate_set
                improved = True
                break
            
            if not improved:
                break
        
        return best
    
    # Run multiple randomized greedy attempts
    best_sets = None
    best_score = 0
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        balance = attempt % 2 == 0
        sets = _greedy_random(attempt, balance=balance)
        
        if _is_valid(sets):
            s = _score(sets)
            if s > best_score:
                best_score = s
                best_sets = [list(sets[0]), list(sets[1])]
    
    # Try local search on best found
    if best_sets is not None and time.time() < deadline:
        improved = _local_search(best_sets, max_iters=50)
        if _is_valid(improved):
            s = _score(improved)
            if s > best_score:
                best_score = s
                best_sets = improved
    
    # Ensure nonempty
    if best_sets is None or len(best_sets[0]) == 0:
        best_sets = [[1], [2]]
    if len(best_sets[1]) == 0:
        best_sets[1] = [3]
    
    # Final validation
    if not _is_valid(best_sets):
        # Fallback to simple valid sets
        best_sets = [[1, 2, 4, 8, 16, 32, 64, 128], [3, 5, 9, 17, 33, 65, 129]]
    
    return best_sets
# EVOLVE-BLOCK-END
