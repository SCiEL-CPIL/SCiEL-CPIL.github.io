"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    def score_set(U):
        U_set = set(U)
        max_U = max(U)
        if max_U == 0:
            return 0.0
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        import math
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent
    
    # Start with a small known good set
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Try to improve by adding elements
    # Use a greedy/local search approach
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        current_max = max(best_set)
        # Try adding elements in a range
        candidates = []
        for _ in range(50):
            # Random candidate
            candidate = random.randint(1, current_max * 2 + 10)
            candidates.append(candidate)
        
        # Also try some structured candidates
        for k in range(1, 6):
            candidates.append(current_max * k + 1)
            candidates.append(current_max * k - 1 if current_max * k - 1 > 0 else 1)
        
        best_candidate = None
        best_candidate_score = best_score
        
        for cand in set(candidates):
            if cand in best_set:
                continue
            new_set = sorted(best_set + [cand])
            new_score = score_set(new_set)
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = new_set
        
        if best_candidate is not None:
            best_set = best_candidate
            best_score = best_candidate_score
        else:
            # Try removing an element (except 0)
            removable = [x for x in best_set if x != 0]
            if not removable:
                break
            for _ in range(20):
                if time.time() > deadline:
                    break
                idx = random.randint(0, len(removable) - 1)
                elem = removable[idx]
                new_set = [x for x in best_set if x != elem]
                new_score = score_set(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    break
    
    return best_set
# EVOLVE-BLOCK-END
