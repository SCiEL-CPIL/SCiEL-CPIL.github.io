<<<<<<< SEARCH
import time
import random

N = 40
_DEADLINE = time.time() + 85

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    target = set(range(1, max_val - 1))  # 1..38
    rng = random.Random(seed)
    
    def compute_covered(S):
        s_list = sorted(S)
        covered = set()
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                covered.add(s_list[j] - s_list[i])
        return covered
    
    # Always include 1 and max_val-1
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    
    best = None
    best_score = -1
    
    for attempt in range(1000):
        if time.time() > _DEADLINE - 5:
            break
        # Choose remaining elements
        candidates = [x for x in range(2, max_val - 1) if x not in fixed]
        S = set(fixed)
        S.update(rng.sample(candidates, remaining_size))
        covered = compute_covered(S)
        score = len(covered & target)
        
        if score > best_score:
            best_score = score
            best = set(S)
        
        if target <= covered:
            return sorted(S)
        
        # Local search
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            s_list = sorted(S)
            covered = compute_covered(S)
            missing = target - covered
            if not missing:
                return sorted(S)
            
            for idx in range(len(s_list)):
                old = s_list[idx]
                if old in fixed:
                    continue  # Don't move fixed elements
                for nv in range(2, max_val - 1):
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    new_covered = compute_covered(S)
                    new_score = len(new_covered & target)
                    if new_score > score:
                        score = new_score
                        covered = new_covered
                        improved = True
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if not (target - covered):
                    return sorted(S)
    
    if best is not None and target <= compute_covered(best):
        return sorted(best)
    return None
=======
import time
import random

N = 40
_DEADLINE = time.time() + 85

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    target = set(range(1, max_val - 1))  # 1..38
    rng = random.Random(seed)
    
    def compute_covered(S):
        s_list = sorted(S)
        covered = set()
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                covered.add(s_list[j] - s_list[i])
        return covered
    
    # Always include 1 and max_val-1
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    
    best = None
    best_score = -1
    
    for attempt in range(2000):
        if time.time() > _DEADLINE - 5:
            break
        
        # Smart initial construction: greedy to maximize coverage
        candidates = [x for x in range(2, max_val - 1) if x not in fixed]
        S = set(fixed)
        covered = compute_covered(S)
        
        # Greedily add elements that maximize new coverage
        for _ in range(remaining_size):
            if time.time() > _DEADLINE - 5:
                break
            best_gain = -1
            best_elem = None
            for x in candidates:
                if x in S:
                    continue
                S.add(x)
                new_covered = compute_covered(S)
                gain = len(new_covered & target) - len(covered & target)
                S.remove(x)
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            if best_elem is not None:
                S.add(best_elem)
                covered = compute_covered(S)
            else:
                break
        
        score = len(covered & target)
        
        if score > best_score:
            best_score = score
            best = set(S)
        
        if target <= covered:
            return sorted(S)
        
        # Local search with swap moves
        improved = True
        max_iters = 500
        iters = 0
        while improved and time.time() < _DEADLINE - 5 and iters < max_iters:
            improved = False
            iters += 1
            s_list = sorted(S)
            covered = compute_covered(S)
            missing = target - covered
            if not missing:
                return sorted(S)
            
            # Try single swaps
            for idx in range(len(s_list)):
                old = s_list[idx]
                if old in fixed:
                    continue
                for nv in range(2, max_val - 1):
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    new_covered = compute_covered(S)
                    new_score = len(new_covered & target)
                    if new_score > score:
                        score = new_score
                        covered = new_covered
                        improved = True
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if not (target - covered):
                    return sorted(S)
            
            # Try 2-swap moves if no improvement from single swaps
            if not improved:
                s_list = sorted(S)
                non_fixed = [x for x in s_list if x not in fixed]
                for i in range(len(non_fixed)):
                    if time.time() > _DEADLINE - 5:
                        break
                    old1 = non_fixed[i]
                    for j in range(i + 1, len(non_fixed)):
                        old2 = non_fixed[j]
                        for nv1 in range(2, max_val - 1):
                            if nv1 in S and nv1 != old1:
                                continue
                            for nv2 in range(2, max_val - 1):
                                if nv2 in S and nv2 != old2:
                                    continue
                                if nv1 == nv2:
                                    continue
                                S.remove(old1)
                                S.remove(old2)
                                S.add(nv1)
                                S.add(nv2)
                                new_covered = compute_covered(S)
                                new_score = len(new_covered & target)
                                if new_score > score:
                                    score = new_score
                                    covered = new_covered
                                    improved = True
                                    break
                                else:
                                    S.remove(nv1)
                                    S.remove(nv2)
                                    S.add(old1)
                                    S.add(old2)
                            if not (target - covered):
                                return sorted(S)
                        if improved:
                            break
                    if improved:
                        break
    
    if best is not None and target <= compute_covered(best):
        return sorted(best)
    return None
>>>>>>> REPLACE

