import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covered_mask(S):
        diffs = get_diffs(S)
        mask = 0
        for d in diffs:
            if 1 <= d <= 30:
                mask |= (1 << (d - 1))
        return mask

    def try_greedy(max_size=32, random_seed=0):
        """Greedy construction with random restarts"""
        rng = random.Random(random_seed)
        S = [0]
        used_diffs = set()
        
        for _ in range(max_size - 1):
            if time.time() - start_time > deadline:
                break
                
            missing = target_diffs - used_diffs
            if not missing:
                break
                
            # Generate candidates: s + d for s in S, d in missing
            candidates = set()
            for s in S:
                for d in missing:
                    candidates.add(s + d)
            
            # Also add some random candidates near existing elements
            for _ in range(20):
                s = rng.choice(S)
                d = rng.choice(list(missing))
                candidates.add(s + d)
                candidates.add(s - d)
            
            valid_candidates = []
            for x in candidates:
                if x <= 0 or x > 10**6:
                    continue
                if x in S:
                    continue
                
                new_diffs = set()
                valid = True
                for s in S:
                    d = abs(x - s)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                new_used = used_diffs | new_diffs
                new_missing_count = len(target_diffs - new_used)
                valid_candidates.append((new_missing_count, x, new_diffs))
            
            if not valid_candidates:
                # Try adding a large number to unblock
                x = 2 * max(S) + rng.randint(1, 100)
                new_diffs = set()
                valid = True
                for s in S:
                    d = abs(x - s)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    S.append(x)
                    S.sort()
                    used_diffs |= new_diffs
                continue
            
            # Sort by missing count, then by value
            valid_candidates.sort(key=lambda item: (item[0], item[1]))
            
            # Try top candidates
            for _, x, new_diffs in valid_candidates[:10]:
                S.append(x)
                S.sort()
                used_diffs |= new_diffs
                break
        
        return S

    def local_search(S, max_iterations=1000):
        """Local search to minimize max element while maintaining Sidon property and coverage"""
        S = sorted(S)
        used_diffs = get_diffs(S)
        
        if not used_diffs.issuperset(target_diffs):
            return S
        
        best_S = S[:]
        best_max = max(S)
        
        for _ in range(max_iterations):
            if time.time() - start_time > deadline:
                break
                
            improved = False
            n = len(S)
            
            # Try removing each element and see if we can maintain coverage
            for i in range(n):
                if S[i] == 0:
                    continue  # Can't remove 0
                
                candidate = S[:i] + S[i+1:]
                if len(candidate) < 2:
                    continue
                
                new_diffs = get_diffs(candidate)
                if new_diffs.issuperset(target_diffs) and is_sidon(candidate):
                    if max(candidate) < best_max:
                        best_S = candidate[:]
                        best_max = max(candidate)
                        S = candidate
                        used_diffs = new_diffs
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try replacing an element with a smaller one
            for i in range(n):
                if S[i] == 0:
                    continue
                
                # Try values smaller than S[i]
                for new_val in range(1, S[i]):
                    if new_val in S:
                        continue
                    
                    candidate = S[:i] + [new_val] + S[i+1:]
                    candidate.sort()
                    
                    if not is_sidon(candidate):
                        continue
                    
                    new_diffs = get_diffs(candidate)
                    if new_diffs.issuperset(target_diffs):
                        if max(candidate) < best_max:
                            best_S = candidate[:]
                            best_max = max(candidate)
                            S = candidate
                            used_diffs = new_diffs
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_S

    # Phase 1: Greedy construction with multiple random restarts
    best_set = None
    best_max = float('inf')
    
    for seed in range(50):
        if time.time() - start_time > deadline - 20:
            break
        
        S = try_greedy(max_size=32, random_seed=seed)
        
        if len(S) > 32:
            continue
        
        if not is_sidon(S):
            continue
        
        diffs = get_diffs(S)
        if not diffs.issuperset(target_diffs):
            continue
        
        if max(S) < best_max:
            best_max = max(S)
            best_set = S[:]
    
    # Phase 2: Local search on best set
    if best_set is not None:
        best_set = local_search(best_set, max_iterations=500)
        best_max = max(best_set)
    
    # Phase 3: More aggressive search if time remains
    if time.time() - start_time < deadline - 30 and best_set is not None:
        # Try to improve with more local search iterations
        for _ in range(10):
            if time.time() - start_time > deadline - 10:
                break
            
            improved_set = local_search(best_set, max_iterations=200)
            if max(improved_set) < best_max:
                best_set = improved_set
                best_max = max(best_set)
    
    if best_set is not None:
        return best_set
    else:
        # Fallback: Return a valid set even if not optimal
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in sorted(missing):
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add a large number to unblock?
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        return S
