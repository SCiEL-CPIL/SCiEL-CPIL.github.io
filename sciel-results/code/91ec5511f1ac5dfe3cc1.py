"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def get_diffs(s):
        """Get all positive differences from sorted list s."""
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if s is a Sidon set (all positive differences distinct)."""
        n = len(s)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(s[j] - s[i])
        return len(diffs) == len(set(diffs))
    
    def covers_all(s, target=30):
        """Check if all differences 1..target are covered."""
        diffs = get_diffs(s)
        return all(d in diffs for d in range(1, target + 1))
    
    def valid_solution(s):
        """Check if s is a valid Sidon set covering 1..30 with at most 32 elements."""
        if len(s) > 32:
            return False
        if 0 not in s:
            return False
        if not is_sidon(s):
            return False
        return covers_all(s, 30)
    
    def greedy_construct():
        """Greedy construction of a valid Sidon set covering 1..30."""
        s = [0]
        target_diffs = set(range(1, 31))
        
        for _ in range(32):
            if not target_diffs:
                break
            if time.time() > DEADLINE:
                break
            
            # Find candidate that covers most missing diffs
            best_cand = None
            best_cover = 0
            current_diffs = get_diffs(s)
            
            # Try candidates from 1 to max(s) + 50
            max_try = max(s) + 50 if s else 50
            for cand in range(1, min(max_try, 10000)):
                if cand in s:
                    continue
                new_s = sorted(s + [cand])
                if not is_sidon(new_s):
                    continue
                new_diffs = get_diffs(new_s)
                cover = len(new_diffs & target_diffs)
                if cover > best_cover:
                    best_cover = cover
                    best_cand = cand
            
            if best_cand is None:
                # Try larger candidates
                for cand in range(max_try + 1, max_try + 200):
                    if cand in s:
                        continue
                    new_s = sorted(s + [cand])
                    if not is_sidon(new_s):
                        continue
                    new_diffs = get_diffs(new_s)
                    cover = len(new_diffs & target_diffs)
                    if cover > best_cover:
                        best_cover = cover
                        best_cand = cand
            
            if best_cand is None:
                break
            
            s = sorted(s + [best_cand])
            current_diffs = get_diffs(s)
            target_diffs -= current_diffs
        
        if covers_all(s, 30) and is_sidon(s) and len(s) <= 32:
            return s
        return None
    
    def local_search(s, max_iters=10000):
        """Local search to minimize max element while maintaining validity."""
        if not valid_solution(s):
            return s
        
        best_s = list(s)
        best_max = max(best_s)
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Try moving each element to a smaller value
            improved = False
            for i in range(len(best_s)):
                old_val = best_s[i]
                if old_val == 0:
                    continue
                
                # Try decreasing this element
                for new_val in range(old_val - 1, -1, -1):
                    if new_val in best_s:
                        continue
                    new_s = sorted([best_s[j] if j != i else new_val for j in range(len(best_s))])
                    if valid_solution(new_s):
                        if max(new_s) < best_max:
                            best_s = new_s
                            best_max = max(best_s)
                            improved = True
                        break
            
            if not improved:
                # Try swapping two non-zero elements
                for i in range(1, len(best_s)):
                    for j in range(i + 1, len(best_s)):
                        # Try swapping values
                        new_s = list(best_s)
                        new_s[i], new_s[j] = new_s[j], new_s[i]
                        new_s = sorted(new_s)
                        if valid_solution(new_s) and max(new_s) < best_max:
                            best_s = new_s
                            best_max = max(best_s)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try removing an element if coverage is still maintained
                for i in range(1, len(best_s)):
                    new_s = [best_s[j] for j in range(len(best_s)) if j != i]
                    if len(new_s) >= 2 and valid_solution(new_s):
                        if max(new_s) < best_max:
                            best_s = new_s
                            best_max = max(best_s)
                            improved = True
                        break
            
            if not improved:
                # Try adding a new small element and then removing a large one
                if len(best_s) < 32:
                    for new_val in range(1, best_max + 1):
                        if new_val in best_s:
                            continue
                        new_s = sorted(best_s + [new_val])
                        if not is_sidon(new_s):
                            continue
                        # Now try removing the largest element if coverage is maintained
                        if covers_all(new_s, 30):
                            # Try removing largest
                            for rem in range(len(new_s) - 1, 0, -1):
                                trial = [new_s[j] for j in range(len(new_s)) if j != rem]
                                if len(trial) >= 2 and valid_solution(trial):
                                    if max(trial) < best_max:
                                        best_s = trial
                                        best_max = max(best_s)
                                        improved = True
                                    break
                        if improved:
                            break
                
                if not improved:
                    break
        
        return best_s
    
    def random_restart_search():
        """Try random constructions and local search."""
        best_result = None
        best_max = float('inf')
        
        for restart in range(100):
            if time.time() > DEADLINE:
                break
            
            random.seed(restart * 12345 + 67890)
            
            # Random greedy with randomization
            s = [0]
            target_diffs = set(range(1, 31))
            
            for _ in range(32):
                if not target_diffs:
                    break
                if time.time() > DEADLINE:
                    break
                
                best_cand = None
                best_cover = 0
                current_diffs = get_diffs(s)
                
                max_try = max(s) + 30 if s else 30
                candidates = list(range(1, min(max_try, 500)))
                random.shuffle(candidates)
                
                for cand in candidates:
                    if cand in s:
                        continue
                    new_s = sorted(s + [cand])
                    if not is_sidon(new_s):
                        continue
                    new_diffs = get_diffs(new_s)
                    cover = len(new_diffs & target_diffs)
                    if cover > best_cover:
                        best_cover = cover
                        best_cand = cand
                
                if best_cand is None:
                    break
                
                s = sorted(s + [best_cand])
                current_diffs = get_diffs(s)
                target_diffs -= current_diffs
            
            if covers_all(s, 30) and is_sidon(s) and len(s) <= 32:
                s = local_search(s, max_iters=5000)
                if max(s) < best_max:
                    best_max = max(s)
                    best_result = list(s)
        
        return best_result
    
    # First, try greedy construction
    result = greedy_construct()
    
    if result is not None:
        result = local_search(result, max_iters=50000)
        best_result = list(result)
        best_max = max(best_result)
    else:
        best_result = None
        best_max = float('inf')
    
    # Then try random restarts
    random_result = random_restart_search()
    if random_result is not None and max(random_result) < best_max:
        best_result = random_result
        best_max = max(best_result)
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple greedy
    a = [0]
    for d in range(1, 31):
        diffs = get_diffs(a)
        if d in diffs:
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
