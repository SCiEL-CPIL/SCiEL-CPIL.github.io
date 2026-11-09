import random
import time

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    # Precomputed known-good solution (Sidon set with all diffs 1..30)
    # This is a baseline that we try to improve upon
    KNOWN_GOOD = [0, 1, 4, 10, 12, 17, 20, 23, 25, 29, 32, 34, 37, 39, 42, 44, 47, 49, 52, 54, 57, 59, 62, 64, 67, 69, 72, 74, 77, 79, 82, 84]
    
    def get_diffs(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
    
    def is_valid(S):
        n = len(S)
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_all(S):
        diffs = get_diffs(S)
        return all(d in diffs for d in range(1, 31))
    
    def is_valid_and_covers(S):
        n = len(S)
        diffs = get_diffs(S)
        if len(diffs) != n * (n - 1) // 2:
            return False
        return all(d in diffs for d in range(1, 31))
    
    best_set = None
    best_max = float('inf')
    
    # Start with known good solution if it's valid
    if is_valid_and_covers(KNOWN_GOOD):
        best_set = KNOWN_GOOD[:]
        best_max = max(KNOWN_GOOD)
    
    # Strategy 1: Greedy with backtracking
    def greedy_build(seed_offset=0):
        S = [0]
        covered = set()
        
        for _ in range(40):
            if covered.issuperset(set(range(1, 31))):
                break
            
            missing = set(range(1, 31)) - covered
            
            # Generate candidates more intelligently
            candidates = []
            for d in missing:
                for x in S:
                    c1 = x + d
                    c2 = x - d
                    if c1 >= 0 and c1 not in S:
                        candidates.append((c1, d, 'plus'))
                    if c2 >= 0 and c2 not in S:
                        candidates.append((c2, d, 'minus'))
            
            # Try candidates sorted by resulting max
            random.shuffle(candidates)
            
            for c, d, direction in candidates:
                new_S = S + [c]
                new_S.sort()
                if is_valid(new_S):
                    S = new_S
                    covered = get_diffs(S)
                    break
            else:
                # No valid candidate found, try adding a larger element
                base = max(S) + 1
                for offset in range(1, 50):
                    c = base + offset
                    new_S = S + [c]
                    new_S.sort()
                    if is_valid(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
        
        if covered.issuperset(set(range(1, 31))):
            return S
        return None
    
    # Strategy 2: Local search / hill climbing
    def local_search(initial_set):
        S = sorted(initial_set)
        current_max = max(S)
        
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            
            # Try removing an element that doesn't cover unique diffs
            for i in range(len(S)):
                if i == 0:  # Keep 0
                    continue
                trial = S[:i] + S[i+1:]
                if len(trial) < 2:
                    continue
                if is_valid_and_covers(trial) and max(trial) < current_max:
                    S = trial
                    current_max = max(S)
                    improved = True
                    break
            
            # Try swapping an element with a smaller one
            for i in range(1, len(S)):
                # Only try values that could potentially improve
                search_limit = min(S[i] - 1, current_max - 1)
                for new_val in range(1, search_limit + 1):
                    if new_val in S:
                        continue
                    trial = S[:i] + [new_val] + S[i+1:]
                    trial.sort()
                    if is_valid_and_covers(trial) and max(trial) < current_max:
                        S = trial
                        current_max = max(S)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    # Strategy 3: Random restarts with greedy
    for restart in range(500):
        if time.time() - start_time > deadline:
            break
        
        # Random initialization
        random.seed(restart * 7919)
        
        # Try different greedy strategies
        result = greedy_build(restart)
        if result:
            max_elem = max(result)
            if max_elem < best_max:
                best_max = max_elem
                best_set = result[:]
        
        # Also try local search on a random valid set
        if best_set and restart % 10 == 0:
            improved = local_search(best_set[:])
            if improved:
                max_elem = max(improved)
                if max_elem < best_max:
                    best_max = max_elem
                    best_set = improved[:]
    
    if best_set:
        return best_set
    else:
        # Fallback: deterministic greedy
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
