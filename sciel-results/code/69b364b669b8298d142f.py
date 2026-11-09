"""Sidon set containing 0 with differences 1..30, minimizing the largest element."""
import time

TIME_LIMIT = 110  # wall-clock seconds, safety margin under 120s

def construct():
    t0 = time.time()
    DEADLINE = t0 + TIME_LIMIT
    
    # Target: differences 1..30 must be present.
    # Sidon set: all positive differences distinct.
    # Minimize max element.
    
    # We know size must be at least 9.
    # Diameter must be at least 36 (derived above).
    
    best_solution = None
    best_diameter = float('inf')
    
    target_diffs = set(range(1, 31))
    
    def check_time():
        return time.time() < DEADLINE
        
    def solve(n, max_diam):
        """Find Sidon set of size n in [0, max_diam] covering target_diffs."""
        # State: S (list), used (set of diffs)
        # We use DFS.
        # Pruning:
        # 1. If len(S) == n, check if target_diffs subset of used.
        # 2. If used contains a diff > max_diam, invalid (impossible).
        # 3. If number of remaining slots < number of missing target diffs?
        #    Each new element adds (len(S)) new diffs.
        #    Total new diffs = sum_{i=1}^{n-len(S)} i = k(k+1)/2 where k = n-len(S).
        #    We need to cover missing = target_diffs - used.
        #    If len(missing) > k(k+1)/2, prune.
        
        S = [0]
        used = set()
        
        def dfs(S, used):
            nonlocal best_solution, best_diameter
            if not check_time():
                return False
            
            if len(S) == n:
                if target_diffs.issubset(used):
                    diam = max(S)
                    if diam < best_diameter:
                        best_diameter = diam
                        best_solution = S[:]
                    return True
                return False
            
            missing = target_diffs - used
            k = n - len(S)
            max_new_diffs = k * (k + 1) // 2
            if len(missing) > max_new_diffs:
                return False
            
            # Generate candidates
            # x must be in [1, max_diam]
            # To reduce search space, we can restrict x to be "useful".
            # x should create diffs that are in missing OR small enough to allow future coverage?
            # Actually, any x that doesn't conflict is a candidate.
            # But we can sort candidates by how many missing diffs they cover.
            
            candidates = set()
            # We can generate candidates from missing diffs:
            # If d in missing, x = y + d or x = y - d.
            for d in missing:
                for y in S:
                    c1 = y + d
                    if c1 <= max_diam:
                        candidates.add(c1)
                    c2 = y - d
                    if c2 >= 0:
                        candidates.add(c2)
            
            # Also, we might need to add elements that don't directly cover a missing diff
            # but are necessary for structure?
            # For now, let's just try these candidates.
            # If this fails, we might need to expand.
            
            valid_candidates = []
            for x in candidates:
                if x in S:
                    continue
                ok = True
                new_diffs = []
                for y in S:
                    diff = abs(x - y)
                    if diff in used:
                        ok = False
                        break
                    new_diffs.append(diff)
                if ok:
                    # Check if any new diff is > max_diam? No, x <= max_diam, y >= 0, so diff <= max_diam.
                    # Count how many missing diffs are covered
                    covered = len(set(new_diffs) & missing)
                    valid_candidates.append((x, new_diffs, covered))
            
            # Sort by covered (desc), then x (asc)
            valid_candidates.sort(key=lambda t: (-t[2], t[0]))
            
            for x, new_diffs, _ in valid_candidates:
                if not check_time():
                    return False
                new_used = used | set(new_diffs)
                if dfs(S + [x], new_used):
                    return True
            
            return False
        
        return dfs([0], set())
    
    # Try sizes 9, 10, ...
    # Try diameters 36, 37, ...
    # We can interleave: for each size, try increasing diameters.
    # Or for each diameter, try increasing sizes.
    # Since smaller size is better for diameter, try size 9 first with increasing D.
    
    for n in range(9, 33):
        if not check_time():
            break
        # Lower bound on D for this n
        # n(n-1)/2 diffs. Need to cover 30.
        # D must be at least 30.
        # Also D must be at least n(n-1)/2 if we want to pack them tightly?
        # No, D is the max element.
        # Let's just start from 30.
        for D in range(30, 100):
            if not check_time():
                break
            if D >= best_diameter:
                break
            if solve(n, D):
                # Found solution with diameter D
                # We can stop searching for this n with larger D?
                # Yes, because we want min D.
                # But maybe a larger n allows smaller D?
                # No, larger n requires larger D generally.
                # But to be safe, we continue to check other n?
                # Actually, if we found a solution with D, we can update best.
                # We can break the D loop for this n.
                break
        if best_solution is not None:
            # If we found a solution, we might still find a better one with different n?
            # Unlikely, but let's check a few more n if time permits.
            # However, if D is small, it's likely optimal.
            # Let's just break if D is very small?
            # Or just let the time limit handle it.
            pass
    
    if best_solution is not None:
        return sorted(best_solution)
    
    # Fallback greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
