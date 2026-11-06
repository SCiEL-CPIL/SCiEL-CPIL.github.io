import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_diam = float('inf')
    best_set = None
    
    # We search for Sidon sets of size n=9, 10, ...
    # Start with n=9
    
    def solve(n):
        nonlocal best_diam, best_set
        
        # a[0] = 0 is fixed
        a = [0]
        used_diffs = set()
        covered = 0  # bitmask or count of 1..30 covered
        
        # We need to cover 1..30.
        # We track which of 1..30 are covered.
        # For efficiency, we can use a set of missing diffs from 1..30.
        missing = set(range(1, 31))
        
        def rec(idx, current_max):
            nonlocal best_diam, best_set, missing
            
            if time.time() > deadline:
                return
                
            # If we have enough elements to cover all missing diffs?
            # Max new diffs from adding (n - idx) elements to current set of size idx:
            # Each new element adds (idx) diffs to existing, and 1 diff to previous new ones?
            # Total diffs in final set of size n is n*(n-1)/2.
            # Current diffs: idx*(idx-1)/2.
            # Remaining diffs to add: n*(n-1)/2 - idx*(idx-1)/2.
            # We need to cover len(missing) diffs.
            # If remaining possible diffs < len(missing), prune.
            
            if len(missing) == 0:
                # We covered 1..30.
                # Check if we have reached size n?
                # We can stop adding if we want, but we need exactly n?
                # The problem says "at most 32".
                # Minimizing diameter.
                # If we have a valid set, record it.
                if current_max < best_diam:
                    best_diam = current_max
                    best_set = list(a)
                return

            if idx == n:
                # Reached max size but didn't cover all
                return

            # Pruning:
            # 1. If current_max >= best_diam, prune.
            if current_max >= best_diam:
                return

            # 2. Check if it's possible to cover missing diffs.
            # Max new diffs we can generate with (n - idx) more elements:
            # New element x adds (a_i - x) for all i < idx.
            # So each new element adds at most idx new diffs (to existing) + diffs to other new elements.
            # Rough upper bound on new diffs: (n - idx) * idx + (n - idx) * (n - idx - 1) / 2
            max_new_diffs = (n - idx) * idx + (n - idx) * (n - idx - 1) // 2
            if max_new_diffs < len(missing):
                return

            # Determine lower bound for next element
            # Must be > current_max
            # Also, to cover small missing diffs, we might need specific values.
            # Simple bound: start from current_max + 1
            
            start_val = current_max + 1
            
            # Upper bound for next element
            # We can't exceed best_diam
            end_val = best_diam - 1
            
            # Optimization: 
            # If the smallest missing diff is d_min, and we add x, 
            # one of the new diffs is x - a[i].
            # If x - a[i] == d_min, then x = a[i] + d_min.
            # We should prioritize candidates that cover missing diffs.
            
            # Generate candidates
            candidates = set()
            # 1. Simple increment
            candidates.add(start_val)
            
            # 2. Candidates derived from missing diffs
            # If we add x, and x - a[i] = d, then x = a[i] + d.
            # We want x to be small and cover missing diffs.
            for d in missing:
                for i in range(idx):
                    x = a[i] + d
                    if x > current_max:
                        candidates.add(x)
            
            # Also, we might just need to fill gaps.
            # Sort candidates
            for x in sorted(candidates):
                if x > end_val:
                    continue
                if time.time() > deadline:
                    return
                
                # Check if x creates duplicate diffs
                new_diffs = []
                ok = True
                for i in range(idx):
                    d = x - a[i]
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                
                if not ok:
                    continue
                
                # Update state
                a.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                    if d in missing:
                        missing.remove(d)
                
                rec(idx + 1, x)
                
                # Backtrack
                a.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
                    if d <= 30:
                        missing.add(d)

        rec(1, 0)
        
    # Try size 9
    solve(9)
    
    # If not found or time is running out, try size 10 with tighter bounds?
    # If best_diam is inf, we need a valid set.
    if best_set is None:
        # Fallback: Greedy construction
        a = [0]
        used = set()
        for _ in range(30): # Try to get a valid set
            added = False
            for x in range(1, 1000):
                if x in a:
                    continue
                ok = True
                for v in a:
                    d = abs(x - v)
                    if d in used:
                        ok = False
                        break
                if ok:
                    a.append(x)
                    for v in a[:-1]:
                        used.add(abs(x - v))
                    added = True
                    break
            if not added:
                break
        # Check if it covers 1..30
        # If not, just return it? The task requires covering 1..30.
        # The fallback should be a valid set.
        # Let's force a valid set by adding elements until 1..30 are covered.
        missing = set(range(1, 31)) - set(used)
        while missing:
            d = min(missing)
            # Find x such that x - a[i] = d or a[i] - x = d (x > a[i])
            # Try x = a[i] + d
            found = False
            for i in range(len(a)):
                x = a[i] + d
                # Check if x is valid
                if x in a:
                    continue
                ok = True
                for v in a:
                    diff = abs(x - v)
                    if diff in used:
                        ok = False
                        break
                if ok:
                    a.append(x)
                    for v in a[:-1]:
                        used.add(abs(x - v))
                    missing = set(range(1, 31)) - set(used)
                    found = True
                    break
            if not found:
                # Try larger x
                x = max(a) + d + 1
                while True:
                    if x in a:
                        x += 1
                        continue
                    ok = True
                    for v in a:
                        diff = abs(x - v)
                        if diff in used:
                            ok = False
                            break
                    if ok:
                        a.append(x)
                        for v in a[:-1]:
                            used.add(abs(x - v))
                        missing = set(range(1, 31)) - set(used)
                        break
                    x += 1
        best_set = a
        best_diam = max(a)

    return best_set

# EVOLVE-BLOCK-END
