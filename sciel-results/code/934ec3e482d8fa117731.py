import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    N = 4096
    MAX_K = 80
    
    # Start with the seed program's result to have a baseline
    # Greedy from the seed
    def greedy_build():
        selected = []
        from itertools import combinations_with_replacement
        used_sums = set()
        for x in range(1, 513):
            # Check all triples with x
            ok = True
            # Triple (x, x, x)
            s3 = 3 * x
            if s3 in used_sums:
                ok = False
            else:
                # Triples (x, x, y) for y in selected
                for y in selected:
                    s2 = 2 * x + y
                    if s2 in used_sums:
                        ok = False
                        break
                if ok:
                    # Triples (x, y, z) for y <= z in selected
                    for i, y in enumerate(selected):
                        for z in selected[i:]:
                            s1 = x + y + z
                            if s1 in used_sums:
                                ok = False
                                break
                        if not ok:
                            break
            if ok:
                selected.append(x)
                # Add all new sums
                used_sums.add(s3)
                for y in selected:
                    if y == x:
                        continue
                    used_sums.add(2 * x + y)
                for i, y in enumerate(selected):
                    for z in selected[i:]:
                        used_sums.add(x + y + z)
            if len(selected) >= 80:
                break
        return selected, used_sums
    
    # Quick greedy to get a starting point
    selected, used_sums = greedy_build()
    best_k = len(selected)
    
    # If we already have a good answer, try to improve
    # Use randomized greedy with restarts
    import random
    
    def try_randomized_greedy(max_x):
        """Try a randomized greedy approach."""
        selected = []
        used_sums = set()
        
        # Shuffle candidates
        candidates = list(range(1, max_x + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(selected) >= MAX_K:
                break
            if time.time() > deadline:
                break
            
            # Check if x can be added
            ok = True
            s3 = 3 * x
            if s3 in used_sums:
                ok = False
            else:
                # Check (x, x, y)
                for y in selected:
                    s2 = 2 * x + y
                    if s2 in used_sums:
                        ok = False
                        break
                if ok:
                    # Check (x, y, z)
                    n = len(selected)
                    for i in range(n):
                        y = selected[i]
                        for j in range(i, n):
                            z = selected[j]
                            s1 = x + y + z
                            if s1 in used_sums:
                                ok = False
                                break
                        if not ok:
                            break
            
            if ok:
                selected.append(x)
                selected.sort()
                used_sums.add(s3)
                for y in selected:
                    if y != x:
                        used_sums.add(2 * x + y)
                for i in range(len(selected)):
                    y = selected[i]
                    for j in range(i, len(selected)):
                        z = selected[j]
                        used_sums.add(x + y + z)
        
        return selected, used_sums
    
    # Try several randomized attempts
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Vary the max_x for diversity
        if attempt < 10:
            max_x = 513 + attempt * 50
        elif attempt < 25:
            max_x = 1000 + attempt * 100
        elif attempt < 40:
            max_x = 2500 + attempt * 200
        else:
            max_x = 4096
        
        max_x = min(max_x, N)
        
        selected_new, used_sums_new = try_randomized_greedy(max_x)
        
        if len(selected_new) > best_k:
            best_k = len(selected_new)
            selected = selected_new
            used_sums = used_sums_new
            # If we got a better result, try to extend it
            if best_k < MAX_K:
                # Try to extend by adding more elements
                for x in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if x in selected:
                        continue
                    ok = True
                    s3 = 3 * x
                    if s3 in used_sums:
                        ok = False
                    else:
                        for y in selected:
                            s2 = 2 * x + y
                            if s2 in used_sums:
                                ok = False
                                break
                        if ok:
                            n = len(selected)
                            for i in range(n):
                                y = selected[i]
                                for j in range(i, n):
                                    z = selected[j]
                                    s1 = x + y + z
                                    if s1 in used_sums:
                                        ok = False
                                        break
                                if not ok:
                                    break
                    if ok:
                        selected.append(x)
                        selected.sort()
                        used_sums.add(s3)
                        for y in selected:
                            if y != x:
                                used_sums.add(2 * x + y)
                        for i in range(len(selected)):
                            y = selected[i]
                            for j in range(i, len(selected)):
                                z = selected[j]
                                used_sums.add(x + y + z)
                        if len(selected) >= MAX_K:
                            break
    
    # Final check: ensure all elements are <= 4096 and distinct
    selected = sorted(set([x for x in selected if 1 <= x <= N]))
    
    # Verify the result is valid
    from itertools import combinations_with_replacement
    sums = [sum(t) for t in combinations_with_replacement(selected, 3)]
    if len(sums) != len(set(sums)):
        # Should not happen, but if it does, fall back to greedy
        selected, _ = greedy_build()
    
    return selected
