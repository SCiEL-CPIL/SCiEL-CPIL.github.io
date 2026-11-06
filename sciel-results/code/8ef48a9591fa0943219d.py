"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Backtracking search to find Sidon set with min max element
    # containing 0 and all differences 1..30
    
    def solve(max_elem_limit):
        """Find Sidon set in [0, max_elem_limit] with 0 and diffs 1..30."""
        used_diffs = set()
        elements = [0]
        
        def backtrack(remaining, cur_max):
            if time.time() > deadline:
                return None
            if not remaining:
                return list(elements)
            
            # Generate candidates: x = e + d for e in elements, d in remaining
            candidates = set()
            for e in elements:
                for d in remaining:
                    x = e + d
                    if 0 < x <= max_elem_limit:
                        candidates.add(x)
            
            # Score candidates by how many remaining diffs they cover
            def score(x):
                cnt = 0
                for e in elements:
                    if x - e in remaining:
                        cnt += 1
                return cnt
            
            # Try high-scoring candidates first
            for x in sorted(candidates, key=lambda v: (-score(v), v)):
                if time.time() > deadline:
                    return None
                
                # Check Sidon: no diff repeated
                ok = True
                new_covered = set()
                for e in elements:
                    d = x - e
                    if d <= 0:
                        continue
                    if d in used_diffs:
                        ok = False
                        break
                    if d in remaining:
                        new_covered.add(d)
                
                if not ok:
                    continue
                
                elements.append(x)
                used_diffs.update(new_covered)
                rem2 = remaining - new_covered
                
                res = backtrack(rem2, x)
                if res is not None:
                    return res
                
                elements.pop()
                used_diffs -= new_covered
            
            return None
        
        return backtrack(set(range(1, 31)), 0)
    
    # Binary search for minimum max element
    lo, hi = 30, 500  # reasonable bounds
    
    # First find any solution to set hi
    for test_hi in [100, 200, 500, 1000]:
        if time.time() > deadline - 10:
            break
        sol = solve(test_hi)
        if sol is not None:
            hi = test_hi
            break
    
    best_sol = None
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        sol = solve(mid)
        if sol is not None:
            best_sol = sol
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_sol is None:
        # Fallback: simple greedy
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            placed = False
            for e in sorted(a):
                x = e + d
                if x in a:
                    continue
                ok = True
                for e2 in a:
                    if x - e2 in used and x != e2:
                        ok = False
                        break
                if ok:
                    a.append(x)
                    for e2 in a[:-1]:
                        dd = x - e2
                        if dd > 0:
                            used.add(dd)
                    placed = True
                    break
            if not placed:
                x = max(a) + 31
                while True:
                    ok = True
                    for e in a:
                        if x - e in used:
                            ok = False
                            break
                    if ok:
                        a.append(x)
                        for e in a[:-1]:
                            dd = x - e
                            if dd > 0:
                                used.add(dd)
                        break
                    x += 1
        return a
    
    return best_sol
# EVOLVE-BLOCK-END
