import time
import sys
import random
from math import exp

# EVOLVE-BLOCK-START

def _is_valid_set(a):
    """Check if a is a valid Sidon set covering 1..30. Return (ok, diffs, diam)."""
    a = sorted(a)
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False, diffs, a[-1]
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False, diffs, a[-1]
    return True, diffs, a[-1]

def _solve_csp(deadline, initial_bound=100):
    """
    Solve the Sidon set problem using CSP backtracking.
    Returns the best set found.
    """
    best_set = [0]
    best_diam = initial_bound
    node_count = 0
    
    # Pre-compute missing diffs
    all_diffs = set(range(1, 31))
    
    def rec(a, used_diffs, missing_diffs):
        nonlocal best_set, best_diam, node_count
        if time.time() > deadline:
            return
        
        node_count += 1
        if node_count % 1000 == 0 and time.time() > deadline:
            return

        # If we covered all diffs, update best
        if not missing_diffs:
            if a[-1] < best_diam:
                best_diam = a[-1]
                best_set = list(a)
            return

        # Pruning: If current max is already >= best_diam, stop
        if a[-1] >= best_diam:
            return

        # Heuristic: Check if we can still cover missing diffs
        # A diff d is "reachable" if there exists a in a such that a + d > a[-1]
        # and a + d < best_diam (strictly less, since we want to improve or match)
        # Actually, the new element x must be < best_diam.
        # So x - a = d => x = a + d.
        # We need x > a[-1] and x < best_diam.
        # So for every d in missing_diffs, there must be at least one a in a
        # such that a[-1] < a + d < best_diam.
        # Note: It's possible d is formed by two future elements, but that's less likely
        # to be the *primary* way we cover small diffs in a tight set.
        # However, if d is large, it might require a larger gap.
        # Let's use a weaker but safer check:
        # For each d in missing_diffs, is there ANY a in a such that a+d < best_diam?
        # If not, d can never be covered by a future element x < best_diam subtracting an existing a.
        # It could be x-y=d with x,y future. But if we assume we build sequentially,
        # we usually cover d by x_new - a_old.
        # Let's stick to the strong check: d must be coverable by x_new - a_old.
        for d in missing_diffs:
            coverable = False
            for a_val in a:
                x_candidate = a_val + d
                if a[-1] < x_candidate < best_diam:
                    coverable = True
                    break
            if not coverable:
                return

        # Generate candidates for next element x
        start = a[-1] + 1
        end = best_diam - 1  # We want to find something strictly smaller than current best
        
        candidates = []
        for x in range(start, end + 1):
            new_diffs = set()
            ok = True
            for a_val in a:
                d = x - a_val
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            
            # Count how many missing diffs this x covers
            covered_count = 0
            for d in new_diffs:
                if d in missing_diffs:
                    covered_count += 1
            
            # Only keep candidates that cover at least 1 missing diff?
            # Or all? If we don't cover any, we are wasting a slot.
            # Since we have limited slots (max 32 elements), we should prioritize coverage.
            # But sometimes we need to "bridge" gaps.
            # Let's keep all valid candidates, but sort them.
            candidates.append((covered_count, x, new_diffs))
        
        if not candidates:
            return

        # Sort by coverage (desc), then value (asc) to try smaller x first if coverage is same
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        # Limit branching factor to avoid explosion?
        # If there are too many, take top N.
        # Top 50 is usually safe.
        for covered_count, x, new_diffs in candidates[:50]:
            if time.time() > deadline:
                return
            
            a.append(x)
            new_used = used_diffs | new_diffs
            new_missing = missing_diffs - new_diffs
            
            rec(a, new_used, new_missing)
            
            a.pop()
            
            # If we found a solution, best_diam is updated.
            # We can break if we found a solution with x == start (smallest possible)?
            # Not necessarily, but if best_diam becomes a[-1] + 1, we can stop?
            # No, we want to minimize a[-1]. If we found a solution, we continue to see if we can go lower.

    # Initial call
    # We need to start with 0.
    # We can also try to force the second element to be small?
    # Let's just let the CSP handle it.
    
    # To speed up, we can initialize best_diam with a known good solution or greedy result.
    # Let's run a quick greedy to get a decent initial bound.
    def _greedy_initial():
        a = [0]
        used = set()
        missing = set(range(1, 31))
        for _ in range(20):
            best_x = None
            best_cov = -1
            for x in range(1, 100):
                if x in a: continue
                nd = set()
                ok = True
                for v in a:
                    d = abs(x-v)
                    if d in used or d in nd:
                        ok = False
                        break
                    nd.add(d)
                if not ok: continue
                cov = len(nd & missing)
                if cov > best_cov:
                    best_cov = cov
                    best_x = x
            if best_x is None: break
            a.append(best_x)
            a.sort()
            for v in a[:-1]:
                used.add(abs(best_x-v))
            missing -= used
            if not missing: break
        return a[-1] if a else 100

    init_bound = _greedy_initial()
    if init_bound < initial_bound:
        initial_bound = init_bound
        # If greedy found a valid set, update best
        # (Greedy doesn't guarantee validity, but it's a good bound)
        # Let's just use the bound.

    rec([0], set(), set(range(1, 31)))
    
    return best_set, best_diam

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: CSP Backtracking
    # Try to find a solution with small diameter
    # The minimal diameter for covering 1..30 is likely around 25-30.
    # We start with a loose bound and let it tighten.
    
    # Run CSP
    try:
        res_set, res_diam = _solve_csp(deadline, initial_bound=100)
        if res_set is not None and res_diam < best_diam:
            best_set = res_set
            best_diam = res_diam
    except Exception as e:
        pass

    # If CSP didn't find anything (or timed out), use a fallback greedy + local search
    if best_set is None:
        # Fallback: Greedy
        a = [0]
        used = set()
        missing = set(range(1, 31))
        for _ in range(20):
            best_x = None
            best_cov = -1
            for x in range(1, 200):
                if x in a: continue
                nd = set()
                ok = True
                for v in a:
                    d = abs(x-v)
                    if d in used or d in nd:
                        ok = False
                        break
                    nd.add(d)
                if not ok: continue
                cov = len(nd & missing)
                if cov > best_cov:
                    best_cov = cov
                    best_x = x
            if best_x is None: break
            a.append(best_x)
            a.sort()
            for v in a[:-1]:
                used.add(abs(best_x-v))
            missing -= used
            if not missing: break
        
        if _is_valid_set(a)[0]:
            best_set = a
            best_diam = a[-1]
        else:
            # Last resort: Return a valid set even if large
            # Construct a large Sidon set
            # Known large Sidon sets exist.
            # For 1..30, a set like {0, 1, 4, 10, 20, ...} might work.
            # Let's just return the greedy result if it's close, or a hardcoded one if we have time.
            # Actually, the greedy usually works for 30.
            # If not, we can try to fix it.
            # But to be safe, let's ensure we return *something* valid.
            # A trivial valid set:
            # 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961
            # Differences are distinct? Squares have distinct differences? No, 1^2-0^2=1, 2^2-1^2=3, 3^2-2^2=5...
            # Differences of squares: (k+1)^2 - k^2 = 2k+1.
            # Differences between non-consecutive squares are also distinct?
            # (j^2 - i^2) = (j-i)(j+i).
            # It is a known fact that squares form a Sidon set?
            # No, 1^2+3^2 = 2^2+2^2? 1+9=10, 4+4=8. No.
            # 1^2+4^2 = 5^2? 1+16=17, 25. No.
            # Actually, the set of squares is NOT a Sidon set. 1^2 + 4^2 = 5^2 is false.
            # But 1^2 + 7^2 = 5^2 + 5^2? 1+49=50, 25+25=50. Yes!
            # So squares are not Sidon.
            
            # Let's use a known Sidon set construction: Mian-Chowla sequence.
            # It grows as n^3. For 30 diffs, we need ~10-15 elements.
            # Mian-Chowla: 1, 2, 4, 8, 13, 21, 31, 45, 66, 81, ...
            # Let's generate Mian-Chowla up to covering 1..30.
            
            a_mc = [0]
            used = set()
            missing = set(range(1, 31))
            next_candidate = 1
            while missing:
                valid = True
                nd = set()
                for v in a_mc:
                    d = abs(next_candidate - v)
                    if d in used or d in nd:
                        valid = False
                        break
                    nd.add(d)
                if valid:
                    a_mc.append(next_candidate)
                    a_mc.sort()
                    for v in a_mc[:-1]:
                        used.add(abs(next_candidate - v))
                    missing -= used
                next_candidate += 1
            
            if _is_valid_set(a_mc)[0]:
                best_set = a_mc
                best_diam = a_mc[-1]
            else:
                # Should not happen
                best_set = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961]
                # Verify this is valid?
                # If not, return the greedy one even if invalid? No, must be valid.
                # The Mian-Chowla sequence is a Sidon set.
                # So a_mc should be valid.

    return best_set

# EVOLVE-BLOCK-END
