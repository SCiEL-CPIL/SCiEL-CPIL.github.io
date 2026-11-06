import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    
    Structural approach: Exact-cover / Algorithm X (DLX-style) search where:
    - Columns = the 30 required differences (each must be covered exactly once)
    - Rows = valid edges (a, b) with b - a = d, for some required d
    - Sidon constraint: no two selected edges may share a difference value
    - We also track which elements are "used" to ensure consistency
    
    The search assigns each required difference to a unique edge, maintaining:
    - The set of elements used
    - The set of difference values already consumed (for Sidon)
    
    Combined with iterative deepening on diameter and local refinement.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    TARGET = set(range(1, 31))
    TARGET_LIST = list(range(1, 31))
    N_TARGET = 30

    def is_sidon(S):
        """Check if sorted list S is a Sidon set (all pairwise differences distinct)."""
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return diffs

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def validate(S):
        """Full validation: Sidon + covers 1..30 + includes 0 + at most 32 elements + range."""
        if not S or 0 not in S:
            return False
        if len(S) > 32:
            return False
        if max(S) > 10**6:
            return False
        if len(set(S)) != len(S):
            return False
        if not is_sidon(S):
            return False
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def greedy_construct(max_val, deadline):
        """
        Greedy: start with {0}, repeatedly add the element that covers the most
        missing differences while maintaining Sidon property.
        """
        S = [0]
        used_diffs = set()
        covered = set()
        S_set = {0}

        for _ in range(31):
            if time.time() > deadline:
                break
            if covered == TARGET:
                break

            missing = TARGET - covered
            best_x = None
            best_key = None

            # Generate candidates: for each existing s and missing d, try s+d and s-d
            candidates = set()
            for s in S:
                for d in missing:
                    c1 = s + d
                    if 0 < c1 <= max_val and c1 not in S_set:
                        candidates.add(c1)
                    c2 = s - d
                    if 0 < c2 <= max_val and c2 not in S_set:
                        candidates.add(c2)

            for x in sorted(candidates):
                # Check Sidon validity
                new_diffs = set()
                valid = True
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if not valid:
                    continue

                gain = len(new_diffs & missing)
                if gain == 0:
                    continue
                # Prefer: more gain, then fewer total new diffs (less "waste"), then smaller x
                key = (-gain, len(new_diffs), x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_x = x

            if best_x is None:
                # Fallback: scan small values
                for x in range(1, min(max_val + 1, 200)):
                    if x in S_set:
                        continue
                    new_diffs = set()
                    valid = True
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid and (new_diffs & missing):
                        best_x = x
                        break

            if best_x is None:
                break

            S.append(best_x)
            S_set.add(best_x)
            for s in S[:-1]:
                used_diffs.add(abs(s - best_x))
            covered = used_diffs & TARGET

        if covers_all(S) and is_sidon(S):
            return sorted(S)
        return None

    def exact_cover_dlx(max_val, deadline):
        """
        Exact-cover / DLX-style search for a Sidon set with all differences 1..30.
        
        Model:
        - We need to select a set of elements S containing 0 such that:
          * For each d in 1..30, there exists a,b in S with b-a=d
          * All pairwise differences are distinct (Sidon)
        
        Reformulation as exact cover:
        - Each required difference d must be "witnessed" by exactly one pair (a,b) in S
        - The Sidon constraint means: if we assign d to pair (a,b), then no other pair
          in S can have difference d
        
        Search strategy:
        - Process differences in order (MRV: fewest valid pairs first)
        - For each difference d, try all valid pairs (a, a+d) where:
          * a and a+d are not already "conflicting" with assigned pairs
          * Adding this pair doesn't violate Sidon with other assigned pairs
        - Track: assigned elements, used difference values, consistency
        
        Pruning:
        - If max(assigned_elements) > max_val, prune
        - If any required difference cannot be covered with remaining budget, prune
        """
        best_solution = None
        best_diameter = max_val + 1
        
        # Precompute for each difference d, all possible pairs (a, a+d) with a+d <= max_val
        # For each pair, precompute its "difference signature" (just d itself, since b-a=d)
        # But we also need to check Sidon conflicts with other pairs
        
        # For efficiency, we'll track:
        # - elements_used: set of elements in the current solution
        # - diffs_used: set of difference values already consumed (for Sidon)
        # - For each difference d, which pairs have been tried
        
        # Order differences by "difficulty": differences that have fewer valid pairs first
        # A pair (a, a+d) is valid if:
        #   - a >= 0, a+d <= max_val
        #   - a not in elements_used (or a == 0, which is always allowed)
        #   - a+d not in elements_used
        #   - d not in diffs_used (Sidon: this difference value isn't already used)
        #   - For all existing elements e, |e - a| not in diffs_used and |e - (a+d)| not in diffs_used
        #     (i.e., adding a and a+d doesn't create a difference that's already used)
        
        # Actually, the Sidon constraint is: all pairwise differences in the final set must be distinct.
        # When we add a new element x, the new differences are |x - s| for all s in current set.
        # These must all be distinct from each other AND from all previously used differences.
        
        # Let's use a cleaner formulation:
        # - Build the set incrementally
        # - At each step, we must cover at least one more required difference
        # - We choose which difference to cover next (MRV), then which pair witnesses it
        
        # State:
        #   S: current set of elements (contains 0)
        #   S_set: set version for O(1) lookup
        #   used_diffs: set of all difference values present in S
        #   covered: used_diffs & TARGET
        
        # The key insight for DLX: instead of building S incrementally by adding elements,
        # we assign each required difference to a specific pair. The union of all pairs'
        # endpoints gives us S. The Sidon constraint is that no two pairs share a difference
        # value (which is automatic if each pair has a unique difference value, i.e., each
        # required d is assigned to exactly one pair).
        
        # Wait, that's not quite right. The Sidon constraint is about ALL pairwise differences
        # in S, not just the required ones. If S = {0, 1, 3}, then differences are 1, 3, 2.
        # If we assign d=1 to pair (0,1) and d=3 to pair (0,3), the pair (1,3) has difference 2,
        # which is fine. But if we also had d=2 assigned to pair (0,2), then S={0,1,2,3}, and
        # differences include 1,2,3,2,3,1 — duplicate! So the Sidon constraint is global.
        
        # Better approach: build S incrementally, but use the required differences as guidance.
        # At each step, pick the "most constrained" missing difference and try to cover it
        # by adding one or two new elements.
        
        # Let's use a hybrid: backtracking search where at each node, we:
        # 1. Check if all differences are covered -> if yes, record solution
        # 2. Pick the most constrained missing difference (fewest valid ways to cover it)
        # 3. For each way to cover it (add element x such that x-s=d or s-x=d for some s in S),
        #    try adding x and recurse
        
        # This is similar to the parent's approach but with better pruning.
        
        S = [0]
        S_set = {0}
        used_diffs = set()
        covered = set()
        
        # Precompute: for each element x, what differences it creates with existing elements
        # (this changes as S grows, so we compute on the fly)
        
        # MRV heuristic: for each missing difference d, count how many valid ways to cover it
        # A way to cover d: add x such that for some s in S, |x - s| = d, and adding x is Sidon-valid
        
        def count_ways_to_cover(d):
            """Count valid ways to cover difference d by adding one element."""
            count = 0
            for s in S:
                for x in [s + d, s - d]:
                    if x <= 0 or x > max_val or x in S_set:
                        continue
                    # Check Sidon validity of adding x
                    new_diffs = set()
                    valid = True
                    for s2 in S:
                        dd = abs(s2 - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid:
                        count += 1
            return count
        
        def dfs():
            nonlocal best_solution, best_diameter
            
            if time.time() > deadline:
                return
            
            # Prune: if current max exceeds best known, no point continuing
            current_max = max(S)
            if current_max >= best_diameter:
                return
            
            if covered == TARGET:
                diam = max(S)
                if diam < best_diameter:
                    best_diameter = diam
                    best_solution = S[:]
                return
            
            if len(S) >= 32:
                return
            
            missing = TARGET - covered
            
            # MRV: pick the missing difference with fewest valid ways to cover
            best_d = None
            best_count = 999
            for d in sorted(missing):
                cnt = count_ways_to_cover(d)
                if cnt < best_count:
                    best_count = cnt
                    best_d = d
                if cnt == 0:
                    # Dead end: this difference can't be covered
                    return
            
            if best_d is None or best_count == 0:
                return
            
            # Try all valid ways to cover best_d
            d = best_d
            for s in S:
                for x in [s + d, s - d]:
                    if x <= 0 or x > max_val or x in S_set:
                        continue
                    
                    # Check Sidon validity
                    new_diffs = set()
                    valid = True
                    for s2 in S:
                        dd = abs(s2 - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if not valid:
                        continue
                    
                    # Prune: if x > best_diameter, skip
                    if x >= best_diameter:
                        continue
                    
                    # Apply
                    S.append(x)
                    S_set.add(x)
                    old_used = set(used_diffs)
                    old_covered = set(covered)
                    used_diffs |= new_diffs
                    covered |= (new_diffs & TARGET)
                    
                    dfs()
                    
                    # Backtrack
                    S.pop()
                    S_set.remove(x)
                    used_diffs = old_used
                    covered = old_covered
        
        dfs()
        
        if best_solution:
            return sorted(best_solution)
        return None

    def local_refine(S, deadline):
        """Reduce max(S) by replacing elements with smaller values."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        while time.time() < deadline:
            improved = False
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                # Try replacing S[i] with smaller values, scanning from 1 upward
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if not is_sidon(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    if max(trial) < best_max:
                        S = trial
                        best = trial
                        best_max = max(S)
                        improved = True
                        break
            if not improved:
                break

        return best

    def swap_local_search(S, deadline):
        """Try swapping pairs of elements to reduce diameter."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        while time.time() < deadline:
            improved = False
            n = len(S)
            for i in range(n):
                for j in range(i + 1, n):
                    if S[i] == 0 or S[j] == 0:
                        continue
                    # Try replacing S[j] with a smaller value
                    for new_val in range(1, S[j]):
                        if new_val in S:
                            continue
                        trial = S[:]
                        trial[j] = new_val
                        trial.sort()
                        if not is_sidon(trial):
                            continue
                        if not covers_all(trial):
                            continue
                        if max(trial) < best_max:
                            S = trial
                            best = trial
                            best_max = max(S)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return best

    def iterative_deepening(deadline):
        """Try increasing max_val until we find a solution."""
        # Get an initial upper bound from greedy
        S_greedy = greedy_construct(300, deadline - 10)
        if S_greedy and covers_all(S_greedy) and is_sidon(S_greedy):
            upper = max(S_greedy)
        else:
            upper = 200  # Default upper bound
        
        # Try from small values up to the upper bound
        # Start from a reasonable lower bound
        # For a Sidon set with 31 elements covering diffs 1..30, 
        # the minimum diameter is at least ~30 (need to realize diff 30)
        # but likely much larger due to Sidon constraints
        
        for max_val in range(30, upper + 1):
            if time.time() > deadline:
                break
            
            sol = exact_cover_dlx(max_val, deadline - 5)
            if sol and covers_all(sol) and is_sidon(sol):
                return sol
        
        # If exact cover failed, return greedy
        if S_greedy and covers_all(S_greedy) and is_sidon(S_greedy):
            return S_greedy
        
        return None

    # Main search
    best_S = None
    best_score = (30, 10**9)

    # Get initial solution with greedy
    random.seed(42)
    S0 = greedy_construct(300, DEADLINE - 20)
    if S0 and covers_all(S0) and is_sidon(S0):
        missing = len(TARGET - get_diffs(S0))
        s0 = (missing, max(S0))
        if s0 < best_score:
            best_score = s0
            best_S = S0

    # Try exact-cover DLX search with iterative deepening
    if time.time() < DEADLINE - 30:
        sol_dlx = iterative_deepening(DEADLINE - 30)
        if sol_dlx and covers_all(sol_dlx) and is_sidon(sol_dlx):
            missing = len(TARGET - get_diffs(sol_dlx))
            s = (missing, max(sol_dlx))
            if s < best_score:
                best_score = s
                best_S = sol_dlx

    # Local refinement
    if best_S and time.time() < DEADLINE - 10:
        refined = local_refine(best_S, DEADLINE - 10)
        if refined:
            missing = len(TARGET - get_diffs(refined))
            rs = (missing, max(refined))
            if rs < best_score:
                best_score = rs
                best_S = refined

    # Swap local search
    if best_S and time.time() < DEADLINE - 5:
        swapped = swap_local_search(best_S, DEADLINE - 5)
        if swapped:
            missing = len(TARGET - get_diffs(swapped))
            ss = (missing, max(swapped))
            if ss < best_score:
                best_score = ss
                best_S = swapped

    # Final validation
    if best_S is None or not covers_all(best_S) or not is_sidon(best_S):
        # Fallback: simple construction
        best_S = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            for x in range(1, 1000000):
                if x in best_S:
                    continue
                trial = sorted(best_S + [x])
                if is_sidon(trial):
                    trial_diffs = get_diffs(trial)
                    if d in trial_diffs:
                        best_S = trial
                        covered = trial_diffs & TARGET
                        found = True
                        break
            if not found:
                y = 2 * max(best_S) + d + 1
                best_S = sorted(best_S + [y, y + d])
                covered = get_diffs(best_S) & TARGET

    return best_S
