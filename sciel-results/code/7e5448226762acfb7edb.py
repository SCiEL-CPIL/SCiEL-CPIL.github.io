import time
import sys
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 140.0

    def elapsed():
        return time.time() - start

    def find_solution(n_max):
        """
        Find a Sidon set of size n_max (including 0) with max element <= n_max
        that covers differences 1..30.
        Uses backtracking with forward checking.
        """
        used_diffs = set()
        covered = set()
        need = set(range(1, 31))
        S = [0]
        remaining_needed = 30

        def backtrack(idx, max_val):
            nonlocal used_diffs, covered, need, remaining_needed
            if elapsed() > deadline:
                return None
            if len(covered) == 30:
                return list(S)
            if idx >= n_max:
                return None
            # We need at least 'remaining_needed' more differences from
            # (n_max - idx) more elements. Each new element can contribute
            # at most idx new differences (one with each existing element).
            # But we also need the differences to be in 1..30 and not already used.
            # Simple bound: need at least ceil(remaining_needed / idx) more elements if idx > 0
            if idx > 0 and remaining_needed > (n_max - idx) * idx:
                return None
            
            # Generate candidates: for each missing needed difference d,
            # and each existing s in S, candidate = s + d must be > S[-1]
            # Also try small values just above S[-1]
            candidates = set()
            for d in need:
                for s in S:
                    cand = s + d
                    if cand > S[-1] and cand <= n_max:
                        candidates.add(cand)
            # Also add a few values just above current max for flexibility
            for v in range(S[-1] + 1, min(S[-1] + 10, n_max + 1)):
                candidates.add(v)
            
            # Sort candidates: prefer those that cover more needed differences
            def score_candidate(x):
                sc = 0
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    if d in need:
                        sc += 1
                return (sc if valid else -1, x)
            
            valid_candidates = []
            for x in candidates:
                if x <= S[-1] or x > n_max:
                    continue
                sc = 0
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    if d in need:
                        sc += 1
                if valid:
                    valid_candidates.append((sc, x))
            
            if not valid_candidates:
                return None
            
            # Sort by score descending, then by value ascending
            valid_candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit branching factor for speed
            max_branch = min(len(valid_candidates), 15)
            
            for sc, x in valid_candidates[:max_branch]:
                # Try this candidate
                new_diffs = []
                for s in S:
                    d = x - s
                    new_diffs.append(d)
                
                S.append(x)
                old_used = set(used_diffs)
                for d in new_diffs:
                    used_diffs.add(d)
                old_covered = set(covered)
                for d in new_diffs:
                    if 1 <= d <= 30:
                        covered.add(d)
                old_need = set(need)
                need = need - covered
                old_remaining = remaining_needed
                remaining_needed = len(need)
                
                result = backtrack(idx + 1, n_max)
                if result is not None:
                    return result
                
                # Undo
                S.pop()
                used_diffs = old_used
                covered = old_covered
                need = old_need
                remaining_needed = old_remaining
            
            return None
        
        return backtrack(1, n_max)

    def solve():
        """
        Iterative deepening: try increasing diameter bounds.
        First determine minimum needed size, then search.
        """
        # Lower bound on size: a Sidon set of size n has at most n*(n-1)/2 distinct differences.
        # We need at least 30 differences, so n*(n-1)/2 >= 30 => n >= 8
        # But we also need to fit them in [0, max_val].
        # Try sizes from 8 upward, and for each size try increasing max_val.
        
        # Actually, let's just do iterative deepening on max_val directly.
        # Start with a reasonable lower bound.
        # A Sidon set covering 1..30 needs at least 8 elements (since 8*7/2=28 < 30, need 9: 9*8/2=36)
        # Actually 8*7/2 = 28 < 30, so we need at least 9 elements.
        
        # Let's try different set sizes and max values
        # For a given max_val M and size n, we need to pick n-1 values from 1..M
        # such that all C(n,2) differences are distinct and 1..30 are covered.
        
        # Strategy: for each candidate size n (9 to 12), find the minimum M
        # by trying M from a lower bound upward.
        
        # Lower bound on M: we need differences 1..30, so M >= 30.
        # But with Sidon constraint, M will be larger.
        
        # Let's try a few (n, M) combinations
        # Known: the optimal is likely around n=9-11, M=40-60
        
        # Try increasing M
        for M in range(35, 80):
            if elapsed() > deadline:
                break
            # Try different sizes
            for n in range(9, 13):
                if elapsed() > deadline:
                    break
                # Check if n elements can cover 30 diffs: C(n,2) >= 30
                if n * (n - 1) // 2 < 30:
                    continue
                result = find_solution_with_size(n, M)
                if result is not None:
                    return result
        return None

    def find_solution_with_size(n, M):
        """Find Sidon set of exactly n elements (including 0) with max <= M covering 1..30."""
        used_diffs = set()
        covered = set()
        need = set(range(1, 31))
        S = [0]
        
        def backtrack(idx):
            nonlocal used_diffs, covered, need
            if elapsed() > deadline:
                return None
            if len(covered) == 30:
                if idx >= n:
                    return list(S)
                # We have coverage but need exactly n elements; add more if possible
                # Actually, if we have all 30 covered and idx < n, we still need to fill to size n
                # But the problem says "at most 32", so smaller is fine. Let's just return if covered.
                # Wait, we want to minimize max element, not maximize size. So if covered, return.
                return list(S)
            if idx >= n:
                return None
            
            remaining = n - idx
            remaining_needed = len(need)
            # Each new element adds at most (idx) new differences (with existing elements)
            # But some might be > 30 or already used
            if remaining_needed > remaining * idx:
                return None
            
            # Generate candidates
            candidates = set()
            for d in need:
                for s in S:
                    cand = s + d
                    if cand > S[-1] and cand <= M:
                        candidates.add(cand)
            for v in range(S[-1] + 1, min(S[-1] + 8, M + 1)):
                candidates.add(v)
            
            valid_candidates = []
            for x in candidates:
                if x <= S[-1] or x > M:
                    continue
                sc = 0
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    if d in need:
                        sc += 1
                if valid:
                    valid_candidates.append((sc, x))
            
            if not valid_candidates:
                return None
            
            valid_candidates.sort(key=lambda t: (-t[0], t[1]))
            max_branch = min(len(valid_candidates), 12)
            
            for sc, x in valid_candidates[:max_branch]:
                new_diffs = [x - s for s in S]
                
                S.append(x)
                old_used = set(used_diffs)
                for d in new_diffs:
                    used_diffs.add(d)
                old_covered = set(covered)
                for d in new_diffs:
                    if 1 <= d <= 30:
                        covered.add(d)
                old_need = set(need)
                need = need - covered
                
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                
                S.pop()
                used_diffs = old_used
                covered = old_covered
                need = old_need
            
            return None
        
        return backtrack(1)

    # Main: try to find the best solution
    best_result = None
    
    # Try increasing M values, for each try multiple sizes
    for M in range(35, 75):
        if elapsed() > deadline:
            break
        found = False
        for n in range(9, 13):
            if elapsed() > deadline:
                break
            if n * (n - 1) // 2 < 30:
                continue
            result = find_solution_with_size(n, M)
            if result is not None:
                best_result = result
                found = True
                break
        if found:
            break
    
    # Fallback: if no solution found, use a simple greedy
    if best_result is None:
        best_result = [0]
        used = set()
        need = set(range(1, 31))
        for _ in range(32):
            found_x = False
            for x in range(best_result[-1] + 1, best_result[-1] + 200):
                ok = True
                for s in best_result:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                if ok:
                    best_result.append(x)
                    for s in best_result[:-1]:
                        used.add(x - s)
                    need -= used
                    found_x = True
                    break
            if not found_x:
                break
            if not need:
                break
    
    # Final validation
    def is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_1_to_30(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return all(d in diffs for d in range(1, 31))

    if not is_sidon(best_result) or not covers_1_to_30(best_result) or 0 not in best_result:
        # Absolute fallback
        best_result = [0]
        used = set()
        need = set(range(1, 31))
        for _ in range(32):
            found_x = False
            for x in range(best_result[-1] + 1, best_result[-1] + 500):
                ok = True
                for s in best_result:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                if ok:
                    best_result.append(x)
                    for s in best_result[:-1]:
                        used.add(x - s)
                    need -= used
                    found_x = True
                    break
            if not found_x:
                break
            if not need:
                break
    
    return best_result
