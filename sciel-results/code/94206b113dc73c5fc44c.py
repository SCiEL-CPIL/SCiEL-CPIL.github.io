import time
import random

def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    # We need a Sidon set (all positive differences distinct) containing 0,
    # covering all differences 1..30, at most 32 elements, minimize max element.
    #
    # Strategy: Backtracking search. Build the set element by element.
    # At each step, we track:
    #   - current set S (sorted)
    #   - set of used differences
    #   - set of still-needed differences (subset of 1..30)
    #
    # Heuristic: prioritize adding elements that cover the most needed differences.
    
    def solve():
        needed = set(range(1, 31))
        used_diffs = set()
        S = [0]
        
        def backtrack(S, used_diffs, needed, depth):
            if time.time() > deadline:
                return None
            if not needed:
                return S[:]
            if len(S) >= 32:
                return None
            
            # Try to find candidates that cover needed differences
            # Candidate: for each x in S and each d in needed, try x+d and x-d
            # Also try small new values
            candidates = set()
            for x in S:
                for d in needed:
                    candidates.add(x + d)
                    if x - d >= 0:
                        candidates.add(x - d)
            # Also add some small candidates
            max_s = max(S)
            for v in range(1, min(max_s + 31, 200)):
                candidates.add(v)
            
            # Score candidates by how many needed diffs they cover
            scored = []
            for c in candidates:
                if c in S:
                    continue
                if c < 0 or c > 1000000:
                    continue
                # Check Sidon property: all new differences must be unused
                new_diffs = set()
                valid = True
                for x in S:
                    d = abs(c - x)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                # Count how many needed diffs are covered
                covered = len(new_diffs & needed)
                if covered == 0:
                    continue
                scored.append((covered, c))
            
            if not scored:
                return None
            
            # Sort by most covered first, then by value
            scored.sort(key=lambda t: (-t[0], t[1]))
            
            for covered, c in scored:
                new_S = S + [c]
                new_S.sort()
                new_used = used_diffs | set(abs(c - x) for x in S)
                new_needed = needed - set(abs(c - x) for x in S)
                
                result = backtrack(new_S, new_used, new_needed, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(S, used_diffs, needed, 0)
    
    result = solve()
    if result is not None:
        return result
    
    # Fallback: greedy (should not be reached if backtracking works)
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
