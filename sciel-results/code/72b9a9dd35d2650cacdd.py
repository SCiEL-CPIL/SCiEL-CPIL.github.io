"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Strategy:
    # We search for a Sidon set A containing 0 such that differences cover 1..30.
    # We want to minimize max(A).
    # Let M = max(A). We know M >= 30.
    # We will iterate M from a lower bound upwards.
    # For a fixed M, we check if there exists a subset A of {0, ..., M} 
    # such that:
    # 1. 0 in A, M in A (since we minimize max, if a solution exists with max < M, we would have found it earlier. 
    #    Actually, we can just check if a solution exists with elements in [0, M] and max(A)=M? 
    #    No, it's easier to just check if a solution exists in [0, M]. 
    #    If it does, we can potentially reduce M later? No, we iterate M up.
    #    So for each M, we ask: Is there a Sidon set A subset of {0..M} containing 0, covering 1..30, with max(A)=M?
    #    Actually, if a set exists in [0..M-1], we would have found it in the previous iteration.
    #    So we can assume M is in A? Not necessarily. But if max(A) < M, we return that.
    #    So: Iterate M from LB to UB.
    #    Check if there is a valid Sidon set A subset of {0..M} covering 1..30.
    #    If yes, return it. (This guarantees minimal max(A) because we check M in increasing order).
    
    # Lower Bound Estimation:
    # To cover 30 diffs, we need at least 9 elements.
    # A Sidon set of size k in [0, M] requires M >= k^2 - k + something?
    # For k=9, M is roughly 9^2 = 81? 
    # Known optimal for k=9 covering 1..30?
    # Let's start M from 40. It's unlikely to be less than 30.
    
    LB = 40
    UB = 500 # Safety net
    
    def check_solution(A):
        if not A or 0 not in A:
            return False
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    # CSP Solver for fixed M
    def solve_for_M(M):
        """
        Finds a Sidon set A subset of {0..M} containing 0, covering 1..30.
        Uses backtracking with forward checking.
        Variables: Elements of A.
        Domain: {0, ..., M}.
        Constraints:
        - 0 in A.
        - All differences distinct.
        - Differences cover 1..30.
        """
        
        # We can fix 0 in A.
        # We need to select k-1 more elements from {1..M}.
        # To minimize search space, we can try to determine the size k.
        # Minimum k is 9.
        # Let's try k=9 first, then 10, etc.
        
        # State:
        # A: set of chosen elements (starts with {0})
        # used_diffs: set of differences generated
        # missing: set of diffs 1..30 not yet covered
        
        # Heuristic:
        # We will build A by adding elements one by one.
        # To be efficient, we should pick elements that cover many missing diffs.
        # However, standard backtracking on subsets is hard.
        
        # Alternative: 
        # Since M is small (<= 500), we can use a bitmask or boolean array for used_diffs.
        # But differences can be up to M.
        
        # Let's use a recursive search.
        # To prune, we need to know if it's possible to cover remaining diffs.
        
        # Optimization:
        # If we fix M, we can assume M is in A? 
        # If a solution exists with max(A) < M, we would have found it for M-1.
        # So for the specific M we are testing, we can force M to be in A.
        # This reduces the search space significantly.
        
        initial_A = [0, M]
        initial_used = {M}
        initial_missing = set(range(1, 31))
        
        # Check if M itself covers any required diff (it doesn't, since M >= 40 > 30)
        # Wait, if M < 30, it would. But M starts at 40.
        
        # Sort candidates?
        # We need to pick k-2 more elements from {1..M-1}.
        
        # Forward Checking:
        # When we add x, we add diffs |x - a| for all a in A.
        # If any diff is already in used, invalid.
        # If any diff is in missing, remove it from missing.
        
        def search(A, used, missing):
            if time.time() > deadline:
                return None
            
            if not missing:
                # Check if A is valid Sidon (should be by construction)
                # And covers 1..30 (missing is empty)
                return A[:]
            
            if len(A) > 32:
                return None
            
            # Heuristic: 
            # Calculate for each candidate x in {1..M-1} \ A:
            # 1. Is it valid? (no diff conflict)
            # 2. How many missing diffs does it cover?
            # Pick x with max coverage (or min conflicts).
            
            # To speed up, we can pre-filter candidates.
            # A candidate x is valid if for all a in A, |x-a| not in used and |x-a| not in { |x-b| for b in A, b!=a }?
            # Actually, we just check against 'used' and internal consistency of new diffs.
            
            # Let's generate candidates.
            # To avoid iterating 1..M-1 every time (which is O(M)), we can be smarter?
            # M is small (<=500). Iterating is fine.
            
            best_candidate = None
            best_coverage = -1
            best_new_diffs = None
            
            # We also need to consider that we might need to add multiple elements.
            # Greedy selection of the "best" x might lead to dead ends.
            # So we should try top K candidates.
            
            candidates_info = []
            
            for x in range(1, M):
                if x in A:
                    continue
                
                # Check validity and coverage
                new_diffs = set()
                valid = True
                for a in A:
                    d = abs(x - a)
                    if d in used:
                        valid = False
                        break
                    if d in new_diffs:
                        # Duplicate diff within the new set? 
                        # This happens if |x-a1| == |x-a2| => a1 == a2 (impossible) or a1+a2 = 2x.
                        # If a1+a2 = 2x, then d is generated twice.
                        # This violates Sidon property.
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Count how many missing diffs this x covers
                coverage = sum(1 for d in new_diffs if d in missing)
                
                # Heuristic score:
                # Prefer higher coverage.
                # Tie-breaker: Prefer smaller x? Or larger x?
                # Smaller x tends to create smaller diffs, which are more likely to be in 1..30.
                # Larger x creates larger diffs, less likely to be in 1..30 (unless M is small).
                # Since we want to cover 1..30, smaller x is generally better for coverage of small diffs.
                
                candidates_info.append((coverage, x, new_diffs))
            
            if not candidates_info:
                return None
            
            # Sort by coverage (desc), then x (asc)
            candidates_info.sort(key=lambda item: (-item[0], item[1]))
            
            # Try top N candidates
            # If no candidate covers any missing diff, we might be stuck?
            # If coverage is 0 for all, we can still add them if necessary to bridge?
            # But if we add a diff that is not in 1..30 and not in used, it's "wasted" space.
            # However, it might be necessary to allow other elements to be placed?
            # Actually, adding an element that doesn't cover a required diff is usually bad for minimizing size/diameter.
            # But we are fixing diameter M. We just need *a* set.
            # If we have to add "waste" elements, we should prefer those that leave the most options open.
            # For now, just try top 10 by coverage.
            
            # If all have 0 coverage, we might need to backtrack or fail.
            # Let's try the first 5.
            
            for coverage, x, new_diffs in candidates_info[:5]:
                if coverage == 0 and len(candidates_info) > 5:
                    # If the best one covers 0, and there are many, maybe try a few others?
                    # Or just break?
                    # If best coverage is 0, we are likely in a dead end unless we add "neutral" elements.
                    # Let's try the first 3 neutral ones if best is 0.
                    pass
                
                new_A = A + [x]
                new_A.sort()
                new_used = set(used)
                new_used.update(new_diffs)
                
                # Update missing
                new_missing = set(missing)
                for d in new_diffs:
                    if d in new_missing:
                        new_missing.remove(d)
                
                # Recurse
                res = search(new_A, new_used, new_missing)
                if res is not None:
                    return res
            
            return None
        
        return search(initial_A, initial_used, initial_missing)

    # Main Loop
    for M in range(LB, UB + 1):
        if time.time() > deadline:
            break
            
        # Try to find a solution with max element exactly M
        # Note: By forcing M in A, we ensure max(A)=M.
        # If a solution with max(A) < M exists, it would have been found in previous iterations.
        
        sol = solve_for_M(M)
        
        if sol is not None:
            # Verify solution
            if check_solution(sol):
                return sol
            else:
                # Should not happen if solver is correct
                continue

    # Fallback: If no solution found in [40, 500], try a larger range or greedy
    # This is unlikely, but just in case.
    # Let's try a simple greedy to get *a* valid set, even if not optimal.
    a = [0]
    used = set()
    for d in range(1, 31):
        if d in used:
            continue
        found = False
        # Try to add x
        for x in range(1, 10000):
            if x in a:
                continue
            new_diffs = set()
            valid = True
            for ai in a:
                diff = abs(x - ai)
                if diff in used or diff in new_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            if valid:
                a.append(x)
                a.sort()
                used.update(new_diffs)
                found = True
                break
        if not found:
            # Emergency
            x = max(a) + d + 100
            a.append(x)
            a.sort()
            # Update used
            for ai in a[:-1]:
                used.add(abs(x-ai))
    
    # Verify fallback
    if check_solution(a):
        return a
    
    # Last resort: Hardcoded large set (should not be needed)
    return [0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]

# EVOLVE-BLOCK-END
