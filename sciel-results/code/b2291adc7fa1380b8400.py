import sys
import time

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with the minimum possible largest element.
    Uses a custom branch-and-bound solver with symmetry breaking (fixing max element to D).
    """
    required_diffs = set(range(1, 31))
    
    def find_set_for_D(D, k):
        """
        Tries to find a Sidon set of size k in [0, D] containing 0 and D,
        covering 1..30.
        """
        # We fix 0 and D. We need to choose k-2 elements from 1 to D-1.
        # Initial state
        A = [0, D]
        diffs = {D} # Difference D-0=D
        covered = required_diffs.intersection(diffs)
        
        # If k==2, we just check if covered == required_diffs
        if k == 2:
            if covered == required_diffs:
                return A
            return None

        def backtrack(idx, current_max):
            if idx == k:
                if covered == required_diffs:
                    return A[:]
                return None
            
            # Pruning:
            # 1. Capacity
            remaining_req = required_diffs - covered
            m = k - idx
            # New elements will be added to A.
            # Current size is idx.
            # Each new element adds idx diffs.
            # Between new elements, they add m*(m-1)/2 diffs.
            # Total new diffs capacity: m * idx + m * (m - 1) // 2
            max_new_diffs = m * idx + m * (m - 1) // 2
            if len(remaining_req) > max_new_diffs:
                return None

            # 2. Fit
            # We need m elements in [current_max + 1, D-1] (since D is already taken)
            # Wait, the elements must be sorted.
            # A is sorted. current_max is A[-1].
            # The next elements must be > current_max and < D.
            # So they are in [current_max + 1, D - 1].
            # Number of available slots: (D - 1) - (current_max + 1) + 1 = D - current_max - 1
            if D - current_max - 1 < m:
                return None

            # 3. Forward Checking on Smallest Uncovered Diff
            if remaining_req:
                min_uncovered = min(remaining_req)
                # The next element x must be such that it doesn't make it impossible to form min_uncovered.
                # Actually, min_uncovered must be formed by some pair.
                # If min_uncovered is not formed by (0, min_uncovered) or (D-min_uncovered, D), 
                # it must be formed by two elements in the middle or one middle and one endpoint.
                # This check is complex. We'll rely on the capacity and standard backtracking.
                
            # Generate candidates
            # Start from current_max + 1 to D - 1
            # Optimization: Sort candidates by how many required diffs they cover?
            # For now, simple range.
            
            start_x = current_max + 1
            end_x = D - 1
            
            for x in range(start_x, end_x + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for a in A:
                    d = abs(x - a)
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Update
                A.append(x)
                diffs.update(new_diffs)
                new_covered = required_diffs.intersection(new_diffs)
                covered.update(new_covered)
                
                res = backtrack(idx + 1, x)
                if res is not None:
                    return res
                
                # Backtrack
                A.pop()
                diffs.difference_update(new_diffs)
                covered.difference_update(new_covered)
            
            return None
        
        return backtrack(2, 0) # idx=2 because 0 and D are at indices 0 and 1? 
        # Wait, A = [0, D]. idx represents the number of elements chosen so far?
        # If idx is the index in the final list, and we start with 2 elements, next index is 2.
        # But A is not sorted if we append D at the end?
        # I should maintain A sorted.
        # A = [0, D] is not sorted if D > 0.
        # Let's restart the logic for A.

    # Revised Backtracking State
    def find_set_for_D_v2(D, k):
        # We want a set S of size k, 0 in S, D in S, S subset [0, D].
        # Let's build the set in sorted order.
        # S[0] = 0.
        # S[k-1] = D.
        # We need to choose S[1] ... S[k-2] from 1 to D-1.
        
        A = [0] * k
        A[0] = 0
        A[k-1] = D
        
        diffs = set()
        # Add diffs for 0 and D
        diffs.add(D)
        covered = required_diffs.intersection(diffs)
        
        def backtrack(idx, current_max):
            if idx == k - 1:
                # All elements chosen
                if covered == required_diffs:
                    return A[:]
                return None
            
            # Pruning
            remaining_req = required_diffs - covered
            m = (k - 1) - idx # Number of elements left to choose (excluding the last fixed D)
            # Wait, idx is the index we are filling.
            # Elements filled: 0 to idx-1.
            # Elements to fill: idx to k-2.
            # Count = (k-2) - idx + 1 = k - idx - 1.
            num_to_fill = k - idx - 1
            
            # Capacity check
            # Current set size is idx.
            # Each new element adds idx diffs.
            # Between new elements: num_to_fill * (num_to_fill - 1) / 2
            max_new_diffs = num_to_fill * idx + num_to_fill * (num_to_fill - 1) // 2
            if len(remaining_req) > max_new_diffs:
                return None

            # Fit check
            # Next element x must be > current_max and < D.
            # We need num_to_fill elements in [current_max + 1, D - 1].
            if D - current_max - 1 < num_to_fill:
                return None

            # Candidate generation
            start_x = current_max + 1
            end_x = D - 1
            
            # Optimization: Try values that cover required diffs first?
            # For now, iterate normally.
            for x in range(start_x, end_x + 1):
                valid = True
                new_diffs = []
                # Check against all previously chosen elements A[0]...A[idx-1]
                for i in range(idx):
                    d = x - A[i]
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                A[idx] = x
                diffs.update(new_diffs)
                new_covered = required_diffs.intersection(new_diffs)
                covered.update(new_covered)
                
                res = backtrack(idx + 1, x)
                if res is not None:
                    return res
                
                # Backtrack
                diffs.difference_update(new_diffs)
                covered.difference_update(new_covered)
            
            return None
        
        return backtrack(1, 0)

    # Iterate k and D
    start_time = time.time()
    timeout = 110
    
    # k must be at least 9
    for k in range(9, 16):
        # Lower bound for D
        LB = k * (k - 1) // 2
        # Upper bound heuristic: maybe 2 * LB?
        # We just iterate D upwards
        for D in range(LB, 200):
            if time.time() - start_time > timeout:
                # Fallback
                return greedy_construct()
            
            result = find_set_for_D_v2(D, k)
            if result is not None:
                return sorted(result)
    
    return greedy_construct()

def greedy_construct():
    """Fallback greedy constructor."""
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

sys.setrecursionlimit(10000)
