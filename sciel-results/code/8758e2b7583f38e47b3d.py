import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target: cover differences 1..30
    full_mask = (1 << 30) - 1
    
    def popcount(x):
        return bin(x).count("1")
        
    def solve_for_M(M):
        """
        Find a Sidon set containing 0 and M, with all elements in [0, M],
        covering diffs 1..30.
        Returns the set as a list or None.
        """
        # Initial state: {0, M}
        # Difference M is present.
        A = [0, M]
        all_diffs = {M}
        
        # Mask for covered differences 1..30
        # If M <= 30, bit (M-1) is set.
        covered_mask = (1 << (M - 1)) if M <= 30 else 0
        
        # If M > 30, we still need to cover 1..30.
        # If M <= 30, we have covered M.
        
        # Check if already solved
        if covered_mask == full_mask:
            return A[:]
            
        def backtrack(A, all_diffs, covered_mask, next_min):
            if time.time() > deadline:
                return None
                
            # Check goal
            if covered_mask == full_mask:
                return A[:]
                
            # Pruning: slots
            if len(A) >= 32:
                return None
                
            # Pruning: missing diffs vs capacity
            missing = 30 - popcount(covered_mask)
            if missing == 0:
                return A[:]
                
            slots_left = 32 - len(A)
            # Max new diffs from slots_left elements
            # Each new element x adds diffs with all existing elements in A.
            # If we add k elements, the total new diffs is:
            # Sum_{i=0}^{k-1} (len(A) + i) = k * len(A) + k * (k - 1) / 2
            max_new = slots_left * len(A) + (slots_left * (slots_left - 1)) // 2
            if missing > max_new:
                return None
                
            # Try adding next element x
            # x must be >= next_min and < M
            # We iterate x in increasing order.
            
            for x in range(next_min, M):
                if time.time() > deadline:
                    return None
                    
                valid = True
                new_diffs = []
                for a in A:
                    d = x - a
                    if d <= 0:
                        continue
                    if d in all_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                    
                if not valid:
                    continue
                    
                new_all_diffs = all_diffs | set(new_diffs)
                new_covered_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_covered_mask |= (1 << (d - 1))
                        
                # Heuristic: Skip if no new target diff covered
                # This assumes that every element in an optimal minimal-diameter
                # Sidon set covering 1..30 must contribute at least one new diff
                # to the set {1..30}.
                if new_covered_mask == covered_mask:
                    continue
                    
                new_A = A + [x]
                res = backtrack(new_A, new_all_diffs, new_covered_mask, x + 1)
                if res is not None:
                    return res
                    
            return None
            
        return backtrack(A, all_diffs, covered_mask, 1)

    # Sweep M from lower bound upwards
    # Lower bound is 30 because diff 30 must be present.
    
    # To ensure we don't run out of time without a result,
    # we keep a fallback solution if we find one.
    # But the loop breaks on time.
    
    for M in range(30, 10000):
        if time.time() > deadline:
            break
            
        sol = solve_for_M(M)
        if sol is not None:
            return sol
            
    # Fallback: If no solution found in time, return a valid (but possibly non-optimal) set
    # We can construct a simple valid set using a greedy approach or known properties.
    # A simple valid set: {0, 1, 3, 7, 12, 18, 25, 33, ...} is not guaranteed.
    # Let's use a set that is known to work for larger M.
    # Or just raise an error? No, must return something.
    
    # Greedy fallback
    a = [0]
    used_diffs = set()
    target = set(range(1, 31))
    
    # We need to add elements to cover 1..30.
    # This is a slow fallback, but ensures validity.
    while not target.issubset(used_diffs):
        # Find smallest d in target not in used_diffs
        missing = [d for d in target if d not in used_diffs]
        if not missing:
            break
        d = min(missing)
        
        # Try to find x such that x - a = d or a - x = d for some a in a
        # Or x such that x is new and x - a = d
        found = False
        for a in a:
            x = a + d
            if x in a:
                continue
            # Check Sidon property for x
            valid = True
            new_diffs = set()
            for y in a:
                diff = abs(x - y)
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            
            if valid:
                a.append(x)
                used_diffs.update(new_diffs)
                found = True
                break
        
        if not found:
            # Try x = max(a) + d + 1
            x = max(a) + d + 1
            while True:
                valid = True
                new_diffs = set()
                for y in a:
                    diff = abs(x - y)
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                if valid:
                    a.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
                
    return a

