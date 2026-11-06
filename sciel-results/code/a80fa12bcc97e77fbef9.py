import time
import random

def creates_4ap(S, x):
    """Check if adding x to S creates a 4-AP."""
    for d in range(1, 257):
        # x is 1st term: x, x+d, x+2d, x+3d
        if x + 3 * d <= 256:
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        else:
            break # Since d is increasing, further d will also exceed 256 for x+3d
            
        # x is 2nd term: x-d, x, x+d, x+2d
        if x - d >= 1:
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # Note: if x-d < 1, then for larger d, x-d will be even smaller, so no need to check further for this pattern?
        # Actually, if x-d < 1, then x-2d < 1 etc.
        # But we still need to check x as 1st term for larger d?
        # Yes, x+3d might still be <= 256 for small d, but for large d it breaks.
        # The `break` above handles the x as 1st term case.
        # For x as 2nd, 3rd, 4th, the lower bound constraint `x-kd >= 1` becomes stricter as d increases.
        # So if `x-d < 1`, then `x-2d < 1` and `x-3d < 1` for all larger d.
        # So we can break out of the loop entirely if x-d < 1?
        # No, because x as 1st term check is independent.
        # But wait, if d is such that x-d < 1, then x-2d < 1 and x-3d < 1.
        # So cases 2, 3, 4 are impossible for this d and all larger d.
        # Case 1 (x as 1st) is possible until x+3d > 256.
        # So we can't break the whole loop, but we can skip checks 2,3,4.
        
        # x is 3rd term: x-2d, x-d, x, x+d
        if x - 2 * d >= 1:
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        
        # x is 4th term: x-3d, x-2d, x-d, x
        if x - 3 * d >= 1:
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
                
    return False

def count_3aps(S):
    """Count total number of 3-APs in S."""
    count = 0
    for a in range(1, 257):
        if a not in S:
            continue
        max_d = (256 - a) // 2
        for d in range(1, max_d + 1):
            if (a + d) in S and (a + 2 * d) in S:
                count += 1
    return count

def construct():
    n = 256
    best_A = set()
    best_score = -1
    start_time = time.time()
    time_limit = 55.0
    
    while time.time() - start_time < time_limit:
        # Initialize random set
        # Random size between n/2 and n
        size = random.randint(n // 2, n)
        A = set(random.sample(range(1, n + 1), size))
        
        # Make valid: remove elements that form 4APs
        changed = True
        while changed:
            changed = False
            ap_found = None
            # Find any 4AP
            for a in range(1, n + 1):
                if a not in A:
                    continue
                max_d = (n - a) // 3
                for d in range(1, max_d + 1):
                    a1, a2, a3, a4 = a, a + d, a + 2 * d, a + 3 * d
                    if a2 in A and a3 in A and a4 in A:
                        ap_found = (a1, a2, a3, a4)
                        break
                if ap_found:
                    break
            
            if ap_found:
                # Remove the last element for simplicity
                A.discard(ap_found[3])
                changed = True
        
        # Local Search: Add elements
        score = count_3aps(A)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            candidates = [x for x in range(1, n + 1) if x not in A]
            random.shuffle(candidates)
            for x in candidates:
                if creates_4ap(A, x):
                    continue
                # Calculate delta
                delta = 0
                for d in range(1, 257):
                    # x is 1st
                    if x + 2 * d <= n:
                        if (x + d) in A and (x + 2 * d) in A:
                            delta += 1
                    # x is 2nd
                    if x - d >= 1:
                        if (x - d) in A and (x + d) in A:
                            delta += 1
                    # x is 3rd
                    if x - 2 * d >= 1:
                        if (x - 2 * d) in A and (x - d) in A:
                            delta += 1
                    # Optimization: if x-2d < 1, then x-d < 1 for larger d, so break?
                    # No, x+2d might still be valid for x as 1st.
                    # But x as 2nd and 3rd require x-kd >= 1.
                    # If x-2d < 1, then x-d < 1 is false? No, x-d > x-2d.
                    # If x-2d < 1, then x-d might be >= 1.
                    # Example: x=5, d=3. x-2d = -1. x-d = 2.
                    # So we can't break simply.
                    
                if delta > 0:
                    A.add(x)
                    score += delta
                    improved = True
        
        if score > best_score:
            best_score = score
            best_A = A
    
    return sorted(best_A)
