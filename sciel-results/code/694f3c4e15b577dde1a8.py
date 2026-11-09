import time
import random

def construct():
    N = 256
    DEADLINE = 85.0  # Safety margin
    t_start = time.time()
    
    best_size = 0
    best_A = []
    best_B = []
    
    # Elements to consider
    elements = list(range(1, N + 1))
    
    # Number of restarts is determined by time budget
    # Each restart is fast (O(N^2) worst case, but N=256 is small)
    # Let's aim for thousands of restarts if possible.
    
    while time.time() - t_start < DEADLINE:
        random.shuffle(elements)
        
        A = []
        B = []
        # Sums sets for Sidon check
        sums_A = set()
        sums_B = set()
        # Difference sets for disjoint check
        diffs_A = set()
        diffs_B = set()
        
        # We will iterate through the shuffled elements and try to add them
        # Strategy: Try to add to the set that is currently smaller (balanced growth)
        # If sizes are equal, try A first, then B, or random.
        
        for x in elements:
            # Check if we can add x to A
            can_add_A = True
            # Sidon check for A
            if (x + x) in sums_A:
                can_add_A = False
            else:
                for a in A:
                    if (x + a) in sums_A:
                        can_add_A = False
                        break
            
            # Diff disjoint check for A
            if can_add_A:
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    if d1 in diffs_B or d2 in diffs_B:
                        can_add_A = False
                        break
            
            # Check if we can add x to B
            can_add_B = True
            if (x + x) in sums_B:
                can_add_B = False
            else:
                for b in B:
                    if (x + b) in sums_B:
                        can_add_B = False
                        break
            
            if can_add_B:
                for b in B:
                    d1 = x - b
                    d2 = b - x
                    if d1 in diffs_A or d2 in diffs_A:
                        can_add_B = False
                        break
            
            # Decide where to add
            if can_add_A and can_add_B:
                # Prefer the smaller set
                if len(A) <= len(B):
                    # Add to A
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]: # A[:-1] is old A
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                else:
                    # Add to B
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]: # B[:-1] is old B
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            elif can_add_A:
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            elif can_add_B:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
            
            # Early termination if we hit the theoretical max? 
            # Max size for Sidon in [1,256] is approx 16? 
            # Actually, max Sidon size for n is ~ sqrt(n). 
            # But we have two sets.
            # If |A|+|B| is very high, we can stop.
            # Let's just let it run to the end of the list.
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B
            
            # Optional: If we reach a very high score, we might stop early?
            # Theoretical max for one Sidon set in [1,256]:
            # Number of sums is k(k+1)/2 <= 511 -> k <= 31.
            # So max combined is 62.
            # If we get close to 60, it's good.
            if best_size >= 58:
                # We can break to save time, but let's see if we can do better.
                # Actually, just continue to see if we can hit 60+.
                pass

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
