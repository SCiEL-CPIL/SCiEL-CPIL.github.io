import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    best_size = 0
    best_A = []
    best_B = []
    
    candidates = list(range(1, 257))
    
    while time.time() - start_time < timeout:
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            # Try to add x to A
            added = False
            
            # Check if x can be added to A
            valid_A = True
            # Check sums: x+a for a in A, and 2x
            for a in A:
                if (x + a) in sums_A:
                    valid_A = False
                    break
            if valid_A:
                if (2 * x) in sums_A:
                    valid_A = False
            
            # Check diffs: |x-a| for a in A must not be in diffs_B
            if valid_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_B:
                        valid_A = False
                        break
            
            if valid_A:
                # Add x to A
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                sums_A.add(2 * x)
                added = True
            
            # If not added to A, try B
            if not added:
                valid_B = True
                # Check sums: x+b for b in B, and 2x
                for b in B:
                    if (x + b) in sums_B:
                        valid_B = False
                        break
                if valid_B:
                    if (2 * x) in sums_B:
                        valid_B = False
                
                # Check diffs: |x-b| for b in B must not be in diffs_A
                if valid_B:
                    for b in B:
                        d = abs(x - b)
                        if d in diffs_A:
                            valid_B = False
                            break
                
                if valid_B:
                    # Add x to B
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(abs(x - b))
                    sums_B.add(2 * x)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)

    # Ensure non-empty (should be guaranteed by greedy on 256 elements)
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2] # 2 might conflict with 1? 
        # If A=[1], D_A is empty. D_B for [2] is empty. Sums A={2}, Sums B={4}. Disjoint diffs? Empty sets are disjoint.
        # So [1], [2] is valid.
        
    return [best_A, best_B]
