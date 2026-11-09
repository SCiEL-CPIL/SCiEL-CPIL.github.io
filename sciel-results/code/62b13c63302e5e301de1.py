"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    best_total = 0
    best_A = []
    best_B = []
    
    def try_greedy_random():
        nonlocal best_total, best_A, best_B
        
        # Randomize the order in which we consider elements
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        # Track state for both sets
        # For each set, we need:
        # 1. The set of elements
        # 2. The set of all pair sums (a+b for a,b in set, a<=b)
        # 3. The set of all nonzero differences (|a-b| for a!=b in set)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # We'll alternate or choose randomly which set to add to
        # But we need to ensure disjointness of nonzero differences
        
        for x in elements:
            # Try adding to A
            can_add_A = True
            # Check Sidon condition for A
            for a in A:
                if (x + a) in sums_A:
                    can_add_A = False
                    break
                if (a - x) in diffs_A or (x - a) in diffs_A:
                    can_add_A = False
                    break
            # Check difference disjointness with B
            if can_add_A:
                for b in B:
                    if (x - b) in diffs_B or (b - x) in diffs_B:
                        can_add_A = False
                        break
            
            if can_add_A:
                # Add x to A
                for a in A:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                sums_A.add(2 * x)
                A.append(x)
                continue
            
            # Try adding to B
            can_add_B = True
            for b in B:
                if (x + b) in sums_B:
                    can_add_B = False
                    break
                if (b - x) in diffs_B or (x - b) in diffs_B:
                    can_add_B = False
                    break
            if can_add_B:
                for a in A:
                    if (x - a) in diffs_A or (a - x) in diffs_A:
                        can_add_B = False
                        break
            
            if can_add_B:
                for b in B:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                sums_B.add(2 * x)
                B.append(x)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
        
        return total
    
    # Run multiple random restarts
    attempts = 0
    while time.time() < deadline:
        try_greedy_random()
        attempts += 1
        # Also try a few more structured approaches
    
    # Try a more structured approach: interleave
    for trial in range(10):
        if time.time() >= deadline:
            break
        
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Alternate between A and B
        for i, x in enumerate(elements):
            target = 0 if i % 2 == 0 else 1
            
            if target == 0:
                can_add = True
                for a in A:
                    if (x + a) in sums_A:
                        can_add = False
                        break
                    if abs(x - a) in diffs_A:
                        can_add = False
                        break
                if can_add:
                    for b in B:
                        if abs(x - b) in diffs_B:
                            can_add = False
                            break
                
                if can_add:
                    for a in A:
                        sums_A.add(x + a)
                        diffs_A.add(abs(x - a))
                    sums_A.add(2 * x)
                    A.append(x)
                else:
                    # Try B instead
                    can_add = True
                    for b in B:
                        if (x + b) in sums_B:
                            can_add = False
                            break
                        if abs(x - b) in diffs_B:
                            can_add = False
                            break
                    if can_add:
                        for a in A:
                            if abs(x - a) in diffs_A:
                                can_add = False
                                break
                    if can_add:
                        for b in B:
                            sums_B.add(x + b)
                            diffs_B.add(abs(x - b))
                        sums_B.add(2 * x)
                        B.append(x)
            else:
                can_add = True
                for b in B:
                    if (x + b) in sums_B:
                        can_add = False
                        break
                    if abs(x - b) in diffs_B:
                        can_add = False
                        break
                if can_add:
                    for a in A:
                        if abs(x - a) in diffs_A:
                            can_add = False
                            break
                
                if can_add:
                    for b in B:
                        sums_B.add(x + b)
                        diffs_B.add(abs(x - b))
                    sums_B.add(2 * x)
                    B.append(x)
                else:
                    # Try A instead
                    can_add = True
                    for a in A:
                        if (x + a) in sums_A:
                            can_add = False
                            break
                        if abs(x - a) in diffs_A:
                            can_add = False
                            break
                    if can_add:
                        for b in B:
                            if abs(x - b) in diffs_B:
                                can_add = False
                                break
                    if can_add:
                        for a in A:
                            sums_A.add(x + a)
                            diffs_A.add(abs(x - a))
                        sums_A.add(2 * x)
                        A.append(x)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
