"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 85  # safety margin before 90s limit
    
    best_total = 0
    best_A = [1]
    best_B = [2]
    
    def validate(A, B):
        """Full validation: Sidon for both, disjoint nonzero diffs."""
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        if diffs_A & diffs_B:
            return False
        return True
    
    def fast_greedy(seed):
        """Fast bitmask-based greedy construction."""
        rng = random.Random(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidate pool
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            placed = False
            # Try A first
            if len(A) < max_size:
                ok = True
                for a in A:
                    s = a + x
                    if s in sums_A:
                        ok = False
                        break
                if ok:
                    if x + x in sums_A:
                        ok = False
                if ok:
                    for a in A:
                        d = abs(a - x)
                        if d > 0 and d in diffs_B:
                            ok = False
                            break
                if ok:
                    # Update
                    for a in A:
                        sums_A.add(a + x)
                        d = abs(a - x)
                        if d > 0:
                            diffs_A.add(d)
                    sums_A.add(x + x)
                    A.append(x)
                    placed = True
            
            # Try B
            if not placed and len(B) < max_size:
                ok = True
                for b in B:
                    s = b + x
                    if s in sums_B:
                        ok = False
                        break
                if ok:
                    if x + x in sums_B:
                        ok = False
                if ok:
                    for b in B:
                        d = abs(b - x)
                        if d > 0 and d in diffs_A:
                            ok = False
                            break
                if ok:
                    for b in B:
                        sums_B.add(b + x)
                        d = abs(b - x)
                        if d > 0:
                            diffs_B.add(d)
                    sums_B.add(x + x)
                    B.append(x)
        
        return A, B
    
    def local_improve(A, B, time_budget):
        """Local search: try swapping elements to increase total size."""
        if not A or not B:
            return A, B
        
        A = A[:]
        B = B[:]
        start = time.time()
        
        # Compute current state
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        
        in_A = set(A)
        in_B = set(B)
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            
            # Try adding elements to A
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x in in_A:
                        continue
                    # Check Sidon
                    ok = True
                    for a in A:
                        if a + x in sums_A:
                            ok = False
                            break
                    if ok and x + x in sums_A:
                        ok = False
                    if ok:
                        for a in A:
                            d = abs(a - x)
                            if d > 0 and d in diffs_B:
                                ok = False
                                break
                    if ok:
                        # Add x to A
                        for a in A:
                            sums_A.add(a + x)
                            d = abs(a - x)
                            if d > 0:
                                diffs_A.add(d)
                        sums_A.add(x + x)
                        A.append(x)
                        in_A.add(x)
                        improved = True
                        break
            
            # Try adding elements to B
            if len(B) < max_size:
                for x in range(1, n + 1):
                    if x in in_B:
                        continue
                    ok = True
                    for b in B:
                        if b + x in sums_B:
                            ok = False
                            break
                    if ok and x + x in sums_B:
                        ok = False
                    if ok:
                        for b in B:
                            d = abs(b - x)
                            if d > 0 and d in diffs_A:
                                ok = False
                                break
                    if ok:
                        for b in B:
                            sums_B.add(b + x)
                            d = abs(b - x)
                            if d > 0:
                                diffs_B.add(d)
                        sums_B.add(x + x)
                        B.append(x)
                        in_B.add(x)
                        improved = True
                        break
            
            # Try removing from A and re-adding elsewhere, or vice versa
            # Try swapping: remove one from A, add one to B (if it helps)
            if len(A) > 1:
                for idx in range(len(A) - 1, -1, -1):
                    if time.time() - start > time_budget:
                        break
                    x = A[idx]
                    # Remove x from A
                    A.remove(x)
                    in_A.discard(x)
                    # Recompute sums_A and diffs_A
                    sums_A_new = set()
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            sums_A_new.add(A[i] + A[j])
                    diffs_A_new = set()
                    for i in range(len(A)):
                        for j in range(len(A)):
                            if i != j:
                                d = abs(A[i] - A[j])
                                if d > 0:
                                    diffs_A_new.add(d)
                    
                    # Try adding x to B
                    ok = True
                    for b in B:
                        if b + x in sums_B:
                            ok = False
                            break
                    if ok and x + x in sums_B:
                        ok = False
                    if ok:
                        for b in B:
                            d = abs(b - x)
                            if d > 0 and d in diffs_A_new:
                                ok = False
                                break
                    if ok:
                        # Add x to B
                        for b in B:
                            sums_B.add(b + x)
                            d = abs(b - x)
                            if d > 0:
                                diffs_B.add(d)
                        sums_B.add(x + x)
                        B.append(x)
                        in_B.add(x)
                        sums_A = sums_A_new
                        diffs_A = diffs_A_new
                        improved = True
                        break
                    else:
                        # Put x back in A
                        A.append(x)
                        in_A.add(x)
                        sums_A = sums_A_new
                        diffs_A = diffs_A_new
            
            # Similarly try swapping from B to A
            if len(B) > 1:
                for idx in range(len(B) - 1, -1, -1):
                    if time.time() - start > time_budget:
                        break
                    x = B[idx]
                    B.remove(x)
                    in_B.discard(x)
                    sums_B_new = set()
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            sums_B_new.add(B[i] + B[j])
                    diffs_B_new = set()
                    for i in range(len(B)):
                        for j in range(len(B)):
                            if i != j:
                                d = abs(B[i] - B[j])
                                if d > 0:
                                    diffs_B_new.add(d)
                    
                    ok = True
                    for a in A:
                        if a + x in sums_A:
                            ok = False
                            break
                    if ok and x + x in sums_A:
                        ok = False
                    if ok:
                        for a in A:
                            d = abs(a - x)
                            if d > 0 and d in diffs_B_new:
                                ok = False
                                break
                    if ok:
                        for a in A:
                            sums_A.add(a + x)
                            d = abs(a - x)
                            if d > 0:
                                diffs_A.add(d)
                        sums_A.add(x + x)
                        A.append(x)
                        in_A.add(x)
                        sums_B = sums_B_new
                        diffs_B = diffs_B_new
                        improved = True
                        break
                    else:
                        B.append(x)
                        in_B.add(x)
                        sums_B = sums_B_new
                        diffs_B = diffs_B_new
        
        return A, B
    
    # Phase 1: Fast greedy with multiple seeds
    num_trials = 300
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        A, B = fast_greedy(trial)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Phase 2: Local improvement on best solution
    if time.time() < deadline - 5:
        budget = min(20, deadline - 5 - time.time())
        if budget > 1:
            A2, B2 = local_improve(best_A, best_B, budget)
            if validate(A2, B2):
                total2 = len(A2) + len(B2)
                if total2 > best_total:
                    best_total = total2
                    best_A = A2
                    best_B = B2
    
    # Phase 3: More greedy trials with different strategies if time permits
    if time.time() < deadline - 3:
        for trial in range(num_trials, num_trials + 200):
            if time.time() > deadline - 3:
                break
            # Alternate: prefer filling the smaller set
            A, B = fast_greedy(trial)
            # Try reversed preference
            A2, B2 = fast_greedy(trial + 10000)
            total = len(A) + len(B)
            total2 = len(A2) + len(B2)
            if total > best_total:
                best_total = total
                best_A = A[:]
                best_B = B[:]
            if total2 > best_total:
                best_total = total2
                best_A = A2[:]
                best_B = B2[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    if not validate(best_A, best_B):
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
