import time
import numpy as np
import random

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_A = []
    best_B = []
    best_score = 0
    
    def is_valid(A, B):
        """Verify the sets are valid Sidon sets with disjoint differences."""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False
        
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
                if i != j:
                    d = abs(A[i] - A[j])
                    diffs_A.add(d)
        
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
                if i != j:
                    d = abs(B[i] - B[j])
                    diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return False
            
        return True

    def greedy_build(seed):
        """Run a single randomized greedy construction using bitsets."""
        rng = random.Random(seed)
        A = []
        B = []
        
        # Bitsets: sums use bits 0..511, diffs use bits 0..255
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        # Candidate pool
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Decide which set to try to add to
            if len(A) < len(B):
                order = [0, 1]
            elif len(B) < len(A):
                order = [1, 0]
            else:
                if rng.random() < 0.5:
                    order = [0, 1]
                else:
                    order = [1, 0]
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    valid = True
                    for a in A:
                        if (sums_A >> (x + a)) & 1:
                            valid = False
                            break
                    if valid and ((sums_A >> (2 * x)) & 1):
                        valid = False
                    if not valid:
                        continue
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (((diffs_A >> d) & 1) or ((diffs_B >> d) & 1)):
                            valid = False
                            break
                    if not valid:
                        continue
                    A.append(x)
                    for a in A:
                        sums_A |= (1 << (x + a))
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A |= (1 << d)
                    added = True
                    break
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    valid = True
                    for b in B:
                        if (sums_B >> (x + b)) & 1:
                            valid = False
                            break
                    if valid and ((sums_B >> (2 * x)) & 1):
                        valid = False
                    if not valid:
                        continue
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (((diffs_A >> d) & 1) or ((diffs_B >> d) & 1)):
                            valid = False
                            break
                    if not valid:
                        continue
                    B.append(x)
                    for b in B:
                        sums_B |= (1 << (x + b))
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B |= (1 << d)
                    added = True
                    break
            
        return A, B

    def local_search_polish(A, B, max_iters=200):
        """Try to add more elements via local search / polish."""
        A = list(A)
        B = list(B)
        if not A or not B:
            return A, B
        
        # Rebuild bitsets from current A, B
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A |= (1 << (A[i] + A[j]))
                if i != j:
                    diffs_A |= (1 << abs(A[i] - A[j]))
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B |= (1 << (B[i] + B[j]))
                if i != j:
                    diffs_B |= (1 << abs(B[i] - B[j]))
        
        in_A = set(A)
        in_B = set(B)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in in_A and x not in in_B]
            if not candidates:
                break
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                x = random.choice(candidates)
                
                if len(A) < MAX_EACH:
                    valid = True
                    for a in A:
                        if (sums_A >> (x + a)) & 1:
                            valid = False
                            break
                    if valid and ((sums_A >> (2 * x)) & 1):
                        valid = False
                    if valid:
                        for a in A:
                            d = abs(x - a)
                            if d > 0 and (((diffs_A >> d) & 1) or ((diffs_B >> d) & 1)):
                                valid = False
                                break
                    if valid:
                        A.append(x)
                        in_A.add(x)
                        for a in A:
                            sums_A |= (1 << (x + a))
                        for a in A:
                            d = abs(x - a)
                            if d > 0:
                                diffs_A |= (1 << d)
                        improved = True
                        break
                
                if len(B) < MAX_EACH:
                    valid = True
                    for b in B:
                        if (sums_B >> (x + b)) & 1:
                            valid = False
                            break
                    if valid and ((sums_B >> (2 * x)) & 1):
                        valid = False
                    if valid:
                        for b in B:
                            d = abs(x - b)
                            if d > 0 and (((diffs_A >> d) & 1) or ((diffs_B >> d) & 1)):
                                valid = False
                                break
                    if valid:
                        B.append(x)
                        in_B.add(x)
                        for b in B:
                            sums_B |= (1 << (x + b))
                        for b in B:
                            d = abs(x - b)
                            if d > 0:
                                diffs_B |= (1 << d)
                        improved = True
                        break
            
            if not improved:
                break
        
        return A, B

    # Run multiple restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        A, B = greedy_build(i)
        
        # Polish
        A2, B2 = local_search_polish(A, B)
        
        # Verify
        if is_valid(A2, B2):
            score = len(A2) + len(B2)
            if score > best_score:
                best_score = score
                best_A = A2
                best_B = B2
                
    # Fallback
    if not best_A:
        best_A = [1]
        best_B = [3]
        
    return [best_A, best_B]
