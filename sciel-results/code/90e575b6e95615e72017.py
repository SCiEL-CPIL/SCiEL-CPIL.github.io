import time
import random
import numpy as np

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s wall clock limit)
    
    best_A = None
    best_B = None
    best_score = 0
    
    def check_and_update(A, B):
        nonlocal best_A, best_B, best_score
        if not A or not B:
            return
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return
        
        # Verify Sidon property for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return
                sums_A.add(s)
        
        # Verify Sidon property for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return
                sums_B.add(s)
        
        # Compute nonzero differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    diffs_B.add(d)
        
        # Check disjointness of difference sets
        if diffs_A & diffs_B:
            return
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    def greedy_with_lookahead(seed):
        """Greedy construction with lookahead and multiple strategies."""
        rng = random.Random(seed)
        
        # Strategy 1: Build A first with spacing, then B with complementary differences
        # Strategy 2: Interleave A and B construction
        # Strategy 3: Build one set, then find best complement for the other
        
        strategy = seed % 3
        
        if strategy == 0:
            return _build_A_then_B(rng)
        elif strategy == 1:
            return _interleave_build(rng)
        else:
            return _build_with_complement(rng)
    
    def _build_A_then_B(rng):
        """Build A first using sparse spacing, then find B with non-conflicting differences."""
        A = []
        sums_A = [False] * (2 * N + 1)
        diffs_A = [False] * (N + 1)
        
        # Choose spacing for A to minimize difference conflicts
        # Start with a base spacing and perturb it
        base_spacing = rng.choice([3, 4, 5, 6])
        spacing = base_spacing
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if len(A) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            
            # Check if x can be added to A
            valid_sum = True
            for a in A:
                if sums_A[x + a]:
                    valid_sum = False
                    break
            if valid_sum and sums_A[2 * x]:
                valid_sum = False
            
            if not valid_sum:
                continue
            
            # Check if adding x creates differences that would block B too much
            # (we'll just add it for now, B construction will handle conflicts)
            A.append(x)
            for a in A:
                sums_A[x + a] = True
            sums_A[2 * x] = True
            for a in A:
                d = abs(x - a)
                if d > 0:
                    diffs_A[d] = True
        
        # Now build B avoiding differences in diffs_A
        B = []
        sums_B = [False] * (2 * N + 1)
        diffs_B = [False] * (N + 1)
        
        candidates_B = list(range(1, N + 1))
        rng.shuffle(candidates_B)
        
        for x in candidates_B:
            if len(B) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            
            # Check Sidon property for B
            valid_sum = True
            for b in B:
                if sums_B[x + b]:
                    valid_sum = False
                    break
            if valid_sum and sums_B[2 * x]:
                valid_sum = False
            
            if not valid_sum:
                continue
            
            # Check difference disjointness with A
            valid_diff = True
            for b in B:
                d = abs(x - b)
                if d > 0 and (diffs_A[d] or diffs_B[d]):
                    valid_diff = False
                    break
            
            if not valid_diff:
                continue
            
            B.append(x)
            for b in B:
                sums_B[x + b] = True
            sums_B[2 * x] = True
            for b in B:
                d = abs(x - b)
                if d > 0:
                    diffs_B[d] = True
        
        return A, B
    
    def _interleave_build(rng):
        """Interleave building A and B, balancing sizes."""
        A = []
        B = []
        sums_A = [False] * (2 * N + 1)
        sums_B = [False] * (2 * N + 1)
        diffs_A = [False] * (N + 1)
        diffs_B = [False] * (N + 1)
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Decide which set to try: prefer the smaller one, with some randomness
            if len(A) < len(B) - 1:
                order = [0]
            elif len(B) < len(A) - 1:
                order = [1]
            elif len(A) == len(B):
                order = [0, 1] if rng.random() < 0.5 else [1, 0]
            else:
                order = [0, 1] if len(A) <= len(B) else [1, 0]
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    valid_sum = True
                    for a in A:
                        if sums_A[x + a]:
                            valid_sum = False
                            break
                    if valid_sum and sums_A[2 * x]:
                        valid_sum = False
                    if not valid_sum:
                        continue
                    valid_diff = True
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (diffs_A[d] or diffs_B[d]):
                            valid_diff = False
                            break
                    if not valid_diff:
                        continue
                    A.append(x)
                    for a in A:
                        sums_A[x + a] = True
                    sums_A[2 * x] = True
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A[d] = True
                    added = True
                    break
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    valid_sum = True
                    for b in B:
                        if sums_B[x + b]:
                            valid_sum = False
                            break
                    if valid_sum and sums_B[2 * x]:
                        valid_sum = False
                    if not valid_sum:
                        continue
                    valid_diff = True
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (diffs_A[d] or diffs_B[d]):
                            valid_diff = False
                            break
                    if not valid_diff:
                        continue
                    B.append(x)
                    for b in B:
                        sums_B[x + b] = True
                    sums_B[2 * x] = True
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B[d] = True
                    added = True
                    break
        
        return A, B
    
    def _build_with_complement(rng):
        """Build a smaller set first, then greedily find the best complement."""
        # First, build a moderate-sized set (say 20-30 elements)
        target_size = rng.randint(20, 35)
        
        A = []
        sums_A = [False] * (2 * N + 1)
        diffs_A = [False] * (N + 1)
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if len(A) >= target_size:
                break
            if time.time() > deadline:
                break
            
            valid_sum = True
            for a in A:
                if sums_A[x + a]:
                    valid_sum = False
                    break
            if valid_sum and sums_A[2 * x]:
                valid_sum = False
            if not valid_sum:
                continue
            
            A.append(x)
            for a in A:
                sums_A[x + a] = True
            sums_A[2 * x] = True
            for a in A:
                d = abs(x - a)
                if d > 0:
                    diffs_A[d] = True
        
        # Now build B to maximize size given A's difference constraints
        B = []
        sums_B = [False] * (2 * N + 1)
        diffs_B = [False] * (N + 1)
        
        candidates_B = list(range(1, N + 1))
        rng.shuffle(candidates_B)
        
        for x in candidates_B:
            if len(B) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            
            valid_sum = True
            for b in B:
                if sums_B[x + b]:
                    valid_sum = False
                    break
            if valid_sum and sums_B[2 * x]:
                valid_sum = False
            if not valid_sum:
                continue
            
            valid_diff = True
            for b in B:
                d = abs(x - b)
                if d > 0 and (diffs_A[d] or diffs_B[d]):
                    valid_diff = False
                    break
            if not valid_diff:
                continue
            
            B.append(x)
            for b in B:
                sums_B[x + b] = True
            sums_B[2 * x] = True
            for b in B:
                d = abs(x - b)
                if d > 0:
                    diffs_B[d] = True
        
        # Try to also add more to A now
        candidates_A2 = list(range(1, N + 1))
        rng.shuffle(candidates_A2)
        
        for x in candidates_A2:
            if len(A) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            if x in A:
                continue
            
            valid_sum = True
            for a in A:
                if sums_A[x + a]:
                    valid_sum = False
                    break
            if valid_sum and sums_A[2 * x]:
                valid_sum = False
            if not valid_sum:
                continue
            
            valid_diff = True
            for a in A:
                d = abs(x - a)
                if d > 0 and (diffs_A[d] or diffs_B[d]):
                    valid_diff = False
                    break
            if not valid_diff:
                continue
            
            A.append(x)
            for a in A:
                sums_A[x + a] = True
            sums_A[2 * x] = True
            for a in A:
                d = abs(x - a)
                if d > 0:
                    diffs_A[d] = True
        
        return A, B
    
    # Run multiple restarts with different strategies
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        A, B = greedy_with_lookahead(i)
        if A and B:
            check_and_update(A, B)
    
    # Fallback
    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [4]
    
    return [best_A, best_B]
