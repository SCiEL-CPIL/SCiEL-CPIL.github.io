import time
import numpy as np

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_A = []
    best_B = []
    best_score = 0
    
    def run_greedy(prefer_a_first, balance, ascending=True):
        A = []
        B = []
        sums_A = np.zeros(513, dtype=bool)
        sums_B = np.zeros(513, dtype=bool)
        diffs_A = np.zeros(256, dtype=bool)
        diffs_B = np.zeros(256, dtype=bool)
        
        range_fn = range(1, N + 1) if ascending else range(N, 0, -1)
        
        for x in range_fn:
            if time.time() > deadline:
                break
                
            if balance:
                if len(A) > len(B):
                    try_A_first = False
                elif len(B) > len(A):
                    try_A_first = True
                else:
                    try_A_first = prefer_a_first
            else:
                try_A_first = prefer_a_first
            
            ordered_sides = [0, 1] if try_A_first else [1, 0]
            
            for side in ordered_sides:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    ok = True
                    for a in A:
                        s = x + a
                        if sums_A[s]:
                            ok = False
                            break
                    if ok and 2 * x <= 512:
                        if sums_A[2 * x]:
                            ok = False
                    if ok:
                        for a in A:
                            d = abs(x - a)
                            if diffs_A[d] or diffs_B[d]:
                                ok = False
                                break
                    
                    if ok:
                        A.append(x)
                        for a in A:
                            sums_A[x + a] = True
                        if 2 * x <= 512:
                            sums_A[2 * x] = True
                        for a in A:
                            d = abs(x - a)
                            diffs_A[d] = True
                        break
                        
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    ok = True
                    for b in B:
                        s = x + b
                        if sums_B[s]:
                            ok = False
                            break
                    if ok and 2 * x <= 512:
                        if sums_B[2 * x]:
                            ok = False
                    if ok:
                        for b in B:
                            d = abs(x - b)
                            if diffs_B[d] or diffs_A[d]:
                                ok = False
                                break
                    
                    if ok:
                        B.append(x)
                        for b in B:
                            sums_B[x + b] = True
                        if 2 * x <= 512:
                            sums_B[2 * x] = True
                        for b in B:
                            d = abs(x - b)
                            diffs_B[d] = True
                        break
        
        return A, B
    
    def local_search_polish(A, B, max_iters=50):
        """Try to add more elements to A and B via local search."""
        sums_A = np.zeros(513, dtype=bool)
        sums_B = np.zeros(513, dtype=bool)
        diffs_A = np.zeros(256, dtype=bool)
        diffs_B = np.zeros(256, dtype=bool)
        
        # Rebuild constraint structures from current A and B
        for a in A:
            for a2 in A:
                sums_A[a + a2] = True
            for a2 in A:
                d = abs(a - a2)
                if d > 0:
                    diffs_A[d] = True
        for b in B:
            for b2 in B:
                sums_B[b + b2] = True
            for b2 in B:
                d = abs(b - b2)
                if d > 0:
                    diffs_B[d] = True
        
        A_set = set(A)
        B_set = set(B)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try adding to A
            for x in range(1, N + 1):
                if x in A_set:
                    continue
                if len(A) >= MAX_EACH:
                    break
                ok = True
                for a in A:
                    s = x + a
                    if sums_A[s]:
                        ok = False
                        break
                if ok and sums_A[2 * x]:
                    ok = False
                if ok:
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (diffs_A[d] or diffs_B[d]):
                            ok = False
                            break
                if ok:
                    A.append(x)
                    A_set.add(x)
                    for a in A:
                        sums_A[x + a] = True
                    if 2 * x <= 512:
                        sums_A[2 * x] = True
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A[d] = True
                    improved = True
                    break
            
            # Try adding to B
            for x in range(1, N + 1):
                if x in B_set:
                    continue
                if len(B) >= MAX_EACH:
                    break
                ok = True
                for b in B:
                    s = x + b
                    if sums_B[s]:
                        ok = False
                        break
                if ok and sums_B[2 * x]:
                    ok = False
                if ok:
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (diffs_B[d] or diffs_A[d]):
                            ok = False
                            break
                if ok:
                    B.append(x)
                    B_set.add(x)
                    for b in B:
                        sums_B[x + b] = True
                    if 2 * x <= 512:
                        sums_B[2 * x] = True
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B[d] = True
                    improved = True
                    break
        
        return A, B
    
    # Run multiple variations
    variations = [
        (True, False, True),
        (False, False, True),
        (True, True, True),
        (False, True, True),
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
    ]
    
    for pref, bal, asc in variations:
        if time.time() > deadline:
            break
        
        A1, B1 = run_greedy(pref, bal, asc)
        A1, B1 = local_search_polish(A1, B1)
        score = len(A1) + len(B1)
        if score > best_score:
            best_score = score
            best_A, best_B = A1, B1

    # Fallback if something went wrong
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [3]
        
    return [best_A, best_B]
