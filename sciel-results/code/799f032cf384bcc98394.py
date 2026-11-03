import math
import time

def _score_of(U):
    if not U:
        return 0.0
    u = sorted(set(U))
    if 0 not in u:
        u.append(0)
    n = len(u)
    mx = u[-1]
    if mx <= 0:
        return 0.0
    diffs = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
    sums = set()
    for a in u:
        for b in u:
            sums.add(a + b)
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = len(diffs) / float(len(sums))
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build(b, m):
    U = [0]
    v = 1.0
    for k in range(1, m + 1):
        v *= b
        U.append(int(v))
    return U

def construct():
    deadline = time.time() + 110.0
    best_score = 0.0
    best_set = [0, 1]
    
    # Coarse Grid Search
    # m: number of terms (excluding 0? no, _build adds 0, then m terms)
    # If m=1, U=[0, 1].
    # Let's search m from 2 to 40.
    # b from 1.01 to 3.0 step 0.02
    
    for m in range(2, 41):
        if time.time() > deadline:
            break
        b = 1.01
        while b <= 3.0:
            if time.time() > deadline:
                break
            U = _build(b, m)
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_set = U
            b += 0.02
            
    # Fine Grid Search around best? 
    # Or just refine the whole range with smaller step?
    # Given time, let's do a second pass with smaller step if time permits.
    
    if time.time() < deadline - 20.0:
        # Refine
        for m in range(2, 41):
            if time.time() > deadline:
                break
            b = 1.01
            while b <= 3.0:
                if time.time() > deadline:
                    break
                U = _build(b, m)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                b += 0.005
                
    return best_set
