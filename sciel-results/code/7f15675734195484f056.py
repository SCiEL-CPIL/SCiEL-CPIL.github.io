"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import math

# EVOLVE-BLOCK-START

def _sumset_size(U):
    if not U:
        return 0
    Us = set(U)
    s = set()
    for a in Us:
        for b in Us:
            s.add(a + b)
    return len(s)

def _diffset_size(U):
    if not U:
        return 0
    Us = set(U)
    d = set()
    for a in Us:
        for b in Us:
            d.add(a - b)
    return len(d)

def _score_set(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    s_sz = _sumset_size(U)
    d_sz = _diffset_size(U)
    if s_sz == 0:
        return 0.0
    ratio = d_sz / s_sz
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    val = math.log(ratio) / denom
    if val < 0:
        val = 0.0
    return 1.0 + val

def construct():
    start = time.time()
    deadline = 135  # safety margin
    
    best_set = [0, 1, 4, 6]
    best_score = _score_set(best_set)
    
    # Phase 1: Exhaustive search over small sets with small max
    # We know {0,1,4,6} scores ~1.102. Let's search all sets of size 3-5 with max up to 20.
    # To keep it fast, we restrict to sets that include 0 and are sorted.
    
    # Pre-compute some hard-coded good candidates first to have a high baseline
    candidates = [
        [0, 1, 4, 6],       # Score ~1.102
        [0, 1, 3],          # Score ~1.096 (actually lower, let's check: S={0,1,2,3,4,6}, D={-3,-2,-1,0,1,2,3}. |S|=6, |D|=7. M=3. log(7/6)/log(7) = 0.1438/1.9459 = 0.0739. Score 1.0739)
        [0, 1, 5, 7],       # Score ~1.096 (S=10, D=13, M=7. log(1.3)/log(15) = 0.2624/2.708 = 0.0969. Score 1.0969)
        [0, 1, 4, 7],       # Let's check: S={0,1,2,4,5,8,10,11}, wait. 0,1,4,7. S: 0,1,2,4,5,8,10,11? No.
                            # 0+0=0, 0+1=1, 0+4=4, 0+7=7
                            # 1+1=2, 1+4=5, 1+7=8
                            # 4+4=8, 4+7=11
                            # 7+7=14
                            # S={0,1,2,4,5,7,8,11,14}. Size 9.
                            # D: 1,4,7,3,6,2. Pos: 1,2,3,4,6,7. D size 13.
                            # Ratio 13/9 = 1.44. M=7. log(1.44)/log(15) = 0.366/2.708 = 0.135. Score 1.135.
        [0, 1, 4, 7]
    ]
    
    for c in candidates:
        s = _score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
    
    # Search for better sets
    # Size 4 sets, max up to 15
    for max_val in range(5, 16):
        if time.time() > deadline:
            break
        # Generate combinations of 3 numbers from 1..max_val
        # This is C(15,3) = 455 for max_val=15. Very fast.
        for comb in itertools.combinations(range(1, max_val + 1), 3):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            # Optional pruning: check if max element is actually max_val? No, just search all.
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Size 5 sets, max up to 12
    # C(11,4) = 330.
    for max_val in range(8, 13):
        if time.time() > deadline:
            break
        for comb in itertools.combinations(range(1, max_val + 1), 4):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    # Size 3 sets, max up to 30
    for max_val in range(4, 31):
        if time.time() > deadline:
            break
        for comb in itertools.combinations(range(1, max_val + 1), 2):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    # Size 6 sets, max up to 10
    # C(9,5) = 126.
    for max_val in range(9, 11):
        if time.time() > deadline:
            break
        for comb in itertools.combinations(range(1, max_val + 1), 5):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set

# EVOLVE-BLOCK-END
