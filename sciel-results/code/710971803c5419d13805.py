import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit
    
    def is_valid(S):
        # S is a list of distinct integers
        n = len(S)
        if n < 2:
            return True
        S_set = set(S)
        for a in S:
            Pos = []
            Neg = []
            for t in S_set:
                if t == a:
                    continue
                d = t - a
                if d > 0:
                    Pos.append(d)
                elif d < 0:
                    Neg.append(-d)
            
            # Compute subset sums of Pos
            sums_pos = 1
            for d in Pos:
                sums_pos |= sums_pos << d
            
            # Compute subset sums of Neg
            sums_neg = 1
            for d in Neg:
                sums_neg |= sums_neg << d
            
            # Check intersection
            # If any common sum > 0 exists, then a subset of Pos and a subset of Neg
            # sum to the same value, so their difference is 0.
            # The total number of elements is |P| + |N|.
            # Since sums > 0, P and N are non-empty.
            # So |P| >= 1 and |N| >= 1, total size >= 2.
            # This satisfies the "at least two other elements" condition for invalidity.
            intersection = sums_pos & sums_neg
            if intersection & ~1:
                return False
        return True

    best_set = []
    
    # Try multiple random orders
    candidates = list(range(1, 513))
    
    # First, try a deterministic greedy (smallest first) to get a baseline
    S = []
    for x in candidates:
        if time.time() > deadline:
            break
        S_new = S + [x]
        if is_valid(S_new):
            S = S_new
    best_set = S
    
    # Random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        S = []
        for x in candidates:
            if time.time() > deadline:
                break
            S_new = S + [x]
            if is_valid(S_new):
                S = S_new
        if len(S) > len(best_set):
            best_set = S
    
    return best_set
