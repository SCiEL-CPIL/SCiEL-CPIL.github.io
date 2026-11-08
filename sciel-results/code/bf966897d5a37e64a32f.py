import itertools
import math
import time

def _score_set(U):
    """Compute the score for a set U of nonnegative integers."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))  # ensure uniqueness, keep 0
    U.sort()
    n = len(U)
    if n < 1:
        return 0.0
    
    # Compute sumset |U+U|
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    P = len(sumset)
    
    # Compute difference set |U-U|
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    D = len(diffset)
    
    max_U = max(U)
    if max_U == 0:
        return 0.0
    
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max_U + 1)
    # We want to maximize this. Note that D >= P always (since 0 in U-U means symmetric, and |U-U| >= |U+U| for sets containing 0? Not necessarily, but usually D >= P for structured sets)
    # Actually for 0 in U, |U-U| is at least |U+U|? No. Example: U={0,1}. U+U={0,1,2}, |P|=3. U-U={-1,0,1}, |D|=3. Equal.
    # U={0,1,3}. U+U={0,1,2,3,4,6}, |P|=6. U-U={-3,-2,-1,0,1,2,3}, |D|=7. D>P.
    
    ratio = D / P
    if ratio <= 1:
        return 0.0  # or 1? The formula is 1 + log(ratio)/log(2*max_U+1). If ratio<1, log is negative, score < 1. We want to maximize.
    
    denom = math.log(2 * max_U + 1)
    if denom == 0:
        return 0.0
    
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Strategy: 
    # 1. Exhaustive search for small sizes (k=2 to 6) with bounded max element.
    # 2. Local search / simulated annealing for larger sizes or larger ranges.
    
    # Phase 1: Exhaustive search for small sets
    # Try sizes k from 2 to 7
    # Max element range: up to 30 for small k, larger for bigger k?
    # Actually, optimal sets for this problem tend to be "Mersenne-like" or similar.
    # Let's try exhaustive for k=2,3,4,5 with max <= 20.
    # k=6 with max <= 15.
    
    # Helper to get combinations
    def search_exhaustive(k, max_val):
        nonlocal best_score, best_set
        # Elements from 1 to max_val, choose k-1 (since 0 is fixed)
        # Total combos: C(max_val, k-1)
        for combo in itertools.combinations(range(1, max_val + 1), k - 1):
            U = [0] + list(combo)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U
            
            if time.time() > deadline:
                return True  # timeout
        
        return False
    
    # Search parameters
    search_configs = [
        (2, 20),
        (3, 20),
        (4, 20),
        (5, 20),
        (6, 15),
        (7, 12),
    ]
    
    for k, max_val in search_configs:
        if search_exhaustive(k, max_val):
            break
    
    # Phase 2: Local search / Hill climbing
    # Start from best_set found so far, and try adding/removing elements or shifting.
    # We can also try to grow the set by adding elements that improve the score.
    # Or try to optimize the "tail" element.
    
    # Let's do a greedy addition: try adding each possible new element x in range(best_max+1, 100)
    # and see if it improves the score. If so, add it and repeat.
    
    def greedy_grow(U):
        nonlocal best_score, best_set
        U = list(U)
        improved = True
        while improved:
            improved = False
            current_score = _score_set(U)
            max_U = max(U)
            # Try adding x in range(1, max_U + 10) not in U
            # Also try x up to 100 to allow larger jumps
            for x in range(1, 101):
                if x not in U:
                    new_U = U + [x]
                    score = _score_set(new_U)
                    if score > current_score + 1e-9:
                        U = new_U
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = U
                        if time.time() > deadline:
                            return
            # Also try removing an element? No, we're growing.
            # But maybe we should also try to replace elements?
        
        # After growing, try local moves: swap elements
        # Or just return

    greedy_grow(best_set)
    
    # Phase 3: Simulated Annealing / Random restarts
    # If we have time, do some random perturbations of the best set.
    if time.time() < deadline - 10:
        # Try to optimize by adjusting the largest element
        # Or try a few random sets
        pass
    
    return best_set
