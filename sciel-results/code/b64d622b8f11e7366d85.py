import math
import time
import itertools

def _score(U):
    """
    Compute the score for a finite set U of nonnegative integers containing 0.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Efficiently compute sumset and difference set
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
            
    n_plus = len(plus)
    n_minus = len(minus)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _exhaustive_search(max_size, max_val, deadline):
    """
    Exhaustively search for sets of size up to max_size with elements up to max_val.
    We always include 0. We choose the rest from {1, ..., max_val}.
    """
    best_score = 0.0
    best_U = [0]
    
    # To speed up, we can iterate over combinations of {1..max_val}
    # We need to pick k-1 elements from max_val options.
    # Total combos = sum_{k=2}^{max_size} C(max_val, k-1)
    
    # We'll iterate by size
    for size in range(2, max_size + 1):
        if time.time() > deadline:
            break
            
        # Choose size-1 elements from 1..max_val
        # Using itertools.combinations
        # This might be too slow if max_val is large and size is large.
        # C(20, 5) = 15504. C(20, 6) = 38760. C(30, 5) = 142506.
        # Let's cap max_val dynamically or just use a reasonable bound.
        
        for combo in itertools.combinations(range(1, max_val + 1), size - 1):
            if time.time() > deadline:
                break
                
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U
                
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s buffer for safety
    
    best_U = [0, 1, 2, 4]
    best_score = _score(best_U)
    
    # Strategy: Exhaustive search for small sets
    # The score function is 1 + log(ratio) / log(2mx+1).
    # To maximize this, we want high ratio (sparse sumset) and low mx.
    # Small sets (size 3-6) with small elements are the prime candidates.
    
    # Let's try a few configurations of exhaustive search
    # Config 1: Small max_val, moderate size
    # C(15, 5) = 3003. Very fast.
    # C(20, 5) = 15504. Fast.
    # C(25, 5) = 53130. Still fast.
    # C(30, 5) = 142506. ~1-2 seconds.
    # C(30, 6) = 593775. ~5-10 seconds.
    
    # We will expand the search space gradually until time runs out.
    
    search_configs = [
        (10, 6),  # max_val=10, max_size=6
        (15, 6),  # max_val=15, max_size=6
        (20, 6),  # max_val=20, max_size=6
        (20, 7),  # max_val=20, max_size=7
        (25, 6),  # max_val=25, max_size=6
        (25, 7),  # max_val=25, max_size=7
        (30, 6),  # max_val=30, max_size=6
        (30, 7),  # max_val=30, max_size=7
        (35, 6),  # max_val=35, max_size=6
        (35, 7),  # max_val=35, max_size=7
        (40, 6),  # max_val=40, max_size=6
        (40, 7),  # max_val=40, max_size=7
    ]
    
    for max_val, max_size in search_configs:
        if time.time() > deadline:
            break
            
        # If we've already found a very high score, we might skip smaller searches?
        # No, smaller searches are fast.
        
        # Estimate time:
        # C(max_val, max_size-1) * cost_per_eval
        # cost_per_eval is roughly O(n^2) where n is size. n<=7. So very fast.
        # 100k evals * 49 ops = 5M ops. < 0.1s.
        
        U, score = _exhaustive_search(max_size, max_val, deadline)
        
        if score > best_score:
            best_score = score
            best_U = U
            
    # If we still have time, try larger max_val with smaller sizes
    if time.time() < deadline - 5.0:
        for max_val in range(40, 60, 5):
            if time.time() > deadline:
                break
            U, score = _exhaustive_search(5, max_val, deadline) # size up to 5
            if score > best_score:
                best_score = score
                best_U = U

    return best_U

if __name__ == "__main__":
    # Quick self-test
    u = construct()
    print(f"Best U: {u}")
    print(f"Score: {_score(u)}")
