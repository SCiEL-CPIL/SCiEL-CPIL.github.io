import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Initial set: {0}
    # We must ensure 0 is in the set.
    # If we start with {0}, max_U=0, log(1)=0, score is undefined or 0?
    # Score formula: 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
    # If U={0}, |U-U|=1, |U+U|=1, ratio=1, log(1)=0.
    # If max U = 0, log(1)=0. 0/0 is undefined.
    # The task says "containing 0".
    # Let's start with {0, 1} to have a valid score.
    # U={0,1}: 
    # U+U = {0, 1, 2}, size 3
    # U-U = {-1, 0, 1}, size 3
    # max U = 1. log(3) in denominator.
    # Score = 1 + log(1)/log(3) = 1.
    
    # Let's try to build a set that maximizes the ratio |U-U|/|U+U|.
    # |U-U| is at most n(n-1)+1.
    # |U+U| is at least n(n+1)/2.
    # For large n, ratio is ~ 2n / (n/2) = 4? No.
    # Max |U-U| ~ n^2. Min |U+U| ~ n^2/2. Ratio ~ 2.
    # log(2) / log(2 max U + 1).
    # To maximize this, we want max U to be small relative to n.
    # This suggests a dense set, but dense sets have small |U-U|/|U+U|?
    # For arithmetic progression {0, 1, ..., n-1}:
    # |U+U| = 2n-1.
    # |U-U| = 2n-1.
    # Ratio = 1. Score = 1.
    # We want ratio > 1.
    # Sets with large difference set relative to sum set are "asymmetric" or "sparse" in a specific way.
    # Actually, |U-U| is always >= |U+U| if 0 in U? No.
    # For U={0, 2, 4}: U+U={0,2,4,6,8} (5), U-U={-4,-2,0,2,4} (5). Ratio 1.
    # For U={0, 1, 3}: U+U={0,1,2,3,4,6} (6), U-U={-3,-2,-1,0,1,2,3} (7). Ratio 7/6 > 1.
    # Score = 1 + log(7/6)/log(2*3+1) = 1 + 0.154/1.79 = 1.08.
    
    # Greedy Strategy:
    # Start with U = {0, 1}.
    # Repeatedly find x >= 0 not in U that maximizes score(U ∪ {x}).
    # To keep max U low, we should prefer small x.
    # However, adding a large x increases the denominator log(2 max U + 1) significantly.
    # So we should likely stick to small integers.
    
    # Let's search for the best set of size k for k=2..15, using greedy addition.
    # We will also try different starting seeds or restarts if time permits.
    
    best_score = -1.0
    best_set = [0, 1]
    
    def get_score(U):
        if not U:
            return -1.0
        U_set = set(U)
        if 0 not in U_set:
            return -1.0
        max_U = max(U_set)
        if max_U == 0:
            return 0.0 # Avoid division by zero or log(0)
        
        sumset = set()
        diffset = set()
        # Optimization: Use sets for O(1) add
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return -1.0
            
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return -1.0
            
        denom = math.log(2 * max_U + 1)
        if denom == 0:
            return -1.0
            
        return 1 + math.log(ratio) / denom

    # Greedy Construction
    # We'll try to build sets of increasing size.
    # To avoid getting stuck in a local minimum of the greedy choice,
    # we can try a few different "starting" pairs or small sets.
    
    # Candidate pool: We'll consider adding x from 0 up to some limit L.
    # Since max U drives the denominator, L shouldn't be too large.
    # Let's say L=50.
    
    max_candidate = 60
    
    # Try starting with {0, 1}
    U = {0, 1}
    current_score = get_score(U)
    if current_score > best_score:
        best_score = current_score
        best_set = list(U)
    
    # Iteratively add the best x
    for step in range(20): # Try to add up to 20 more elements
        if time.time() > deadline:
            break
            
        best_x = -1
        best_next_score = current_score
        best_next_set = None
        
        # We should consider x not in U.
        # Also, x should be >= 0.
        # To keep max U controlled, we check candidates in increasing order of x?
        # Or just scan the range?
        # Scanning 0..60 is cheap.
        
        for x in range(max_candidate + 1):
            if x in U:
                continue
            
            new_U = U | {x}
            s = get_score(new_U)
            
            if s > best_next_score:
                best_next_score = s
                best_x = x
                best_next_set = new_U
        
        if best_x == -1 or best_next_set is None:
            break
            
        # Check if improvement is significant enough?
        # Or just accept the best.
        if best_next_score > current_score:
            current_score = best_next_score
            U = best_next_set
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            # No improvement found in this step. 
            # Maybe try a different starting point?
            # For now, break the loop for this seed.
            break

    # Try a second seed: {0, 2}
    U = {0, 2}
    current_score = get_score(U)
    if current_score > best_score:
        best_score = current_score
        best_set = list(U)
        
    for step in range(20):
        if time.time() > deadline:
            break
            
        best_x = -1
        best_next_score = current_score
        best_next_set = None
        
        for x in range(max_candidate + 1):
            if x in U:
                continue
            
            new_U = U | {x}
            s = get_score(new_U)
            
            if s > best_next_score:
                best_next_score = s
                best_x = x
                best_next_set = new_U
        
        if best_x == -1 or best_next_set is None:
            break
            
        if best_next_score > current_score:
            current_score = best_next_score
            U = best_next_set
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            break

    # Try a third seed: {0, 1, 3} (known good ratio)
    U = {0, 1, 3}
    current_score = get_score(U)
    if current_score > best_score:
        best_score = current_score
        best_set = list(U)
        
    for step in range(20):
        if time.time() > deadline:
            break
            
        best_x = -1
        best_next_score = current_score
        best_next_set = None
        
        for x in range(max_candidate + 1):
            if x in U:
                continue
            
            new_U = U | {x}
            s = get_score(new_U)
            
            if s > best_next_score:
                best_next_score = s
                best_x = x
                best_next_set = new_U
        
        if best_x == -1 or best_next_set is None:
            break
            
        if best_next_score > current_score:
            current_score = best_next_score
            U = best_next_set
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            break

    # Local Search on the best set found so far
    # Try swapping elements or removing elements?
    # Removing elements might reduce max U significantly, improving denominator.
    # Let's try removing the largest element if it doesn't hurt the ratio too much.
    
    if best_set and len(best_set) > 2:
        U = set(best_set)
        current_score = best_score
        
        # Try removing elements
        # Sort by value, try removing the largest
        sorted_U = sorted(U)
        for x_to_remove in reversed(sorted_U):
            if x_to_remove == 0:
                continue
            new_U = U - {x_to_remove}
            if not new_U:
                continue
            s = get_score(new_U)
            if s > current_score:
                current_score = s
                U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)
                # Continue removing?
                # Re-sort
                sorted_U = sorted(U)
            else:
                # If removing the largest didn't help, maybe removing a smaller one?
                # But removing smaller ones keeps max U high.
                # So only removing the largest is a good heuristic for reducing max U.
                break

    # Ensure 0 is in the final set
    if 0 not in best_set:
        best_set.append(0)
        
    return best_set

# EVOLVE-BLOCK-END
