import time
import random

N = 32
MAX_VAL = 512
DEADLINE_MARGIN = 3.0

def _score_mask(mask):
    """
    Score a bitset mask (Python int) representing a set of integers in 1..512.
    Returns the number of distinct d > 0 such that there exists a, a+d, a+2d in the set.
    """
    count = 0
    # Max possible d is 255 (if a=1, a+2d<=512 => d<=255)
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> 2*d)) != 0:
            count += 1
    return count

def _mask_to_list(mask):
    """Convert a bitset mask to a sorted list of integers."""
    return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # --- Phase 1: Greedy Construction ---
    # Build a good initial set using greedy selection with lookahead.
    
    current_mask = 0
    candidates = list(range(1, MAX_VAL + 1))
    current_set = []
    
    for step in range(N):
        if time.time() > deadline:
            break
            
        best_candidate = None
        best_new_score = -1
        
        # To speed up greedy, we can limit candidates to a subset if we have many,
        # but 512 is small. However, evaluating _score_mask 512 times per step is fast.
        # Let's optimize: if we have a lot of time, check all. Otherwise, sample?
        # 512 * 32 = 16384 scores. Each score is ~256 bit ops. ~4M ops. Very fast (<1s).
        
        # Optimization: Pre-compute potential contributions?
        # For a new element c, new APs are formed if:
        # 1. c is the end: c, c-d, c-2d in S
        # 2. c is the middle: c+d, c-d in S (d = c - (c-d) = (c+d)-c)
        # 3. c is the start: c, c+d, c+2d in S
        # But distinct d count makes incremental update tricky.
        # Stick with full re-score for correctness and simplicity, it's fast enough.
        
        for c in candidates:
            new_mask = current_mask | (1 << c)
            new_score = _score_mask(new_mask)
            if new_score > best_new_score:
                best_new_score = new_score
                best_candidate = c
                
        if best_candidate is None:
            best_candidate = random.choice(candidates)
            best_new_score = _score_mask(current_mask | (1 << best_candidate))
            
        current_mask |= (1 << best_candidate)
        current_set.append(best_candidate)
        candidates.remove(best_candidate)
        
    current_score = _score_mask(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # --- Phase 2: Local Search (Swaps) ---
    # Try to improve by swapping elements in and out of the set.
    # Use a "best improvement" or "first improvement" strategy?
    # Best improvement is safer but slower. First improvement can get stuck.
    # Let's do a mix: random swaps + hill climbing.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        set_elems = _mask_to_list(best_mask)
        not_in_set = [i for i in range(1, MAX_VAL + 1) if not ((best_mask >> i) & 1)]
        
        # Try all swaps in this pass? 32 * 480 = 15360 swaps.
        # Each swap scores 256 times. 15360 * 256 ~ 4M ops.
        # This is fast. We can do multiple passes.
        
        for x in set_elems:
            if time.time() > deadline:
                break
            for y in not_in_set:
                if time.time() > deadline:
                    break
                    
                # Swap x with y
                new_mask = (best_mask & ~(1 << x)) | (1 << y)
                new_score = _score_mask(new_mask)
                
                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask
                    set_elems = _mask_to_list(best_mask)
                    not_in_set = [i for i in range(1, MAX_VAL + 1) if not ((best_mask >> i) & 1)]
                    improved = True
                    # Restart loops to avoid using stale lists
                    # We break out of both loops effectively by the logic below
                    break
            if improved:
                break
                
    # --- Phase 3: Simulated Annealing (if time remains) ---
    # If we have more than 20 seconds left, do a SA run to escape local optima.
    if time.time() < deadline - 20:
        current_mask = best_mask
        current_score = best_score
        T = 5.0
        cooling = 0.999
        
        while time.time() < deadline:
            # Generate a random neighbor: swap a random element in with a random element out
            set_elems = _mask_to_list(current_mask)
            not_in_set = [i for i in range(1, MAX_VAL + 1) if not ((current_mask >> i) & 1)]
            
            if not set_elems or not not_in_set:
                break
                
            x = random.choice(set_elems)
            y = random.choice(not_in_set)
            
            new_mask = (current_mask & ~(1 << x)) | (1 << y)
            new_score = _score_mask(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
            T *= cooling
            if T < 0.1:
                break

    # --- Phase 4: Final Polish ---
    # One more pass of strict hill climbing to ensure we're at a local maximum
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            set_elems = _mask_to_list(best_mask)
            not_in_set = [i for i in range(1, MAX_VAL + 1) if not ((best_mask >> i) & 1)]
            
            for x in set_elems:
                for y in not_in_set:
                    new_mask = (best_mask & ~(1 << x)) | (1 << y)
                    new_score = _score_mask(new_mask)
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                        improved = True
                        break
                if improved:
                    break

    result = _mask_to_list(best_mask)
    
    # Safety check: ensure exactly 32 distinct integers in 1..512
    if len(result) != N or any(x < 1 or x > MAX_VAL for x in result) or len(set(result)) != N:
        # Fallback: return a valid set (e.g., first 32)
        # This should never happen with the logic above, but just in case
        return list(range(1, N+1))
        
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    mask = 0
    for r in res:
        mask |= (1 << r)
    print(f"Result: {res}")
    print(f"Score: {_score_mask(mask)}")
