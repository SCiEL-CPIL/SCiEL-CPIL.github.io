import time
import random
import math

N = 32
MAX_VAL = 512

def _score_mask(mask):
    """
    Fast score using Python big integers as bitsets.
    mask: int where bit i is 1 if (i+1) is in the set.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    count = 0
    # d can range from 1 to 255 (since 1 + 2*255 = 511 <= 512)
    # For a given d, we need to check if there exists a such that
    # bit a, bit a+d, bit a+2d are all set.
    # This is equivalent to (mask & (mask >> d) & (mask >> 2*d)) != 0
    # We iterate d from 1 to 255.
    # Optimization: Python int bit ops are very fast.
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _mask_to_list(mask):
    """Convert mask to list of integers (1..512)."""
    res = []
    m = mask
    i = 0
    while m:
        if m & 1:
            res.append(i + 1)
        m >>= 1
        i += 1
    return res

def _list_to_mask(S):
    """Convert list of integers (1..512) to mask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def _get_candidates(S, mask):
    """
    Generate a list of candidate values y (1..512, not in S)
    that would complete a 3-term AP with two elements from S.
    """
    candidates = set()
    # S is a list of size 32.
    # Iterate over all pairs (u, v)
    for i in range(len(S)):
        u = S[i]
        for j in range(i + 1, len(S)):
            v = S[j]
            # AP: u, v, w => w = 2v - u
            w1 = 2 * v - u
            if 1 <= w1 <= MAX_VAL and not ((mask >> (w1 - 1)) & 1):
                candidates.add(w1)
            
            # AP: w, u, v => w = 2u - v
            w2 = 2 * u - v
            if 1 <= w2 <= MAX_VAL and not ((mask >> (w2 - 1)) & 1):
                candidates.add(w2)
            
            # AP: v, u, w => w = 2u - v (Same as above, but order doesn't matter for set membership)
            # AP: u, w, v => w = (u+v)/2. If (u+v) is even, w = (u+v)//2.
            s = u + v
            if s % 2 == 0:
                w3 = s // 2
                if 1 <= w3 <= MAX_VAL and not ((mask >> (w3 - 1)) & 1):
                    candidates.add(w3)
                
    return list(candidates)

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0
    
    best_score = 0
    best_mask = 0
    
    # Strategy:
    # 1. Run multiple Simulated Annealing chains.
    # 2. Use heuristic move generation to focus on high-value swaps.
    
    num_chains = 20  # We can afford more chains if each is fast enough
    
    for chain_idx in range(num_chains):
        if time.time() > deadline:
            break
        
        # Initialize a random set of 32 distinct integers
        # Sometimes start with a structured set to help convergence?
        # Let's stick to random for diversity, but maybe bias towards smaller numbers initially?
        # No, uniform random is safer.
        S = random.sample(range(1, MAX_VAL + 1), N)
        mask = _list_to_mask(S)
        current_score = _score_mask(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
        
        # Simulated Annealing parameters
        T = 2.0
        T_min = 0.05
        cooling = 0.9995
        step = 0
        
        while time.time() < deadline:
            if T < T_min:
                break
            
            step += 1
            
            # Generate candidate moves
            # To keep it fast, we don't regenerate candidates every single step if T is high?
            # Actually, generating candidates is O(N^2) ~ 1000 ops. Scoring is ~255 bit ops.
            # Scoring is likely faster than candidate generation?
            # Let's check: 255 big-int ANDs vs 1000 integer arithmetic + set lookups.
            # Big-int ops are C-speed. 1000 Python loop iterations is slower.
            # So, maybe generate candidates less frequently? Or just use random y if candidates are too expensive?
            # Let's try generating candidates every 10 steps or so, or just use a hybrid:
            # 50% random y, 50% heuristic y.
            
            # Pick random element to remove
            idx_out = random.randint(0, N - 1)
            x = S[idx_out]
            
            # Decide on y
            if random.random() < 0.5:
                # Heuristic move
                candidates = _get_candidates(S, mask)
                if candidates:
                    y = random.choice(candidates)
                else:
                    y = random.randint(1, MAX_VAL)
                    if y in S:
                        y = random.randint(1, MAX_VAL) # Retry once, if still in S, skip move?
                        if y in S:
                            continue
            else:
                # Random move
                y = random.randint(1, MAX_VAL)
                if y in S:
                    # If y is in S, this isn't a swap, it's a no-op or invalid.
                    # We need y NOT in S.
                    # Resample
                    while y in S:
                        y = random.randint(1, MAX_VAL)
            
            if y == x:
                continue
            
            # Apply move
            # Remove x
            mask &= ~(1 << (x - 1))
            # Add y
            mask |= (1 << (y - 1))
            
            # Update S list
            S[idx_out] = y
            
            new_score = _score_mask(mask)
            delta = new_score - current_score
            
            # Acceptance probability
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
            else:
                # Revert
                mask &= ~(1 << (y - 1))
                mask |= (1 << (x - 1))
                S[idx_out] = x
            
            T *= cooling
            
            # Occasional check for time
            if step % 100 == 0 and time.time() > deadline:
                break

    return _mask_to_list(best_mask)
