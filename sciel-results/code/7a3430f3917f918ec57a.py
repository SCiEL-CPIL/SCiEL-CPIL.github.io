"""Optimized program: Bitset scoring + fast SA for 3-AP differences."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score_set_bitset(S):
    """
    Count distinct d > 0 such that a, a+d, a+2d all in S.
    Uses a 512-bit integer mask for O(D) bit operations.
    """
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    
    count = 0
    # d can be at most 255 since a+2d <= 512 => 2d <= 511 => d <= 255
    # Actually a>=1, so a+2d <= 512 => 2d <= 511 is not quite right.
    # Max d is when a=1, 1+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are in S
        # This is equivalent to checking if S & (S >> d) & (S >> 2*d) is non-zero
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _greedy_polish(S, time_limit=5.0):
    """
    Greedy local search: try swapping each element with every other candidate.
    """
    S = list(S)
    current_score = _score_set_bitset(S)
    s_set = set(S)
    
    start_time = time.time()
    
    # We will iterate until no improvement or time limit
    # To speed up, we can limit the number of full passes or just run until time limit
    # Since scoring is very fast, we can do many iterations.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            s_set.remove(old_val)
            
            # Try adding each candidate not in s_set
            # To be faster, we can just iterate 1..512
            best_cand = None
            best_score = current_score
            
            # Optimization: Only try candidates that might help?
            # For now, brute force 1..512 is fast enough with bitset scoring?
            # 32 * 512 * (cost of score)
            # Cost of score is ~256 bit ops. Very fast.
            
            for cand in range(1, 513):
                if cand in s_set:
                    continue
                s_set.add(cand)
                # Update S temporarily for scoring? 
                # _score_set_bitset takes a list/set. 
                # We can pass a list but it iterates. 
                # Better to maintain the mask?
                # Let's just use the function, it's fast.
                # But converting set to list is O(N).
                
                # Let's optimize _score_set_bitset to accept a mask directly?
                # No, let's just pass the set. It's small.
                
                # Actually, let's make a helper that scores a mask.
                # But here we are in the middle of a swap.
                
                # Let's refine: 
                # We can compute the new mask by removing old_val and adding cand.
                # mask = (mask & ~(1 << (old_val-1))) | (1 << (cand-1))
                
                # Let's refactor to use masks in the search loop.
                pass
            # If we didn't find an improvement in this i, restore old_val
            if best_cand is None:
                s_set.add(old_val)
            else:
                # This logic above was pseudo-code. Let's implement the mask version properly.
                pass
        
        # The loop above was a sketch. Let's implement the real one below.
        
    return S, current_score

def _greedy_polish_fast(S, time_limit=5.0):
    """
    Fast greedy local search using bit masks.
    """
    S = list(S)
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    
    current_score = _score_set_bitset(S)
    s_set = set(S)
    
    start_time = time.time()
    
    # Repeat passes until no improvement or time limit
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            s_set.remove(old_val)
            mask &= ~(1 << (old_val - 1))
            
            best_cand = None
            best_score = current_score
            
            # Try all candidates
            # We can vectorize this? No, bit ops are per d.
            # But 512 candidates * 256 d * 3 bit ops = 393k ops per i.
            # 32 i's = 12.5M ops per pass.
            # Python loop overhead is high. 
            # Let's optimize the inner loop.
            
            # Pre-compute valid candidates? No.
            
            # Just loop.
            for cand in range(1, 513):
                if cand in s_set:
                    continue
                
                new_mask = mask | (1 << (cand - 1))
                
                # Score new_mask
                score = 0
                # Unroll the d loop? 
                # for d in 1..255:
                #   if (new_mask & (new_mask>>d) & (new_mask>>2*d)): score++
                
                # This is still a Python loop of 255.
                # 512 * 255 = 130k iterations per i.
                # 32 * 130k = 4M iterations per pass.
                # Might be too slow in Python if not optimized.
                
                # Let's try to speed up the scoring.
                # We can use numpy?
                # Or just accept it's slower than C but faster than set lookups?
                # Set lookups were O(N*D) = 32*256 = 8k lookups.
                # Bit ops are 256 ops.
                # Bit ops should be faster.
                
                # Let's just use the bitset function but inline it for speed?
                # No, function call overhead.
                
                # Let's stick to the bitset function and hope it's fast enough.
                # Or reduce candidates?
                
                # Actually, let's just use the bitset function.
                # If it's too slow, we'll see.
                
                score = _score_set_bitset_from_mask(new_mask)
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand is not None:
                S[i] = best_cand
                s_set.add(best_cand)
                mask |= (1 << (best_cand - 1))
                current_score = best_score
                improved = True
            else:
                # Restore
                s_set.add(old_val)
                mask |= (1 << (old_val - 1))
        
        if not improved:
            break
    
    return S, current_score

def _score_set_bitset_from_mask(mask):
    """Score a mask directly."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _simulated_annealing(S_init, time_budget, rng):
    """SA with swap moves using bitset scoring."""
    S = list(S_init)
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    
    current_score = _score_set_bitset_from_mask(mask)
    best_S = list(S)
    best_score = current_score
    best_mask = mask
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.01:
            break
        
        i = rng.randint(0, 31)
        old_elem = S[i]
        new_elem = rng.randint(1, 512)
        
        if new_elem == old_elem:
            continue
        
        # Check if new_elem is already in S (excluding i)
        # We can check if (mask >> (new_elem-1)) & 1 is 1.
        # If it is, and new_elem != old_elem, then it's a duplicate.
        # If new_elem == old_elem, we handled that.
        # So if bit is set, it's a duplicate (since we are replacing old_elem, 
        # the bit for old_elem is still set in mask until we clear it).
        # Wait, if new_elem is in S at j != i, then bit is set.
        # We should clear old_elem first?
        # No, just check if new_elem is in S and new_elem != old_elem.
        
        if (mask >> (new_elem - 1)) & 1:
            # It's in the set.
            # Is it the one we are replacing?
            # If new_elem == old_elem, we skipped.
            # So if bit is set, it's a duplicate.
            continue
        
        # Apply move
        mask &= ~(1 << (old_elem - 1))
        mask |= (1 << (new_elem - 1))
        
        new_score = _score_set_bitset_from_mask(mask)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            S[i] = new_elem
            
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
                best_mask = mask
        else:
            # Revert
            mask &= ~(1 << (new_elem - 1))
            mask |= (1 << (old_elem - 1))
            # S[i] remains old_elem
    
    return best_S, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = random.Random(42)
    
    best_S = list(range(1, 33))
    best_score = _score_set_bitset(best_S)
    
    # Try multiple restarts
    num_restarts = 20
    per_restart_time = 6.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            S_init = list(range(1, 33))
        else:
            # Random init
            S_init = sorted(rng.sample(range(1, 513), 32))
        
        remaining = min(per_restart_time, deadline - time.time() - 5.0)
        if remaining < 1.0:
            break
        
        S, score = _simulated_annealing(S_init, remaining, rng)
        
        # Greedy polish
        if time.time() < deadline - 3.0:
            S, score = _greedy_polish_fast(S, time_limit=2.0)
        
        if score > best_score:
            best_score = score
            best_S = S
    
    # Final polish
    if time.time() < deadline - 1.0:
        best_S, best_score = _greedy_polish_fast(best_S, time_limit=5.0)
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
