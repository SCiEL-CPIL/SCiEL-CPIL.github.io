import time
import numpy as np
import random

def _precompute_aps(n):
    """
    Precompute masks for all arithmetic progressions.
    Returns a list of (mask, length) tuples.
    """
    ap_data = []
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            # AP is r, r+d, r+2d, ... < n
            mask = 0
            length = 0
            idx = r
            while idx < n:
                mask |= (1 << idx)
                length += 1
                idx += d
            ap_data.append((mask, length))
    return ap_data

def _compute_max_discrepancy_fast(mask_pos, ap_data):
    """
    Compute max abs sum using bitmask representation.
    mask_pos: int with bits set for +1 signs.
    ap_data: list of (mask, length)
    """
    max_disc = 1
    for mask, length in ap_data:
        # Count bits in intersection
        k = (mask_pos & mask).bit_count()
        # Sum = k - (length - k) = 2k - length
        s = 2 * k - length
        if abs(s) > max_disc:
            max_disc = abs(s)
            if max_disc == length: # Max possible for this AP is length, but we need global max
                # We can't break early because another AP might have larger length and max sum
                # Actually max sum is bounded by n. 
                # If we hit n, we can stop? No, we want to find the max.
                # If max_disc reaches n, it's the theoretical max.
                pass
    return max_disc

def _greedy_construct(n):
    """
    Greedy construction using the fast evaluator.
    """
    ap_data = _precompute_aps(n)
    mask_pos = 0 # All 0s initially (all -1)
    # Wait, greedy needs to decide sign by sign.
    # We need to be able to query the discrepancy given a partial assignment?
    # The fast evaluator assumes full assignment.
    # So greedy with fast evaluator is hard because we don't have full mask yet.
    # We can use the old greedy for initialization, or random init.
    # Let's use random init + local search. It's often better with fast eval.
    
    # Or we can keep the old greedy which is O(n^3) but n=256 is small enough for one run.
    # Old greedy:
    signs = [0] * n
    for i in range(n):
        best_sign = 1
        best_disc = 10**9
        for s in [1, -1]:
            signs[i] = s
            # Recompute max disc? Too slow for greedy inner loop if O(n^3) per step.
            # Old code did a partial check.
            # Let's just use a simple heuristic or random init.
            pass
            
    # Let's use the Thue-Morse or just random.
    # Random init is fine if local search is strong.
    return None

def construct():
    n = 256
    start_time = time.time()
    deadline = 58.0
    
    ap_data = _precompute_aps(n)
    
    rng = random.Random(42)
    
    best_mask = None
    best_disc = n + 1
    
    # Try a few initializations
    # 1. Random
    # 2. Thue-Morse
    # 3. Alternating
    
    inits = []
    
    # Random
    m = 0
    for _ in range(n):
        if rng.random() < 0.5:
            m |= (1 << rng.randint(0, n-1)) # This is wrong, need unique indices or just iterate
    # Proper random init
    m_rand = 0
    for i in range(n):
        if rng.random() < 0.5:
            m_rand |= (1 << i)
    inits.append(m_rand)
    
    # Thue-Morse
    m_tm = 0
    for i in range(n):
        c = bin(i).count('1')
        if c % 2 == 0:
            m_tm |= (1 << i)
    inits.append(m_tm)
    
    # Alternating
    m_alt = 0
    for i in range(n):
        if i % 2 == 0:
            m_alt |= (1 << i)
    inits.append(m_alt)
    
    # Run local search on each init
    for init_mask in inits:
        current_mask = init_mask
        current_disc = _compute_max_discrepancy_fast(current_mask, ap_data)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_mask = current_mask
            
        no_improve = 0
        max_no_improve = 500 # Can do more with fast eval
        
        while time.time() - start_time < deadline:
            if no_improve > max_no_improve:
                break
            
            # Find a random index to flip? Or find worst AP and flip in it?
            # Finding worst AP requires scanning all APs.
            # We can just pick a random index and flip it.
            # If it improves, keep it. If not, revert.
            # This is simple hill climbing.
            
            # Better: Find the current max discrepancy and the AP causing it?
            # We can modify _compute_max_discrepancy_fast to return the worst AP index.
            
            # Let's implement a helper to find worst AP index
            max_d = 1
            worst_idx = 0
            for idx, (mask, length) in enumerate(ap_data):
                k = (current_mask & mask).bit_count()
                s = 2 * k - length
                if abs(s) > max_d:
                    max_d = abs(s)
                    worst_idx = idx
            
            if max_d < best_disc:
                best_disc = max_d
                best_mask = current_mask
                no_improve = 0
            else:
                no_improve += 1
            
            # Get the mask of the worst AP
            worst_mask, worst_len = ap_data[worst_idx]
            
            # Try flipping each bit in the worst AP?
            # Or just a random bit in the worst AP?
            # Flipping a bit in the worst AP is likely to help.
            
            # Extract indices set in worst_mask
            # This is expensive to do every time.
            # We can precompute the list of indices for each AP mask?
            # ap_indices[ap_idx] = list of indices
            
            # Let's precompute indices for APs
            # We already have ap_data. Let's make ap_indices.
            
            # Actually, we can just iterate through bits of worst_mask.
            # But we want to try flipping them.
            
            # Let's try a random index from the worst AP.
            # To get a random index from a bitmask:
            # 1. Count bits k.
            # 2. Pick random r in [0, k).
            # 3. Find r-th set bit.
            
            # This is getting complex.
            # Simpler: Just pick a random index from 0..n-1 and flip it.
            # With fast eval, we can do many iterations.
            
            idx = rng.randint(0, n - 1)
            bit = 1 << idx
            new_mask = current_mask ^ bit
            
            new_disc = _compute_max_discrepancy_fast(new_mask, ap_data)
            
            if new_disc < current_disc:
                current_mask = new_mask
                current_disc = new_disc
                no_improve = 0
            else:
                no_improve += 1
                # If no improvement for a while, we might get stuck.
                # But we break out of loop if no_improve > max_no_improve.
                
    # Convert best_mask to list of signs
    signs = [0] * n
    for i in range(n):
        if (best_mask >> i) & 1:
            signs[i] = 1
        else:
            signs[i] = -1
            
    return signs
