import numpy as np
import time

N = 100
TARGET_SUM = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Initial best (Seed)
    best_h = np.full(N, 0.5)
    best_val = 0.5 # Seed value: max_corr=25, 25/100*2 = 0.5
    
    # Helper to evaluate
    def evaluate(h):
        c = np.correlate(h, 1 - h, mode="full")
        max_c = np.max(c)
        return max_c / N * 2

    # Continuous Local Search
    # We perturb h slightly, ensuring sum remains 50 and bounds [0,1]
    # To maintain sum, we can pick two indices i, j and move delta from i to j.
    # h[i] -= delta, h[j] += delta.
    
    h = best_h.copy()
    
    # Pre-calculate 1-h for speed? 
    # np.correlate is C-optimized, it's fast.
    
    # We will try random pair swaps with continuous deltas.
    # This is essentially a 2-point move.
    
    # To be faster, we can try to fix the structure.
    # But random search in continuous space might find a "smooth" x that has positive autocorrelation.
    
    # Let's try a few structured initializations first, they are cheap.
    # 1. Sine wave (low freq)
    # 2. Sine wave (high freq)
    
    # Actually, let's just do a very fast random search with small steps.
    # 10000 iterations of random 2-point move.
    # Time per iter:
    # 2 random ints: ~1us
    # Modify h: ~1us
    # Correlate: ~10-20us
    # Total ~25us.
    # 10000 * 25us = 0.25s. Very fast.
    # We can do 100,000 iterations in 2.5s.
    
    count = 0
    while time.time() < deadline - 5: # Leave 5s for cleanup/return
        # Pick two distinct indices
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        if i == j:
            continue
            
        # Pick a small delta
        # Max possible delta is limited by bounds
        # h[i] in [0,1], h[j] in [0,1]
        # delta must be such that h[i]-delta >= 0 and h[j]+delta <= 1
        # Also we want to explore, so random delta in [-0.1, 0.1]
        delta = np.random.uniform(-0.1, 0.1)
        
        new_hi = h[i] - delta
        new_hj = h[j] + delta
        
        if new_hi < 0 or new_hi > 1 or new_hj < 0 or new_hj > 1:
            continue
            
        h[i] = new_hi
        h[j] = new_hj
        
        val = evaluate(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        else:
            # Revert
            h[i] = new_hi + delta # original
            h[j] = new_hj - delta # original
            # Note: floating point error might accumulate, so better to keep a copy or revert carefully.
            # Actually, h[i] was h[i]_old. h[i]_new = h[i]_old - delta.
            # So h[i]_old = h[i]_new + delta. Correct.
            
        count += 1
        if count % 1000 == 0:
            # Optional: Check time more frequently if needed, but time.time() is cheap enough.
            pass

    # Final check
    # Ensure sum is close to 50
    # The moves preserve sum exactly in real arithmetic, but float errors might creep in.
    # sum(best_h) should be very close to 50.
    # If not, we can correct by distributing the error.
    err = TARGET_SUM - np.sum(best_h)
    if abs(err) > 1e-12:
        # Distribute error evenly? Or add to one element?
        # Adding to one element might push it out of bounds.
        # Distribute evenly is safer for bounds if err is small.
        best_h += err / N
        # Clamp to [0,1] just in case, though err is tiny
        best_h = np.clip(best_h, 0, 1)
        # Re-correct sum if clamping changed it? 
        # For such small errors, clamping shouldn't change sum significantly.
        # But to be safe, if we clamp, we might break sum.
        # Given err is tiny, we can just adjust one element if needed.
        # Actually, the random walk preserves sum. The error is < 1e-15.
        # The requirement is within 1e-9 * 100 = 1e-7.
        # So no correction is likely needed.
        
    return best_h
