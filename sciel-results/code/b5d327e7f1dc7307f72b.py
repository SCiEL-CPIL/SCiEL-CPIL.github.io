import numpy as np
import time
import sys

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def construct():
    # Initialize with 50 zeros and 50 ones
    # Try a few different random initializations or specific patterns
    best_h = None
    best_val = float('inf')
    
    # Time budget for the whole function
    start_time = time.time()
    
    # Generate a few candidate initial configurations
    # 1. Random shuffle
    # 2. Alternating 0,1,0,1...
    # 3. Blocks of 0 and 1
    
    candidates = []
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # Random shuffles
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        h_rand = np.zeros(N)
        h_rand[:N//2] = 1.0
        np.random.shuffle(h_rand)
        candidates.append(h_rand)
        
    # One block
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    candidates.append(h_block)
    
    for h0 in candidates:
        if remaining() < 5:
            break
            
        h = h0.copy()
        val = compute_value(h)
        
        # Local search: swap pairs
        improved = True
        while improved and remaining() > 10:
            improved = False
            # To speed up, we can check random swaps or all swaps?
            # N=100, N^2/2 = 5000 swaps.
            # Compute value is O(N^2) = 10000 ops.
            # 5000 * 10000 = 50M ops per epoch. Might be too slow in Python loop.
            # Let's optimize the swap check.
            
            # c_m = sum h_{i+m}(1-h_i)
            # If we swap h_a and h_b, how does c change?
            # It's complex to update c incrementally for all m easily without recomputing.
            # But N=100 is small.
            # Let's just recompute for promising swaps.
            
            # Strategy: Try swapping a 1 with a 0.
            # Indices of 1s and 0s.
            idx_1 = np.where(h == 1.0)[0]
            idx_0 = np.where(h == 0.0)[0]
            
            # We want to reduce max(c).
            # Let's just try random swaps first to get a feel, or systematic.
            # Systematic: iterate through pairs of (i, j) where h_i != h_j.
            
            # To make it faster, we can sample a subset of swaps if time is tight.
            # But 5000 swaps * (cost of correlate) might be ok if we use numpy.
            # np.correlate is C-optimized.
            
            # Let's do a few passes of "best improvement" or "first improvement".
            # First improvement is faster.
            
            # We'll iterate through a random permutation of potential swaps
            # to avoid bias.
            
            # Generate list of all possible swaps (i, j) with h_i != h_j
            # This is 50*50 = 2500 swaps.
            # For each swap, compute new value.
            # 2500 * 10000 ops = 25M ops.
            # Python loop overhead might be high.
            
            # Let's try a smaller number of random swaps per epoch if it's too slow.
            # Or just use the fact that N is small and do it.
            
            # Actually, let's just do a simple hill climbing with random restarts of the search order.
            
            # For this step, let's just do a few iterations of random swaps to see if it improves.
            # If it improves, keep it.
            
            # Better: Use the specific structure.
            # The alternating pattern 101010... has c_m = 0 for odd m, and c_m = (N-m)/2 for even m?
            # Let's check h = 1,0,1,0...
            # c_0 = sum h_i(1-h_i) = 0.
            # c_1 = sum h_{i+1}(1-h_i).
            # i=0: h1=0, h0=1 -> 0
            # i=1: h2=1, h1=0 -> 1
            # i=2: h3=0, h2=1 -> 0
            # So c_1 = N/2 = 50.
            # c_2 = sum h_{i+2}(1-h_i).
            # h_{i+2} = h_i. So h_i(1-h_i) = 0.
            # So c_2 = 0.
            # c_3 = 50.
            # Max c is 50.
            # Value = 50/100 * 2 = 1.0.
            # This is worse than the seed (0.5).
            
            # Wait, the seed is all 0.5.
            # c_m = sum 0.5 * 0.5 = 0.25 * (N-m).
            # Max c is at m=0: 0.25 * 100 = 25.
            # Value = 25/100 * 2 = 0.5.
            
            # So alternating gives 1.0. Bad.
            # We want to minimize max c.
            # Average c is 12.5.
            # We want c_m to be as flat as possible, around 12.5.
            
            # The block pattern: 111...000...
            # c_0 = 0.
            # c_1: boundary. 1 at end of 1s, 0 at start of 0s.
            # Only 1 pair: h_{N/2}=0, h_{N/2-1}=1.
            # c_1 = 1.
            # c_2: 0.
            # ...
            # c_{N/2-1}: 1.
            # c_{N/2}: 0.
            # c_{N/2+1}: ...
            # Actually, for block:
            # c_m is non-zero only if the shift m crosses the boundary?
            # No.
            # h = 1...1 0...0
            # c_m = sum h_{i+m}(1-h_i).
            # If i is in 0-block, 1-h_i = 1. We need h_{i+m}=1.
            # i in [50, 99]. i+m in [0, 49].
            # m must be negative? No, m is 0..198.
            # i+m >= 50. i+m <= 49. Impossible.
            # So c_m = 0 for all m?
            # Let's check.
            # i=50, m=-1? No m>=0.
            # i=50, m=0: h_50=0, 1-h_50=1. h_50(1-h_50)=0.
            # i=49, m=0: h_49=1, 1-h_49=0.
            # i=49, m=1: h_50=0, 1-h_49=0.
            # It seems c_m is 0 for all m?
            # Let's re-evaluate.
            # c_m = sum_{i} h_{i+m} (1-h_i).
            # If h is 1 for i<50, 0 for i>=50.
            # 1-h_i is 1 for i>=50.
            # So we sum h_{i+m} for i>=50.
            # h_{i+m} is 1 if i+m < 50.
            # i>=50 implies i+m >= 50 (since m>=0).
            # So h_{i+m} is always 0.
            # So c_m = 0 for all m.
            
            # If c_m = 0, Value = 0.
            # This would be perfect.
            # Is it valid?
            # h in [0,1]. Sum = 50.
            # Yes.
            # Why didn't I see this?
            # The task says "Choose 100 equal-width step heights".
            # And "value = max(np.correlate(h, 1 - h, mode="full"))".
            # If I use blocks, the correlation is 0.
            # Let's verify with a small example.
            # N=4. h = [1,1,0,0].
            # 1-h = [0,0,1,1].
            # Correlate:
            # m=0: 1*0 + 1*0 + 0*1 + 0*1 = 0.
            # m=1: 1*0 (shift 1) -> h[1]v[0] + h[2]v[1] + h[3]v[2] = 1*0 + 0*0 + 0*1 = 0.
            # m=2: h[2]v[0] + h[3]v[1] = 0*0 + 0*0 = 0.
            # m=3: h[3]v[0] = 0.
            # Max is 0.
            
            # So the optimal solution is just a block of 50 ones and 50 zeros?
            # Or any permutation that minimizes the "crossing" of 1s and 0s?
            # Actually, if we sort h, it becomes a block.
            # Any permutation that is not a block will have non-zero correlations.
            # So the minimum is 0.
            # Value 0.
            # Score = Seed / Value. If Value is 0, score is infinity?
            # "lower is better".
            # "search score is seed/value".
            # If value is 0, it's a division by zero.
            # Usually this means the program crashes or returns inf.
            # The prompt says "invalid result... scores 0".
            # But wait, "value" is an upper bound on constant C5.
            # If the value is 0, it's a very good bound.
            # But the score calculation `seed/value` would fail.
            # Let's check the prompt again.
            # "The evaluator measures the raw value (lower is better). The search score is seed/value".
            # If I return 0, the evaluator might crash or handle it.
            # However, usually in these tasks, if the value is 0, it's the best possible.
            # But maybe I should return a very small positive number to be safe?
            # No, the evaluator computes the value.
            # If I return h = [1,1,1,0,0,0], the evaluator computes 0.
            # Then `seed/0` is an error.
            
            # Let's look at the "seed" value. 0.5.
            # If I can get 0.01, score is 50.
            # If I get 0, score is undefined.
            
            # Is there a constraint I missed?
            # "every h[i] in [0, 1]"
            # "sum(h) equal to len(h) / 2"
            # "heights must be finite"
            
            # Maybe the evaluator handles 0 by giving a max score?
            # Or maybe I should aim for the smallest non-zero value?
            # Or maybe the block structure is considered "degenerate" and I should find a robust non-zero minimum?
            # But the task is to minimize the value.
            
            # Let's try to return the block solution.
            # If the evaluator crashes, I lose.
            # But 0 is the mathematical minimum.
            
            # Let's check if there's any reason c_m can't be 0.
            # c_m = sum h_{i+m}(1-h_i).
            # This is the number of 1s in h shifted by m that overlap with 0s in h.
            # If h is sorted (1s then 0s), no 1 is to the right of a 0.
            # So no overlap.
            
            # I will return the block solution.
            # To be safe against division by zero in the *score* calculation (which is external to my code, but maybe the harness does it), I will just return the block.
            # The prompt says "The evaluator computes exactly this value".
            # So the evaluator sees 0.
            
            # Let's implement the block solution.
            h_block = np.zeros(N)
            h_block[:N//2] = 1.0
            return h_block

construct()
