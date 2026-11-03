"""Rewritten program for signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400
# We need to return a sequence of signed step heights.
# The task is to minimize the maximum absolute autoconvolution divided by squared integral.
# Let x be the step heights.
# Autoconvolution c = x * x (linear convolution).
# Integral I = sum(x).
# Score metric: max(|c|) / I^2.
# We want to minimize this value.
# Note: The evaluator measures the "raw value" (lower is better).
# The "score" in the prompt is seed/value. So if we get a lower raw value, the score is higher.
# The prompt says: "score is the upper bound on C3' (maximum absolute autoconvolution)".
# Wait, let's re-read carefully:
# "score is the upper bound on C3' (maximum absolute autoconvolution)"
# "The evaluator measures the raw value (lower is better)."
# "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
# This implies the "raw value" is the quantity we want to minimize.
# The quantity is likely: max(|autocorr|) / (sum(x))^2.
# Or just max(|autocorr|) if the integral is normalized?
# "minimizing the maximum absolute autoconvolution divided by squared integral"
# So RawValue = max(|conv(x, x)|) / (sum(x))^2.

def compute_raw_value(x):
    # Linear convolution
    conv = np.convolve(x, x, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if integral == 0:
        return float('inf')
    return max_abs_conv / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Initial solution: all ones
    best_x = np.ones(N)
    best_val = compute_raw_value(best_x)
    
    # Strategy:
    # 1. Random restarts with structured patterns.
    # 2. Local search / hill climbing.
    
    # Let's try a few heuristic patterns first to see if we can beat the all-ones baseline.
    # All ones:
    # conv = [1, 2, ..., N, ..., 2, 1]
    # max_abs_conv = N
    # integral = N
    # raw_val = N / N^2 = 1/N = 1/400 = 0.0025
    
    # What if we use alternating signs?
    # x = [1, -1, 1, -1, ...]
    # conv will have small values.
    # Let's test a few candidates.
    
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Alternating signs
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    candidates.append(x_alt)
    
    # 3. Random binary signs
    for _ in range(100):
        x_rand = np.random.choice([1, -1], N)
        candidates.append(x_rand)
    
    # 4. Block patterns
    # Try blocks of 1s and -1s
    for block_len in [2, 4, 8, 16, 32, 64, 128]:
        x_block = np.ones(N)
        x_block[1::2] = -1 # Start with alternating, then modify?
        # Let's just create random block patterns
        for _ in range(20):
            x = np.ones(N)
            pos = 0
            while pos < N:
                length = np.random.randint(1, block_len+1)
                sign = np.random.choice([1, -1])
                x[pos:pos+length] = sign
                pos += length
            candidates.append(x)
    
    # Evaluate candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        if time.time() > deadline:
            break
    
    # Local search: flip individual bits
    # We can also try flipping small blocks.
    # Since N=400, flipping one bit changes the convolution by a small amount.
    # However, recomputing the full convolution is O(N^2) which is 160,000 ops.
    # 100,000 flips would be 16e9 ops, which might be too slow.
    # We need to be careful with the compute budget.
    
    # Let's estimate: np.convolve of size 400 is fast in C.
    # Let's time it.
    # Actually, let's just do a simple hill climbing with a time check.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try random flips
        for _ in range(5000): # Limit iterations per pass
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            new_x = best_x.copy()
            new_x[i] = -new_x[i]
            val = compute_raw_value(new_x)
            if val < best_val:
                best_val = val
                best_x = new_x
                improved = True
    
    return best_x

# Run construct once to ensure it works and doesn't crash
# But we should not print anything unless necessary.
# The function construct is the entry point.

if __name__ == "__main__":
    # Optional: quick self-test
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
