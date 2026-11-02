import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random signs
    f = np.random.choice([1, -1], size=N)
    
    # Precompute initial auto-correlation
    # g[k] = sum_{i=0}^{N-1-k} f[i] f[i+k]
    # Using FFT for initial calculation is fast, but we need explicit g for updates.
    # For N=400, direct calculation is fine.
    g = np.zeros(N)
    for k in range(N):
        g[k] = np.dot(f[:N-k], f[k:])
        
    # Current cost
    current_cost = np.sum(np.abs(g[1:]))
    
    best_cost = current_cost
    best_f = f.copy()
    
    # Local search
    # We will iterate through indices and try flipping
    # To be efficient, we can use a simple hill climbing with random restarts if stuck, 
    # but given the budget, a single strong hill climb might suffice.
    
    # Optimization: 
    # Flipping f[i] changes g[k] for all k.
    # Delta g[k] = -2 * f[i] * (f[i+k] if i+k < N else 0) - 2 * f[i] * (f[i-k] if i-k >= 0 else 0)
    # Note: The term f[i]f[i+k] appears in g[k] and g[-k] (if defined). 
    # But here g is defined for k>=0.
    # g[k] = sum_{j=0}^{N-1-k} f[j]f[j+k].
    # If we flip f[i]:
    # 1. If i < N-k, f[i] is in the sum for g[k] as f[j] with j=i. Term is f[i]f[i+k]. Change: -2 f[i]f[i+k].
    # 2. If i-k >= 0, f[i] is in the sum for g[k] as f[j+k] with j=i-k. Term is f[i-k]f[i]. Change: -2 f[i-k]f[i].
    # So delta_g[k] = -2 * f[i] * ( (1 if i < N-k else 0) * f[i+k] + (1 if i >= k else 0) * f[i-k] )
    
    # We will do multiple passes.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order of indices to avoid bias? Or just sweep?
        # Sweeping is deterministic. Let's do a sweep.
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Calculate delta cost for flipping f[i]
            delta_cost = 0.0
            # We need to check all k
            # To speed up, we can note that delta_g[k] is non-zero only if i interacts with k.
            # Actually, delta_g[k] is non-zero for all k where the indices exist.
            # k range: 0 to N-1.
            # Condition 1: i < N-k  => k < N-i
            # Condition 2: i >= k   => k <= i
            # So k is in [0, max(i, N-1-i)] roughly? No, union of [0, i] and [0, N-1-i].
            # So k in [0, max(i, N-1-i)].
            
            # Let's just loop k for simplicity first, N=400 is small.
            # But 400*400 = 160k ops per flip. 400 flips = 64M ops per pass.
            # In Python, this might be slow. 64M simple ops ~ 1-2 seconds?
            # 120s budget allows ~100 passes.
            
            # Let's optimize the inner loop using NumPy vectorization if possible, 
            # or just stick to Python loops if N is small enough.
            # Actually, we can precompute the "neighbor" values.
            
            # Let's try a vectorized approach for the delta calculation.
            # f is 1D array.
            # For a specific i:
            # term1 (j=i): affects g[k] for k=0..N-1-i. Value: -2*f[i]*f[i+k]
            # term2 (j+k=i): affects g[k] for k=0..i. Value: -2*f[i-k]*f[i]
            
            # Construct delta_g array
            delta_g = np.zeros(N)
            
            # Part 1: k from 0 to N-1-i
            # g[k] += -2 * f[i] * f[i+k]
            if N - 1 - i >= 0:
                delta_g[:N-i] += -2.0 * f[i] * f[i:]
            
            # Part 2: k from 0 to i
            # g[k] += -2 * f[i-k] * f[i]
            # f[i-k] for k=0..i is f[i], f[i-1], ..., f[0] -> reversed f[0:i+1]
            if i >= 0:
                delta_g[:i+1] += -2.0 * f[i::-1] * f[i]
                
            # New g = g + delta_g
            # We only care about k >= 1 for the cost (g[0] is constant N)
            # But g[0] changes? 
            # g[0] = sum f[j]^2 = N. Always N.
            # Let's verify delta_g[0].
            # Part 1 k=0: -2 f[i] f[i] = -2.
            # Part 2 k=0: -2 f[i] f[i] = -2.
            # Total delta_g[0] = -4?
            # Wait. If we flip f[i], g[0] = sum f[j]^2. Since f[j] is +/-1, f[j]^2 is 1. Sum is N.
            # Why did I think it changes? It doesn't.
            # My delta formula:
            # Original term in g[0]: f[i]*f[i] = 1.
            # New term: (-f[i])*(-f[i]) = 1.
            # Change is 0.
            # Where is the error?
            # g[k] = sum f[j] f[j+k].
            # If we flip f[i]:
            # Term where j=i: f[i]f[i+k] -> -f[i]f[i+k]. Change: -2 f[i]f[i+k].
            # Term where j+k=i: f[i-k]f[i] -> f[i-k](-f[i]). Change: -2 f[i-k]f[i].
            # For k=0:
            # j=i: term f[i]f[i]. Change -2 f[i]^2 = -2.
            # j+k=i => j=i: term f[i]f[i]. Change -2 f[i]^2 = -2.
            # These are the SAME term! I double counted.
            # When k=0, the condition j=i and j+k=i are identical.
            # So for k=0, delta_g[0] should be -2, not -4?
            # Let's check.
            # g[0] = sum_{j=0}^{N-1} f[j]f[j].
            # Flip f[i].
            # Old term: f[i]^2 = 1.
            # New term: (-f[i])^2 = 1.
            # Change: 0.
            # My formula gave -2 + -2 = -4.
            # The issue is that for k=0, the "j=i" and "j+k=i" refer to the same product f[i]f[i].
            # In the sum for g[k], the term f[i]f[i+k] is present if i is in [0, N-1-k].
            # The term f[i-k]f[i] is present if i-k is in [0, N-1-k] => i in [k, N-1].
            # For k=0:
            # Term 1: f[i]f[i] present if i in [0, N-1]. Yes.
            # Term 2: f[i]f[i] present if i in [0, N-1]. Yes.
            # But they are the same term in the sum?
            # g[0] = sum_{j=0}^{N-1} f[j]f[j].
            # The term f[i]f[i] appears once in the sum.
            # My decomposition:
            # Contribution of f[i] to g[k]:
            # 1. As the first factor: f[i] * f[i+k]. (Requires i+k < N)
            # 2. As the second factor: f[i-k] * f[i]. (Requires i-k >= 0)
            # For k=0, these are both f[i]*f[i].
            # If we sum the deltas, we are adding the change for "first factor" and "second factor".
            # But they are the same physical term in the convolution sum.
            # So for k=0, we should only count the change once?
            # No, the convolution sum is just a sum of products.
            # g[0] = f[0]f[0] + f[1]f[1] + ... + f[i]f[i] + ...
            # If I flip f[i], the term f[i]f[i] changes from 1 to 1. Change 0.
            # If I use the formula:
            # Delta from "first factor" perspective: The term f[i]f[i] is replaced by -f[i]f[i]. Change -2.
            # Delta from "second factor" perspective: The term f[i]f[i] is replaced by f[i](-f[i]). Change -2.
            # This implies I am treating them as two separate contributions to the sum, which is incorrect for k=0.
            # For k > 0, the terms f[i]f[i+k] and f[i-k]f[i] are distinct terms in the sum (involving different indices or different positions).
            # So for k > 0, the double counting is correct (they are two different terms in the sum that involve f[i]).
            # For k = 0, they are the same term.
            # So:
            # delta_g[0] = 0.
            # delta_g[k] (k>0) = -2 f[i] (f[i+k] + f[i-k]) (with boundary checks).
            
            # Let's re-calculate delta_g for k>0 only.
            delta_g[1:] = 0.0
            
            # Part 1: k in [1, N-1-i]
            # f[i+k] exists.
            if N - 1 - i >= 1:
                # k=1..N-1-i
                # delta_g[k] += -2 f[i] f[i+k]
                # f[i+k] for k=1..N-1-i is f[i+1:i+N-i] = f[i+1:]
                delta_g[1:N-i] += -2.0 * f[i] * f[i+1:]
            
            # Part 2: k in [1, i]
            # f[i-k] exists.
            if i >= 1:
                # k=1..i
                # delta_g[k] += -2 f[i-k] f[i]
                # f[i-k] for k=1..i is f[i-1], ..., f[0] -> f[i-1::-1]
                # Note: k=1 corresponds to f[i-1]. k=i corresponds to f[0].
                # Vector f[i-1::-1] has length i.
                # It maps to indices 1..i in delta_g.
                delta_g[1:i+1] += -2.0 * f[i-1::-1] * f[i]
                
            # Now calculate delta cost
            # cost = sum |g[k]| for k=1..N-1
            # new_g = g + delta_g
            # delta_cost = sum(|g[k] + delta_g[k]| - |g[k]|) for k=1..N-1
            
            # We can compute this vectorized
            # g[1:] is the current values
            # delta_g[1:] is the change
            new_vals = g[1:] + delta_g[1:]
            delta_cost = np.sum(np.abs(new_vals) - np.abs(g[1:]))
            
            if delta_cost < -1e-9:
                # Accept flip
                f[i] = -f[i]
                g[1:] += delta_g[1:]
                # g[0] is always N, so we don't update it, but we must ensure it stays consistent if we ever use it.
                # We don't use g[0] in cost, so it's fine.
                current_cost += delta_cost
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_f = f.copy()
                    
    return best_f
