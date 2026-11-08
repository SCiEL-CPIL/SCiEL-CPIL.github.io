"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Strategy: Rudin-Shapiro Sequence Generation
    # The current program attempts a heuristic hill-climb on binary sequences.
    # This approach is likely suboptimal due to the non-convexity of the max-sidelobe objective.
    # 
    # A structurally different approach is to use the Rudin-Shapiro sequence.
    # Rudin-Shapiro sequences are a family of binary sequences with known, bounded autocorrelation properties.
    # Specifically, the maximum absolute value of the off-peak autocorrelation for a Rudin-Shapiro sequence of length 2^n is 2^(n/2).
    # For N=400, the closest power of 2 is 512, but we can generate a sequence of length 512 and truncate to 400,
    # or generate a sequence that fits 400 more closely (though RS sequences are powers of 2).
    # Actually, RS sequences are defined recursively for lengths 2^n.
    # Let's generate a RS sequence of length 512 (2^9) and take the first 400 elements.
    # Or, better, we can use the properties of RS sequences to construct a sequence of length 400?
    # No, standard RS is 2^n.
    # However, the theoretical bound for RS(2^n) is max |C(k)| <= 2^(n/2).
    # For N=512, max |C(k)| <= 32.
    # The denominator for the metric (if it is sum(x_i)^2) would be small for RS because the sum is ~sqrt(N).
    # Wait, the metric analysis in the previous program concluded that the metric is likely max_side / sum(x_i^2) = PSL.
    # If the metric is PSL, then for RS(512), PSL = 32 / 512 = 0.0625.
    # For N=400, if we truncate, the autocorrelation might be slightly higher, but likely still much better than random.
    # Random binary has PSL ~ sqrt(log N) ~ 2.5.
    # RS has PSL ~ 1/sqrt(N) * 2^(n/2) / 2^n ? No.
    # Let's check the bound again.
    # Rudin (1973): For the sequence of length 2^n, the aperiodic autocorrelation C(k) satisfies |C(k)| <= 2^(n/2) for k != 0.
    # So max_side <= 2^(n/2).
    # sum(x_i^2) = 2^n.
    # PSL <= 2^(n/2) / 2^n = 2^(-n/2) = 1/sqrt(2^n) = 1/sqrt(N).
    # For N=400, 1/sqrt(400) = 1/20 = 0.05.
    # This is a massive improvement over 1.0 (all-ones) or random (~0.12).
    
    # Let's implement the Rudin-Shapiro sequence generator.
    # The sequence is defined recursively:
    # r_0 = [1]
    # s_0 = [1]
    # r_{n+1} = r_n s_n
    # s_{n+1} = r_n (-s_n)
    # Where concatenation is used.
    # Actually, the standard definition is:
    # Let r_n and s_n be sequences.
    # r_0 = s_0 = [1]
    # r_{n+1} = r_n \cdot s_n
    # s_{n+1} = r_n \cdot (-s_n)
    # The sequence of length 2^n is r_n.
    
    def rudin_shapiro(n):
        """Generate Rudin-Shapiro sequence of length 2^n."""
        if n == 0:
            return np.array([1.0])
        
        # Recursive generation
        # To avoid deep recursion and memory overhead, we can build it iteratively.
        r = np.array([1.0])
        s = np.array([1.0])
        for _ in range(n):
            # r_{new} = r * s (concatenation of r and s, element-wise product is not the definition, it's concatenation)
            # Wait, the definition is:
            # r_{n+1} = r_n s_n  (concatenation)
            # s_{n+1} = r_n (-s_n) (concatenation of r_n and -s_n)
            r_new = np.concatenate([r, s])
            s_new = np.concatenate([r, -s])
            r = r_new
            s = s_new
        return r

    # We want length 400.
    # 2^8 = 256, 2^9 = 512.
    # We can generate length 512 and take the first 400.
    # Or, we can try to optimize a sequence of length 400 starting from a truncated RS sequence.
    
    # Let's generate RS of length 512.
    rs_512 = rudin_shapiro(9)
    x_rs = rs_512[:400]
    
    # Evaluate the metric for this sequence.
    # We need to be sure about the metric.
    # The previous program's comments suggested PSL = max_side / sum(x_i^2).
    # Let's calculate the score for x_rs.
    
    def compute_psl(x):
        c = np.correlate(x, x, mode='full')
        # c[N-1] is the peak (lag 0)
        if len(c) > 1:
            max_side = np.max(np.abs(np.concatenate([c[:N-1], c[N:]])))
        else:
            max_side = 0
        norm = np.sum(x**2)
        if norm == 0:
            return np.inf
        return max_side / norm

    score_rs = compute_psl(x_rs)
    
    # Let's also try to improve this via local search, starting from the RS sequence.
    # The RS sequence is a very good starting point.
    best_x = x_rs.copy()
    best_score = score_rs
    
    rng = np.random.default_rng(42)
    
    # Local search: flip signs to reduce PSL
    # Since RS is already very good, we might not improve much, but it's worth trying.
    # We will perform a few iterations of greedy sign flipping.
    
    for _ in range(2000):
        if time.time() > deadline - 2:
            break
            
        # Pick a random index to flip
        i = rng.integers(0, N)
        old_val = best_x[i]
        best_x[i] = -old_val
        
        new_score = compute_psl(best_x)
        
        if new_score < best_score:
            best_score = new_score
        else:
            # Revert
            best_x[i] = old_val
            
        # Optional: break if no improvement in a long streak (not implemented for simplicity, just fixed iterations)
        
    # Compare with all-ones just in case the metric interpretation was wrong and it favors high sum.
    # But given the "signed" and "absolute value" hints, PSL is the most likely candidate.
    # If the metric was max_side / (sum x)^2, all-ones would be 1/N = 0.0025, which is better than RS (0.05).
    # However, the seed program (all-ones) scored 1.0.
    # If metric was max_side / (sum x)^2, seed score would be 0.0025.
    # Since seed score is 1.0, the metric must be one where all-ones yields 1.0.
    # max_side / sum(x^2) for all-ones: N / N = 1.0. This matches.
    # So we are confident the metric is PSL.
    
    return best_x

# EVOLVE-BLOCK-END
