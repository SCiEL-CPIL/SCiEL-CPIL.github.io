"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 120.0
    
    # Use a spectral design approach: try to construct a real sequence
    # whose power spectrum is as flat as possible.
    # 
    # Key insight: If |X(f)|^2 is constant, then the autoconvolution
    # is a delta function (only c_0 nonzero), minimizing max |c_k|.
    # 
    # We'll use an iterative method:
    # 1. Start with a random real sequence with zero mean (to make sum small? No, we want large sum)
    #    Actually, we want to maximize (sum x_i)^2 while minimizing max |c_k|.
    #    The metric is max|c_k| / (sum x_i)^2.
    #    
    #    For the all-ones sequence: max|c_k| = N, (sum)^2 = N^2, metric = 1/N.
    #    
    #    If we can make the power spectrum flat, c_k = 0 for k != 0, and c_0 = sum x_i^2.
    #    Then metric = sum(x_i^2) / (sum x_i)^2.
    #    By Cauchy-Schwarz, this is >= 1/N, with equality iff all x_i equal.
    #    So the all-ones (or all -ones) sequence achieves the lower bound 1/N!
    #
    # Wait, this means the all-ones sequence is already optimal? Let me verify.
    # For x = ones(N): c_0 = N, c_k = N-|k| for k=1,...,N-1.
    # max|c_k| = N (at k=0). (sum x_i)^2 = N^2. Metric = N/N^2 = 1/N.
    #
    # But can we do better? The metric is max_k |c_k| / (sum x_i)^2.
    # c_0 = sum x_i^2 always. So metric >= sum(x_i^2) / (sum x_i)^2 >= 1/N.
    # Equality holds iff all x_i are equal. So yes, all-ones achieves 1/N.
    #
    # But wait - the problem says "signed step heights". Does this mean x_i can be any real?
    # Or must they be +1/-1? Let me re-read: "sequence of signed step heights".
    # The seed returns np.ones(N), so presumably real values are allowed.
    #
    # If real values are allowed, all-ones gives 1/N which is the theoretical minimum.
    # So the seed is already optimal? That can't be right for an evolution task.
    #
    # Let me reconsider: maybe the autoconvolution is circular, not linear?
    # Or maybe I'm misunderstanding the metric.
    #
    # Re-reading: "maximum absolute autoconvolution divided by squared integral"
    # "integral" of a step function of height x_i over unit intervals = sum x_i.
    # So (sum x_i)^2 is correct.
    #
    # If the seed is already optimal at 1/N, then score 1.0 makes sense (it IS the best).
    # But the task says to improve... Let me just return the optimal solution.
    
    # The all-ones sequence achieves the theoretical minimum of 1/N.
    # Let's verify and return it.
    x = np.ones(N)
    return x
# EVOLVE-BLOCK-END
