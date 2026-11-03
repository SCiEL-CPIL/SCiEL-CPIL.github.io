# EVOLVE-BLOCK-START
def construct():
    """
    Analytical solution for minimizing the peak of the autoconvolution of a non-negative
    step function on a discrete circle.
    
    Mathematical Derivation:
    Let h be a vector of length N with h_i >= 0 and sum(h) = 1.
    The circular autoconvolution c is given by c_k = sum_i h_i h_{k-i}.
    We want to minimize M = max_k c_k.
    
    1. Sum of c: sum_k c_k = sum_k sum_i h_i h_{k-i} = sum_i h_i (sum_k h_{k-i}) = sum_i h_i * 1 = 1.
    2. Since c_k >= 0, we have M >= (1/N) * sum_k c_k = 1/N.
    3. The uniform distribution h_i = 1/N yields c_k = 1/N for all k.
    4. Thus, the uniform distribution achieves the lower bound M = 1/N.
    
    Therefore, the uniform distribution is the global optimum.
    """
    N = 600
    return [1.0 / N] * N
# EVOLVE-BLOCK-END
