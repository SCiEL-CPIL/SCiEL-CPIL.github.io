import numpy as np

N = 400

def construct():
    # The task involves minimizing the upper bound on C3, which is related to the
    # third autocorrelation inequality. The score is the raw value (lower is better).
    # The seed program uses uniform distribution x_i = 1/sqrt(N), which gives:
    # C3 = sum_i (sum_j x_j^2 x_{i-j})^3 or similar convolution-based metric.
    # 
    # For the uniform case, the autoconvolution of x with x^2 (or x with itself)
    # produces a triangular-like shape. The peak of the autoconvolution determines the bound.
    #
    # The directive suggests using a sparse combinatorial design: select a small subset
    # of intervals to have non-zero values. This decouples the search from the continuous
    # 400-dimensional space.
    #
    # Key insight: If we use a sparse signal with k non-zero entries, each with value 1/sqrt(k),
    # the autoconvolution will have a different peak structure. For sparse signals with
    # well-separated non-zero entries, the autoconvolution peak can be lower than the uniform case.
    #
    # Let's try a sparse design where we select k indices from the 400 intervals.
    # The autoconvolution peak for a sparse signal depends on the spacing of the non-zero entries.
    # If we space them evenly, we can reduce the peak of the autoconvolution.
    #
    # Strategy: Use a random permutation to select k active intervals, then optimize k.
    # For simplicity, let's start with a fixed k and see if it improves over uniform.
    #
    # Actually, let's think more carefully. The metric is the "upper bound on C3".
    # For a signed step function, the third autocorrelation inequality likely involves:
    # C3 = max_i |sum_j x_j x_{i-j} x_{i-2j}| or similar triple correlation.
    #
    # For the uniform case x_i = 1/sqrt(N), the triple autocorrelation sum is:
    # sum_j (1/sqrt(N))^3 = N * (1/N^(3/2)) = 1/sqrt(N)
    # But this is for the peak value of the triple convolution.
    #
    # If we use a sparse signal with k non-zero entries, each 1/sqrt(k):
    # The triple autocorrelation at the origin would be sum over j where x_j, x_{i-j}, x_{i-2j} are all non-zero.
    # For well-spaced entries, this sum is small.
    #
    # Let's try k = 200 (half the intervals). This might reduce the peak.
    # Or even sparser: k = 100.
    #
    # Let's implement a sparse design with evenly spaced non-zero entries.
    
    # Try different sparsity levels
    best_x = None
    best_score = float('inf')
    
    # We need to compute the actual metric. Since we don't have the exact formula,
    # let's assume it's related to the peak of the autoconvolution or triple correlation.
    # 
    # For now, let's try a few sparse configurations and return the one that seems best.
    # The evaluation will tell us if it's better.
    
    # Try sparse with evenly spaced entries
    for k in [50, 100, 200, 300]:
        # Create a sparse signal with k non-zero entries, evenly spaced
        indices = np.linspace(0, N-1, k, dtype=int)
        x = np.zeros(N)
        x[indices] = 1.0 / np.sqrt(k)
        
        # Compute a proxy for the metric: peak of autoconvolution
        # autoconv = np.convolve(x, x, mode='full')
        # score = np.max(np.abs(autoconv))
        
        # For triple correlation:
        # We need sum_j x_j * x_{i-j} * x_{i-2j} for each i
        # This is more complex. Let's use a simpler proxy.
        
        # Actually, let's just return a sparse signal and let the evaluator judge.
        # The uniform case scores 1.0, so anything lower is an improvement.
        
        if best_x is None:
            best_x = x
            best_score = 0  # Placeholder
    
    # For the first iteration, let's try k=100 with evenly spaced entries
    k = 100
    indices = np.linspace(0, N-1, k, dtype=int)
    x = np.zeros(N)
    x[indices] = 1.0 / np.sqrt(k)
    
    return x
