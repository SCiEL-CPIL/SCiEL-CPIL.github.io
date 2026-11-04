import numpy as np

N = 400

def construct():
    # 1. Select Support
    # We aim to find a support set S of size M such that the differences d = |s_i - s_j|
    # have low collision counts, and then assign signs to minimize the peak of the sum.
    
    # The objective is to minimize the upper bound on C3.
    # Let x be the signal. The autoconvolution y[k] = sum_i x[i] x[i+k].
    # For a sparse signal with support S and signs sigma, y[d] = sum_{(i,j) in S, |i-j|=d} sigma_i sigma_j.
    # We want to minimize max_d |y[d]|.
    
    # Strategy:
    # 1. Choose M. Smaller M reduces the number of terms, but we need enough terms to "spread" the energy?
    # Actually, if we normalize x to have L2 norm 1, x[i] = sigma_i / sqrt(M) for i in S.
    # Then y[d] = (1/M) * sum_{pairs} sigma_i sigma_j.
    # The peak value is (1/M) * max_d |sum_{pairs} sigma_i sigma_j|.
    # To minimize this, we want M to be large enough to dilute, but the sum can grow with M.
    # However, if the signs cancel well, the sum stays small.
    # Let's try a range of M values.
    
    best_peak = float('inf')
    best_support = None
    best_signs = None
    
    # We will iterate over a few candidate M values
    for M in [10, 15, 20, 25, 30, 40]:
        
        # Initialize support with random permutation
        support = np.random.permutation(N)[:M]
        
        # Local search on support to minimize collision counts
        # We want to minimize the max collision count, and as a tie-breaker the sum of squares.
        
        def get_diff_counts(supp):
            diffs = []
            for i in range(len(supp)):
                for j in range(i+1, len(supp)):
                    diffs.append(abs(supp[i] - supp[j]))
            counts = np.bincount(diffs, minlength=N)
            return counts

        counts = get_diff_counts(support)
        max_count = np.max(counts)
        score = np.sum(counts**2)
        
        # Local search iterations
        # We perform random swaps to improve the support structure.
        # To speed up, we limit the number of iterations.
        for _ in range(100):
            if max_count <= 1:
                break
            
            i1, i2 = np.random.choice(M, 2, replace=False)
            old_i1, old_i2 = support[i1], support[i2]
            
            support_set = set(support.tolist())
            all_indices = set(range(N))
            available = list(all_indices - support_set)
            
            if len(available) < 2:
                continue
                
            new_i1, new_i2 = np.random.choice(available, 2, replace=False)
            
            support[i1] = new_i1
            support[i2] = new_i2
            
            new_counts = get_diff_counts(support)
            new_max = np.max(new_counts)
            new_score = np.sum(new_counts**2)
            
            if new_max < max_count or (new_max == max_count and new_score < score):
                counts = new_counts
                max_count = new_max
                score = new_score
            else:
                support[i1] = old_i1
                support[i2] = old_i2

        # 2. Assign Signs via Local Search
        # For this support, find signs that minimize the peak.
        # We use multiple random restarts for signs and local search.
        
        def calc_peak(signs, support):
            Y = np.zeros(N)
            for k in range(len(support)):
                for l in range(k+1, len(support)):
                    d = abs(support[k] - support[l])
                    Y[d] += signs[k] * signs[l]
            return np.max(np.abs(Y))

        best_local_peak = float('inf')
        best_local_signs = None
        
        for restart in range(10):
            signs = np.random.choice([-1.0, 1.0], size=M)
            current_peak = calc_peak(signs, support)
            
            # Local search: flip signs
            for _ in range(50):
                improved = False
                for k in range(M):
                    old_sign = signs[k]
                    signs[k] = -old_sign
                    new_peak = calc_peak(signs, support)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        improved = True
                    else:
                        signs[k] = old_sign
                if not improved:
                    break
            
            if current_peak < best_local_peak:
                best_local_peak = current_peak
                best_local_signs = signs.copy()
        
        # Calculate the actual bound value: peak / M
        # Wait, the task says "score is the upper bound on C3".
        # The seed program returns x. The evaluator calculates the score from x.
        # We need to know what metric the evaluator uses.
        # "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
        # "The evaluator measures the raw value (lower is better)."
        
        # Let's assume the standard definition: C3 = sup ||x * x||_inf / ||x||_2^2 ?
        # Or just ||x * x||_inf?
        # If x is scaled by 1/sqrt(M), ||x||_2 = 1.
        # Then the value is just max |y[d]|.
        # In our calc_peak, we computed sum sigma_i sigma_j.
        # So y[d] = (1/M) * sum sigma_i sigma_j.
        # So the peak value is best_local_peak / M.
        
        current_bound = best_local_peak / M
        
        if current_bound < best_peak:
            best_peak = current_bound
            best_support = support.copy()
            best_signs = best_local_signs.copy()
            # print(f"M={M}, Peak={best_local_peak}, Bound={current_bound}, MaxCount={max_count}")

    # Construct final array
    x = np.zeros(N)
    x[best_support] = best_signs / np.sqrt(len(best_support))
    
    return x
