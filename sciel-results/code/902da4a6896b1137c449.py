import numpy as np

N = 400

def construct():
    # We aim for a support size M that balances the number of terms in the convolution
    # and the ability to find a sparse structure.
    # For N=400, a Sidon set size is roughly sqrt(N) ~ 20.
    # Let's try M = 20.
    M = 20
    
    # 1. Construct Support
    # Goal: Minimize max difference multiplicity.
    # Greedy construction: Start with 0. Add next index that minimizes the max collision count.
    # This is O(M^2 * N) which is fast.
    
    support = np.zeros(M, dtype=int)
    support[0] = 0
    
    def get_diff_counts(supp):
        diffs = np.zeros(N, dtype=int)
        for i in range(len(supp)):
            for j in range(i+1, len(supp)):
                d = abs(supp[i] - supp[j])
                diffs[d] += 1
        return diffs

    # Greedy support selection with random restarts
    # We want to spread indices out.
    
    best_support = None
    best_max_count = N
    best_score = np.inf
    
    for restart in range(100):
        # Random initial support
        curr_support = np.random.permutation(N)[:M]
        counts = get_diff_counts(curr_support)
        curr_max = np.max(counts)
        curr_score = np.sum(counts**2)
        
        # Local search on support: swap one element
        improved = True
        while improved:
            improved = False
            # Try swapping each element in support with a random element outside
            for i in range(M):
                # Pick a random candidate to swap with
                for _ in range(20):
                    new_val = np.random.randint(N)
                    if new_val in curr_support:
                        continue
                    
                    old_val = curr_support[i]
                    curr_support[i] = new_val
                    
                    new_counts = get_diff_counts(curr_support)
                    new_max = np.max(new_counts)
                    new_score = np.sum(new_counts**2)
                    
                    if new_max < curr_max or (new_max == curr_max and new_score < curr_score):
                        curr_max = new_max
                        curr_score = new_score
                        counts = new_counts
                        improved = True
                    else:
                        curr_support[i] = old_val
            
            if not improved:
                break
        
        if curr_max < best_max_count or (curr_max == best_max_count and curr_score < best_score):
            best_max_count = curr_max
            best_score = curr_score
            best_support = curr_support.copy()
            
    support = best_support

    # 2. Assign Signs
    # We want to minimize peak of Y[d] = sum_{i<j, d_i-d_j=d} s_i s_j
    
    def calc_peak(signs, supp):
        Y = np.zeros(N)
        for k in range(M):
            for l in range(k+1, M):
                d = abs(supp[k] - supp[l])
                Y[d] += signs[k] * signs[l]
        return np.max(np.abs(Y))

    best_signs = None
    best_peak = np.inf
    
    for restart in range(50):
        signs = np.random.choice([-1.0, 1.0], size=M)
        current_peak = calc_peak(signs, support)
        
        improved = True
        while improved:
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
                    
        if current_peak < best_peak:
            best_peak = current_peak
            best_signs = signs.copy()

    # 3. Construct Final Array
    x = np.zeros(N)
    # Scale by 1/sqrt(N) to keep L2 norm consistent with seed
    x[support] = best_signs / np.sqrt(N)
    
    return x
