import numpy as np

N = 400
M = 20

def construct():
    # We aim to find a subset S of {0, ..., N-1} of size M
    # that minimizes the maximum number of pairs (i, j) in S with i - j = d (for d != 0).
    # This minimizes the peak of the autoconvolution (excluding lag 0).
    
    def max_diff_count(support):
        # support: np.array of shape (M,)
        diffs = support[:, None] - support[None, :]
        diffs_shifted = diffs + (N - 1)
        counts = np.bincount(diffs_shifted.flatten(), minlength=2 * N - 1)
        counts[N - 1] = 0  # Ignore lag 0
        return np.max(counts)

    best_support = None
    best_score = float('inf')
    
    # Multiple random restarts
    for restart in range(100):
        # Initialize with a random permutation
        perm = np.random.permutation(N)
        support = perm[:M].copy()
        
        current_max_count = max_diff_count(support)
        
        # Local search
        no_improve = 0
        max_iters = 2000
        sup_set = set(support.tolist())
        
        for _ in range(max_iters):
            improved = False
            # Pick a random index to swap
            i_idx = np.random.randint(M)
            old_val = support[i_idx]
            
            # Try to find a better replacement
            # We can sample new values from the permutation
            # Or just random integers
            for _ in range(5):
                new_val = np.random.randint(N)
                if new_val in sup_set:
                    continue
                
                # Check if new_val is in support
                # We need to update sup_set if we accept
                
                new_support = support.copy()
                new_support[i_idx] = new_val
                
                new_max_count = max_diff_count(new_support)
                
                if new_max_count < current_max_count:
                    support = new_support
                    current_max_count = new_max_count
                    sup_set.remove(old_val)
                    sup_set.add(new_val)
                    improved = True
                    break
                elif new_max_count == current_max_count and np.random.rand() < 0.05:
                    support = new_support
                    current_max_count = new_max_count
                    sup_set.remove(old_val)
                    sup_set.add(new_val)
                    improved = True
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0
        
        if current_max_count < best_score:
            best_score = current_max_count
            best_support = support.copy()
            
            # If we found a perfect Sidon set (max count 1), we can stop
            if best_score == 1:
                break

    # Construct the function
    x = np.zeros(N)
    x[best_support] = 1.0 / np.sqrt(M)
    
    return x
