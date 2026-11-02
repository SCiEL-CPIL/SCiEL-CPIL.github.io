import numpy as np
import time

N = 600

def compute_score(h, exclude_lag0=True):
    """
    Compute the normalized max autoconvolution.
    If exclude_lag0 is True, max is over k >= 1.
    Normalization: (sum h)^2.
    """
    S_h = np.sum(h)
    if S_h < 1e-6:
        return np.inf
    
    # Use FFT for speed, though N=600 is small enough for direct correlate.
    # np.correlate is O(N^2) = 360k ops, very fast.
    # FFT is O(N log N), also fast. Let's stick to correlate for simplicity and precision.
    S = np.correlate(h, h, mode='full')
    
    # S is length 2N-1. Center is N-1 (lag 0).
    if exclude_lag0:
        # Max of S[0:N-1] and S[N:2N-1]
        # Due to symmetry S[k] = S[-k], max(S[0:N-1]) == max(S[N:2N-1])
        # We can just check one side to save half the time, but max is cheap.
        max_side = np.max(np.concatenate([S[:N-1], S[N:2*N-1]]))
    else:
        max_side = np.max(S)
    
    return max_side / (S_h ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = np.inf
    best_h = np.ones(N) / N  # Uniform baseline
    
    # 1. Evaluate Uniform
    s = compute_score(best_h, exclude_lag0=True)
    if s < best_score:
        best_score = s
        best_h = best_h.copy()
        
    # 2. Analytic Two-Block Optimization
    # Model: h(x) = A1 * rect(x - c1, w1) + A2 * rect(x - c2, w2)
    # We restrict to symmetric configurations around 0 to simplify.
    # h(x) = A1 * rect(x + D, W) + A2 * rect(x - D, W)
    # where rect(x, W) is 1 on [-W/2, W/2] and 0 otherwise.
    # In discrete indices [0, N-1] representing [-1/4, 1/4]:
    # Center is (N-1)/2.
    # We will iterate over:
    # - W (width in intervals)
    # - D (separation from center in intervals)
    # - R (amplitude ratio A2/A1). We can fix A1=1 and vary A2.
    #   Actually, since the objective is homogeneous of degree 0 in h (scaling h by k doesn't change score),
    #   we can fix the sum or one amplitude. Let's fix A1=1 and vary A2 in a range.
    #   Wait, if A2 is negative? No, h must be nonnegative. So A1, A2 >= 0.
    #   
    #   Let's analyze the autoconvolution of two blocks.
    #   h = h1 + h2.
    #   h*h = h1*h1 + h2*h2 + 2*h1*h2.
    #   h1*h1 is a triangle of height W (if amplitude 1) centered at 0.
    #   h2*h2 is a triangle of height W*A2^2 centered at 0.
    #   h1*h2 is the cross-correlation. It consists of two triangles shifted by +/- 2D.
    #   The peak of the autoconvolution usually occurs at lag 0 (sum of self-corrs) or at lags +/- 2D (cross-corrs).
    #   At lag 0: (W * A1^2 + W * A2^2) / (W(A1+A2))^2 = (A1^2 + A2^2) / (A1+A2)^2.
    #   At lag 2D (approx, if 2D < W): The cross term adds.
    #   We want to minimize the max of these peaks.
    
    # Let's do a grid search over W, D, and A2 (with A1=1).
    # W: 10 to N//2
    # D: W//2 + 1 to N//2 - W//2 (ensure disjoint)
    # A2: 0.1 to 2.0
    
    # Optimization: 
    # For fixed W, D, the shape is determined by A2.
    # We can compute the score for a range of A2.
    
    # To speed up, we can precompute the convolution structure.
    # But N=600 is small. Let's just loop.
    
    # Ranges:
    Ws = np.arange(20, N//2, 5)
    # D must be large enough for disjointness: 2*D > W. So D > W/2.
    # And blocks must fit in [0, N-1].
    # Center is N/2. Block 1: [N/2 - D - W/2, N/2 - D + W/2]
    # Block 2: [N/2 + D - W/2, N/2 + D + W/2]
    # Min index >= 0 => N/2 - D - W/2 >= 0 => D <= N/2 - W/2
    # Max index < N => N/2 + D + W/2 < N => D < N/2 - W/2
    
    A2s = np.linspace(0.0, 1.0, 21) # A1=1, A2 in [0,1]. Symmetry allows A2>1 to be covered by swapping.
    
    # We will iterate. To respect time, we limit the grid.
    # 20 Ws * 20 Ds * 20 A2s = 8000 evaluations.
    # Each eval is ~1ms. 8s total. Safe.
    
    for W in Ws:
        if time.time() > deadline - 10:
            break
        # D range
        D_min = int(W / 2) + 1
        D_max = int(N / 2) - int(W / 2) - 1
        if D_min >= D_max:
            continue
            
        # Sample D values
        Ds = np.linspace(D_min, D_max, 10).astype(int)
        
        for D in Ds:
            if time.time() > deadline - 5:
                break
                
            # Construct base h with A1=1, A2=0 to get indices
            h_base = np.zeros(N)
            # Center is (N-1)/2. Let's use integer centers for simplicity.
            # Actually, let's just place blocks.
            # Block 1 center: N/2 - D
            # Block 2 center: N/2 + D
            c1 = int(N/2 - D)
            c2 = int(N/2 + D)
            
            # Ensure blocks are within bounds
            start1 = c1 - W//2
            end1 = c1 + W//2 + 1
            start2 = c2 - W//2
            end2 = c2 + W//2 + 1
            
            if start1 < 0 or end2 > N:
                continue
            
            h_base[start1:end1] = 1.0
            h_base[start2:end2] = 1.0
            
            # Now vary A2.
            # h = h_base * (1 for block 1, A2 for block 2)
            # We can create a mask for block 2.
            mask2 = np.zeros(N)
            mask2[start2:end2] = 1.0
            
            # h(A2) = h_base + (A2 - 1) * mask2 ? No.
            # h_base has 1s in both blocks.
            # We want 1 in block 1, A2 in block 2.
            # h = h_base + (A2 - 1) * mask2 is correct if h_base is 1 in both.
            # Wait, h_base is 1 in block 1 and 1 in block 2.
            # h = 1*mask1 + A2*mask2.
            # h_base = mask1 + mask2.
            # h = h_base - mask2 + A2*mask2 = h_base + (A2-1)*mask2.
            
            for A2 in A2s:
                h_curr = h_base + (A2 - 1.0) * mask2
                # Ensure non-negative (A2>=0, so ok)
                s = compute_score(h_curr, exclude_lag0=True)
                if s < best_score:
                    best_score = s
                    best_h = h_curr.copy()

    # 3. Local Refinement (Optional)
    # If we found a good 2-block structure, we can try to refine the continuous positions/widths slightly?
    # The grid is coarse. A local search on the discrete indices of the best block found could help.
    # But the analytic structure suggests the optimum is likely well-captured by the grid if we dense enough.
    # Let's do a quick local search on the best_h if it's not uniform.
    # We will try shifting the blocks by 1 pixel and changing width by 1 pixel.
    
    if not np.allclose(best_h, np.ones(N)/N):
        # Identify the two blocks in best_h
        # Simple heuristic: find contiguous non-zero regions
        # For a 2-block function, this is easy.
        non_zero_idx = np.where(best_h > 1e-6)[0]
        if len(non_zero_idx) > 0:
            # Find gaps
            diffs = np.diff(non_zero_idx)
            gaps = np.where(diffs > 1)[0]
            
            if len(gaps) == 1:
                # Two blocks
                split_idx = gaps[0]
                block1_idx = non_zero_idx[:split_idx+1]
                block2_idx = non_zero_idx[split_idx+1:]
                
                # Get properties
                b1_start, b1_end = block1_idx[0], block1_idx[-1]
                b2_start, b2_end = block2_idx[0], block2_idx[-1]
                
                W1 = b1_end - b1_start + 1
                W2 = b2_end - b2_start + 1
                A1 = best_h[b1_start]
                A2 = best_h[b2_start]
                
                # Refine: try shifting b1 and b2 by +/- 1, +/- 2
                for shift1 in [-2, -1, 0, 1, 2]:
                    for shift2 in [-2, -1, 0, 1, 2]:
                        h_test = np.zeros(N)
                        ns1 = b1_start + shift1
                        ne1 = b1_end + shift1
                        ns2 = b2_start + shift2
                        ne2 = b2_end + shift2
                        
                        if ns1 < 0 or ne1 >= N or ns2 < 0 or ne2 >= N:
                            continue
                        if ns1 > ne1 or ns2 > ne2:
                            continue
                        
                        h_test[ns1:ne1+1] = A1
                        h_test[ns2:ne2+1] = A2
                        
                        s = compute_score(h_test, exclude_lag0=True)
                        if s < best_score:
                            best_score = s
                            best_h = h_test.copy()

    return list(best_h)
