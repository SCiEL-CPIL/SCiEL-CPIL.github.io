# EVOLVE-BLOCK-START
import numpy as np
import time


def _eval_and_subgrad(x, n=64, M=20000):
    """
    Evaluate area and compute subgradient for the Kakeya needle problem.
    Returns (area, subgrad) where subgrad has shape (n,).
    
    At each height y_m, triangle i has left edge L_i = x_i + a_i * y_m, width w = (1-y_m)/n.
    Union length = w + sum over consecutive sorted gaps of min(gap, w).
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i
    w_base = 1.0 / n
    
    # y values
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    w = (1.0 - y) / n             # shape (M,)
    
    # L_i at each height: shape (M, n)
    # L[m, i] = x[i] + a[i] * y[m]
    L = x[None, :] + a[None, :] * y[:, None]  # (M, n)
    
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)  # (M, n)
    
    # Gaps between consecutive sorted elements
    gaps = np.diff(L_sorted, axis=1)  # (M, n-1)
    
    # min(gaps, w): w is (M,), broadcast to (M, n-1)
    w_b = w[:, None]  # (M, 1)
    clipped = np.minimum(gaps, w_b)  # (M, n-1)
    
    # Union length at each height
    union_len = w + clipped.sum(axis=1)  # (M,)
    
    # Area = mean of union lengths
    area = union_len.mean()
    
    # Subgradient computation:
    # d(union_len)/d(L_i) = 1 if i is the leftmost, 
    #   + 0.5 if the gap to next is < w (partially overlapping with next),
    #   - 0.5 if the gap to prev is < w (partially overlapping with prev),
    #   - 1 if i is the rightmost
    # More precisely: for sorted order, if gap_j = L_{j+1} - L_j < w, then
    #   d/dL_{sorted_j} gets -0.5 (from min(gap_j, w) derivative w.r.t. L_j)
    #   d/dL_{sorted_{j+1}} gets +0.5
    # Plus the constant w contributes 1/n to each (but w doesn't depend on x)
    # Actually w is independent of x, so only the clipped terms matter.
    # 
    # For each height m:
    #   subgrad[m, i] = sum over adjacent gaps involving i
    #   If i is at sorted position p:
    #     - if p > 0 and gap_{p-1} < w: subgrad -= 0.5
    #     - if p < n-1 and gap_p < w: subgrad += 0.5
    #   Then d(area)/d(x_i) = mean over m of subgrad[m, i] * a_i (chain rule: L_i = x_i + a_i*y)
    #   Wait, dL_i/dx_i = 1, so d(area)/d(x_i) = mean_m subgrad[m, i]
    
    # Compute subgradient at each height
    # For sorted positions, track which original index is at each sorted position
    # But we can work directly: for each gap_j (between sorted_j and sorted_{j+1}),
    # if gap_j < w, it contributes -0.5 to sorted_j and +0.5 to sorted_{j+1}
    
    # We need to map back to original indices. Use argsort.
    # This is expensive but necessary for correct subgradient.
    
    # Alternative: compute subgradient more directly.
    # For each height m, the subgradient w.r.t. L (unsorted) can be computed as:
    #   sg[m, i] = 0.5 * (I[gap_to_next_sorted < w] - I[gap_to_prev_sorted < w])
    # where gap_to_next is the gap from this element to the next in sorted order.
    
    # Let's compute using the sorted order and argsort
    # For efficiency, do this in a loop over heights? No, vectorize.
    
    # Actually, let's compute it properly:
    # sorted_order[m, p] = original index of the p-th smallest L at height m
    sorted_order = np.argsort(L, axis=1)  # (M, n)
    
    # Initialize subgradient at each height to zero
    sg = np.zeros_like(L)  # (M, n)
    
    # For each gap position j (0 to n-2), if gaps[m, j] < w[m]:
    #   sg[m, sorted_order[m, j]] -= 0.5
    #   sg[m, sorted_order[m, j+1]] += 0.5
    
    # Vectorized approach:
    # active = gaps < w_b  (M, n-1)
    active = gaps < w_b  # (M, n-1), boolean
    
    # For each active gap at position j, we need to add to two entries.
    # This is tricky to vectorize fully. Let's use a different approach.
    
    # Alternative: compute the subgradient as:
    # sg[m, i] = 0.5 * (count of active gaps where i is the right element 
    #                  - count of active gaps where i is the left element)
    # 
    # For sorted position p:
    #   - It's the "left" element of gap p (if p < n-1 and active[m,p])
    #   - It's the "right" element of gap p-1 (if p > 0 and active[m,p-1])
    # So sg for sorted position p = 0.5 * (active[m, p-1] - active[m, p])
    #   where active[m, -1] = 0 and active[m, n-1] = 0
    
    # Build active_with_padding: (M, n) where
    #   padded[m, 0] = 0
    #   padded[m, p] = active[m, p-1] for p = 1..n-1
    #   So padded[m, p] = active[m, p-1] for p>=1, and 0 for p=0
    # Similarly for the "right" side:
    #   right_padded[m, p] = active[m, p] for p < n-1, and 0 for p = n-1
    
    # sg_sorted[m, p] = 0.5 * (padded[m, p] - right_padded[m, p])
    
    # Let's build these:
    # left_contrib[m, p] = active[m, p-1] for p >= 1, else 0
    # right_contrib[m, p] = active[m, p] for p <= n-2, else 0
    
    # left_contrib: shift active right by 1, pad with 0 on left
    left_contrib = np.zeros_like(active)  # (M, n-1)
    # Actually, let me think differently.
    
    # For sorted position p (0-indexed, 0..n-1):
    #   It is the right endpoint of gap (p-1) if p >= 1
    #   It is the left endpoint of gap p if p <= n-2
    #   sg = 0.5 * (I[p>=1 and active[m,p-1]] - I[p<=n-2 and active[m,p]])
    
    # Let me create arrays of shape (M, n):
    # is_right_of_active_gap[m, p] = active[m, p-1] if p >= 1 else 0
    # is_left_of_active_gap[m, p] = active[m, p] if p <= n-2 else 0
    
    # Create is_right: (M, n)
    is_right = np.zeros((M, n), dtype=float)
    if n > 1:
        is_right[:, 1:] = active  # active is (M, n-1), assign to columns 1..n-1
    
    # Create is_left: (M, n)
    is_left = np.zeros((M, n), dtype=float)
    is_left[:, :n-1] = active  # assign to columns 0..n-2
    
    # sg_sorted[m, p] = 0.5 * (is_right[m, p] - is_left[m, p])
    sg_sorted = 0.5 * (is_right - is_left)  # (M, n)
    
    # Now map back to original indices:
    # sg[m, original_index] = sg_sorted[m, sorted_position]
    # i.e., sg[m, sorted_order[m, p]] = sg_sorted[m, p]
    
    # Use np.put_along_axis or advanced indexing
    # For each m, for each p: sg[m, sorted_order[m, p]] = sg_sorted[m, p]
    
    # Vectorized:
    # indices = np.arange(M)[:, None]  # (M, 1)
    # sg[m, sorted_order[m, p]] = sg_sorted[m, p]
    
    # This is: sg[arange(M)[:,None], sorted_order] = sg_sorted
    m_idx = np.arange(M)[:, None]  # (M, 1), broadcast to (M, n)
    np.put_along_axis(sg, sorted_order, sg_sorted, axis=1)
    
    # Now d(area)/d(x_i) = mean over m of sg[m, i]
    # Because area = (1/M) * sum_m union_len[m], and d(union_len[m])/d(x_i) = sg[m, i]
    subgrad = sg.mean(axis=0)  # (n,)
    
    return area, subgrad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses subgradient descent starting from the Keich construction.
    """
    n = 64
    M = 20000  # number of quadrature points
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 20 second safety margin
    
    # Initial solution: Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Binary expansion of i in k bits
        bits = [(i >> (k - 1 - jj)) & 1 for jj in range(k)]
        eps = np.array(bits, dtype=np.float64)
        x[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    # Evaluate initial
    best_x = x.copy()
    best_area, _ = _eval_and_subgrad(best_x, n, M)
    
    # Subgradient descent
    # Use diminishing step size
    lr_init = 0.01
    total_steps_budget = 5000
    
    step = 0
    while step < total_steps_budget:
        if time.time() > deadline:
            break
        
        area, sg = _eval_and_subgrad(x, n, M)
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        # Diminishing learning rate
        lr = lr_init / (1.0 + 0.001 * step)
        
        # Take step
        x = x - lr * sg
        
        # Ensure x stays in reasonable bounds (not strictly necessary but helps)
        # x should be in [-1, 1] roughly for the triangles to make sense
        
        step += 1
    
    # Return best found
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
