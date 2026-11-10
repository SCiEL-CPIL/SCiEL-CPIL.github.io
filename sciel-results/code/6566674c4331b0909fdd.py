# EVOLVE-BLOCK-START
import numpy as np
import time


def _keich_positions_1d(N):
    """Keich binary construction for 1D positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _swept_1d_len(x, N, n_samples=4096):
    """Compute the 1D swept union length for base positions x (shape N).
    
    Tube i sweeps interval [x_i + c*i/N, x_i + (i+1)/N + c*i/N] for c in [0,1].
    The union over c of these intervals is what we measure.
    """
    # For each i, the interval at height c is [x_i + c*i/N, x_i + (i+1)/N + c*i/N]
    # As c goes from 0 to 1, the left endpoint goes from x_i to x_i + i/N
    # and the right endpoint goes from x_i + (i+1)/N to x_i + (i+1)/N + i/N
    # So the full swept region for tube i is [x_i, x_i + (i+1)/N + i/N]
    # But we need the union over ALL tubes and ALL c.
    # 
    # Better approach: sample c values and compute union of intervals at each c.
    # The 1D union length at height c is the measure of union of intervals
    # [x_i + c*i/N, x_i + (i+1)/N + c*i/N] for i=0..N-1.
    # The 3D volume contribution in x is the integral over c of this 1D union length.
    # By Fubini, the 3D volume = (integral over c of 1D_x_union(c)) * (integral over c of 1D_y_union(c))
    # Wait, that's not quite right. Let me think more carefully.
    #
    # Actually, the 3D union volume is NOT simply a product. The union of products is not
    # the product of unions. However, if we can make the x-union and y-union independent...
    #
    # Actually, for the Monte Carlo estimate: a point (a,b,c) is in the union if there exists
    # (i,j) such that a is in [x_i + c*i/N, x_i + (i+1)/N + c*i/N] AND b is in [y_j + c*j/N, y_j + (j+1)/N + c*j/N].
    # This is NOT separable in general.
    #
    # But if base_x depends only on i and base_y depends only on j, then:
    # P(point in union) = P(exists (i,j): a in I_i(c) AND b in J_j(c))
    # where I_i(c) = [x_i + c*i/N, x_i + (i+1)/N + c*i/N] and J_j(c) similarly.
    #
    # This is still not simply a product. However, we can compute the 2D cross-section area
    # at each c: Area(c) = measure of (I_union(c) x J_union(c)) where I_union(c) is the union
    # of all I_i(c) intervals and J_union(c) is the union of all J_j(c) intervals.
    # Wait no! The union of (I_i x J_j) over all (i,j) is I_union x J_union only if every
    # (i,j) pair is present, which it is! Since we have all 64 tubes, the union at height c
    # is indeed I_union(c) x J_union(c).
    #
    # So Volume = integral_0^1 |I_union(c)| * |J_union(c)| dc
    #
    # This IS separable in the sense that if we use the same construction for x and y,
    # Volume = (integral_0^1 |I_union(c)| dc)^2
    #
    # So we just need to minimize the 1D "swept measure" S = integral_0^1 |union of intervals| dc.
    
    # Sample c values
    c_samples = np.linspace(0, 1, n_samples, endpoint=False) + 0.5 / n_samples
    c_samples = c_samples[:n_samples]
    
    total_len = 0.0
    for c in c_samples:
        # Intervals: [x_i + c*i/N, x_i + (i+1)/N + c*i/N] for i=0..N-1
        lefts = x + c * np.arange(N) / N
        rights = x + (np.arange(N) + 1) / N + c * np.arange(N) / N
        # Compute union length of these intervals
        order = np.argsort(lefts)
        l = lefts[order]
        r = rights[order]
        # Merge intervals
        merged_len = 0.0
        cur_l, cur_r = l[0], r[0]
        for k in range(1, N):
            if l[k] > cur_r:
                merged_len += cur_r - cur_l
                cur_l, cur_r = l[k], r[k]
            else:
                cur_r = max(cur_r, r[k])
        merged_len += cur_r - cur_l
        total_len += merged_len
    
    return total_len / n_samples


def _local_search_1d(x0, N, max_iters=2000, step=0.01):
    """Local search to minimize the 1D swept measure."""
    x = x0.copy()
    best_len = _swept_1d_len(x, N)
    
    for it in range(max_iters):
        # Try moving one coordinate
        i = it % N
        for delta in [step, -step, 2*step, -2*step]:
            x_new = x.copy()
            x_new[i] = x[i] + delta
            L = _swept_1d_len(x_new, N)
            if L < best_len - 1e-12:
                x = x_new
                best_len = L
                break
    
    return x, best_len


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    N = 8
    t_start = time.time()
    deadline = t_start + 580  # safety margin
    
    # Start with Keich construction
    x_1d = _keich_positions_1d(N)
    
    # Local search refinement
    x_opt, L_opt = _local_search_1d(x_1d, N, max_iters=3000, step=0.005)
    
    elapsed = time.time() - t_start
    if elapsed > deadline - t_start:
        x_opt = x_1d  # fall back to Keich if over time
    
    # Build 64 positions: row k = (i, j) with i = k//8, j = k%8
    # base_x depends only on i, base_y depends only on j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = x_opt[j_flat]  # same construction for y
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
