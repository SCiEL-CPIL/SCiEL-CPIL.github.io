import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 580  # Leave margin for the 600s limit

    # Precompute shifts for i and j
    shifts = np.arange(N) * frac  # [0, 1/8, ..., 7/8]

    # Keich construction for initial guess
    def keich_positions(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    u_init = keich_positions(N)
    v_init = keich_positions(N)

    # We optimize 14 variables: u_1..u_7, v_1..v_7 (u_0, v_0 fixed at 0)
    # This fixes the "center of mass" or one point to remove translational invariance
    # and keeps the box size reasonable.

    # Smooth union length approximation using soft-max
    # L(base, c) = sum_k (end_k - start_k) * P(k is "active" and not covered)
    # A simpler smooth approximation for the measure of union of intervals:
    # Measure ≈ sum_k len_k * sigmoid(beta * (len_k - overlap_k)) ? 
    # Actually, a robust differentiable approximation for union length of small N:
    # L ≈ max(end) - min(start) - sum_{gaps} max(0, gap)
    # But max and min are non-smooth.
    # Let's use the "soft indicator" approach.
    # The union of intervals I_k = [s_k, e_k].
    # The measure is the integral of 1_{x in U} dx.
    # 1_{x in U} = 1 - prod_k (1 - 1_{x in I_k}).
    # This is hard to integrate analytically smoothly.
    
    # Alternative: Since N=8 is small, we can use a "soft merge" logic.
    # However, a very effective smooth approximation for union length is:
    # L ≈ sum_k (e_k - s_k) * w_k
    # where w_k is a weight that is 1 if the interval is "unique" and less if it's covered.
    
    # Let's try a different smooth formulation:
    # L = max_k e_k - min_k s_k - sum_{pairs} overlap? No.
    
    # Let's use the fact that for intervals of equal length, the union length is:
    # L = max(e) - min(s) - sum_{i<j} max(0, min(e_i, e_j) - max(s_i, s_j)) * correction?
    # No, inclusion-exclusion is complex.
    
    # Let's use a "Soft Max" approximation of the sorted endpoints.
    # If we sort starts s_(1)..s_(N) and ends e_(1)..e_(N), the union length is:
    # L = sum_{k=1}^N max(0, e_(k) - max(s_(k), e_(k-1)))
    # This is still non-smooth due to sorting and max.
    
    # Let's use a high-temperature limit of a partition function?
    # Or simply: L ≈ (1/beta) * log( sum_k exp(beta * (e_k - s_k)) ) ? No, that's max length.
    
    # Let's try the "Soft Coverage" model.
    # For each interval k, define its "contribution" as the length of the part not covered by others.
    # This is hard.
    
    # Let's go back to the finite difference idea but make it faster?
    # No, the directive asks for a structural change: "Represent the 1D interval union length as a smooth, differentiable function".
    
    # A known smooth approximation for the union of intervals is based on the "max" operation.
    # L = max(e) - min(s) - sum_{i} max(0, s_{(i+1)} - e_{(i)})
    # The gaps are the non-overlapping parts.
    # Gap_i = max(0, s_{(i+1)} - e_{(i)})
    # We can approximate max(0, x) with softplus(x, beta).
    # And we can approximate sorting with a "soft sort" or just assume the order is preserved?
    # In the Keich construction, the order of starts might change with c.
    
    # Let's use a "Soft Sort" or just rely on the fact that for small N, we can use a
    # permutation-invariant smooth function?
    # Actually, there is a simpler smooth approximation:
    # L ≈ sum_k (e_k - s_k) * sigmoid(beta * (e_k - max_{j != k} min(e_j, e_k) + min(s_j, s_k))) ?
    # This is getting too complicated and error-prone.
    
    # Let's use the "Gaussian Blur" approximation of the indicator function.
    # L ≈ sum_k (e_k - s_k) * (1 - exp(-beta * (e_k - s_k))) ? No.
    
    # Let's try this:
    # The union length is the integral of the "coverage" function C(x) = 1 - prod_k (1 - I_k(x)).
    # If we approximate I_k(x) with a smooth step function, we can integrate analytically?
    # I_k(x) ≈ 0.5 * (tanh(beta*(x-s_k)) - tanh(beta*(x-e_k)))
    # Then C(x) is a product of tanh terms. Integrating this analytically is hard.
    
    # Let's use a numerical quadrature inside the objective, but make the *inner* function smooth.
    # If we use a smooth approximation for the union length, we can differentiate through it.
    
    # Let's use the "Soft Max" of the endpoints to define the "effective" union.
    # L ≈ soft_max(e, beta) - soft_min(s, beta) - sum_{i} soft_gap(i, beta)
    # This is still hard because of the sorting.
    
    # Let's try a different structural change:
    # Instead of optimizing positions to minimize the *exact* volume, we optimize to minimize a
    # *relaxed* volume that is strictly convex and smooth, and then snap to the exact optimal
    # positions for that relaxed problem? No, the relaxed problem's optimum might be far.
    
    # Let's use the "Log-Sum-Exp" approximation for the max in the merge algorithm.
    # We can define the union length as:
    # L = max(e) - min(s) - sum_{gaps} gap
    # Where gaps are determined by the sorted intervals.
    # If we assume the order of intervals by start position is fixed (which is true for small perturbations
    # around the Keich solution), we can compute the union length as a smooth function of the parameters.
    # We can detect the order at the current point and use that fixed order for the gradient calculation.
    # This is a "local linearization" of the combinatorial structure.
    
    def compute_union_length_smooth(base, c, beta=100.0):
        # base: (N,)
        # c: scalar
        starts = base + c * shifts
        ends = starts + frac
        
        # Sort by starts
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Merge with soft gaps
        # L = e[0] - s[0] + sum_{k=1}^{N-1} soft_gap(k)
        # soft_gap(k) = max(0, s[k] - max(e[0..k-1]))
        # Let M_k = max(e[0..k-1])
        # gap_k = max(0, s[k] - M_k)
        # L = e[0] - s[0] + sum_{k=1}^{N-1} gap_k
        
        # To make this smooth, we approximate max(0, x) with softplus(x, beta) = log(1 + exp(beta*x))/beta
        # And we approximate max(e[0..k-1]) with a running max? 
        # Running max is non-smooth.
        
        # Let's just use the exact union length but provide a subgradient?
        # No, L-BFGS-B needs smooth.
        
        # Let's use the "Soft Merge" where we blend the ends.
        # This is a heuristic.
        
        # Actually, let's use a different approach:
        # The volume is the integral of L_x(c) * L_y(c).
        # L_x(c) is the union length of 8 intervals.
        # We can approximate L_x(c) by the length of the convex hull of the intervals? No.
        
        # Let's use the "Average Coverage" approximation.
        # L ≈ sum_k (e_k - s_k) * P(k is not fully covered)
        # P(k is not fully covered) ≈ 1 - prod_{j != k} (1 - overlap(k, j) / len_k)
        # This is a smooth function!
        
        # overlap(k, j) = max(0, min(e_k, e_j) - max(s_k, s_j))
        # Let's implement this.
        
        # s: (N,), e: (N,)
        # Overlaps matrix O_{kj} = max(0, min(e_k, e_j) - max(s_k, s_j))
        # This is non-smooth due to max/min.
        # We can smooth max/min with softplus/softmin.
        
        def soft_max(x, y, beta):
            # approx max(x, y)
            return logsumexp(beta * np.array([x, y])) / beta
            
        def soft_min(x, y, beta):
            # approx min(x, y)
            return -logsumexp(-beta * np.array([x, y])) / beta
            
        # This will be slow if not vectorized.
        # N=8, so 64 pairs.
        
        s_mat = np.repeat(s[:, None], N, axis=1)
        e_mat = np.repeat(e[:, None], N, axis=1)
        
        # min(e_k, e_j)
        min_e = np.minimum(e_mat, e_mat.T)
        # max(s_k, s_j)
        max_s = np.maximum(s_mat, s_mat.T)
        
        # raw_overlap = min_e - max_s
        raw_overlap = min_e - max_s
        
        # smooth overlap = softplus(raw_overlap, beta)
        # softplus(x) = log(1 + exp(x)) / beta
        # For large beta, this approximates max(0, x)
        
        # We need to be careful with numerical stability in logsumexp or softplus.
        # scipy.special.softplus is available? No, use log1p(exp(x)).
        # For large x, log1p(exp(x)) ~ x.
        
        beta = 50.0
        # smooth_overlap = np.where(raw_overlap > 0, raw_overlap, np.log1p(np.exp(beta * raw_overlap)) / beta)
        # Vectorized softplus:
        # x = raw_overlap
        # softplus(x) = max(x, 0) + log1p(exp(-abs(x))) / beta ? No.
        # Standard softplus: log(1 + exp(x))
        # We want max(0, x) approximation.
        # f(x) = log(1 + exp(beta*x)) / beta
        
        # To avoid overflow in exp(beta*x):
        # If x > 0, f(x) ~ x.
        # If x < 0, f(x) ~ exp(beta*x)/beta.
        
        # Let's use np.logaddexp(0, beta * raw_overlap) / beta
        
        smooth_overlap = np.logaddexp(0, beta * raw_overlap) / beta
        
        # len_k = e_k - s_k = frac (constant)
        # P(k not fully covered) = 1 - prod_{j != k} (1 - overlap(k, j) / frac)
        # Let r_{kj} = overlap(k, j) / frac
        # We want prod_{j != k} (1 - r_{kj})
        
        # This product is over j != k.
        # We can compute the product over all j and divide by (1 - r_{kk})?
        # r_{kk} = 1, so 1 - r_{kk} = 0. This is a problem.
        # We should exclude k from the product.
        
        # Let's compute the product for each k.
        # prod_k = prod_{j != k} (1 - r_{kj})
        
        # r matrix: (N, N)
        r = smooth_overlap / frac
        
        # Set diagonal to 0 to ignore self-overlap in the "coverage" logic?
        # No, the formula is:
        # Contribution of k is len_k * (1 - P(k is covered))
        # P(k is covered) is NOT simply 1 - prod(1 - r).
        # If k is covered by the union of others, its contribution is 0.
        # This approximation is flawed.
        
        # Let's go back to the "Fixed Order" assumption.
        # For the Keich construction, the order of intervals by start position is stable
        # for small perturbations. We can compute the exact union length assuming the
        # order is fixed, and provide the analytic gradient for that fixed order.
        # This is a valid local optimization strategy.
        
        # If the order changes, the gradient is wrong, but L-BFGS-B can handle it
        # by taking small steps or restarting.
        
        # Let's implement the "Fixed Order" smooth union length.
        
        # L = e[0] - s[0] + sum_{k=1}^{N-1} max(0, s[k] - M_{k-1})
        # where M_{k-1} = max(e[0..k-1])
        
        # Since the order is fixed, M_{k-1} is a smooth function of the parameters?
        # No, max is non-smooth.
        # But we can use soft_max for M_{k-1}.
        
        # M_0 = e[0]
        # M_k = soft_max(M_{k-1}, e[k], beta)
        
        # gap_k = soft_plus(s[k] - M_{k-1}, beta)
        
        # L = e[0] - s[0] + sum_{k=1}^{N-1} gap_k
        
        # This is smooth!
        
        M = np.zeros(N)
        M[0] = e[0]
        for k in range(1, N):
            # M[k] = max(M[k-1], e[k])
            # Using soft_max
            M[k] = logsumexp(beta * np.array([M[k-1], e[k]])) / beta
            
        gaps = np.zeros(N)
        for k in range(1, N):
            # gap_k = max(0, s[k] - M[k-1])
            # Using soft_plus
            x = s[k] - M[k-1]
            gaps[k] = np.logaddexp(0, beta * x) / beta
            
        L = e[0] - s[0] + np.sum(gaps)
        return L

    def volume_objective(params):
        u = np.zeros(N)
        v = np.zeros(N)
        u[1:] = params[:7]
        v[1:] = params[7:]
        
        # Integrate over c
        # Use Gauss-Legendre quadrature
        c_points, weights = np.polynomial.legendre.leggauss(50)
        c_vals = 0.5 * (c_points + 1.0)
        
        vol = 0.0
        for c, w in zip(c_vals, weights):
            lx = compute_union_length_smooth(u, c)
            ly = compute_union_length_smooth(v, c)
            vol += w * lx * ly
        return 0.5 * vol

    # Initial guess: Keich
    u0 = np.zeros(N)
    v0 = np.zeros(N)
    u0[1:] = u_init[1:]
    v0[1:] = v_init[1:]
    x0 = np.concatenate([u0[1:], v0[1:]])
    
    # Bounds
    bounds = [(-2.0, 2.0)] * 14
    
    best_x = x0
    best_f = volume_objective(x0)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            volume_objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception as e:
        pass
    
    # If time remains, try a second phase
    if time.time() - start_time < timeout:
        try:
            res_nm = minimize(
                volume_objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_f:
                best_f = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass

    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() - start_time > timeout - 60:
            break
        x_rand = x0 + rng.normal(0, 0.1, size=14)
        x_rand = np.clip(x_rand, -2.0, 2.0)
        
        try:
            res_r = minimize(
                volume_objective, 
                x_rand, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res_r.fun < best_f:
                best_f = res_r.fun
                best_x = res_r.x
        except Exception:
            pass

    # Construct final positions
    u_final = np.zeros(N)
    v_final = np.zeros(N)
    u_final[1:] = best_x[:7]
    v_final[1:] = best_x[7:]
    
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = u_final[i]
        positions[k, 1] = v_final[j]
        
    return positions

if __name__ == "__main__":
    pos = kakeya_needle_3d8()
    print(pos.shape)
