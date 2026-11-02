import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 115.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _max_overlap_fast(h, r_cache=None):
    """Compute max overlap, optionally using cached autocorrelation."""
    if r_cache is not None:
        r = r_cache
    else:
        r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _update_autocorr_after_swap(r, h, i, j):
    """
    Update autocorrelation after swapping h[i] and h[j].
    Only shifts k where |i-j| mod N != k are affected.
    Actually, all shifts are potentially affected, but we can compute delta efficiently.
    
    For a swap of positions i and j, the change in autocorrelation at shift k is:
    delta_r[k] = (h[j]-h[i]) * (h[(i+k)%N] - h[(j+k)%N]) + (h[i]-h[j]) * (h[(i-k)%N] - h[(j-k)%N])
    Wait, this isn't quite right. Let me think more carefully.
    
    r[k] = sum_m h[m] * h[(m+k) % N]
    
    After swapping h[i] <-> h[j], let a = h[i], b = h[j].
    New h'[i] = b, h'[j] = a, others unchanged.
    
    r'[k] = r[k] - a*h[(i+k)%N] - b*h[(j+k)%N] + a*h[(j-k)%N] + b*h[(i-k)%N]
           Wait, this is getting complicated. Let me just recompute.
    """
    # For simplicity and correctness, just recompute. N=100 is small.
    return _autocorr(h)

def _soft_min_surrogate(h, beta=10.0):
    """
    Compute a smooth approximation of the maximum overlap.
    Overlap at shift k is S - autocorr[k].
    We want to minimize max_k (S - autocorr[k]).
    This is equivalent to maximizing min_k autocorr[k].
    
    Let y_k = autocorr[k] for k in 1..N-1.
    We want to maximize min(y).
    Soft min: (1/beta) * log(sum(exp(-beta * y_k))) is a soft MIN of y.
    Wait, soft min of y is (1/beta) log(sum(exp(-beta y_k))).
    We want to MINIMIZE the max overlap, which is S - min(y).
    So we want to MAXIMIZE min(y).
    Maximizing min(y) is equivalent to minimizing -min(y).
    So we minimize (1/beta) log(sum(exp(-beta * y_k)))? No.
    
    Let's check:
    min(y) approx (1/beta) log(sum(exp(-beta y_k))) is FALSE.
    Soft min of y is (1/beta) log(sum(exp(-beta y_k)))?
    If y = [1, 100], beta large.
    exp(-beta*1) is big, exp(-beta*100) is small.
    sum ~ exp(-beta). log ~ -beta. 1/beta * -beta = -1.
    So soft min approximates the minimum.
    
    We want to maximize min(y).
    So we want to minimize -min(y).
    Objective to minimize: - (1/beta) log(sum(exp(-beta y_k)))
    """
    r = _autocorr(h)
    y = r[1:] # Autocorrelation for shifts 1..N-1
    # To avoid overflow/underflow issues, shift y
    y_min = np.min(y)
    # Soft min of y is approx y_min - (1/beta) log(sum(exp(-beta(y - y_min))))
    # Actually, standard soft min: m_beta(y) = (1/beta) log(sum(exp(-beta y_i)))
    # This converges to min(y_i) as beta -> inf.
    # We want to maximize min(y), so minimize -m_beta(y).
    
    # Numerical stability:
    # log(sum(exp(-beta y))) = log(sum(exp(-beta (y - y_ref)))) - beta y_ref
    # Let y_ref = max(y) to keep exponents <= 0?
    # If we use y_ref = min(y), exponents are <= 0.
    # exp(-beta (y - y_min)) <= 1.
    # sum(exp(...)) >= 1.
    # log(sum) >= 0.
    # m_beta = y_min - (1/beta) log(sum(exp(-beta (y - y_min))))
    
    y_shifted = y - y_min
    exp_terms = np.exp(-beta * y_shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = y_min - log_sum / beta
    
    # We want to minimize the maximum overlap.
    # Max overlap = HALF - min(y).
    # So we want to minimize (HALF - soft_min_val) ?
    # Minimizing HALF - soft_min_val is same as maximizing soft_min_val.
    # So objective function for minimizer should be: -soft_min_val
    # Or just return the value we want to minimize: HALF - soft_min_val
    
    return HALF - soft_min_val

def _grad_soft_min_surrogate(h, beta=10.0):
    """
    Gradient of the soft min surrogate.
    f(h) = HALF - (1/beta) log(sum_k exp(-beta * y_k))
    where y_k = sum_i h_i h_{i+k}
    
    df/dh_j = - (1/beta) * (1/sum_k exp(-beta y_k)) * sum_k (-beta exp(-beta y_k) * dy_k/dh_j)
    df/dh_j = (sum_k exp(-beta y_k) * dy_k/dh_j) / sum_k exp(-beta y_k)
    
    dy_k/dh_j = h_{j+k} + h_{j-k} (circular)
    """
    r = _autocorr(h)
    y = r[1:]
    y_min = np.min(y)
    y_shifted = y - y_min
    exp_terms = np.exp(-beta * y_shifted)
    denom = np.sum(exp_terms)
    
    # We need dy_k/dh_j for all j.
    # dy_k/dh_j = h_{j+k} + h_{j-k}
    # This is a convolution-like operation.
    # Let's compute the gradient numerically or analytically via FFT?
    # Analytical is better for speed.
    # 
    # Let w_k = exp(-beta y_k) / denom. Note: w_k depends on h.
    # Actually, the formula derived above is for the derivative of the log-sum-exp part.
    # Let S = sum_k exp(-beta y_k).
    # d/dh_j [ (1/beta) log S ] = (1/beta) (1/S) sum_k (-beta exp(-beta y_k)) dy_k/dh_j
    # = - (1/S) sum_k exp(-beta y_k) dy_k/dh_j
    # 
    # We want to minimize HALF - soft_min.
    # soft_min = y_min - (1/beta) log(sum(exp(-beta(y-y_min))))
    # This is getting complicated with the y_min shift.
    # Let's stick to the unshifted version for gradient derivation, assuming stability.
    # Or just use numerical gradient? N=100, gradient is 100 points.
    # Numerical gradient is 2*100 evaluations. Each eval is FFT(100).
    # 200 FFTs per gradient step. L-BFGS might take 100 steps. 20,000 FFTs.
    # FFT(100) is ~microseconds. 20k FFTs is ~0.1s. Feasible.
    # But analytical is faster.
    
    # Let's try a simpler approach:
    # Minimize J(h) = (1/beta) log(sum_{k=1}^{N-1} exp(-beta * r_k))
    # where r_k is autocorrelation at shift k.
    # grad J = - (1/sum(exp(-beta r_k))) * sum_k exp(-beta r_k) * grad r_k
    # grad r_k wrt h_j = h_{j+k} + h_{j-k}
    
    # To compute sum_k exp(-beta r_k) * (h_{j+k} + h_{j-k}) for all j:
    # This is a convolution of w_k = exp(-beta r_k) with h.
    # Specifically, it's 2 * (w * h) at lag 0? No.
    # sum_k w_k h_{j+k} is (w * h)_j if w is symmetric?
    # w_k is defined for k=1..N-1. w_0 is not used in the sum.
    # Let's construct W as a vector of length N with W_k = exp(-beta r_k) for k != 0, and 0 for k=0?
    # No, the sum is over k=1..N-1.
    # Let's use FFT to compute the convolution.
    
    W = np.zeros(N)
    W[1:] = exp_terms # exp(-beta (y - y_min)) ... wait, the shift y_min affects the derivative?
    # The shift y_min is a constant with respect to h (locally, if min index doesn't change).
    # If we use the shifted version, the weights are exp(-beta (y_k - y_min)).
    # The derivative is proportional to these weights.
    
    # Convolution: C_j = sum_k W_k h_{j-k}
    # We need sum_k W_k (h_{j+k} + h_{j-k})
    # = sum_k W_k h_{j+k} + sum_k W_k h_{j-k}
    # Since W is symmetric (autocorrelation is symmetric, so y_k = y_{N-k}), W is symmetric.
    # sum_k W_k h_{j-k} is (W * h)_j.
    # sum_k W_k h_{j+k} is (W * h)_j as well if W is symmetric?
    # Let's check.
    # (W * h)_j = sum_k W_k h_{j-k}.
    # If W is symmetric, W_k = W_{-k}.
    # sum_k W_k h_{j+k} = sum_k W_{-k} h_{j+k} = sum_m W_m h_{j-m} (let m=-k).
    # Yes, they are the same.
    # So grad term is 2 * (W * h)_j / denom.
    
    # Compute W * h via FFT
    W_fft = np.fft.fft(W)
    H_fft = np.fft.fft(h)
    conv = np.fft.ifft(W_fft * H_fft).real
    
    grad = 2.0 * conv / denom
    # Objective is HALF - soft_min.
    # soft_min approx min(y).
    # We are minimizing HALF - min(y).
    # So we want to decrease this value.
    # The gradient of the objective J = HALF - (1/beta)log(...) 
    # dJ/dh = - d(soft_min)/dh.
    # My derivation above was for d(soft_min)/dh?
    # Let's re-evaluate.
    # J = HALF - m_beta(y).
    # m_beta(y) = (1/beta) log(sum exp(-beta y)). (Ignoring shift for gradient direction, it scales the gradient)
    # dm/dh = - (1/sum) sum exp(-beta y) dy/dh.
    # dJ/dh = - dm/dh = (1/sum) sum exp(-beta y) dy/dh.
    # dy/dh_j = h_{j+k} + h_{j-k}.
    # So grad J = (1/denom) * sum_k W_k (h_{j+k} + h_{j-k}) = 2 * conv / denom.
    
    return grad

def _random_bitseq(rng):
    x = rng.random(N)
    idx = np.argsort(x)
    seq = np.zeros(N)
    seq[idx[HALF:]] = 1.0
    return seq

def _quantize_and_refine(h_cont, rng, deadline):
    """
    Quantize continuous h to binary, then run local search.
    """
    # Quantize: if h > 0.5, set to 1, else 0.
    # Ensure exactly HALF ones.
    h_bin = (h_cont > 0.5).astype(float)
    sum_h = np.sum(h_bin)
    if sum_h != HALF:
        # Adjust to ensure exactly HALF ones
        # Sort by h_cont values, pick top HALF
        idx = np.argsort(h_cont)[::-1]
        h_bin = np.zeros(N)
        h_bin[idx[:HALF]] = 1.0
    
    # Run local search to refine
    h_ls = h_bin.copy()
    cur = _max_overlap(h_ls)
    
    # Effective local search with restarts
    improved = True
    stall_count = 0
    max_stall = 50
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(h_ls == 1.0)[0]
        zeros = np.where(h_ls == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        # Try multiple random swaps per iteration
        for _ in range(200):
            if time.time() > deadline:
                break
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
            newc = _max_overlap(h_ls)
            if newc < cur:
                cur = newc
                improved = True
                stall_count = 0
                break
            else:
                h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
        
        if not improved:
            stall_count += 1
            if stall_count >= max_stall:
                # Try a random perturbation
                for _ in range(10):
                    i = ones[rng.integers(len(ones))]
                    j = zeros[rng.integers(len(zeros))]
                    h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
                stall_count = 0
                improved = True  # Force another round
    
    return h_ls, cur

def _sa_with_ls(h_init, rng, deadline, t0_temp=1.0, t1_temp=0.01, ls_steps=1000):
    """
    Simulated annealing with local search refinement.
    """
    h = h_init.copy()
    cur = _max_overlap(h)
    best_h = h.copy()
    best_c = cur
    
    T = t0_temp
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    for it in range(ls_steps):
        if time.time() > deadline:
            break
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        h[i], h[j] = h[j], h[i]
        newc = _max_overlap(h)
        
        if newc < cur or (newc > cur and rng.random() < np.exp(-(newc - cur) / T)):
            cur = newc
            if cur < best_c:
                best_c = cur
                best_h = h.copy()
        else:
            h[i], h[j] = h[j], h[i]
        
        # Cool down
        T = max(t1_temp, T * 0.999)
    
    # Local search polish
    h_ls = best_h.copy()
    cur_ls = best_c
    improved = True
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(h_ls == 1.0)[0]
        zeros = np.where(h_ls == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        for _ in range(300):
            if time.time() > deadline:
                break
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
            newc = _max_overlap(h_ls)
            if newc < cur_ls:
                cur_ls = newc
                improved = True
                break
            else:
                h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
    
    return h_ls, cur_ls

def _exhaustive_2swap_ls(h, deadline):
    """
    Exhaustive 2-swap local search: try all pairs of (1->0) and (0->1) swaps.
    This is more thorough than single-swap.
    """
    cur = _max_overlap(h)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        for i in ones:
            if time.time() > deadline:
                break
            for j in zeros:
                if i == j:
                    continue
                h[i], h[j] = h[j], h[i]
                newc = _max_overlap(h)
                if newc < cur:
                    cur = newc
                    improved = True
                    break
                else:
                    h[i], h[j] = h[j], h[i]
            if improved:
                break
    
    return h, cur

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    best = None
    bestc = np.inf
    deadline = t0 + DEADLINE

    # Generate diverse initial solutions
    starts = []
    
    # Alternating pattern
    alt = np.zeros(N); alt[::2] = 1.0
    starts.append(alt)
    
    # Block pattern
    blk = np.zeros(N); blk[:HALF] = 1.0
    starts.append(blk)
    
    # Multiple random starts
    for _ in range(10):
        starts.append(_random_bitseq(rng))
    
    # Phase 1: Run SA + LS on each start
    for idx, s in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time based on remaining budget
        time_per_start = max(5.0, (deadline - time.time()) / max(1, len(starts) - idx))
        sa_deadline = min(deadline - 5.0, time.time() + time_per_start)
        
        h_sa, c_sa = _sa_with_ls(s, rng, sa_deadline, t0_temp=0.5, t1_temp=0.005, ls_steps=2000)
        
        if c_sa < bestc:
            bestc = c_sa
            best = h_sa

    # Phase 2: Gradient descent refinement on best solution
    if best is not None and time.time() < deadline - 20.0:
        h0 = best.astype(float)
        bounds = [(0, 1)] * N
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF, 'jac': lambda x: np.ones(N)}]
        
        try:
            gd_deadline = deadline - 10.0
            res = minimize(
                _soft_min_surrogate,
                h0,
                method='L-BFGS-B',
                jac=_grad_soft_min_surrogate,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 20}
            )
            h_gd = np.clip(res.x, 0, 1)
            s_sum = np.sum(h_gd)
            if s_sum > 0:
                h_gd = h_gd * (HALF / s_sum)
            h_gd = np.clip(h_gd, 0, 1)
            
            c_gd = _max_overlap(h_gd)
            
            # Quantize and refine
            h_ref, c_ref = _quantize_and_refine(h_gd, rng, deadline - 5.0)
            
            if c_ref < bestc:
                bestc = c_ref
                best = h_ref
        except Exception:
            pass

    # Phase 3: Final exhaustive 2-swap local search
    if best is not None and time.time() < deadline - 2.0:
        h_final, c_final = _exhaustive_2swap_ls(best.copy(), deadline - 1.0)
        if c_final < bestc:
            bestc = c_final
            best = h_final

    if best is None:
        best = np.full(N, 0.5)
        
    # Final safety check: Ensure sum is exactly HALF
    # If best is continuous, quantize it. If binary, verify sum.
    if np.sum(best) != HALF:
        # Quantize to binary with exactly HALF ones
        idx = np.argsort(best)[::-1]
        best = np.zeros(N)
        best[idx[:HALF]] = 1.0
        
    return best
