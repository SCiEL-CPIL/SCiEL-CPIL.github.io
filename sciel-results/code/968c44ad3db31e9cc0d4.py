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
    
    # Run a few steps of local search to refine
    h_ls = h_bin.copy()
    cur = _max_overlap(h_ls)
    
    # Simple hill climbing / local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(h_ls == 1.0)[0]
        zeros = np.where(h_ls == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        # Try a few random swaps
        for _ in range(100):
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
            newc = _max_overlap(h_ls)
            if newc < cur:
                cur = newc
                improved = True
                # Update lists? For simplicity, recompute or just break and restart loop
                # Recomputing is O(N). Breaking is safer.
                break
            else:
                h_ls[i], h_ls[j] = h_ls[j], h_ls[i]
        
    return h_ls, cur

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    bestc = np.inf
    deadline = t0 + DEADLINE

    # Phase 1: Gradient Descent on Soft-Min Surrogate
    # Start from diverse initial points
    starts = []
    alt = np.zeros(N); alt[::2] = 1.0
    starts.append(alt)
    blk = np.zeros(N); blk[:HALF] = 1.0
    starts.append(blk)
    for _ in range(5):
        starts.append(_random_bitseq(rng))

    # Allocate time for GD
    gd_deadline = t0 + (DEADLINE * 0.5)
    
    for s in starts:
        if time.time() > gd_deadline:
            break
        
        # Start from binary, but allow continuous optimization
        h0 = s.astype(float)
        
        # Constraints: sum(h) = HALF, 0 <= h <= 1
        # L-BFGS-B supports bounds and linear constraints
        bounds = [(0, 1)] * N
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF, 'jac': lambda x: np.ones(N)}]
        
        try:
            res = minimize(
                _soft_min_surrogate,
                h0,
                method='L-BFGS-B',
                jac=_grad_soft_min_surrogate,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            h_gd = res.x
            # Enforce exact sum and bounds
            h_gd = np.clip(h_gd, 0, 1)
            # Normalize sum
            s_sum = np.sum(h_gd)
            if s_sum > 0:
                h_gd = h_gd * (HALF / s_sum)
            h_gd = np.clip(h_gd, 0, 1)
            
            # Evaluate true objective
            c_gd = _max_overlap(h_gd)
            
            # Refine via quantization and discrete search
            h_ref, c_ref = _quantize_and_refine(h_gd, rng, deadline - 5.0)
            
            if c_ref < bestc:
                bestc = c_ref
                best = h_ref
            elif c_gd < bestc:
                # Sometimes continuous is better? Unlikely for this problem, but check.
                # The task asks for step heights in [0,1]. Binary is a subset.
                # But if continuous gives better score, we should keep it.
                # However, the evaluator might prefer simple structures?
                # "score is the upper bound on C5".
                # Let's keep the best of both.
                if c_gd < bestc:
                    bestc = c_gd
                    best = h_gd
                    
        except Exception as e:
            pass

    # Phase 2: If GD didn't find anything good, fallback to SA/LS from current program logic
    if best is None or bestc > 0.95:
        # Run a quick SA on random starts
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            s = _random_bitseq(rng)
            h_sa = s.copy()
            cur = _max_overlap(h_sa)
            
            # Simple SA
            T = 1.0
            for it in range(5000):
                if time.time() > deadline - 5.0:
                    break
                ones = np.where(h_sa == 1.0)[0]
                zeros = np.where(h_sa == 0.0)[0]
                if len(ones) == 0 or len(zeros) == 0:
                    break
                i = ones[rng.integers(len(ones))]
                j = zeros[rng.integers(len(zeros))]
                h_sa[i], h_sa[j] = h_sa[j], h_sa[i]
                newc = _max_overlap(h_sa)
                if newc < cur or rng.random() < np.exp(-(newc - cur) / T):
                    cur = newc
                else:
                    h_sa[i], h_sa[j] = h_sa[j], h_sa[i]
                T *= 0.995
                
            if cur < bestc:
                bestc = cur
                best = h_sa

    # Final Polish: Local Search on the best candidate
    if best is not None:
        h_final = best.copy()
        cur_final = _max_overlap(h_final)
        
        # Exhaustive single-swap local search
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            ones = np.where(h_final == 1.0)[0]
            zeros = np.where(h_final == 0.0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            for i in ones:
                if time.time() > deadline - 1.0:
                    break
                for j in zeros:
                    h_final[i], h_final[j] = h_final[j], h_final[i]
                    newc = _max_overlap(h_final)
                    if newc < cur_final:
                        cur_final = newc
                        improved = True
                        break
                    else:
                        h_final[i], h_final[j] = h_final[j], h_final[i]
                if improved:
                    break
        
        if cur_final < bestc:
            bestc = cur_final
            best = h_final

    if best is None:
        best = np.full(N, 0.5)

    return best
