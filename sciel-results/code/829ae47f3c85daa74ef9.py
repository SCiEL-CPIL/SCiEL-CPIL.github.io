import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    """
    Minimizes the maximum overlap of h with shifted 1-h.
    
    Strategy:
    1. Formulate the problem as a smooth continuous optimization.
    2. Objective: Minimize a log-sum-exp approximation of the maximum overlap.
       Overlap(s) = N/2 - autocorr(h)[s].
       We minimize log(sum(exp(Overlap(s)/tau))).
    3. Constraints: h_i in [0, 1], sum(h) = N/2.
    4. Method: SLSQP with analytic gradient.
    5. Initialization: Multiple random starts to escape local minima.
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin for safety
    
    # Precompute indices for circular shifts
    # We need gradients of autocorrelation.
    # acorr[s] = sum_j h[j] h[(j+s)%N]
    # d acorr[s] / d h[i] = h[(i-s)%N] + h[(i+s)%N]
    
    def objective_and_grad(h, tau=0.05):
        """
        Computes the log-sum-exp of overlaps and its gradient.
        """
        h = np.asarray(h)
        # Compute circular autocorrelation using FFT
        Fh = np.fft.fft(h)
        S = Fh * np.conj(Fh)
        acorr = np.fft.ifft(S).real
        
        # Overlaps for shifts 1 to N-1
        # acorr[0] is the energy, we ignore shift 0.
        overlaps = N / 2.0 - acorr[1:N]
        
        # Log-sum-exp for numerical stability
        max_ov = np.max(overlaps)
        if max_ov > 0:
            # log(sum(exp((ov - max_ov)/tau))) + max_ov/tau
            # We want to minimize this.
            # Actually, standard log-sum-exp: log(sum(exp(x/tau))) = max(x)/tau + log(sum(exp((x-max(x))/tau)))
            # We can just use:
            # val = np.log(np.sum(np.exp(overlaps / tau)))
            # But this overflows.
            # Stable version:
            shifted = overlaps / tau
            max_shifted = np.max(shifted)
            val = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            
            # Gradient
            # d/dh_i log(sum_s exp(O_s/tau))
            # = 1/tau * sum_s exp(O_s/tau) / sum_s exp(O_s/tau) * dO_s/dh_i
            # = sum_s P_s * dO_s/dh_i  where P_s = exp(O_s/tau) / sum(exp(O_s/tau))
            
            weights = np.exp(shifted - max_shifted)
            weights /= np.sum(weights)
            
            # dO_s/dh_i = - (h[(i-s)%N] + h[(i+s)%N])
            # We need to sum over s of weights[s] * (h[(i-s)%N] + h[(i+s)%N])
            
            # Let's compute the sum efficiently.
            # Sum_s weights[s] h[(i-s)%N] is a circular convolution of weights and h?
            # Let w_s = weights[s-1] for s=1..N-1. w_0 = 0.
            # term1_i = sum_s w_s h[(i-s)%N]
            # This is convolution of w and h.
            # Using FFT:
            # W = FFT(w), H = FFT(h)
            # conv = ifft(W * H)
            
            w = np.zeros(N)
            w[1:N] = weights
            
            W = np.fft.fft(w)
            H = np.fft.fft(h)
            
            # term1 = ifft(W * H).real  -> sum_j w_j h[i-j]
            # term2 = ifft(W * conj(H) * ?) No.
            # term2_i = sum_s w_s h[(i+s)%N]
            # This is cross-correlation or convolution with flipped w?
            # sum_s w_s h[i+s]. Let k = -s. sum_k w_-k h[i-k].
            # This is convolution of w_flipped and h.
            # w_flipped[k] = w[-k % N].
            
            w_flip = np.zeros(N)
            w_flip[0] = 0
            w_flip[1:N] = weights[::-1] # w[s] goes to index N-s? 
            # Let's check indices carefully.
            # We want sum_{s=1}^{N-1} w_s h_{i+s}.
            # Let j = i+s => s = j-i.
            # sum_{j} w_{j-i} h_j.
            # This is convolution of w and h at index i?
            # (w * h)[i] = sum_j w_j h_{i-j}.
            # If we define w_rev[j] = w[-j % N], then (w_rev * h)[i] = sum_j w_rev[j] h_{i-j} = sum_j w[-j] h_{i-j}.
            # Let k = -j. sum_k w[k] h_{i+k}. Yes.
            
            # So term2 is convolution of w_rev and h.
            # w_rev[0] = 0.
            # w_rev[k] = w[-k % N].
            # w[1]=w1, ..., w[N-1]=w_{N-1}.
            # w_rev[1] = w[-1] = w[N-1].
            # w_rev[2] = w[-2] = w[N-2].
            # ...
            # w_rev[N-1] = w[-(N-1)] = w[1].
            
            w_rev = np.zeros(N)
            w_rev[1:] = weights[::-1]
            
            W_rev = np.fft.fft(w_rev)
            
            term1 = np.fft.ifft(W * H).real
            term2 = np.fft.ifft(W_rev * H).real
            
            # Gradient of objective w.r.t h_i
            # dObj/dh_i = - (term1[i] + term2[i])
            # Note: The factor 1/tau is absorbed?
            # val = log(sum(exp(O/tau)))
            # dval/dO_s = exp(O_s/tau) / sum(exp(O_s/tau)) = P_s
            # dval/dh_i = sum_s P_s dO_s/dh_i
            # dO_s/dh_i = - (h_{i-s} + h_{i+s})
            # So grad = - (term1 + term2)
            
            grad = -(term1 + term2)
            
        else:
            # If all overlaps are negative (unlikely given N/2 - acorr), 
            # or very small, handle gracefully.
            val = np.sum(overlaps)
            # Gradient is just sum of dO_s/dh_i
            # dO_s/dh_i = -(h_{i-s} + h_{i+s})
            # Sum_s dO_s/dh_i = - sum_s (h_{i-s} + h_{i+s})
            # sum_s h_{i-s} = sum(h) = N/2
            # sum_s h_{i+s} = sum(h) = N/2
            # So grad = - (N/2 + N/2) = -N
            # Wait, this is constant?
            # If overlaps are all negative, exp is small.
            # Let's just use the stable formula anyway.
            shifted = overlaps / tau
            max_shifted = np.max(shifted)
            val = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            
            weights = np.exp(shifted - max_shifted)
            weights /= np.sum(weights)
            
            w = np.zeros(N)
            w[1:N] = weights
            W = np.fft.fft(w)
            H = np.fft.fft(h)
            term1 = np.fft.ifft(W * H).real
            
            w_rev = np.zeros(N)
            w_rev[1:] = weights[::-1]
            W_rev = np.fft.fft(w_rev)
            term2 = np.fft.ifft(W_rev * H).real
            
            grad = -(term1 + term2)
            
        return val, grad

    def score(h):
        """Calculate the actual max overlap for final reporting/scoring."""
        Fh = np.fft.fft(h)
        S = Fh * np.conj(Fh)
        acorr = np.fft.ifft(S).real
        overlaps = N / 2.0 - acorr[1:N]
        return np.max(overlaps)

    # Constraints
    bounds = [(0, 1)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)}
    ]

    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Try a few random initializations
    # 1. Uniform 0.5
    # 2. Alternating 0.2, 0.8
    # 3. Random
    
    inits = []
    inits.append(np.full(N, 0.5))
    
    alt = np.zeros(N)
    alt[0::2] = 0.2
    alt[1::2] = 0.8
    inits.append(alt)
    
    # Add a few random inits
    for _ in range(5):
        r = rng.random(N)
        # Normalize to sum N/2
        r = r / np.sum(r) * (N/2.0)
        r = np.clip(r, 0, 1)
        # Fix sum again if clipping changed it
        diff = N/2.0 - np.sum(r)
        if abs(diff) > 1e-6:
            # Distribute diff evenly, might violate bounds, but SLSQP will fix
            r += diff / N
            r = np.clip(r, 0, 1)
        inits.append(r)

    tau = 0.05
    
    for h0 in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use SLSQP
            res = minimize(
                objective_and_grad,
                h0,
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                h_opt = res.x
                # Ensure validity
                h_opt = np.clip(h_opt, 0, 1)
                # Fix sum
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt * (N/2.0 / s)
                    h_opt = np.clip(h_opt, 0, 1)
                    # If clipping broke sum, do a simple projection
                    # For N=100, a few iterations of averaging or simple adjustment works
                    for _ in range(10):
                        diff = N/2.0 - np.sum(h_opt)
                        if abs(diff) < 1e-6:
                            break
                        h_opt += diff / N
                        h_opt = np.clip(h_opt, 0, 1)
                        
                sc = score(h_opt)
                if sc < best_score:
                    best_score = sc
                    best_h = h_opt.copy()
                    
        except Exception as e:
            pass
            
        # Check time
        if time.time() > deadline - 5:
            break

    if best_h is None:
        # Fallback to uniform
        return np.full(N, 0.5)
        
    return best_h
