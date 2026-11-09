"""
Program for the Erdos minimum-overlap task using spectral domain optimization.

Strategy:
1. Represent the step heights h in the frequency domain (FFT).
2. The overlap profile C[k] is the inverse FFT of H * conj(1-H) = H * (N - H) ... wait.
   Let's derive:
   C[k] = sum_i h[i] * (1 - h[(i-k) mod N])
   This is the circular cross-correlation of h and (1-h).
   In Fourier domain:
   FFT(C) = H * conj(FFT(1-h))
   FFT(1-h) = N - H (since FFT of constant 1 is N at k=0, 0 elsewhere, but 1-h = 1 - h, so FFT(1) - FFT(h) = N*delta_0 - H).
   Actually, FFT(1) is N at index 0 and 0 elsewhere. So FFT(1-h) = [N, 0, ..., 0] - H.
   Let G = FFT(1-h).
   Then C = IFFT(H * conj(G)).
   
   We want to minimize max_k C[k].
   Subject to:
   - h[i] in [0, 1]
   - sum h[i] = N/2 => H[0] = N/2.
   
   Since h is real, H is Hermitian symmetric. We can optimize the real and imaginary parts of H[1..N/2-1] and H[N/2] (if N even).
   
   However, ensuring h[i] in [0,1] in the spatial domain is hard in the frequency domain.
   Instead, we can optimize h directly but use the spectral insight to guide the gradient or use a different parametrization.
   
   Alternative: Optimize h directly but use a "spectral flattening" objective or gradient.
   The current program uses gradient descent on max C[k].
   A different approach: Use the fact that we want C[k] to be as flat as possible.
   Ideally, we want C[k] to be constant for all k.
   If C[k] = c for all k, then FFT(C) = c * N * delta_0.
   So H * conj(G) = c * N * delta_0.
   This implies that for k != 0, H[k] * conj(G[k]) = 0.
   Since G[k] = -H[k] for k != 0 (because G[0]=N-H[0]=N/2, and for k!=0, G[k] = -H[k]).
   Wait, G = FFT(1-h).
   G[0] = N - H[0] = N - N/2 = N/2.
   For k != 0, G[k] = -H[k].
   So H * conj(G) has components:
   k=0: H[0] * conj(G[0]) = (N/2) * (N/2) = N^2/4.
   k!=0: H[k] * conj(-H[k]) = -|H[k]|^2.
   
   So C = IFFT( [N^2/4, -|H[1]|^2, -|H[2]|^2, ...] ).
   We want to minimize max C[k].
   The DC component of C is C[0] = sum_i h[i](1-h[i]) = sum h[i] - sum h[i]^2 = N/2 - sum h[i]^2.
   Also C[0] is the mean of C[k]. So sum C[k] = N * C[0].
   If we make C[k] constant, then C[k] = C[0] for all k.
   This requires -|H[k]|^2 = 0 for all k != 0, which means H[k] = 0 for k != 0.
   This implies h is constant, h[i] = 1/2.
   Then C[k] = 1/2 * 1/2 = 1/4 for all k.
   Max C[k] = 1/4.
   
   Wait, the current program says best_score for constant 0.5 is 25.0.
   Let's check the definition of overlap in the current program.
   _overlap_profile returns r = ifft(H * conj(OMH)).real.
   For h=0.5, H = [50, 0, ..., 0]. OMH = 0.5, OMH_fft = [50, 0, ...].
   H * conj(OMH) = [2500, 0, ...].
   ifft([2500, 0, ...]) = [25, 25, ..., 25].
   So max is 25.
   
   The task asks to minimize the maximum overlap.
   If we can make C[k] constant at 25, that's a baseline.
   Can we do better?
   If we introduce high-frequency components, |H[k]|^2 > 0, then the FFT of C has negative values at k!=0.
   This creates oscillations in C[k].
   The mean is still C[0] = N/2 - sum h[i]^2.
   If we make h non-constant, sum h[i]^2 increases (since sum h[i] is fixed), so C[0] decreases.
   But the oscillations might make some C[k] larger than the mean.
   
   We want to minimize max C[k].
   Ideally, we want C[k] to be as flat as possible, and the mean as low as possible.
   But lowering the mean (by increasing variance of h) introduces negative spectral components, causing positive peaks in spatial domain.
   
   Let's analyze the trade-off.
   Let h = 0.5 + u, where sum u = 0.
   C[k] = IFFT( [N^2/4, -|U[k]|^2, ...] ).
   C[k] = N/4 - IFFT( [0, |U[1]|^2, |U[2]|^2, ...] ).
   Let v = IFFT( [0, |U[1]|^2, |U[2]|^2, ...] ).
   Then C[k] = N/4 - v[k].
   We want to minimize max_k (N/4 - v[k]) = N/4 - min_k v[k].
   So we want to maximize min_k v[k].
   
   v is the inverse FFT of the power spectrum of u (excluding DC).
   v[0] = (1/N) sum_{k!=0} |U[k]|^2 = (1/N) sum u[i]^2 = variance of u * N?
   sum u[i]^2 = (1/N) sum |U[k]|^2 = (1/N) sum_{k!=0} |U[k]|^2 (since U[0]=0).
   So v[0] = (1/N) * sum u[i]^2.
   
   We want to maximize the minimum value of v[k].
   This is equivalent to making v[k] as flat as possible and as large as possible.
   But v is determined by the power spectrum of u.
   If we choose u to be a single frequency, say U[m] = A, then v[k] = (A^2/N) cos(2 pi m k / N).
   min v[k] = -A^2/N.
   Then max C[k] = N/4 - (-A^2/N) = N/4 + A^2/N.
   This is worse than the constant case (N/4).
   
   What if we spread the power?
   If we make |U[k]|^2 proportional to 1, then v[k] is a delta function? No.
   If |U[k]|^2 is constant for k!=0, then v[k] is a delta function at k=0?
   IFFT of constant sequence is a delta.
   So v[0] is large, v[k!=0] = 0.
   Then min v[k] = 0.
   max C[k] = N/4 - 0 = N/4.
   Same as constant case.
   
   Wait, if v[k] >= 0 for all k, then max C[k] = N/4 - min v[k] <= N/4.
   To get max C[k] < N/4, we need min v[k] > 0.
   This requires v[k] to be positive everywhere.
   v is the autocorrelation of u?
   v[k] = (1/N) sum_i u[i] u[(i-k) mod N] ?
   Let's check.
   FFT(v) = |U|^2 (with DC removed).
   So v is the circular autocorrelation of u.
   v[k] = (1/N) sum_i u[i] u[i-k].
   We want to maximize min_k v[k].
   Since v[0] = (1/N) sum u[i]^2, and v is autocorrelation, v[0] is the maximum.
   We want the minimum to be as high as possible.
   This is a known problem: maximizing the minimum autocorrelation.
   For a real sequence with zero mean, can min autocorrelation be positive?
   If u is a "sinc" like function?
   
   Actually, if we can make v[k] constant for all k, then min v[k] = v[0].
   This requires |U[k]|^2 = 0 for k!=0, so u=0.
   Then v=0, max C = N/4.
   
   It seems N/4 is a lower bound?
   Let's check the constraint h[i] in [0,1].
   If we choose u such that v[k] is flat and positive, we might improve.
   
   Let's try a different strategy: Directly optimize h to make C[k] as flat as possible.
   Objective: min (max C[k] - min C[k]) + lambda * max C[k].
   Or simply min max C[k].
   
   The current program does gradient descent on max C[k].
   A different approach: Use the spectral domain to explicitly control the shape of C[k].
   We can optimize the power spectrum P[k] = |U[k]|^2 to shape v[k].
   Then reconstruct u from P[k] (with random phases), and h = 0.5 + u.
   This is a "spectral synthesis" approach.
   
   Let's implement this:
   1. Initialize P[k] (k=1..N/2-1, and N/2 if even).
   2. Compute v = IFFT([0, P[1], P[2], ...]).
   3. Compute C = N/4 - v.
   4. Score = max C.
   5. Gradient of Score w.r.t P?
      Score = max_k (N/4 - v[k]).
      Let k* be the argmax.
      d Score / d P[m] = - d v[k*] / d P[m].
      v = IFFT(S), where S[0]=0, S[k]=P[k] for k!=0.
      v[k] = (1/N) sum_m S[m] exp(2 pi i m k / N).
      d v[k*] / d P[m] = (1/N) exp(2 pi i m k* / N).
      So d Score / d P[m] = - (1/N) exp(2 pi i m k* / N).
      But P[m] must be real and non-negative.
      Also, we need to reconstruct u from P. This requires choosing phases.
      We can use random phases and keep the best one, or optimize phases too.
      
   This is complex. Let's stick to spatial domain but use a smarter optimizer.
   The current program's gradient might be stuck in local minima.
   Let's try a "spectral flattening" gradient in the spatial domain.
   Instead of just minimizing max C[k], we can minimize the variance of C[k] as well.
   Objective: max C[k] + alpha * std(C[k]).
   This encourages flatness.
   
   Another idea: The overlap C[k] is the autocorrelation of h? No, cross-correlation of h and 1-h.
   C[k] = sum h[i] (1-h[i-k]).
   If we set h[i] = 1/2 + u[i], C[k] = N/4 - sum u[i] u[i-k].
   We want to minimize max_k (N/4 - autocorr_u[k]).
   So we want to maximize min_k autocorr_u[k].
   
   Let's implement an optimizer that maximizes the minimum autocorrelation of u, subject to h in [0,1].
   This is a max-min problem.
   We can use gradient ascent on min_k autocorr_u[k].
   min_k f(k) is non-smooth, but we can use the subgradient or focus on the argmin.
   
   Let's try this:
   1. Start with random h.
   2. Compute u = h - 0.5.
   3. Compute autocorr_u = IFFT(|FFT(u)|^2).real.
   4. Find k_min = argmin autocorr_u.
   5. Gradient of autocorr_u[k_min] w.r.t u.
      autocorr_u[k] = sum_i u[i] u[i-k].
      d/d u[j] autocorr_u[k_min] = u[(j-k_min)%N] + u[(j+k_min)%N].
   6. Update u in the direction of the gradient (since we want to maximize the minimum).
   7. Project h = 0.5 + u to [0,1] and fix sum.
   
   This is different from the current program which minimizes max C[k] (equivalent to maximizing min autocorr_u[k] if mean is fixed, but the current program's gradient might be different).
   Actually, minimizing max C[k] is exactly maximizing min autocorr_u[k] because C[k] = N/4 - autocorr_u[k].
   So the objective is the same.
   The difference is in the gradient computation and the optimization method.
   The current program uses:
   g = 1.0 - shifted_minus - shifted_plus
   which is d C[k_max] / d h.
   Since C[k] = N/4 - autocorr_u[k], and u = h - 0.5,
   d C[k] / d h = - d autocorr_u[k] / d u.
   d autocorr_u[k] / d u[j] = u[(j-k)%N] + u[(j+k)%N].
   So d C[k] / d h = - (u[(j-k)%N] + u[(j+k)%N]).
   The current program computes g = 1 - h[(j-k)] - h[(j+k)] = 1 - (0.5+u[j-k]) - (0.5+u[j+k]) = -u[j-k] - u[j+k].
   This matches!
   
   So the current program is already doing this.
   Why is it stuck at 1.0?
   Maybe the local minimum is flat, or the step size is bad.
   Or maybe the constraint h in [0,1] is active and causing issues.
   
   Let's try a different optimization method: Simulated Annealing or Random Restarts with many iterations.
   Or use scipy.optimize.minimize with a different method (e.g., L-BFGS-B) on the spatial domain.
   
   Let's use scipy.optimize.minimize with L-BFGS-B, minimizing max C[k] + small * variance(C[k]).
   This might escape the flat regions.
   
   Also, we can try to explicitly construct u with specific spectral properties.
   For example, make |U[k]|^2 proportional to 1/k^2, which gives a smoother u.
   
   Let's implement a hybrid:
   1. Generate candidates with different spectral shapes (power laws).
   2. Refine with L-BFGS-B.
   
   This is materially different from the current simple gradient descent with random restarts.

"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 140.0
_t0 = time.time()


def _overlap_profile(h):
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    OMH = np.fft.fft(one_minus_h)
    r = np.fft.ifft(H * np.conj(OMH)).real
    return r


def _score(h):
    h = np.asarray(h, dtype=float)
    return float(np.max(_overlap_profile(h)))


def _fix_sum(h):
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-12:
        return h
    step = diff / N
    h2 = h + step
    if h2.min() >= -1e-12 and h2.max() <= 1.0 + 1e-12:
        h2 = np.clip(h2, 0.0, 1.0)
        residual = N / 2.0 - h2.sum()
        if abs(residual) < 1e-9:
            h2 += residual / N
        return h2
    h2 = np.clip(h + step, 0.0, 1.0)
    return h2


def _objective(h_flat):
    h = h_flat.reshape(N)
    h = _fix_sum(h)
    C = _overlap_profile(h)
    # Minimize max C and also variance to encourage flatness
    return np.max(C) + 0.01 * np.var(C)


def construct():
    global _t0
    best = np.full(N, 0.5)
    best_score = _score(best)

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates based on spectral shapes
    candidates = [best.copy()]
    
    # Power law spectra
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for phase_seed in range(5):
            if not deadline_ok():
                break
            # Generate random phases
            phases = rng.uniform(0, 2*np.pi, size=N)
            # Power spectrum
            k = np.arange(N)
            # Avoid k=0
            mask = k != 0
            P = np.zeros(N)
            P[mask] = 1.0 / (k[mask] ** alpha + 1)
            # Make it Hermitian
            P = (P + P[::-1]) / 2
            P[0] = 0
            # Reconstruct u
            U = np.sqrt(P) * np.exp(1j * phases)
            U = (U + np.conj(U[::-1])) / 2  # Ensure Hermitian
            U[0] = 0
            u = np.fft.ifft(U).real
            h = 0.5 + u
            h = _fix_sum(h)
            candidates.append(h)

    # Random candidates
    for _ in range(50):
        if not deadline_ok():
            break
        h = rng.uniform(0.0, 1.0, size=N)
        h = _fix_sum(h)
        candidates.append(h)

    # Evaluate and refine
    for h0 in candidates:
        if not deadline_ok():
            break
        h0 = _fix_sum(h0)
        sc0 = _score(h0)
        
        # Try L-BFGS-B refinement
        try:
            res = minimize(_objective, h0.flatten(), method='L-BFGS-B', 
                           bounds=[(0.0, 1.0)] * N, options={'maxiter': 200, 'ftol': 1e-10})
            h_opt = _fix_sum(res.x.reshape(N))
            sc_opt = _score(h_opt)
            if sc_opt < best_score - 1e-12:
                best = h_opt.copy()
                best_score = sc_opt
        except Exception:
            pass

    # Final local search with small steps
    if deadline_ok():
        h = best.copy()
        eta = 0.01
        no_improve = 0
        while deadline_ok() and no_improve < 20:
            C = _overlap_profile(h)
            k_min = int(np.argmin(C))  # We want to maximize min C, so focus on min
            # Gradient of C[k_min]
            shifted_minus = np.roll(h, k_min)
            shifted_plus = np.roll(h, -k_min)
            g = 1.0 - shifted_minus - shifted_plus
            # We want to increase C[k_min], so move in direction of -g (since we minimize max C, but here we maximize min C)
            # Actually, minimizing max C is equivalent to maximizing min C if mean is fixed.
            # d C[k_min] / d h = g. We want to increase C[k_min], so h += eta * g.
            h_new = h + eta * g
            h_new = _fix_sum(h_new)
            sc_new = _score(h_new)
            if sc_new < best_score - 1e-10:
                best_score = sc_new
                best = h_new.copy()
                h = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1
                eta *= 0.9
                h = best + rng.normal(0, 0.01, size=N)
                h = _fix_sum(h)

    return best
