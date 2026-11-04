"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _autocorr(h):
    """Compute circular autocorrelation via FFT."""
    H = np.fft.fft(h)
    return np.real(np.fft.ifft(H * np.conj(H)))


def _max_overlap_fast(h):
    """Compute max overlap of h with shifted (1-h)."""
    autocorr = _autocorr(h)
    min_autocorr = np.min(autocorr)
    return N / 2.0 - min_autocorr


def _lse_max_overlap(h, beta=10.0):
    """
    Smooth approximation of max overlap using log-sum-exp.
    max_overlap = N/2 - min(autocorr)
    min(autocorr) ≈ -1/beta * log(sum(exp(-beta * autocorr)))
    So max_overlap ≈ N/2 + 1/beta * log(sum(exp(-beta * autocorr)))
    """
    autocorr = _autocorr(h)
    # Numerically stable log-sum-exp of -beta * autocorr
    x = -beta * autocorr
    x_max = np.max(x)
    lse = x_max + np.log(np.sum(np.exp(x - x_max)))
    return N / 2.0 + lse / beta


def _lse_max_overlap_grad(h, beta=10.0):
    """
    Compute smooth max overlap and its gradient w.r.t. h.
    """
    autocorr = _autocorr(h)
    x = -beta * autocorr
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    weights = exp_x / sum_exp  # softmax weights over shifts
    
    # d(lse)/d(autocorr_k) = -beta * weights[k]
    # We need d(autocorr_k)/d(h_j)
    # autocorr[k] = sum_i h[i] * h[(i-k)%N]
    # d(autocorr_k)/d(h_j) = h[(j-k)%N] + h[(j+k)%N] (for k != 0)
    # For k=0: d(autocorr_0)/d(h_j) = 2*h[j]
    
    # Gradient of max_overlap w.r.t. h:
    # d(max_overlap)/d(h_j) = -1/beta * sum_k (d lse/d autocorr_k) * (d autocorr_k/d h_j)
    # = -1/beta * sum_k (-beta * weights[k]) * (d autocorr_k/d h_j)
    # = sum_k weights[k] * (d autocorr_k/d h_j)
    
    # d autocorr_k / d h_j = h[(j-k)%N] + h[(j+k)%N] for k != 0
    # But this is tricky to compute efficiently. Let's use a different approach.
    
    # Alternative: use the fact that autocorr = ifft(|FFT(h)|^2)
    # The gradient can be computed via FFT.
    
    # For now, let's compute gradient numerically if needed, or use a simpler approach.
    # Actually, let's compute it properly.
    
    # autocorr[k] = sum_i h[i]*h[(i-k)%N]
    # Let's define: for each k, autocorr[k] = sum_i h[i]*h[(i-k)%N]
    # d(autocorr[k])/d(h[j]) = h[(j-k)%N] + h[(j+k)%N] if k != 0
    #                         = 2*h[j] if k == 0
    
    # This is O(N^2) which is fine for N=100.
    grad = np.zeros(N)
    for j in range(N):
        for k in range(N):
            if k == 0:
                d_ac = 2.0 * h[j]
            else:
                d_ac = h[(j - k) % N] + h[(j + k) % N]
            grad[j] += weights[k] * d_ac
    
    lse = x_max + np.log(sum_exp)
    value = N / 2.0 + lse / beta
    return value, grad


def _project_to_feasible(h):
    """Project h to satisfy: sum = N/2, 0 <= h[i] <= 1."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            room = 1.0 - h
            if np.sum(room) > 1e-12:
                h += diff * room / np.sum(room)
        else:
            if np.sum(h) > 1e-12:
                h += diff * h / np.sum(h)
        h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if abs(current_sum - target) > 1e-10:
        residual = target - current_sum
        if residual > 0:
            room = 1.0 - h
            mask = room > 1e-10
            if np.sum(room[mask]) > 0:
                h[mask] += residual / np.sum(mask)
        else:
            mask = h > 1e-10
            if np.sum(h[mask]) > 0:
                h[mask] += residual / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h


def _gradient_descent(h0, beta=10.0, lr=0.01, max_iters=5000, time_limit=10.0):
    """
    Gradient descent on the smooth LSE objective.
    """
    h = h0.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute smooth objective and gradient
        val, grad = _lse_max_overlap_grad(h, beta=beta)
        
        # Project gradient to feasible space:
        # We need to move in direction -grad but stay in [0,1] and sum=N/2
        # Simple approach: take step, then project
        step = lr * grad
        h_new = h - step
        h_new = _project_to_feasible(h_new)
        
        # Check if it improved
        score = _max_overlap_fast(h_new)
        if score < best_score:
            best_score = score
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.8
    
    return h, best_score


def _smooth_candidate(freq, phase=0.0):
    """Generate a smooth sinusoidal candidate with exact sum."""
    x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
    h = 0.5 + 0.4 * np.sin(x + phase)
    h = _project_to_feasible(h)
    return h


def _random_candidate():
    """Generate a random feasible candidate."""
    h = np.random.uniform(0, 1, N)
    h = _project_to_feasible(h)
    return h


def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fast(best_h)  # Should be 25.0
    
    # Generate diverse starting candidates
    candidates = []
    
    # Sinusoidal candidates with various frequencies and phases
    for freq in range(1, 15):
        for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 7 * np.pi / 4]:
            h = _smooth_candidate(freq, phase)
            candidates.append(h)
    
    # Random candidates
    for _ in range(100):
        candidates.append(_random_candidate())
    
    # Binary-like candidates
    for pattern in ['alternating', 'blocks']:
        for _ in range(20):
            h = np.random.uniform(0, 1, N)
            if pattern == 'alternating':
                h = np.where(h > 0.5, 1.0, 0.0)
                n_ones = np.sum(h)
                if n_ones > N / 2:
                    idx = np.where(h == 1.0)[0]
                    to_zero = idx[:int(n_ones - N / 2)]
                    h[to_zero] = 0.0
                elif n_ones < N / 2:
                    idx = np.where(h == 0.0)[0]
                    to_one = idx[:int(N / 2 - n_ones)]
                    h[to_one] = 1.0
            else:
                block_size = np.random.randint(2, 10)
                h = np.zeros(N)
                for i in range(0, N, block_size * 2):
                    h[i:i + block_size] = 1.0
                h = _project_to_feasible(h)
            candidates.append(h)
    
    # Evaluate all candidates to find good starting points
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 30:
            break
        score = _max_overlap_fast(h)
        scored_candidates.append((score, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    top_candidates = [h for _, h in scored_candidates[:15]]
    
    # Run gradient descent from each top candidate with different betas
    for h0 in top_candidates:
        if time.time() > deadline - 10:
            break
        for beta in [5.0, 10.0, 20.0]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time() - 3
            if remaining < 1.0:
                break
            h_opt, score_opt = _gradient_descent(
                h0, beta=beta, lr=0.01, 
                max_iters=3000,
                time_limit=min(8.0, remaining)
            )
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
    
    # Final gradient descent polish from best
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 1.0:
            h_final, score_final = _gradient_descent(
                best_h, beta=20.0, lr=0.005,
                max_iters=5000,
                time_limit=min(10.0, remaining)
            )
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    # Ensure the result is valid
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
