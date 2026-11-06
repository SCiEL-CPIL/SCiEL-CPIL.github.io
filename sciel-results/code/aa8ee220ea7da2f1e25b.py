import numpy as np
import time
import math

def _supnorm_fast(coeffs, n_pts=1024):
    """
    Fast supnorm estimate using dense sampling.
    coeffs: array of length n
    """
    n = len(coeffs)
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    # Use matrix multiplication for speed
    # V[i, j] = z[i]^j
    # P(z_i) = sum_j c_j * z_i^j
    # We can precompute z_powers once if coeffs don't change, but here they do.
    # However, n=24 is small.
    # Let's use direct broadcasting which is quite fast for n=24
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    return np.max(np.abs(vals))

def _supnorm_refined(coeffs, n_pts=4096, refine_rounds=8, refine_points=5):
    """
    More accurate supnorm estimate for final verification.
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    # Dense sampling
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    if n_pts > refine_points:
        top_idx = np.argpartition(magnitudes, -refine_points)[-refine_points:]
    else:
        top_idx = np.arange(n_pts)
    
    best_mag = max(magnitudes)
    
    # Refine around the top peaks
    window = 2.0 * np.pi / n_pts * 2  # 2 sample spacings
    inv_golden = 1.0 - 1.0 / ((1.0 + np.sqrt(5.0)) / 2.0)
    
    for idx in top_idx:
        center = angles[idx]
        a = max(0.0, center - window)
        b = min(2.0 * np.pi, center + window)
        
        if b <= a:
            continue
            
        c = b - inv_golden * (b - a)
        d = a + inv_golden * (b - a)
        
        fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * c)))
        fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * d)))
        
        for _ in range(refine_rounds):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + inv_golden * (b - a)
                fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * d)))
            else:
                b = d
                d = c
                fd = fc
                c = b - inv_golden * (b - a)
                fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * c)))
        
        fa = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * a)))
        fb = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * b)))
        
        local_best = max(fa, fb, fc, fd)
        if local_best > best_mag:
            best_mag = local_best
    
    return best_mag

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    N = 24
    sqrt_N = np.sqrt(N)
    
    # Initial best: Random or Rudin-Shapiro
    def _rs_sequence(length):
        p, q = [1], [1]
        while len(p) < length:
            p_new = p + q
            q_new = p + [-c for c in q]
            p, q = p_new, q_new
        return p[:length]
    
    init_coeffs = np.array(_rs_sequence(N), dtype=np.int8)
    
    best_coeffs = init_coeffs.copy()
    best_score = _supnorm_refined(best_coeffs) / sqrt_N
    
    # Precompute powers for fast evaluation? 
    # Actually, for SA, we change one coeff at a time.
    # P(z) = sum c_k z^k.
    # If we flip c_i, P_new(z) = P_old(z) - 2*c_i*z^i.
    # This allows O(n_pts) update instead of O(n_pts * N).
    
    n_pts_fast = 512
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts_fast, endpoint=False)
    # Precompute z^k for all k and all points
    # z_powers[k, i] = exp(1j * angles[i] * k)
    k = np.arange(N)
    # Shape: (N, n_pts_fast)
    z_powers = np.exp(1j * np.outer(k, angles))
    
    # Current P values on the grid
    # P_vals[i] = sum_k c_k * z_powers[k, i]
    # We can maintain this vector.
    
    def evaluate_fast(c, P_vals):
        # P_vals is already computed for c
        return np.max(np.abs(P_vals))
    
    def update_P_vals(P_vals, c, i, z_powers_i):
        # Flip c[i]
        # P_new = P_old - 2 * c[i] * z^i
        # c[i] is +1 or -1
        # delta = -2 * c[i] * z^i
        # z_powers_i is the row for power i
        P_vals -= 2 * c[i] * z_powers_i
        c[i] = -c[i]
        return P_vals, c

    # Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Start SA
    T_start = 0.05
    T_end = 0.001
    alpha = 0.9995 # Cooling rate
    
    current_coeffs = init_coeffs.copy().astype(np.int8)
    # Compute initial P_vals
    current_P_vals = np.sum(current_coeffs[None, :] * z_powers, axis=0)
    current_score = np.max(np.abs(current_P_vals)) / sqrt_N
    
    best_coeffs_sa = current_coeffs.copy()
    best_score_sa = current_score
    
    # If RS was better, start from there
    if best_score < current_score:
        current_coeffs = best_coeffs.copy().astype(np.int8)
        current_P_vals = np.sum(current_coeffs[None, :] * z_powers, axis=0)
        current_score = np.max(np.abs(current_P_vals)) / sqrt_N
        best_coeffs_sa = current_coeffs.copy()
        best_score_sa = current_score
    
    T = T_start
    
    while T > T_end and time.time() < deadline:
        # Pick a random coefficient to flip
        i = rng.integers(0, N)
        
        # Calculate new P_vals
        new_P_vals = current_P_vals - 2 * current_coeffs[i] * z_powers[i]
        new_score = np.max(np.abs(new_P_vals)) / sqrt_N
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept
            current_coeffs[i] = -current_coeffs[i]
            current_P_vals = new_P_vals
            current_score = new_score
            
            if current_score < best_score_sa - 1e-12:
                best_score_sa = current_score
                best_coeffs_sa = current_coeffs.copy()
                # Update global best if significantly better
                if best_score_sa < best_score - 1e-6:
                    best_score = best_score_sa
                    best_coeffs = best_coeffs_sa.copy()
        
        T *= alpha
        
        # Occasionally, do a random restart if stuck? 
        # Or just let SA cool.
        
    # After SA, do a local polish (hill climbing) on the best found
    # Use the fast grid
    polish_coeffs = best_coeffs_sa.copy().astype(np.int8)
    polish_P_vals = np.sum(polish_coeffs[None, :] * z_powers, axis=0)
    polish_score = np.max(np.abs(polish_P_vals)) / sqrt_N
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            new_P_vals = polish_P_vals - 2 * polish_coeffs[i] * z_powers[i]
            new_score = np.max(np.abs(new_P_vals)) / sqrt_N
            if new_score < polish_score - 1e-12:
                polish_coeffs[i] = -polish_coeffs[i]
                polish_P_vals = new_P_vals
                polish_score = new_score
                improved = True
    
    if polish_score < best_score - 1e-6:
        best_score = polish_score
        best_coeffs = polish_coeffs.copy()
    
    # Final verification with refined supnorm
    # Only if we have time
    if time.time() < deadline - 5.0:
        final_score = _supnorm_refined(best_coeffs) / sqrt_N
        if final_score < best_score:
             best_score = final_score
    
    # Return integers
    return [int(c) for c in best_coeffs]

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
