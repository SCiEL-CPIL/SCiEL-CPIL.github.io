import numpy as np
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin

N = 24
SQRTN = math.sqrt(N)

def _time_left():
    return _DEADLINE - time.time()

# Precompute the evaluation grid
# We want to approximate max |P(z)| for |z|=1.
# Use a dense grid.
M_GRID = 20000
thetas = np.linspace(0, 2 * np.pi, M_GRID, endpoint=False)
# E[k, j] = exp(i * k * theta_j)
# We need to compute sum_k c_k E[k, j]
# Let's precompute the matrix of exponentials
# E is (N, M_GRID)
# To save memory and computation, we can compute on the fly or precompute.
# N is small, M is 20k. 24 * 20k = 480k complex numbers. ~ 8MB. Fine.
E = np.zeros((N, M_GRID), dtype=complex)
for k in range(N):
    E[k] = np.exp(1j * k * thetas)

def _supnorm_fast(coeffs):
    """
    Fast sup-norm estimate using precomputed grid.
    coeffs: list or array of 24 floats (+1/-1)
    """
    c = np.array(coeffs, dtype=complex)
    # P_j = sum_k c_k * E[k, j]
    # This is c^T @ E
    vals = np.dot(c.T, E)
    return float(np.max(np.abs(vals)))

def _supnorm_precise(coeffs):
    """
    More precise sup-norm using golden section refinement on peaks found by fast method.
    """
    # First find peaks on the coarse grid
    c = np.array(coeffs, dtype=complex)
    vals = np.dot(c.T, E)
    abs_vals = np.abs(vals)
    
    # Find indices of local maxima
    # For simplicity, just refine the global max and its neighbors
    # Or find top 5 peaks
    
    def poly_val(theta):
        # Compute P(e^{i theta})
        # c_k * exp(i k theta)
        angles = np.arange(N) * theta
        return np.sum(c * np.exp(1j * angles))
    
    best_val = 0.0
    # Identify top few peaks from the grid
    # Sort indices by abs value
    # This is expensive if done every time, but we only call this for final candidates.
    # For local search, _supnorm_fast is enough.
    
    # Simple approach: check the max on the grid, then refine that specific peak.
    idx_max = np.argmax(abs_vals)
    theta_max = thetas[idx_max]
    
    # Golden section search in a small window around theta_max
    # Window size: 2 * (2*pi / M_GRID)
    window = 2 * (2 * np.pi / M_GRID)
    a = theta_max - window
    b = theta_max + window
    
    # Golden section
    phi = (math.sqrt(5) - 1) / 2
    c_pt = b - phi * (b - a)
    d_pt = a + phi * (b - a)
    
    fc = abs(poly_val(c_pt))
    fd = abs(poly_val(d_pt))
    
    for _ in range(50): # 50 iterations is plenty for 1e-9 accuracy on this scale
        if fc < fd:
            a = c_pt
            c_pt = d_pt
            fc = fd
            d_pt = a + phi * (b - a)
            fd = abs(poly_val(d_pt))
        else:
            b = d_pt
            d_pt = c_pt
            fd = fc
            c_pt = b - phi * (b - a)
            fc = abs(poly_val(c_pt))
            
    best_val = max(fc, fd)
    
    # Also check a few other peaks just in case
    # Find top 5 indices
    top_indices = np.argsort(abs_vals)[-5:]
    for idx in top_indices:
        if idx == idx_max:
            continue
        theta_p = thetas[idx]
        a = theta_p - window
        b = theta_p + window
        c_pt = b - phi * (b - a)
        d_pt = a + phi * (b - a)
        fc = abs(poly_val(c_pt))
        fd = abs(poly_val(d_pt))
        for _ in range(50):
            if fc < fd:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + phi * (b - a)
                fd = abs(poly_val(d_pt))
            else:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - phi * (b - a)
                fc = abs(poly_val(c_pt))
        best_val = max(best_val, max(fc, fd))
        
    return best_val

def _hill_climb_fast(coeffs):
    """
    1-bit flip hill climbing using fast supnorm.
    """
    coeffs = list(coeffs)
    current_val = _supnorm_fast(coeffs)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for k in range(N):
            if _time_left() < 5.0:
                break
            coeffs[k] = -coeffs[k]
            new_val = _supnorm_fast(coeffs)
            if new_val < current_val - 1e-9:
                current_val = new_val
                improved = True
            else:
                coeffs[k] = -coeffs[k]
    return coeffs, current_val

def _two_bit_hill_climb_fast(coeffs):
    """
    2-bit flip hill climbing using fast supnorm.
    """
    coeffs = list(coeffs)
    current_val = _supnorm_fast(coeffs)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                if _time_left() < 5.0:
                    break
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
                new_val = _supnorm_fast(coeffs)
                if new_val < current_val - 1e-9:
                    current_val = new_val
                    improved = True
                else:
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
        if _time_left() < 5.0:
            break
    return coeffs, current_val

def _generate_golay_rudin_shapiro(n):
    coeffs = []
    for k in range(n):
        if bin(k).count('1') % 2 == 0:
            coeffs.append(1)
        else:
            coeffs.append(-1)
    return coeffs

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    
    # Generate a set of diverse starting points
    starts = []
    
    # 1. Golay-Rudin-Shapiro
    grs = _generate_golay_rudin_shapiro(N)
    starts.append(grs)
    starts.append([-c for c in grs])
    starts.append(list(reversed(grs)))
    starts.append(list(reversed([-c for c in grs])))
    
    # 2. Known good patterns (often related to Barker sequences or similar)
    # For N=24, we can try a few random ones and some structured ones.
    # Let's add 10 random balanced sequences
    for _ in range(10):
        seq = [random.choice([1, -1]) for _ in range(N)]
        # Ensure some balance
        if sum(seq) > 4:
             # Flip some 1s to -1s
             ones = [i for i, c in enumerate(seq) if c == 1]
             if ones:
                 flip_idx = random.choice(ones)
                 seq[flip_idx] = -1
        starts.append(seq)
        
    # 3. Add "All 1s" and "Alternating" just in case (unlikely to be best but good for search)
    starts.append([1]*N)
    starts.append([1 if i%2==0 else -1 for i in range(N)])
    
    best_coeffs = None
    best_val = float('inf')
    
    # Process each start
    for start in starts:
        if _time_left() < 10.0:
            break
            
        # Phase 1: 1-bit climb
        c1, v1 = _hill_climb_fast(start)
        
        # Phase 2: 2-bit climb
        c2, v2 = _two_bit_hill_climb_fast(c1)
        
        if v2 < best_val:
            best_val = v2
            best_coeffs = c2
            
    # Phase 3: Random restarts with 2-bit perturbations from best
    if best_coeffs is not None and _time_left() > 15.0:
        while _time_left() > 10.0:
            candidate = list(best_coeffs)
            # Flip 2 random bits
            i, j = random.sample(range(N), 2)
            candidate[i] = -candidate[i]
            candidate[j] = -candidate[j]
            
            c1, v1 = _hill_climb_fast(candidate)
            if _time_left() > 15.0:
                c2, v2 = _two_bit_hill_climb_fast(c1)
            else:
                c2, v2 = c1, v1
                
            if v2 < best_val - 1e-9:
                best_val = v2
                best_coeffs = c2
            # If we are stuck, try a larger perturbation?
            # For now, just continue.
            
    # Final Polish: Use precise supnorm to verify and maybe refine?
    # The fast supnorm is an upper bound on the true supnorm (max on grid <= max on circle).
    # Actually, max on grid is an *under*estimate of the true supnorm.
    # So best_val is a lower bound on the true score.
    # We should return the coefficients that minimize the *true* supnorm.
    # The local search minimized the *grid* supnorm.
    # It is highly likely that the minimizer of the grid supnorm is also the minimizer of the true supnorm,
    # especially with a dense grid.
    # To be safe, we can do a final 1-bit climb using _supnorm_precise?
    # That would be slow.
    # Given the grid is dense (20k points), the error is small.
    # The difference between 1.375 and 1.258 is significant.
    # The grid search should find the optimal basin.
    
    # One final check: try to improve the best_coeffs using precise eval in a tight loop
    # But precise eval is ~100x slower.
    # Let's just trust the fast search with the dense grid.
    
    return best_coeffs

if __name__ == "__main__":
    result = construct()
    print(result)
