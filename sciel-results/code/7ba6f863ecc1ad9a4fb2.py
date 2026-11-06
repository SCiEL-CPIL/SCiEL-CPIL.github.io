import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0
_NUM_SAMPLES = 4096

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return np.array(p[:_N], dtype=np.int8)

def _init_eval(coeffs, num_samples):
    """Initialize the evaluation vectors."""
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    z = np.exp(1j * angles)
    # P(z) = sum c_k z^k
    # We store the current values P_j for j in range(num_samples)
    # P_j = sum_{k=0}^{N-1} c_k z_j^k
    # We can compute this as a matrix-vector product
    # Z is a (num_samples, N) matrix where Z[j, k] = z_j^k
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    P_vals = np.dot(Z, coeffs)
    return Z, P_vals

def _max_norm(P_vals):
    return np.max(np.abs(P_vals))

def construct():
    # Start with Rudin-Shapiro
    coeffs = _rs_poly()
    Z, P_vals = _init_eval(coeffs, _NUM_SAMPLES)
    
    best_coeffs = coeffs.copy()
    best_score = _max_norm(P_vals)
    
    # Simulated Annealing / Hill Climbing
    # We flip one coefficient at a time
    # Flipping c_k changes P_j by -2 * c_k * z_j^k
    # We can precompute the update vectors for each k
    # update_k = -2 * z^k (complex vector of length _NUM_SAMPLES)
    # If c_k is +1, new P = P - 2*z^k
    # If c_k is -1, new P = P + 2*z^k
    # So delta_P = -2 * c_k * z^k
    
    # Precompute z^k for all k
    # Z is already (num_samples, N), so Z[:, k] is z^k
    # We need to be careful with memory and speed
    
    current_coeffs = coeffs.copy()
    current_P_vals = P_vals.copy()
    
    # Temperature schedule
    temp = 0.01
    cool_rate = 0.999
    
    # We will iterate until time runs out
    while _time_left() > 1.0:
        # Pick a random index to flip
        k = np.random.randint(0, _N)
        c_k = current_coeffs[k]
        
        # Compute the change in P_vals
        # delta = -2 * c_k * Z[:, k]
        delta = -2 * c_k * Z[:, k]
        
        # New P_vals
        new_P_vals = current_P_vals + delta
        
        # New score
        new_score = _max_norm(new_P_vals)
        
        # Acceptance criterion
        if new_score < best_score:
            # Update best
            best_score = new_score
            best_coeffs = current_coeffs.copy()
            
        # Decide whether to accept this move for the current state
        # If it improves, accept. If it worsens, accept with probability exp(-delta/temp)
        if new_score < _max_norm(current_P_vals) or (np.random.random() < math.exp(-((new_score - _max_norm(current_P_vals)) / temp))):
            current_coeffs[k] = -c_k
            current_P_vals = new_P_vals
            temp *= cool_rate
            if temp < 0.001:
                temp = 0.001 # Floor for temperature
        
    # Return the best coefficients found
    # The task asks for coefficients in {+1, -1}
    # best_coeffs is an int8 array
    return list(best_coeffs)
# EVOLVE-BLOCK-END
