"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import time

# Global state for dynamic feedback: track symbol frequencies in selected set
_symbol_counts = [0, 0, 0]
_total_selected = 0
_start_time = None

def priority(vector, n):
    """
    Dynamic feedback control priority for cap set greedy insertion.
    
    Tracks the marginal line-completion risk by maintaining a global histogram
    of symbol frequencies in the selected set. Penalizes vectors that would
    further skew the distribution away from the balanced (1/3, 1/3, 1/3) ratio.
    
    Uses smooth quadratic surrogate: penalty on increase in variance of symbol counts.
    """
    global _symbol_counts, _total_selected, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety: if we've been running too long, just use a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple hash-based priority
        vec_str = ''.join(str(v) for v in vector)
        h = int(hashlib.md5(vec_str.encode()).hexdigest(), 16)
        return (h % 1000000) / 1000000.0
    
    # Count symbols in this vector
    c0 = c1 = c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        elif v == 2:
            c2 += 1
    
    # Current total symbols in selected set
    T = _total_selected
    
    # If no vectors selected yet, use structural bias
    if T == 0:
        # Prefer balanced vectors (close to n/3 of each symbol)
        target = n / 3.0
        deviation = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
        # Small hash tiebreaker
        vec_str = ''.join(str(v) for v in vector)
        h = int(hashlib.md5(vec_str.encode()).hexdigest(), 16)
        random_part = (h % 1000000) / 1000000.0
        return -deviation * 0.1 + random_part * 0.01
    
    # Compute current mean symbol count per symbol
    # The ideal asymptotic ratio is balanced: each symbol appears T/3 times
    mean_count = T / 3.0
    
    # Current variance-like measure: sum of (count - mean)^2
    current_penalty = 0.0
    for i in range(3):
        diff = _symbol_counts[i] - mean_count
        current_penalty += diff * diff
    
    # New counts if we add this vector
    new_counts = [
        _symbol_counts[0] + c0,
        _symbol_counts[1] + c1,
        _symbol_counts[2] + c2
    ]
    new_T = T + n
    new_mean = new_T / 3.0
    
    # New penalty
    new_penalty = 0.0
    for i in range(3):
        diff = new_counts[i] - new_mean
        new_penalty += diff * diff
    
    # Marginal increase in penalty (line-completion risk)
    marginal_risk = new_penalty - current_penalty
    
    # Smooth quadratic penalty on marginal risk
    # Higher marginal risk = lower priority
    structural_score = -marginal_risk
    
    # Add a small deterministic hash-based component to break ties
    # and avoid clustering in the greedy selection
    vec_str = ''.join(str(v) for v in vector)
    h = int(hashlib.md5(vec_str.encode()).hexdigest(), 16)
    random_part = (h % 1000000) / 1000000.0
    
    # Combine: structural dominates, small random perturbation
    # Scale structural to be in a reasonable range
    # marginal_risk is typically O(n) or O(n^2), so normalize
    scale = max(1.0, n)
    combined = structural_score / scale + random_part * 0.05
    
    # Update global state: this vector is being "considered" for insertion
    # We update the histogram to reflect the dynamic feedback
    _symbol_counts[0] += c0
    _symbol_counts[1] += c1
    _symbol_counts[2] += c2
    _total_selected += n
    
    return combined
# EVOLVE-BLOCK-END
