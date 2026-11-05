"""
Structural change: Direct element-wise optimization of the 400-dim vector h.
Instead of restricting to a low-dimensional parametric family (chirp, CAZAC),
we optimize the full h vector directly using a hybrid strategy:
1. Initialize with a two-level sequence (±1) with DC offset for good baseline
2. Use local perturbation search (simulated annealing-style) on individual
   elements to reduce the maximum positive autoconvolution peak
3. Exploit the asymmetry: only positive peaks in conv matter for the objective

The key structural difference: we work in the full 400-dim space rather than
a 3-parameter family, allowing the optimizer to find non-parametric structures
that exploit the signed-max asymmetry.
"""

import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _compute_value(h):
    """Compute the evaluator's exact value."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _init_two_level():
    """
    Initialize with a two-level sequence: h[i] = a * s[i] + c
    where s[i] is a pseudo-random ±1 sequence and c is chosen to maximize sum.
    We try several bases and pick the best.
    """
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Try different random sign patterns
    rng = np.random.default_rng(42)
    for trial in range(50):
        signs = rng.choice([-1.0, 1.0], size=N)
        # Try different a and c values
        for a in [0.5, 1.0, 1.5, 2.0]:
            for c in [0.0, 0.5, 1.0, 1.5, 2.0]:
                h = a * signs + c
                s = np.sum(h)
                if abs(s) < 1e-10:
                    continue
                val = _compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    # Also try a structured pattern: alternating blocks
    for block_size in [2, 4, 8, 16, 32]:
        n_blocks = N // block_size
        signs = np.zeros(N)
        for b in range(n_blocks):
            # Alternate block signs
            signs[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -1.0
        for a in [0.5, 1.0, 1.5]:
            for c in [0.0, 0.5, 1.0, 1.5]:
                h = a * signs + c
                s = np.sum(h)
                if abs(s) < 1e-10:
                    continue
                val = _compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    return best_h, best_val

def _local_perturbation_search(h0, deadline):
    """
    Local search by perturbing individual elements.
    Uses a simulated annealing approach: at each step, pick a random index,
    perturb its value, and accept if it improves the objective or with
    some probability if it doesn't (to escape local minima).
    """
    h = h0.copy()
    current_val = _compute_value(h)
    best_h = h.copy()
    best_val = current_val
    
    rng = np.random.default_rng(123)
    T = 0.1  # Initial temperature
    cooling = 0.999
    
    # Precompute for efficiency
    def _eval_h(h_vec):
        s = np.sum(h_vec)
        if abs(s) < 1e-15:
            return np.inf
        conv = np.convolve(h_vec, h_vec)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    step_count = 0
    while time.time() < deadline - 5.0:
        # Pick a random index to perturb
        idx = rng.integers(0, N)
        old_val = h[idx]
        
        # Perturbation: random step
        delta = rng.normal(0, 0.5)
        h[idx] = old_val + delta
        
        new_val = _eval_h(h)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
            T = max(T * cooling, 0.001)
        else:
            # Accept with probability exp(-(new_val - current_val) / T)
            if T > 0.001:
                prob = np.exp(-(new_val - current_val) / T)
                if rng.random() < prob:
                    current_val = new_val
                    T = max(T * cooling, 0.001)
                else:
                    h[idx] = old_val
            else:
                h[idx] = old_val
        
        step_count += 1
        # Check time more frequently
        if step_count % 1000 == 0 and time.time() > deadline - 10.0:
            break
    
    return best_h, best_val

def _coordinate_descent(h0, deadline):
    """
    Coordinate descent: iterate over each element, optimize it given the others.
    For element i, the autoconvolution is a quadratic function of h[i],
    so we can find the optimal value analytically or via a 1D search.
    """
    h = h0.copy()
    current_val = _compute_value(h)
    best_h = h.copy()
    best_val = current_val
    
    def _eval_h(h_vec):
        s = np.sum(h_vec)
        if abs(s) < 1e-15:
            return np.inf
        conv = np.convolve(h_vec, h_vec)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            
            old_val = h[i]
            
            # 1D optimization over h[i]
            # The value depends on h[i] through conv and sum
            # Do a small grid search around current value
            best_local = old_val
            best_local_val = current_val
            
            for delta in [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]:
                h[i] = old_val + delta
                v = _eval_h(h)
                if v < best_local_val:
                    best_local_val = v
                    best_local = old_val + delta
            
            h[i] = best_local
            if best_local != old_val:
                improved = True
                current_val = best_local_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    """
    Returns the optimal step heights.
    Strategy:
    1. Initialize with best two-level sequence from random search
    2. Coordinate descent to refine each element
    3. Local perturbation search (simulated annealing) for further improvement
    4. Final coordinate descent pass
    """
    deadline = START_TIME + TIME_LIMIT
    
    # Phase 1: Initialize with two-level sequence
    h_init, val_init = _init_two_level()
    
    # Phase 2: Coordinate descent
    h_cd, val_cd = _coordinate_descent(h_init, deadline - 30.0)
    
    # Phase 3: Simulated annealing / local perturbation
    h_sa, val_sa = _local_perturbation_search(h_cd, deadline - 10.0)
    
    # Phase 4: Final coordinate descent
    h_final, val_final = _coordinate_descent(h_sa, deadline - 2.0)
    
    # Safety check
    final_val = _compute_value(h_final)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
    
    return h_final

# EVOLVE-BLOCK-END
