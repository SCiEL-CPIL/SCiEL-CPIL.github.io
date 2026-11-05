"""
Strategy: Low-dimensional evolutionary search on structured "blocky" candidates.

The current approach uses smooth surrogates (log-sum-exp) and L-BFGS-B on the full 400-dim vector.
This is slow and can get stuck in local minima of the surrogate that aren't minima of the true max.

New approach:
1. Exploit scale invariance: We only care about the shape, not the magnitude. 
   We can fix sum(h) = 1 (or any non-zero constant) to reduce the problem dimensionality effectively.
   Actually, let's just normalize h such that sum(h)=1 before evaluation to keep numbers stable.
   Wait, the formula is abs(2*N*max(conv)/sum^2). 
   If we scale h by k, conv scales by k^2, sum scales by k, sum^2 scales by k^2.
   So the value is scale-invariant. 
   
2. Structure: The optimal solutions for such "autocorrelation" problems often look like:
   - A few positive spikes and a few negative spikes.
   - Or a block of positive values and a block of negative values.
   - Or a "Golay-like" structure.

   I will define a parametric family of functions:
   h(x) = A * 1_{x in P} - B * 1_{x in N}
   where P and N are sets of indices.
   
   To make this optimizable with Nelder-Mead, I'll use a "level set" representation:
   Divide the 400 indices into K blocks (e.g., K=20 blocks of size 20).
   Assign a height to each block.
   This reduces the dimension from 400 to K (e.g., 20).
   Nelder-Mead is very effective in low dimensions (10-30).
   
   I will run multiple Nelder-Mead restarts with different random initializations of the block heights.
   I will also try different block sizes (e.g., 10, 20, 40, 100).
   
   Additionally, I will try a "two-spike + noise" structure but optimize the noise levels using Nelder-Mead.
   
   Key insight: The max of the autocorrelation is dominated by the large values. 
   The "spike" solution (two spikes) gives value ~ 2*N*0.5 = 400 (if sum=1, max conv ~ 0.5, val = 2*400*0.5/1 = 400).
   The seed is 2.0? Wait.
   Seed np.ones(400): sum=400. conv[0]=400. max=400.
   value = abs(2 * 400 * 400 / 400^2) = 800 / 400 = 2.0.
   
   Current best score 1.2742 means value = 2.0 / 1.2742 = 1.57.
   So we are already at value ~ 1.57.
   
   The theoretical lower bound for C3 is likely related to 1.
   Let's aim for value < 1.5.
   
   I will use Nelder-Mead on the block heights.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145  # Safety margin

def _evaluate(h):
    """Compute the exact task value."""
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf
    return val

def _block_objective(params, block_size):
    """
    Objective function for Nelder-Mead.
    params: array of block heights.
    block_size: size of each block.
    Returns the task value.
    """
    n_blocks = N // block_size
    # Ensure we use exactly N elements
    h = np.repeat(params, block_size)
    # If N is not divisible, handle remainder (N=400 is divisible by 1,2,4,5,8,10,16,20,25,40,50,80,100,200,400)
    if len(h) < N:
        h = np.concatenate([h, np.zeros(N - len(h))])
    elif len(h) > N:
        h = h[:N]
    
    # Scale invariance: we can normalize to keep numbers in check, 
    # but Nelder-Mead works on the raw params. 
    # To avoid overflow/underflow, we can normalize h so sum(h)=1 
    # before computing value, but the value is scale invariant anyway.
    # However, if sum(h) is close to 0, value is huge. 
    # We want to encourage sum(h) away from 0.
    
    val = _evaluate(h)
    # Penalty if sum is too small to avoid numerical issues during optimization
    s = np.sum(h)
    if abs(s) < 0.1:
        val += 1e6
    return val

def _try_block_optimization(block_size, n_restarts=5, time_limit=20.0):
    """
    Run Nelder-Mead optimization for a specific block size.
    """
    n_blocks = N // block_size
    best_val = np.inf
    best_h = None
    start_time = time.time()
    
    rng = np.random.RandomState(block_size) # Deterministic per block size
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initial guess: Random signs, random magnitudes
        # Often, alternating signs or sparse spikes are good.
        # Let's try a mix: some blocks 0, some +1, some -1.
        init = rng.choice([-1.0, 0.0, 1.0], size=n_blocks)
        # Add some noise to avoid flat regions
        init += rng.normal(0, 0.1, n_blocks)
        
        try:
            res = minimize(
                lambda x: _block_objective(x, block_size),
                init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.success or res.fun < best_val:
                h = np.repeat(res.x, block_size)
                if len(h) < N:
                    h = np.concatenate([h, np.zeros(N - len(h))])
                elif len(h) > N:
                    h = h[:N]
                val = _evaluate(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
        except Exception:
            pass
            
    return best_val, best_h

def _try_sparse_spike_optimization(n_spikes=2, n_restarts=5, time_limit=20.0):
    """
    Optimize a sparse spike structure.
    h is zero except at n_spikes positions.
    Params: [pos1, pos2, ..., posK, val1, val2, ..., valK]
    This is harder for Nelder-Mead because positions are discrete.
    Instead, I'll stick to the block method which is more robust.
    """
    return np.inf, None

def construct():
    start_time = time.time()
    
    best_val = np.inf
    best_h = None
    
    # 1. Baseline: The known good "two-spike + tail" from the parent program
    # This ensures we have a valid fallback immediately.
    h_base = np.zeros(N)
    h_base[0] = 0.5
    h_base[1] = 0.5
    h_base[2:200] = -0.001
    val_base = _evaluate(h_base)
    if np.isfinite(val_base):
        best_val = val_base
        best_h = h_base.copy()
        
    # 2. Try various block sizes with Nelder-Mead
    # Smaller blocks = higher fidelity, but higher dimension (slower NM).
    # Larger blocks = lower fidelity, but faster NM.
    block_sizes = [10, 20, 40, 80, 100, 200]
    
    # Allocate time based on remaining budget
    # NM is fast for low dims. 
    # 10 blocks (size 40) is very fast.
    # 200 blocks (size 2) is slower but closer to full resolution.
    
    for bs in block_sizes:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Give more time to finer blocks (higher dim) or just fixed time?
        # Let's give 15s per block size, but stop if deadline approaches.
        t_limit = 15.0
        if time.time() - start_time + t_limit > DEADLINE - 5:
            t_limit = max(1.0, DEADLINE - 5 - (time.time() - start_time))
            
        val, h = _try_block_optimization(bs, n_restarts=10, time_limit=t_limit)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # 3. Final polish: If we found a good blocky solution, try to "unblock" it slightly
    # by adding small random noise and re-evaluating, or a quick L-BFGS on the full vector
    # starting from the blocky solution.
    if best_h is not None and time.time() - start_time < DEADLINE - 10:
        # Quick L-BFGS polish on the full vector using the blocky solution as start
        # Use a smooth objective to guide L-BFGS
        def smooth_obj(x):
            s = np.sum(x)
            if abs(s) < 1e-10: return 1e10
            conv = np.convolve(x, x)
            m = np.max(conv)
            # Soft max with small beta to keep it smooth but close to max
            beta = 50.0
            shifted = conv - m
            lse = m + np.log(np.sum(np.exp(beta * shifted))) / beta
            return 2.0 * N * lse / (s * s)
            
        # We don't have the gradient for the smooth obj easily derived in this short time,
        # so we'll use Nelder-Mead on the full vector? No, too slow for 400 dims.
        # Instead, let's just do a few random perturbations of the best blocky solution
        # and keep the best.
        
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            val_pert = _evaluate(h_pert)
            if val_pert < best_val:
                best_val = val_pert
                best_h = h_pert.copy()
                
    # Final validity check
    if best_h is None or not np.isfinite(best_val):
        best_h = h_base
        best_val = val_base
        
    return best_h

# Test locally if run directly
if __name__ == "__main__":
    h = construct()
    print(_evaluate(h))
