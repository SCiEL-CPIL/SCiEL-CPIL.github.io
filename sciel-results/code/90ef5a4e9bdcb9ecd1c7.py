"""Matrix multiplication 3x3x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[(i,j),(j,k),(i,k)] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T):
    """Check if U,V,W exactly decompose T (within tolerance)."""
    rank = U.shape[1]
    approx = np.zeros_like(T)
    for r in range(rank):
        approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T, approx, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _round_to_halfint(x):
    """Round array to nearest half-integer."""
    return np.round(x * 2) / 2

def _round_to_int(x):
    """Round array to nearest integer."""
    return np.round(x)

def _try_rank(n, m, p, T, target_rank, deadline, seed=0):
    """
    Attempt to find a rank-target_rank decomposition.
    Uses structured + random initializations with L-BFGS-B.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    max_restarts = 12
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            return None
        
        if restart == 0:
            # Structured: use subset of schoolbook columns
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            sch_U, sch_V, sch_W = _schoolbook(n, m, p)
            cols_to_use = min(target_rank, n*m*p)
            idx = np.linspace(0, n*m*p-1, cols_to_use).astype(int)
            U_init[:, :cols_to_use] = sch_U[:, idx]
            V_init[:, :cols_to_use] = sch_V[:, idx]
            W_init[:, :cols_to_use] = sch_W[:, idx]
        elif restart == 1:
            # Binary-like initialization
            U_init = rng.choice([0, 1, -1], size=(n*m, target_rank))
            V_init = rng.choice([0, 1, -1], size=(m*p, target_rank))
            W_init = rng.choice([0, 1, -1], size=(p*n, target_rank))
        elif restart == 2:
            # Sparse binary with higher density
            U_init = (rng.rand(n*m, target_rank) < 0.3).astype(float) * rng.choice([-1, 1], size=(n*m, target_rank))
            V_init = (rng.rand(m*p, target_rank) < 0.3).astype(float) * rng.choice([-1, 1], size=(m*p, target_rank))
            W_init = (rng.rand(p*n, target_rank) < 0.3).astype(float) * rng.choice([-1, 1], size=(p*n, target_rank))
        else:
            # Random continuous with varying scales
            scale = 0.3 + 0.4 * rng.rand()
            U_init = rng.randn(n*m, target_rank) * scale
            V_init = rng.randn(m*p, target_rank) * scale
            W_init = rng.randn(p*n, target_rank) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
            diff = T - approx
            return np.sum(diff * diff)
        
        try:
            # Use L-BFGS-B with tighter tolerance
            result = minimize(objective, params_init, method='L-BFGS-B',
                             options={'maxiter': 500, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < 1e-10:
                U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Try multiple rounding strategies
                round_strategies = [
                    ('halfint', _round_to_halfint),
                    ('int', _round_to_int),
                ]
                
                for name, round_fn in round_strategies:
                    U_round = round_fn(U_r)
                    V_round = round_fn(V_r)
                    W_round = round_fn(W_r)
                    
                    if _verify(U_round, V_round, W_round, T):
                        return U_round, V_round, W_round
        except Exception:
            pass
    
    return None

def _fourier_slice_init(n, m, p, target_rank, seed=0):
    """
    Generate initialization using DFT structure over the output dimension.
    This exploits the cyclic structure of the k-dimension.
    """
    rng = np.random.RandomState(seed)
    
    # DFT matrix for Z_p
    omega = np.exp(2j * np.pi / p)
    F = np.array([[omega**(j*k) for k in range(p)] for j in range(p)])
    F_inv = F.conj().T / p
    
    # For each frequency component, try to build a decomposition
    # The idea: the tensor T has structure T[i,j,k] = delta_{j,k} (in the k-index)
    # When we apply DFT to the k-index, we get frequency components
    
    U_init = np.zeros((n*m, target_rank))
    V_init = np.zeros((m*p, target_rank))
    W_init = np.zeros((p*n, target_rank))
    
    # Distribute rank among frequency components
    ranks_per_freq = [target_rank // p + (1 if f < target_rank % p else 0) for f in range(p)]
    
    col_offset = 0
    for f in range(p):
        r_f = ranks_per_freq[f]
        if r_f == 0:
            continue
        
        # For each frequency, create a random structured initialization
        # The W component should have DFT structure
        for r in range(r_f):
            col = col_offset + r
            if col >= target_rank:
                break
            
            # Random values for U and V
            U_init[:, col] = rng.choice([0, 1, -1], size=n*m)
            V_init[:, col] = rng.choice([0, 1, -1], size=m*p)
            
            # W has DFT structure in the k-index
            # W[k*n+i, col] should be related to omega**(f*k)
            for i in range(n):
                for k in range(p):
                    W_init[k*n+i, col] = omega**(f*k)
        
        col_offset += r_f
    
    # Convert complex to real (take real part and adjust)
    U_init = np.real(U_init)
    V_init = np.real(V_init)
    W_init = np.real(W_init)
    
    return U_init, V_init, W_init

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    T = _build_tensor(n, m, p)
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as guaranteed fallback
    U_sch, V_sch, W_sch = _schoolbook(n, m, p)
    best_result = (U_sch, V_s_ch, W_sch) if False else (U_sch, V_sch, W_sch)
    best_rank = schoolbook_rank
    
    # Lower bounds for matrix multiplication rank
    lower_bound = max(n*m, m*p, p*n, n*p)
    
    # For 3x3x5: lower_bound = 15
    # Try ranks from lower_bound upward
    candidate_ranks = list(range(lower_bound, min(schoolbook_rank, 45)))
    candidate_ranks = sorted(set(candidate_ranks))
    
    # Try each candidate rank with multiple seeds and initialization strategies
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            # Use different initialization strategies based on seed
            if seed < 8:
                result = _try_rank(n, m, p, T, target_rank, deadline, seed=seed + target_rank)
            elif seed < 14:
                # Use Fourier-inspired initialization
                U_init, V_init, W_init = _fourier_slice_init(n, m, p, target_rank, seed=seed + target_rank)
                params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                def objective(params):
                    U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    approx = np.zeros_like(T)
                    for r in range(target_rank):
                        approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                    diff = T - approx
                    return np.sum(diff * diff)
                
                try:
                    result_opt = minimize(objective, params_init, method='L-BFGS-B',
                                         options={'maxiter': 500, 'maxfun': 1500, 'ftol': 1e-14})
                    if result_opt.fun < 1e-10:
                        U_r = result_opt.x[:n*m*target_rank].reshape(n*m, target_rank)
                        V_r = result_opt.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                        W_r = result_opt.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                        
                        for round_fn in [_round_to_halfint, _round_to_int]:
                            U_round = round_fn(U_r)
                            V_round = round_fn(V_r)
                            W_round = round_fn(W_r)
                            if _verify(U_round, V_round, W_round, T):
                                result = (U_round, V_round, W_round)
                                break
                        else:
                            result = None
                except Exception:
                    result = None
            else:
                # Additional random restarts
                result = _try_rank(n, m, p, T, target_rank, deadline, seed=seed + target_rank + 100)
            
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # Final verification of best result
    U_final, V_final, W_final = best_result
    
    if not _verify(U_final, V_final, W_final, T):
        # Fallback to schoolbook if something went wrong
        U_final, V_final, W_final = _schoolbook(n, m, p)
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

# EVOLVE-BLOCK-END
