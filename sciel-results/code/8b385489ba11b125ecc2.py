"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds

def _verify_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using rational arithmetic."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(rank):
                    u_val = U[i*m+j][c] if isinstance(U, list) else U[i*m+j, c]
                    v_val = V[j*p+k][c] if isinstance(V, list) else V[j*p+k, c]
                    w_val = W[k*n+i][c] if isinstance(W, list) else W[k*n+i, c]
                    if isinstance(u_val, Fraction):
                        uf = u_val
                    else:
                        uf = Fraction(u_val).limit_denominator(10000)
                    if isinstance(v_val, Fraction):
                        vf = v_val
                    else:
                        vf = Fraction(v_val).limit_denominator(10000)
                    if isinstance(w_val, Fraction):
                        wf = w_val
                    else:
                        wf = Fraction(w_val).limit_denominator(10000)
                    s += uf * vf * wf
                if s != Fraction(1, 1):
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _optimize_block(n, m, p, target_rank, seed=42, max_nfev=3000):
    """
    Try to find a rank-target_rank decomposition for n x m x p multiplication.
    Returns (U, V, W) as numpy arrays or None.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Multiple initialization strategies
    best_result = None
    best_err = np.inf
    
    for init_strategy in range(5):
        if init_strategy == 0:
            # Random small values
            U_init = np.random.randn(n*m, target_rank) * 0.3
            V_init = np.random.randn(m*p, target_rank) * 0.3
            W_init = np.random.randn(p*n, target_rank) * 0.3
        elif init_strategy == 1:
            # Structured: initialize to partially satisfy constraints
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            # Assign some 1s to create partial solutions
            for c in range(min(target_rank, n*m*p)):
                i = (c // (m*p)) % n
                j = (c // p) % m
                k = c % p
                U_init[i*m+j, c] = 1.0
                V_init[j*p+k, c] = 1.0
                W_init[k*n+i, c] = 1.0
            # Add small random perturbation
            U_init += np.random.randn(n*m, target_rank) * 0.1
            V_init += np.random.randn(m*p, target_rank) * 0.1
            W_init += np.random.randn(p*n, target_rank) * 0.1
        elif init_strategy == 2:
            # Block-structured initialization
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            block_size = max(1, target_rank // max(1, n*m*p // target_rank))
            for c in range(target_rank):
                block_idx = c // block_size
                if block_idx < n*m*p:
                    i = (block_idx // (m*p)) % n
                    j = (block_idx // p) % m
                    k = block_idx % p
                    U_init[i*m+j, c] = 1.0
                    V_init[j*p+k, c] = 1.0
                    W_init[k*n+i, c] = 1.0
            U_init += np.random.randn(n*m, target_rank) * 0.2
            V_init += np.random.randn(m*p, target_rank) * 0.2
            W_init += np.random.randn(p*n, target_rank) * 0.2
        elif init_strategy == 3:
            # Sparse random initialization
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            for c in range(target_rank):
                # Choose random positions
                u_pos = random.randint(0, n*m-1)
                v_pos = random.randint(0, m*p-1)
                w_pos = random.randint(0, p*n-1)
                U_init[u_pos, c] = random.choice([-1, 1])
                V_init[v_pos, c] = random.choice([-1, 1])
                W_init[w_pos, c] = random.choice([-1, 1])
            U_init += np.random.randn(n*m, target_rank) * 0.1
            V_init += np.random.randn(m*p, target_rank) * 0.1
            W_init += np.random.randn(p*n, target_rank) * 0.1
        else:
            # Fourier-like structured initialization
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            for c in range(target_rank):
                phase = 2 * np.pi * c / target_rank
                for idx in range(n*m):
                    U_init[idx, c] = np.cos(phase * (idx + 1)) * 0.5
                for idx in range(m*p):
                    V_init[idx, c] = np.cos(phase * (idx + 2)) * 0.5
                for idx in range(p*n):
                    W_init[idx, c] = np.cos(phase * (idx + 3)) * 0.5
            U_init += np.random.randn(n*m, target_rank) * 0.1
            V_init += np.random.randn(m*p, target_rank) * 0.1
            W_init += np.random.randn(p*n, target_rank) * 0.1
        
        def residuals(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        res.append(s - 1.0)
            return res
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=max_nfev)
            err = np.max(np.abs(result.fun))
            if err < best_err:
                best_err = err
                if err < 1e-6:
                    U = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                    W = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                    best_result = (U, V, W)
                    break
        except Exception:
            pass
    
    return best_result

def _snap_and_verify(U, V, W, n, m, p, tolerance=0.15):
    """
    Snap float matrices to half-integers and verify exactness.
    Returns (U_list, V_list, W_list) if successful, else None.
    """
    # Try snapping to half-integers
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    
    # Check if snapping is close enough
    if (np.max(np.abs(U_s - U)) <= tolerance and 
        np.max(np.abs(V_s - V)) <= tolerance and 
        np.max(np.abs(W_s - W)) <= tolerance):
        # Convert to list of lists of Fractions for exact verification
        U_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in U_s]
        V_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in V_s]
        W_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in W_s]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            U_out = [[float(x) for x in row] for row in U_s]
            V_out = [[float(x) for x in row] for row in V_s]
            W_out = [[float(x) for x in row] for row in W_s]
            return U_out, V_out, W_out
    
    # Try snapping to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    if (np.max(np.abs(U_i - U)) <= tolerance and 
        np.max(np.abs(V_i - V)) <= tolerance and 
        np.max(np.abs(W_i - W)) <= tolerance):
        U_frac = [[Fraction(int(x), 1) for x in row] for row in U_i]
        V_frac = [[Fraction(int(x), 1) for x in row] for row in V_i]
        W_frac = [[Fraction(int(x), 1) for x in row] for row in W_i]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            U_out = [[float(x) for x in row] for row in U_i]
            V_out = [[float(x) for x in row] for row in V_i]
            W_out = [[float(x) for x in row] for row in W_i]
            return U_out, V_out, W_out
    
    # Try snapping to quarters
    U_q = np.round(U * 4) / 4.0
    V_q = np.round(V * 4) / 4.0
    W_q = np.round(W * 4) / 4.0
    
    if (np.max(np.abs(U_q - U)) <= tolerance and 
        np.max(np.abs(V_q - V)) <= tolerance and 
        np.max(np.abs(W_q - W)) <= tolerance):
        U_frac = [[Fraction(int(round(x*4)), 4) for x in row] for row in U_q]
        V_frac = [[Fraction(int(round(x*4)), 4) for x in row] for row in V_q]
        W_frac = [[Fraction(int(round(x*4)), 4) for x in row] for row in W_q]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            U_out = [[float(x) for x in row] for row in U_q]
            V_out = [[float(x) for x in row] for row in V_q]
            W_out = [[float(x) for x in row] for row in W_q]
            return U_out, V_out, W_out
    
    return None

# EVOLVE-BLOCK-START
def _direct_search_3x3x2(max_rank=15, time_limit=30.0):
    """
    Direct search for 3x3x2 decomposition using structured patterns.
    """
    start = time.time()
    n, m, p = 3, 3, 2
    
    # Known lower bound: 12 (since 3*3*2 = 18, and we can potentially do better)
    # Try ranks from 12 to max_rank
    
    for rank in range(12, max_rank + 1):
        if time.time() - start > time_limit:
            break
        
        # Try various structured initializations
        for trial in range(20):
            if time.time() - start > time_limit:
                break
            
            np.random.seed(trial * 1000 + rank)
            random.seed(trial * 1000 + rank)
            
            # Initialize with sparse random values
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Create structured pattern
            for c in range(rank):
                # Choose which indices to activate
                if trial % 3 == 0:
                    # Diagonal-like pattern
                    i = (c // (m*p)) % n
                    j = (c // p) % m
                    k = c % p
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                elif trial % 3 == 1:
                    # Random sparse
                    u_idx = random.randint(0, n*m-1)
                    v_idx = random.randint(0, m*p-1)
                    w_idx = random.randint(0, p*n-1)
                    U[u_idx, c] = random.choice([-1, 1])
                    V[v_idx, c] = random.choice([-1, 1])
                    W[w_idx, c] = random.choice([-1, 1])
                else:
                    # Block pattern
                    block = c % 3
                    if block == 0:
                        for i in range(n):
                            U[i*m, c] = 1.0
                        for j in range(m):
                            V[j*p, c] = 1.0
                        for k in range(p):
                            W[k*n, c] = 1.0
                    elif block == 1:
                        for j in range(m):
                            U[j, c] = 1.0
                        for k in range(p):
                            V[k, c] = 1.0
                        for i in range(n):
                            W[i, c] = 1.0
                    else:
                        for i in range(n):
                            U[i, c] = 1.0
                        for j in range(m):
                            V[j, c] = 1.0
                        for k in range(p):
                            W[k*n+1, c] = 1.0
            
            # Add small random perturbation
            U += np.random.randn(n*m, rank) * 0.2
            V += np.random.randn(m*p, rank) * 0.2
            W += np.random.randn(p*n, rank) * 0.2
            
            # Optimize
            sol = _optimize_block(n, m, p, rank, seed=trial, max_nfev=1500)
            if sol is not None:
                snapped = _snap_and_verify(sol[0], sol[1], sol[2], n, m, p)
                if snapped is not None:
                    return snapped, rank
    
    return None, None

def _direct_search_3x3x3(max_rank=23, time_limit=45.0):
    """
    Direct search for 3x3x3 decomposition.
    """
    start = time.time()
    n, m, p = 3, 3, 3
    
    # Known: Strassen gives 7 for 2x2x2, so 3x3x3 might be around 19-23
    for rank in range(19, max_rank + 1):
        if time.time() - start > time_limit:
            break
        
        for trial in range(15):
            if time.time() - start > time_limit:
                break
            
            np.random.seed(trial * 2000 + rank)
            random.seed(trial * 2000 + rank)
            
            sol = _optimize_block(n, m, p, rank, seed=trial, max_nfev=2000)
            if sol is not None:
                snapped = _snap_and_verify(sol[0], sol[1], sol[2], n, m, p)
                if snapped is not None:
                    return snapped, rank
    
    return None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 15.0
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy 1: Direct optimization on full 3x3x5
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But we can do better with structure
    
    # First, try direct search on full problem
    for rank in range(25, 36):
        if time.time() > deadline:
            break
        
        for seed in range(8):
            if time.time() > deadline:
                break
            
            sol = _optimize_block(3, 3, 5, rank, seed=seed, max_nfev=2500)
            if sol is not None:
                snapped = _snap_and_verify(sol[0], sol[1], sol[2], 3, 3, 5)
                if snapped is not None:
                    U_list, V_list, W_list = snapped
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
        if best_rank < 30:  # Found something good, break
            break
    
    # Strategy 2: Decompose into 3x3x3 + 3x3x2
    # This can sometimes give better results due to structure
    if time.time() < deadline - 20:
        # Search for 3x3x3 decomposition
        result1, rank1 = _direct_search_3x3x3(max_rank=22, time_limit=30.0)
        
        if result1 is not None and time.time() < deadline - 10:
            # Search for 3x3x2 decomposition
            result2, rank2 = _direct_search_3x3x2(max_rank=14, time_limit=15.0)
            
            if result2 is not None:
                U1, V1, W1 = result1
                U2, V2, W2 = result2
                total_rank = rank1 + rank2
                
                if total_rank < best_rank:
                    # Construct combined matrices
                    U_comb = np.hstack([np.array(U1), np.array(U2)])
                    
                    V_comb = np.zeros((15, total_rank))
                    for j in range(3):
                        for k in range(3):
                            local_idx = j * 3 + k
                            global_idx = j * 5 + k
                            V_comb[global_idx, :rank1] = np.array(V1)[local_idx, :]
                    
                    for j in range(3):
                        for k_loc in range(2):
                            local_idx = j * 2 + k_loc
                            global_idx = j * 5 + (k_loc + 3)
                            V_comb[global_idx, rank1:] = np.array(V2)[local_idx, :]
                    
                    W_comb = np.zeros((15, total_rank))
                    for k in range(3):
                        for i in range(3):
                            local_idx = k * 3 + i
                            global_idx = k * 3 + i
                            W_comb[global_idx, :rank1] = np.array(W1)[local_idx, :]
                    
                    for k_loc in range(2):
                        for i in range(3):
                            local_idx = k_loc * 3 + i
                            global_idx = (k_loc + 3) * 3 + i
                            W_comb[global_idx, rank1:] = np.array(W2)[local_idx, :]
                    
                    # Verify
                    U_list = [[float(x) for x in row] for row in U_comb]
                    V_list = [[float(x) for x in row] for row in V_comb]
                    W_list = [[float(x) for x in row] for row in W_comb]
                    
                    if _verify_exact(U_list, V_list, W_list, 3, 3, 5):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = total_rank
    
    # Strategy 3: Try a few more direct searches with different seeds if time permits
    if time.time() < deadline - 5 and best_rank > 30:
        for rank in range(28, 32):
            if time.time() > deadline:
                break
            for seed in range(5):
                if time.time() > deadline:
                    break
                sol = _optimize_block(3, 3, 5, rank, seed=seed + 100, max_nfev=1500)
                if sol is not None:
                    snapped = _snap_and_verify(sol[0], sol[1], sol[2], 3, 3, 5)
                    if snapped is not None:
                        U_list, V_list, W_list = snapped
                        if rank < best_rank:
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = rank
                        break
    
    return best_result
# EVOLVE-BLOCK-END
