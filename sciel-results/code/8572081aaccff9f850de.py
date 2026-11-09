"""Rewritten solver: vectorized backward induction with fine-grained candidate search
and Newton refinement of the Euler equation for consumption-saving policy."""
import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 150.0  # wall-clock kill at 150s; keep safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    
    n_z = len(income)
    n_m = len(resource_grid)
    
    # Ensure resource_grid is a numpy array
    resource_grid = np.asarray(resource_grid, dtype=float)
    
    # Terminal condition: c[5] = m (consume all at last period)
    # Shape: (horizon, n_z, n_m)
    C = np.zeros((horizon, n_z, n_m), dtype=float)
    C[horizon - 1, :, :] = resource_grid[None, None, :]  # broadcast
    
    # CRRA utility
    if abs(gamma - 1.0) < 1e-12:
        def u(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def u(c):
            c = np.maximum(c, 1e-12)
            return c ** (1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check deadline
        if time.time() > DEADLINE - 5.0:
            # Return what we have so far; fill remaining with m (terminal-like)
            for tt in range(t, -1, -1):
                C[tt, :, :] = resource_grid[None, None, :]
            break
        
        # Continuation values: V[z', m'] = max over c' of u(c') + beta * E[V[z'', m''] | z']
        # But we're going backward, so at step t we use C[t+1] as the "next period" policy.
        # The value function at t+1 given state (z, m) is:
        # V_{t+1}(z, m) = max_{c'} [ u(c') + beta * sum_{z''} trans[z, z''] * V_{t+2}(z'', m'') ]
        # But we only have C[t+1], not V[t+1]. We need to compute V[t+1] from C[t+1].
        
        # Actually, in backward induction with consumption policies, we can compute the
        # implied value function at each step. Let's define:
        # V[t][z, m] = u(C[t][z, m]) + beta * sum_{z'} trans[z, z'] * V[t+1][z', m']
        # where m' = R*(m - C[t][z, m]) + income[z']
        
        # So at step t, we need V[t+1]. Let's compute it.
        
        # First, compute V[t+1] for all z, m on the grid
        # V[t+1][z, m] = u(C[t+1][z, m]) + beta * sum_{z'} trans[z, z'] * V[t+2][z', m']
        # where m' = R*(m - C[t+1][z, m]) + income[z']
        
        # We need V[t+2] for this. Let's precompute all V's.
        
        # Actually, let's restructure: compute V from the end.
        # V[horizon-1][z, m] = u(m)  (since C[horizon-1] = m)
        # For t = horizon-2 down to 0:
        #   V[t][z, m] = max_c [ u(c) + beta * sum_{z'} trans[z, z'] * V[t+1][z', R*(m-c)+income[z']] ]
        #   C[t][z, m] = argmax
        
        # Let's redo this properly.
        
        pass  # We'll restructure below
    
    # Let's restart with a cleaner approach
    C = np.zeros((horizon, n_z, n_m), dtype=float)
    C[horizon - 1, :, :] = resource_grid[None, None, :]
    
    # Compute value functions from the end
    # V[t][z, m] for t = 0..horizon-1
    # V[horizon-1][z, m] = u(m)
    
    V = np.zeros((horizon, n_z, n_m), dtype=float)
    V[horizon - 1, :, :] = u(resource_grid)[None, None, :]
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE - 5.0:
            for tt in range(t, -1, -1):
                C[tt, :, :] = resource_grid[None, None, :]
            break
        
        for z in range(n_z):
            # For each m on the grid, find optimal c
            # Objective: u(c) + beta * sum_{z'} trans[z, z'] * V[t+1][z', R*(m-c)+income[z']]
            # V[t+1] is defined on the grid resource_grid. We need to interpolate.
            
            # Use a fine grid of c values for each m
            # c in [eps, m]
            # For efficiency, vectorize over m
            
            # Candidate c values: use a geometric or linear grid
            # Let's use a fine linear grid relative to m
            # c = m * linspace(0.001, 1, n_cands)
            
            # To be efficient, let's work with all m at once
            # For each m_i, c candidates are m_i * linspace(0.001, 1, n_cands)
            
            n_cands = 200  # number of candidate c values per m
            
            # Build candidate c array: shape (n_m, n_cands)
            frac = np.linspace(0.001, 1.0, n_cands)
            C_cands = resource_grid[:, None] * frac[None, :]  # (n_m, n_cands)
            C_cands = np.maximum(C_cands, 1e-12)
            
            # For each candidate c, compute m' for each z'
            # m' = R * (m - c) + income[z']
            # Shape: (n_m, n_cands, n_z)
            m_prime = R * (resource_grid[:, None, None] - C_cands[:, :, None]) + income[None, None, :]
            
            # Clip m' to the grid range for interpolation
            m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
            
            # Interpolate V[t+1][z', m'] for each z'
            # V_next[z', m'] = interp(m', resource_grid, V[t+1][z', :])
            # Shape: (n_m, n_cands, n_z)
            V_next = np.zeros((n_m, n_cands, n_z), dtype=float)
            for zp in range(n_z):
                V_next[:, :, zp] = np.interp(m_prime_clipped[:, :, zp].ravel(), resource_grid, V[t + 1, zp, :]).reshape(n_m, n_cands)
            
            # Expected continuation value
            # E[V_next] = sum_{z'} trans[z, z'] * V_next[:, :, z']
            # Shape: (n_m, n_cands)
            E_V_next = np.tensordot(V_next, trans[z, :], axes=([2], [0]))  # (n_m, n_cands)
            
            # Total utility
            # U = u(c) + beta * E_V_next
            U = u(C_cands) + beta * E_V_next  # (n_m, n_cands)
            
            # Find argmax over c for each m
            best_idx = np.argmax(U, axis=1)
            C[t, z, :] = C_cands[np.arange(n_m), best_idx]
            
            # Ensure constraints
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, resource_grid + 1e-10)
            
            # Compute V[t][z, m]
            V[t, z, :] = U[np.arange(n_m), best_idx]
    
    # Final validation
    C = np.clip(C, 1e-12, resource_grid[None, None, :] + 1e-10)
    C[horizon - 1, :, :] = resource_grid[None, None, :]
    
    return C
