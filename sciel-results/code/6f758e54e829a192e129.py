import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 280.0
_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # Value function V[t, q_idx]
    V = np.zeros((N + 1, n_q))
    # Optimal depths opt_b[t, q_idx], opt_a[t, q_idx]
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Grid for depths
    # We use a finer grid but limit the max depth to a reasonable range.
    # The optimal depth d* satisfies roughly: d* + (1/k) + ... = 0 in log space?
    # Let's analyze the condition for optimal d.
    # Let f(d) = d(1-e^{-c e^{-kd}}) + C(1-e^{-c e^{-kd}})
    # f'(d) = 1 - e^{-c e^{-kd}} + d c k e^{-kd} e^{-c e^{-kd}} + C c k e^{-kd} e^{-c e^{-kd}}
    # = 1 - e^{-c e^{-kd}} + c k e^{-kd} e^{-c e^{-kd}} (d + C)
    # As d->inf, e^{-kd} -> 0, f'(d) -> 1.
    # As d->0, e^{-kd} -> 1, f'(d) = 1 - e^{-c} + ck e^{-c} C.
    # If C is large negative, f'(0) < 0, so there is an interior maximum.
    # The maximum is where 1 - e^{-c e^{-kd}} + ck e^{-kd} e^{-c e^{-kd}} (d + C) = 0.
    
    # We will use a grid that covers the likely range.
    # For k=1.5, 5/k = 3.33.
    # For k=0.75, 5/k = 6.66.
    # Let's use 10/k to be safe, but that might be too wide for a fixed number of points.
    # However, the function becomes flat after a certain point.
    # Let's use 800 points on [0, 5/k].
    
    max_depth = 5.0 / k
    grid_size = 800
    depths = np.linspace(0, max_depth, grid_size)
    # Precompute P(d) for grid
    P_grid = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Precompute d * P(d) and P(d) * (d + C) type terms?
    # J(b, a) = b P(b) + a P(a) + V(q) + P(b) delta_plus + P(a) delta_minus + P(b)P(a)(-delta_plus - delta_minus)
    # = P(b)[b + delta_plus - P(a)(delta_plus + delta_minus)] + a P(a) + V(q) + P(a) delta_minus
    
    # For a fixed a, we want to maximize over b:
    # P(b) [ b + C_b ] where C_b = delta_plus - P(a)(delta_plus + delta_minus)
    
    # Let's precompute for each grid point i:
    # P_i, d_i, d_i * P_i
    # We can vectorize the inner loop over b for each a?
    # No, C_b depends on a.
    # But we can precompute a matrix M[i, j] = P_i * (d_j + ...)? No.
    
    # Let's try to vectorize the b-loop for a given a.
    # We need argmax_b P_b * (b + C_b).
    # This is an element-wise product of P_b and (b + C_b).
    # We can compute this for all b in a vectorized operation.
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            # Fallback: use a simple heuristic or seed-like policy
            # Seed was constant 1/k? Or something else.
            # Let's just return current V (which is 0) and default depths.
            # Actually, if we haven't computed V[t], it's 0.
            # We should probably fill the rest with a reasonable guess.
            # 1/k is a decent default.
            for tt in range(t, N):
                opt_b[tt, :] = 1.0 / k
                opt_a[tt, :] = 1.0 / k
            return opt_b, opt_a
        
        V_next = V[t + 1, :]
        
        # Precompute delta_minus and delta_plus for all q
        # delta_minus[q] = V(q-1) - V(q) for q > -Q, else 0
        # delta_plus[q] = V(q+1) - V(q) for q < Q, else 0
        delta_minus = np.zeros(n_q)
        delta_plus = np.zeros(n_q)
        
        # For q in 1..n_q-1 (q_idx), delta_minus corresponds to V[q_idx-1] - V[q_idx]
        # Wait, q_indices[qi] = -Q + qi.
        # If qi > 0, q > -Q.
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        # delta_minus[0] is 0 (q=-Q, no ask fill)
        
        if n_q > 1:
            delta_plus[:-1] = V_next[1:] - V_next[:-1]
        # delta_plus[n_q-1] is 0 (q=Q, no bid fill)
        
        # Now iterate over q
        # We can vectorize over q as well?
        # The optimal b and a depend on delta_minus[qi] and delta_plus[qi].
        # For each qi, we solve the 2D optimization.
        
        # Let's try to vectorize the q-loop.
        # For each qi, we have C_minus = delta_minus[qi], C_plus = delta_plus[qi].
        # We want to find best_b, best_a.
        
        # This might be hard to fully vectorize because argmax is involved.
        # But we can use a "best grid point" approach.
        
        # Let's stick to the loop over qi, but vectorize the inner b-loop.
        for qi in range(n_q):
            c_minus = delta_minus[qi]
            c_plus = delta_plus[qi]
            
            if qi == 0:
                # q = -Q, only bid fill possible (a=0)
                # Maximize b P(b) + P(b) c_plus
                # = P(b) (b + c_plus)
                # Vectorized
                vals = P_grid * (depths + c_plus)
                best_idx = np.argmax(vals)
                opt_b[t, qi] = depths[best_idx]
                opt_a[t, qi] = 0.0
                V[t, qi] = V_next[qi] + vals[best_idx]
            elif qi == n_q - 1:
                # q = Q, only ask fill possible (b=0)
                # Maximize a P(a) + P(a) c_minus
                # = P(a) (a + c_minus)
                vals = P_grid * (depths + c_minus)
                best_idx = np.argmax(vals)
                opt_b[t, qi] = 0.0
                opt_a[t, qi] = depths[best_idx]
                V[t, qi] = V_next[qi] + vals[best_idx]
            else:
                # General case
                # Maximize over a, b:
                # P(b)[b + c_plus - P(a)(c_plus + c_minus)] + a P(a) + V_next[qi] + P(a) c_minus
                
                # Let S = c_plus + c_minus
                S = c_plus + c_minus
                
                # For each a (vectorized over a-grid):
                # C_b = c_plus - P(a) * S
                # We want argmax_b P(b) * (b + C_b)
                # Let M[i, j] = P_i * (d_j + c_plus - P_i * S)  -- No, i is a, j is b?
                # Let's define:
                # For each a-index i:
                #   coeff_b = c_plus - P_grid[i] * S
                #   vals_b = P_grid * (depths + coeff_b)
                #   best_b_idx[i] = argmax(vals_b)
                #   best_val_b[i] = vals_b[best_b_idx[i]]
                #   total_val[i] = best_val_b[i] + a P(a) + V_next[qi] + P(a) c_minus
                #   best_a_idx = argmax(total_val)
                
                # Vectorized computation:
                # P_grid shape (G,)
                # depths shape (G,)
                # coeff_b shape (G,) for each i? No, coeff_b is a scalar for a given i.
                # We need a matrix of vals_b for all i, j.
                # vals_b[i, j] = P_j * (d_j + c_plus - P_i * S)
                # = P_j * d_j + P_j * (c_plus - P_i * S)
                
                # Let's construct this matrix.
                # P_j * d_j is a vector of length G.
                # P_j * (c_plus - P_i * S) = P_j * c_plus - P_j * P_i * S
                # Let P_col = P_grid[:, None]  # (G, 1)
                # Let P_row = P_grid[None, :]  # (1, G)
                # Let d_col = depths[:, None]  # (G, 1) -- Wait, j is the column index for b?
                # Let's say rows are a (i), cols are b (j).
                # P_j is P_grid[j]. d_j is depths[j].
                # P_i is P_grid[i].
                
                # We want M[i, j] = P_grid[j] * (depths[j] + c_plus - P_grid[i] * S)
                
                # P_grid[j] * depths[j] is a vector v_bd of length G.
                # P_grid[j] * c_plus is a vector v_pc of length G.
                # P_grid[i] * S is a vector v_ps of length G.
                # P_grid[j] * P_grid[i] is an outer product P_grid[:, None] * P_grid[None, :]?
                # Wait, i is row, j is col.
                # P_grid[i] is P_grid[:, None] (G, 1)
                # P_grid[j] is P_grid[None, :] (1, G)
                
                # M = (P_grid[None, :] * depths[None, :]) + (P_grid[None, :] * c_plus) - (P_grid[:, None] * S * P_grid[None, :])
                # M[i, j] = P_j d_j + P_j c_plus - P_i S P_j
                
                # Let's compute M.
                Pd = P_grid * depths  # (G,)
                Pc = P_grid * c_plus  # (G,)
                PPs = P_grid[:, None] * P_grid[None, :] * S  # (G, G)
                
                M = (Pd[None, :] + Pc[None, :] - PPs).T  # (G, G) where M[i, j] corresponds to a=i, b=j?
                # Wait, PPs[i, j] = P_i * P_j * S.
                # We want M[i, j] = P_j (d_j + c_plus - P_i S) = P_j d_j + P_j c_plus - P_i P_j S.
                # So M[i, j] = Pd[j] + Pc[j] - PPs[i, j].
                # Pd[None, :] is (1, G) with Pd[j].
                # Pc[None, :] is (1, G) with Pc[j].
                # PPs is (G, G) with PPs[i, j].
                # So M = (Pd[None, :] + Pc[None, :] - PPs).T ?
                # (Pd[None, :] + Pc[None, :]) is (1, G).
                # Broadcasting (1, G) - (G, G) gives (G, G) where row i is (Pd+Pc) - PPs[i, :].
                # This is exactly M[i, j]. So no .T needed if we view rows as i.
                M = (Pd[None, :] + Pc[None, :] - PPs)
                
                # Now, for each i (a-index), we want argmax over j (b-index) of M[i, j].
                best_b_idx = np.argmax(M, axis=1)  # (G,)
                best_val_b = np.take_along_axis(M, best_b_idx[:, None], axis=1).flatten()  # (G,)
                
                # Now compute total value for each i:
                # total_val[i] = best_val_b[i] + d_i P_i + V_next[qi] + P_i c_minus
                # d_i P_i is Pd[i].
                # P_i c_minus is P_grid[i] * c_minus.
                
                total_val = best_val_b + Pd + V_next[qi] + P_grid * c_minus
                
                best_a_idx = np.argmax(total_val)
                
                opt_b[t, qi] = depths[best_b_idx[best_a_idx]]
                opt_a[t, qi] = depths[best_a_idx]
                V[t, qi] = total_val[best_a_idx]
                
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _CACHE:
        _CACHE[key] = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    
    opt_b, opt_a = _CACHE[key]
    
    t_idx = int(t / (T / steps))
    if t_idx >= steps:
        t_idx = steps - 1
        
    # Map inventory to index
    # inventory is in [-Q, Q]
    # inventory is an array of shape (4000,)
    q_idx = inventory + Q
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    return b_depths, a_depths
