import numpy as np
import time
from scipy.optimize import minimize

def _build_idx_cache(n):
    """Precompute index arrays for fast gate application."""
    dim = 1 << n
    ry_cache = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_cache.append((idx0, idx1))
    
    cz_cache = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_cache.append(idx_both)
        
    return {'ry': ry_cache, 'cz': cz_cache}

def _circuit_state(n, layers, params, idx_cache, store_states=False):
    """Compute state from circuit using precomputed indices.
    If store_states is True, returns (state, forward_states) where forward_states[l][q] is the state BEFORE applying Ry(l, q).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    forward_states = None
    if store_states:
        forward_states = [[None for _ in range(n)] for _ in range(layers)]

    for layer in range(layers):
        for q in range(n):
            if store_states:
                forward_states[layer][q] = state.copy()
            
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    if store_states:
        return state, forward_states
    return state

def _hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            H[b, b] += -coupling * (1 - 2 * bi) * (1 - 2 * bj)
    for i in range(n):
        for b in range(dim):
            bf = b ^ (1 << i)
            if b < bf:
                H[b, bf] -= field
                H[bf, b] -= field
    return H

def _energy_and_grad_adjoint(n, layers, params, idx_cache, H, e_ground):
    """Compute energy and analytical gradient using adjoint method."""
    state, fwd_states = _circuit_state(n, layers, params, idx_cache, store_states=True)
    energy = np.vdot(state, H @ state).real
    obj_val = (energy - e_ground) / n
    
    # Initialize adjoint
    # dE/dpsi* = H psi. So adjoint starts as H @ psi / n
    adj = (H @ state) / n
    
    grad = np.zeros((layers, n), dtype=np.float64)
    
    # Backward pass
    # We need to propagate adjoint backward through CZ and Ry gates.
    # Current adj is at the end (after last Ry layer).
    
    for layer in range(layers - 1, -1, -1):
        # Propagate adjoint through Ry gates in reverse order
        # We need to handle the CZ gates between layers.
        
        # First, undo the Ry gates of this layer in reverse order
        # But we also need to undo the CZ gates that preceded this layer (if any)
        # Wait, the structure is:
        # Layer l: Ry -> CZ (if l < L-1)
        # So after Layer l, we have CZ.
        # Backward:
        # Start at end.
        # Undo Layer L-1 Ry gates.
        # Then, if we are going to Layer L-2, we must undo the CZ that was applied after Layer L-2?
        # No, the CZ was applied AFTER Layer L-2 and BEFORE Layer L-1.
        # So in backward pass:
        # 1. Undo Ry of Layer L-1.
        # 2. Undo CZ (which connects L-2 and L-1).
        # 3. Undo Ry of Layer L-2.
        # ...
        
        # Let's refine the loop structure.
        
        # We are at the state after Layer `layer` Ry gates.
        # The adjoint `adj` corresponds to this point.
        
        # Calculate gradients for Ry gates in this layer
        for q in range(n - 1, -1, -1):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            
            # Get forward state BEFORE this Ry gate
            psi_in = fwd_states[layer][q]
            
            # Get adjoint AFTER this Ry gate (which is current `adj`)
            # dE/dtheta = 2 Re < adj | dRy/dtheta | psi_in >
            # dRy/dtheta = 0.5 * [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
            # Let M = dRy/dtheta.
            # M @ psi_in has components:
            # (M psi_in)_0 = -0.5 * s * psi_in[idx0] - 0.5 * c * psi_in[idx1]
            # (M psi_in)_1 = 0.5 * c * psi_in[idx0] - 0.5 * s * psi_in[idx1]
            
            # We need < adj | M psi_in >
            # = conj(adj[idx0]) * (M psi_in)_0 + conj(adj[idx1]) * (M psi_in)_1
            
            # Note: `adj` is the adjoint state AFTER the gate.
            # `psi_in` is the state BEFORE the gate.
            
            m_psi0 = -0.5 * s * psi_in[idx0] - 0.5 * c * psi_in[idx1]
            m_psi1 = 0.5 * c * psi_in[idx0] - 0.5 * s * psi_in[idx1]
            
            # adj is complex, psi is complex
            term = np.vdot(adj[idx0], m_psi0) + np.vdot(adj[idx1], m_psi1)
            # Wait, np.vdot(a, b) = sum(conj(a)*b).
            # We want sum(conj(adj) * (M psi)).
            # So np.vdot(adj[idx0], m_psi0) is correct for the first part.
            
            grad[layer, q] = 2.0 * term.real
            
            # Now update adjoint to BEFORE this Ry gate
            # adj_new = Ry^dag @ adj
            # Ry = [[c, -s], [s, c]]
            # Ry^dag = [[c, s], [-s, c]]
            # adj_new[idx0] = c * adj[idx0] + s * adj[idx1]
            # adj_new[idx1] = -s * adj[idx0] + c * adj[idx1]
            
            old_adj0 = adj[idx0].copy()
            old_adj1 = adj[idx1].copy()
            adj[idx0] = c * old_adj0 + s * old_adj1
            adj[idx1] = -s * old_adj0 + c * old_adj1
        
        # Now adj is BEFORE the Ry gates of this layer.
        # If this is not the first layer, we must undo the CZ gates that were applied BEFORE this layer's Ry gates?
        # No, the CZ gates were applied AFTER the previous layer's Ry gates.
        # So the sequence is:
        # ... -> Ry(L-2) -> CZ -> Ry(L-1) -> End
        # Backward:
        # End -> Undo Ry(L-1) -> [Now at state after CZ] -> Undo CZ -> [Now at state after Ry(L-2)] -> Undo Ry(L-2) ...
        
        # So after undoing all Ry gates of layer `layer`, if `layer > 0`, we are at the state AFTER the CZ gates that connect `layer-1` and `layer`.
        # We need to undo those CZ gates to get to the state AFTER Ry of `layer-1`.
        
        if layer > 0:
            # Undo CZ gates in reverse order
            for q in range(n - 2, -1, -1):
                idx_both = idx_cache['cz'][q]
                adj[idx_both] = -adj[idx_both]
                
    return obj_val, grad

def _mean_field_angle(n, coupling, field):
    """Compute mean-field optimal angle."""
    ratio = field / coupling
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        return np.pi / 2
    elif s_target < -1.0:
        return -np.pi / 2
    else:
        return np.arcsin(s_target)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Precompute
    idx_cache = _build_idx_cache(n)
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p, idx_cache)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        p = xf.reshape((layers, n))
        val, grad = _energy_and_grad_adjoint(n, layers, p, idx_cache, H, e_ground)
        return val, grad.flatten()
    
    # Mean-field optimal angle
    base = _mean_field_angle(n, coupling, field)
    
    # Initial guesses
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    
    # Layer-dependent
    for scale in [0.5, 0.8, 1.2]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.5 + 0.5 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating
    g = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g[l, q] = base * (1 if q % 2 == 0 else -1)
    guesses.append(g)
    
    # Random guesses
    np.random.seed(42)
    for _ in range(3): # Reduced from 5
        g = np.random.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
        
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Optimize from each guess with gradient
    for g in guesses:
        if time.time() > deadline - 10:
            break
        x0 = g.flatten()
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Polish from best with gradient
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B', jac=True,
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
