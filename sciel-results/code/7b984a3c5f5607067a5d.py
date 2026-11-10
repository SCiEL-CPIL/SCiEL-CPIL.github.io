"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START = time.time()

def _check_deadline():
    return (time.time() - _START) > _DEADLINE

def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _maxcut_exact(n, edges):
    if n == 0:
        return 0
    best = 0
    for mask in range(1, 1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _qaoa_expectation(n, edges, p, gammas, betas):
    adj = _build_adj(n, edges)
    # State vector as 2^n complex array
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge masks for efficient update
    edge_list = list(edges)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
        # Z_u Z_v phase is +1 if same bit, -1 if different
        # So phase is -i * gamma * (1 - z_u z_v) / 2
        # If same bit: -i * gamma * 0 = 0
        # If different bit: -i * gamma * 1 = -i * gamma
        
        # Vectorized phase calculation
        # For each basis state, we need to count how many edges have different bits
        # This is expensive to do naively for 2^n states.
        # However, n is small (<=10), so 2^n <= 1024.
        
        phases = np.zeros(dim, dtype=np.float64)
        for u, v in edge_list:
            # Create a mask for bits u and v
            # We can iterate or use vectorization
            # Let's iterate over states for clarity, or better:
            # We can precompute bit values for all states
            
            pass # Placeholder for optimization if needed
            
        # Efficient approach for small n:
        # Precompute bit arrays
        bits_u = np.array([(i >> u) & 1 for i in range(dim)])
        # This is O(n * 2^n) which is fine for n=10.
        
        # Actually, let's just loop over edges and update phases
        # Since n is small, dim is small.
        
        # Reset phases
        phases = np.zeros(dim, dtype=np.float64)
        
        for u, v in edge_list:
            # bits_u[i] is bit u of state i
            # bits_v[i] is bit v of state i
            # diff = 1 if bits differ, 0 if same
            # We can compute this vectorized
            
            # Extract bits
            # bit_u = (state_index >> u) & 1
            # bit_v = (state_index >> v) & 1
            
            # Vectorized extraction
            indices = np.arange(dim)
            bit_u = (indices >> u) & 1
            bit_v = (indices >> v) & 1
            diff = bit_u ^ bit_v # 1 if different
            
            phases += diff * gamma
        
        # Apply phase
        state *= np.exp(-0.5j * phases) # Wait, C = sum (1-ZuZv)/2.
        # Let's re-verify the Hamiltonian.
        # MaxCut Hamiltonian H_C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
        # Eigenvalues are 0 (if same) and 1 (if different).
        # So exp(-i gamma H_C) applies phase exp(-i gamma * 1) if different, exp(0) if same.
        # So phase is -gamma * (number of cut edges in state).
        
        # My phase calculation above: phases += diff * gamma
        # So state *= exp(-i * phases) is correct.
        
        state *= np.exp(-1j * phases)
        
        # Apply exp(-i beta sum X)
        # This is a Hadamard-like rotation on each qubit? No, exp(-i beta X) on a single qubit.
        # X = [[0,1],[1,0]]
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # This is a rotation around X axis.
        # We can apply this to each qubit independently.
        
        # For each qubit, we reshape the state, apply the rotation, and reshape back.
        # This is O(n * 2^n).
        
        c = np.cos(beta)
        s = np.sin(beta)
        
        for q in range(n):
            # Reshape state to isolate qubit q
            # Shape: (2^q, 2, 2^(n-1-q)) ? No, standard qubit ordering.
            # Let's use a general reshape method.
            
            # The state is indexed by bits b_{n-1} ... b_0.
            # Qubit q corresponds to bit q.
            
            # We can reshape to (2^q, 2, 2^(n-q-1)) if qubit q is the middle dimension?
            # Let's check indices.
            # Index i. Bit q is (i >> q) & 1.
            # If we reshape to (2^q, 2, 2^(n-q-1)), the middle index corresponds to bit q.
            # Let's verify.
            # i = a * 2^(n-q-1) + b * 2^q + c ? No.
            # Standard reshape for qubit q:
            # shape = (2**q, 2, 2**(n-q-1))
            # This works if the qubit is in the "middle" of the tensor product in that specific order.
            
            shape = (2**q, 2, 2**(n-q-1))
            state_reshaped = state.reshape(shape)
            
            # Apply rotation: [c, -i s; -i s, c]
            # New state[...0...] = c * state[...0...] - i s * state[...1...]
            # New state[...1...] = -i s * state[...0...] + c * state[...1...]
            
            new_state_reshaped = np.empty_like(state_reshaped)
            new_state_reshaped[..., 0, :] = c * state_reshaped[..., 0, :] - 1j * s * state_reshaped[..., 1, :]
            new_state_reshaped[..., 1, :] = -1j * s * state_reshaped[..., 0, :] + c * state_reshaped[..., 1, :]
            
            state = new_state_reshaped.reshape(dim)
            
        if _check_deadline():
            break
            
    # Calculate expectation of H_C
    # <H_C> = sum_{(u,v) in E} 0.5 * (1 - <Z_u Z_v>)
    # <Z_u Z_v> = sum_state |psi_state|^2 * z_u(state) * z_v(state)
    
    exp_val = 0.0
    probs = np.abs(state)**2
    
    for u, v in edge_list:
        indices = np.arange(dim)
        bit_u = (indices >> u) & 1
        bit_v = (indices >> v) & 1
        z_u = 1 - 2 * bit_u
        z_v = 1 - 2 * bit_v
        zz = z_u * z_v
        exp_zz = np.sum(probs * zz)
        exp_val += 0.5 * (1 - exp_zz)
        
    return exp_val

def _optimize_angles_for_graph(n, edges, p):
    best_val = -np.inf
    best_params = np.zeros(p * 2)
    
    # Initial guess
    x0 = np.tile([0.35, 0.25], p)
    
    # Try a few random restarts if time permits
    restarts = 3
    for r in range(restarts):
        if r > 0:
            x0 = x0 + np.random.normal(0, 0.5, p * 2)
            x0 = np.clip(x0, 0, np.pi)
            
        def objective(x):
            if _check_deadline():
                return 0 # Minimize 0, but we check deadline in loop too
            gammas = x[:p]
            betas = x[p:]
            # Maximize expectation, so minimize negative expectation
            return -_qaoa_expectation(n, edges, p, gammas, betas)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < -best_val: # Wait, objective is -exp. So -res.fun is exp.
                # We want to maximize exp, so minimize -exp.
                # If res.fun (which is -exp) is smaller, exp is larger.
                # So if res.fun < -best_val? No.
                # Let val = -res.fun.
                # If val > best_val, update.
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x
        except Exception:
            pass
            
        if _check_deadline():
            break
            
    return best_params

def angles(n, edges, p):
    # Check deadline
    if _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))
        
    # Optimize
    params = _optimize_angles_for_graph(n, edges, p)
    
    gammas = params[:p]
    betas = params[p:]
    
    result = np.column_stack((gammas, betas))
    return result
# EVOLVE-BLOCK-END
