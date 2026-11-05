"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs.

Vectorized state-vector approach with true analytic gradient via backprop:
- Full state vector simulation with vectorized mixer (reshape-based, no per-qubit loop)
- True analytic gradient via adjoint/backpropagation (1 forward + 1 backward per eval)
- Multiple restarts with polish passes
- For n<=10, dim=2^n <= 1024, which is manageable
"""
import numpy as np
import time
import scipy.optimize

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    edge_list = list(edges)
    for mask in range(dim):
        cut = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def angles(n, edges, p):
    global _start_time, _DEADLINE
    start_time = time.time()
    
    if n > 12:
        # Fallback for large n
        return np.array([[0.4, 0.35], [0.4, 0.35]])
    
    # Precompute cut values once
    dim = 1 << n
    edge_list = list(edges)
    
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cv += 1
        cut_vals[mask] = cv
    
    # Precompute: reshape shape for vectorized mixer
    # State is indexed by bitstring; bit i corresponds to qubit i
    # We reshape to (2,)*n so that axis i corresponds to qubit i
    # In numpy, the last axis varies fastest. For a state indexed by integer mask,
    # bit i of mask selects the i-th qubit. If we reshape to shape (2,)*n,
    # then the element at index [d_{n-1}, d_{n-2}, ..., d_0] corresponds to
    # mask = sum(d_j * 2^j). So axis 0 is qubit n-1, axis n-1 is qubit 0.
    # We want to apply rotation on qubit i, which is axis (n-1-i) in the reshaped array.
    # Actually let's be careful: mask = sum_{j=0}^{n-1} d_j 2^j
    # If we reshape flat array of size 2^n to shape (2,)*n, numpy uses C-order,
    # so the last index varies fastest. Index [a_{n-1}, a_{n-2}, ..., a_0] maps to
    # flat index = a_{n-1}*2^{n-1} + a_{n-2}*2^{n-2} + ... + a_0*2^0.
    # So a_0 is the coefficient of 2^0, which is bit 0 = qubit 0.
    # Therefore axis (n-1) in the reshaped array corresponds to qubit 0.
    # Axis k in reshaped array corresponds to qubit (n-1-k).
    
    def _apply_mixer(psi, b):
        """Apply exp(-i b * sum_i X_i) to state vector psi using vectorized reshape."""
        cb = np.cos(b)
        sb = np.sin(b)
        # Reshape to n-qubit form
        psi_r = psi.reshape([2] * n)
        # For each qubit i, apply rotation on axis (n-1-i)
        # Rotation matrix: [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
        # In terms of the state components:
        # new[0] = cos(b)*old[0] - i*sin(b)*old[1]
        # new[1] = -i*sin(b)*old[0] + cos(b)*old[1]
        for i in range(n):
            axis = n - 1 - i
            # Move target axis to front for easier manipulation
            psi_r = np.moveaxis(psi_r, axis, 0)
            # Apply rotation: shape is (2, 2^{n-1}, ...)
            a = psi_r[0]  # component with qubit i = 0
            b_comp = psi_r[1]  # component with qubit i = 1
            psi_r[0] = cb * a - 1j * sb * b_comp
            psi_r[1] = -1j * sb * a + cb * b_comp
            # Move axis back
            psi_r = np.moveaxis(psi_r, 0, axis)
        return psi_r.reshape(dim)
    
    def _forward(g0, b0, g1, b1):
        """Forward pass returning all intermediate states."""
        psi_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        # Cost layer 0
        phase0 = np.exp(-1j * g0 * cut_vals)
        psi0 = psi_init * phase0
        
        # Mixer 0
        psi1 = _apply_mixer(psi0, b0)
        
        # Cost layer 1
        phase1 = np.exp(-1j * g1 * cut_vals)
        psi2 = psi1 * phase1
        
        # Mixer 1
        psi3 = _apply_mixer(psi2, b1)
        
        return psi0, psi1, psi2, psi3
    
    def _compute_value_and_grad(params):
        """Compute expected cut value and analytic gradient via backprop.
        
        Returns (value, grad) where grad is d(value)/d(params).
        """
        g0, b0, g1, b1 = params
        
        # Forward pass
        psi0, psi1, psi2, psi3 = _forward(g0, b0, g1, b1)
        
        # Value: E = sum_k |psi3[k]|^2 * cut_vals[k]
        probs = np.abs(psi3) ** 2
        value = np.dot(probs, cut_vals)
        
        # Backpropagation for analytic gradient
        # E = sum_k cut_vals[k] * psi3[k] * conj(psi3[k])
        # dE/d(conj(psi3)) = cut_vals * psi3
        # dE/d(psi3) = cut_vals * conj(psi3)
        # We use the convention: dE = sum (dE/dpsi * dpsi + dE/dconj_psi * dconj_psi)
        # For real-valued parameters, dE/dparam = 2*Re(sum conj(dpsi/dparam) * dE/dconj_psi)
        # Actually for complex: dE/dparam = sum (dE/dpsi * dpsi/dparam + dE/dconj_psi * dconj_psi/dparam)
        # Since params are real, dpsi/dparam = conj(dconj_psi/dparam), so:
        # dE/dparam = 2*Re(sum dE/dconj_psi * conj(dpsi/dparam))
        # Wait, let me be more careful.
        # E is real. E = <psi3|C|psi3> where C is diagonal with entries cut_vals.
        # dE/dparam = <dpsi3/dparam|C|psi3> + <psi3|C|dpsi3/dparam>
        #           = 2*Re(<dpsi3/dparam|C|psi3>)
        # <dpsi3/dparam|C|psi3> = sum_k conj(dpsi3[k]/dparam) * cut_vals[k] * psi3[k]
        
        # Backward pass: compute adjoint (co-gradient) at each stage
        # Start: adj_psi3 = C |psi3> = cut_vals * psi3
        adj_psi3 = cut_vals * psi3  # this is C|psi3>, used in <adj|psi3> form
        
        # Back through Mixer 1: psi3 = M(b1) psi2
        # M(b) = exp(-i b sum X_i), which is unitary.
        # dpsi3/db1 = -i * (sum X_i) * M(b1) psi2 = -i * (sum X_i) psi3
        # adj_psi2 = M(b1)^dagger adj_psi3
        # But we also need dE/db1 = 2*Re(<dpsi3/db1|C|psi3>) = 2*Re(<-i sum X psi3|C|psi3>)
        # Actually, let's just do proper backprop.
        
        # The mixer M(b) is unitary. Backprop through M(b):
        # adj_in = M(b)^dagger * adj_out
        # dE/db = 2*Re(<psi_out| (dM/db)^dagger * adj_out>) ... 
        # Let me use a cleaner approach.
        
        # For a unitary U(b) with dU/db, and output y = U(b) x:
        # adj_x = U(b)^dagger * adj_y
        # dE/db = 2*Re(<y| (dU/db)^dagger adj_y>) ... hmm this is getting complicated.
        
        # Let me just use the direct formula for each parameter.
        
        # dE/dg0: g0 appears in psi0 = exp(-i g0 C) |init>
        # dpsi0/dg0 = -i C psi0
        # Need to propagate -i C psi0 through mixer0, cost1, mixer1 to get dpsi3/dg0
        # dpsi3/dg0 = M(b1) * diag(exp(-i g1 C)) * M(b0) * (-i C psi0)
        
        # dE/db0: b0 appears in psi1 = M(b0) psi0
        # dpsi1/db0 = -i (sum X) psi1
        # dpsi3/db0 = M(b1) * diag(exp(-i g1 C)) * (-i (sum X) psi1)
        
        # dE/dg1: g1 appears in psi2 = diag(exp(-i g1 C)) psi1
        # dpsi2/dg1 = -i C psi2
        # dpsi3/dg1 = M(b1) * (-i C psi2)
        
        # dE/db1: b1 appears in psi3 = M(b1) psi2
        # dpsi3/db1 = -i (sum X) psi3
        
        # For each, dE/dparam = 2*Re(sum_k conj(dpsi3[k]/dparam) * cut_vals[k] * psi3[k])
        #            = 2*Re(<dpsi3/dparam | C psi3>)
        #            = 2*Re(dot(conj(dpsi3/dparam), cut_vals * psi3))
        
        # Compute C|psi3> = cut_vals * psi3
        C_psi3 = cut_vals * psi3
        
        # dpsi3/dg0 = M(b1) * diag(exp(-i g1 C)) * M(b0) * (-i C psi0)
        # Step 1: v = -i C psi0 = -i * cut_vals * psi0
        v_g0 = -1j * cut_vals * psi0
        # Step 2: apply M(b0)
        v_g0 = _apply_mixer(v_g0, b0)
        # Step 3: apply diag(exp(-i g1 C))
        phase1_conj = np.exp(-1j * g1 * cut_vals)  # same as phase1
        v_g0 = v_g0 * phase1_conj
        # Step 4: apply M(b1)
        v_g0 = _apply_mixer(v_g0, b1)
        # dE/dg0 = 2*Re(<v_g0 | C psi3>) = 2*Re(sum conj(v_g0) * C_psi3)
        grad_g0 = 2.0 * np.real(np.vdot(v_g0, C_psi3))
        
        # dpsi3/db0 = M(b1) * diag(exp(-i g1 C)) * (-i (sum X) psi1)
        # -i (sum X) psi1: apply X to each qubit of psi1, then multiply by -i
        # X flips each bit. In the state vector, (X_i psi)[k] = psi[k XOR (1<<i)]
        # So sum_i X_i psi1 = sum_i psi1[k XOR (1<<i)] for each k
        # Let's compute this efficiently.
        X_psi1 = np.zeros(dim, dtype=np.complex128)
        for i in range(n):
            bit = 1 << i
            # For each k, X_i psi1[k] = psi1[k XOR bit]
            # We can do this with indexing
            idx = np.arange(dim)
            X_psi1 += psi1[idx ^ bit]
        v_b0 = -1j * X_psi1
        # Apply M(b1)
        v_b0 = _apply_mixer(v_b0, b1)
        # Apply diag(exp(-i g1 C))
        v_b0 = v_b0 * phase1_conj
        # Wait, the order is: psi3 = M(b1) * diag(exp(-i g1 C)) * psi1
        # dpsi3/db0 = M(b1) * diag(exp(-i g1 C)) * dpsi1/db0
        # dpsi1/db0 = -i (sum X) psi1
        # So: dpsi3/db0 = M(b1) * diag(exp(-i g1 C)) * (-i (sum X) psi1)
        # I had the order wrong. Let me redo:
        v_b0 = -1j * X_psi1
        # Apply diag(exp(-i g1 C)): multiply by exp(-i g1 * cut_vals)
        v_b0 = v_b0 * np.exp(-1j * g1 * cut_vals)
        # Apply M(b1)
        v_b0 = _apply_mixer(v_b0, b1)
        grad_b0 = 2.0 * np.real(np.vdot(v_b0, C_psi3))
        
        # dpsi3/dg1 = M(b1) * (-i C psi2)
        v_g1 = -1j * cut_vals * psi2
        v_g1 = _apply_mixer(v_g1, b1)
        grad_g1 = 2.0 * np.real(np.vdot(v_g1, C_psi3))
        
        # dpsi3/db1 = -i (sum X) psi3
        X_psi3 = np.zeros(dim, dtype=np.complex128)
        for i in range(n):
            bit = 1 << i
            idx = np.arange(dim)
            X_psi3 += psi3[idx ^ bit]
        v_b1 = -1j * X_psi3
        grad_b1 = 2.0 * np.real(np.vdot(v_b1, C_psi3))
        
        grad = np.array([grad_g0, grad_b0, grad_g1, grad_b1])
        return value, grad
    
    def eval_fn(params):
        """Evaluate expected cut for given parameters."""
        val, _ = _compute_value_and_grad(params)
        return val
    
    def analytic_gradient(params):
        """Compute analytic gradient (negated for minimization)."""
        _, grad = _compute_value_and_grad(params)
        return -grad
    
    def objective(params):
        return -eval_fn(params)
    
    # Try multiple initial points
    inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.4, 0.4, 0.4],
        [0.6, 0.2, 0.3, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.8, 0.6, 0.5, 0.4],
        [0.3, 0.6, 0.3, 0.6],
        [0.6, 0.4, 0.6, 0.4],
        [0.55, 0.35, 0.35, 0.25],
        [0.45, 0.45, 0.45, 0.35],
        [0.4, 0.35, 0.4, 0.35],
        [0.6, 0.6, 0.3, 0.3],
        [0.3, 0.5, 0.5, 0.3],
    ]
    
    best_score = -np.inf
    best_params = np.array([0.4, 0.35, 0.4, 0.35])
    
    np.random.seed(123)
    
    # Add random inits
    for _ in range(25):
        inits.append(np.random.uniform([0.05, 0.05, 0.05, 0.05], 
                                       [1.5, 1.5, 1.5, 1.5]).tolist())
    
    bounds = [(0.0, np.pi)] * 4
    
    # Phase 1: Broad search with L-BFGS-B
    for init in inits:
        if time.time() - start_time > _DEADLINE - 12:
            break
        
        try:
            res = scipy.optimize.minimize(
                objective, np.array(init, dtype=np.float64),
                jac=analytic_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish pass with tighter tolerances
    if time.time() - start_time < _DEADLINE - 6:
        try:
            res = scipy.optimize.minimize(
                objective, best_params.copy(),
                jac=analytic_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < _DEADLINE - 3:
        try:
            res = scipy.optimize.minimize(
                objective, best_params.copy(),
                jac=analytic_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])
