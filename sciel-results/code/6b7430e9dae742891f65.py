"""QAOA parameter design for fixed-depth (p=2) MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 135.0  # seconds, leave 15s safety margin for 150s limit

def _remaining_time():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force for small n."""
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _expected_cut_fast(n, edges, gammas, betas):
    """
    Fast expected cut computation for p=2 QAOA using combinatorial aggregation.
    
    For p=2 QAOA with MaxCut cost Hamiltonian C = sum_k (1 - Z_k) where Z_k = Z_u Z_v for edge k:
    
    The state after the circuit can be analyzed by noting that:
    - Start: |+>^n = (1/sqrt(2^n)) sum_z |z>
    - Cost unitary: exp(-i*gamma*C) applies phase exp(-i*gamma*(m - sum_k Z_k(z))) to each |z>
    - Mixer: exp(-i*beta*X_sum) where X_sum flips all bits
    
    After first cost unitary: |psi_1> = (1/sqrt(2^n)) sum_z exp(-i*gamma_1*(m - sum_k Z_k(z))) |z>
    After first mixer: |psi_1'> = (1/sqrt(2^n)) sum_z exp(-i*gamma_1*(m - sum_k Z_k(z))) |z XOR all_ones>
    Let w = z XOR all_ones, so z = w XOR all_ones:
    |psi_1'> = (1/sqrt(2^n)) sum_w exp(-i*gamma_1*(m - sum_k Z_k(w XOR all_ones))) |w>
    
    Note Z_k(w XOR all_ones) = Z_u(w XOR all_ones) * Z_v(w XOR all_ones)
    Z_i(x) = (-1)^{x_i}, so Z_i(w XOR all_ones) = (-1)^{w_i XOR 1} = -(-1)^{w_i} = -Z_i(w)
    So Z_k(w XOR all_ones) = (-Z_u(w)) * (-Z_v(w)) = Z_u(w) * Z_v(w) = Z_k(w)
    
    Therefore: |psi_1'> = (1/sqrt(2^n)) sum_w exp(-i*gamma_1*(m - sum_k Z_k(w))) |w>
    
    Interesting! The mixer with all-to-all X doesn't change the phase structure because 
    Z_k is invariant under global bit flip.
    
    After second cost unitary: |psi_2> = (1/sqrt(2^n)) sum_w exp(-i*gamma_2*(m - sum_k Z_k(w))) * exp(-i*gamma_1*(m - sum_k Z_k(w))) |w>
    = (1/sqrt(2^n)) sum_w exp(-i*(gamma_1+gamma_2)*(m - sum_k Z_k(w))) |w>
    
    After second mixer: |psi_2'> = (1/sqrt(2^n)) sum_v exp(-i*(gamma_1+gamma_2)*(m - sum_k Z_k(v XOR all_ones))) |v>
    = (1/sqrt(2^n)) sum_v exp(-i*(gamma_1+gamma_2)*(m - sum_k Z_k(v))) |v>
    
    So the final state is: |psi_final> = (1/sqrt(2^n)) sum_v exp(-i*Gamma*(m - sum_k Z_k(v))) |v>
    where Gamma = gamma_1 + gamma_2
    
    Wait, this means the betas don't matter at all for p=2 with all-to-all mixer and MaxCut!
    Let me verify: the expected cut depends only on Gamma = gamma_1 + gamma_2.
    
    Expected cut = sum_{(u,v) in edges} P(cut edge (u,v))
    P(cut) = 0.5 - 0.5 * <Z_u Z_v>
    
    <Z_u Z_v> = sum_v |psi_v|^2 * Z_u(v) * Z_v(v)
    |psi_v|^2 = 1/2^n (uniform!)
    
    So <Z_u Z_v> = (1/2^n) sum_v Z_u(v) * Z_v(v)
    
    But this is independent of Gamma! That can't be right...
    
    Let me reconsider. The issue is that |psi_v|^2 = 1/2^n always, so the expectation
    values are just the average over all configurations, which for any pair is 0 (since
    half the configs have Z_u=Z_v and half don't).
    
    Hmm, that would give expected cut = m * 0.5, which is wrong.
    
    I think I made an error. Let me redo this more carefully.
    
    Actually wait - the mixer is exp(-i*beta*sum_i X_i), not a simple bit flip. 
    exp(-i*beta*X_sum) = cos(beta)*I - i*sin(beta)*X_sum
    
    So it's a superposition, not a permutation. Let me redo.
    
    After first cost: |psi_1> = (1/sqrt(2^n)) sum_z phi_1(z) |z> where phi_1(z) = exp(-i*gamma_1*(m - sum_k Z_k(z)))
    
    After first mixer: |psi_1'> = cos(beta_1)*|psi_1> - i*sin(beta_1)*X_sum|psi_1>
    X_sum|z> = |z XOR all_ones>
    So |psi_1'> = (1/sqrt(2^n)) sum_w [cos(beta_1)*phi_1(w) - i*sin(beta_1)*phi_1(w XOR all_ones)] |w>
    
    Let phi_1'(w) = cos(beta_1)*phi_1(w) - i*sin(beta_1)*phi_1(w XOR all_ones)
    
    As shown, Z_k(w XOR all_ones) = Z_k(w), so phi_1(w XOR all_ones) = phi_1(w)
    Thus phi_1'(w) = phi_1(w) * (cos(beta_1) - i*sin(beta_1)) = phi_1(w) * exp(-i*beta_1)
    
    So |psi_1'> = exp(-i*beta_1) * |psi_1>
    
    The mixer just adds a global phase! This means for p=2 with all-to-all X mixer and MaxCut,
    the betas are irrelevant (they only contribute global phases that cancel in expectations).
    
    After second cost: |psi_2> = (1/sqrt(2^n)) sum_w phi_2(w) |w> where phi_2(w) = phi_1(w) * exp(-i*gamma_2*(m - sum_k Z_k(w)))
    = (1/sqrt(2^n)) sum_w exp(-i*(gamma_1+gamma_2)*(m - sum_k Z_k(w))) |w>
    
    After second mixer: |psi_2'> = exp(-i*beta_2) * |psi_2> (same global phase argument)
    
    Final state: |psi_final> = exp(-i*(beta_1+beta_2)) * (1/sqrt(2^n)) sum_w exp(-i*Gamma*(m - sum_k Z_k(w))) |w>
    where Gamma = gamma_1 + gamma_2
    
    The global phase exp(-i*(beta_1+beta_2)) doesn't affect expectations.
    
    Now compute <Z_u Z_v> for edge (u,v):
    <Z_u Z_v> = sum_w |psi_final_w|^2 * Z_u(w) * Z_v(w)
    |psi_final_w|^2 = (1/2^n) * |exp(-i*Gamma*(m - sum_k Z_k(w)))|^2 = 1/2^n
    
    So <Z_u Z_v> = (1/2^n) sum_w Z_u(w) Z_v(w)
    
    For a fixed pair (u,v), as w ranges over all 2^n configs, Z_u(w) Z_v(w) = +1 when w_u = w_v and -1 when w_u != w_v.
    Exactly half the configs have w_u = w_v and half have w_u != w_v.
    So (1/2^n) sum_w Z_u(w) Z_v(w) = 0.
    
    This gives expected cut = m * (0.5 - 0.5*0) = m/2.
    
    But that's clearly wrong - QAOA can do better than random. I must be making an error.
    
    OH WAIT. I think the issue is that the cost Hamiltonian for MaxCut is typically 
    C = sum_k (1 - Z_u Z_v)/2 or C = sum_k (1 - Z_u Z_v). Let me check the convention.
    
    The task says "C is unweighted MaxCut". The standard QAOA MaxCut cost is:
    C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    
    But even with this, the analysis is the same - the phases are still diagonal in the computational basis,
    and the mixer still just adds a global phase.
    
    Let me reconsider the mixer. exp(-i*beta*sum_i X_i). Is sum_i X_i really the all-to-all bit flip?
    
    X_i acts on qubit i: X_i|0> = |1>, X_i|1> = |0>.
    sum_i X_i is NOT the same as the tensor product X_1 X_2 ... X_n.
    
    sum_i X_i = X_1 + X_2 + ... + X_n (as an operator sum)
    
    This is NOT a bit-flip operator! exp(-i*beta*sum_i X_i) is NOT a permutation.
    
    I made a critical error. sum_i X_i is the sum of single-qubit X operators, not their product.
    
    exp(-i*beta*sum_i X_i) = exp(-i*beta*X_1) * exp(-i*beta*X_2) * ... * exp(-i*beta*X_n)
    (since all X_i commute)
    = prod_i (cos(beta)*I - i*sin(beta)*X_i)
    
    This is a product of single-qubit rotations, applied to each qubit independently.
    On a basis state |z>, this gives:
    prod_i [cos(beta)*|z_i> - i*sin(beta)*|1-z_i>]
    = (1/sqrt(2^n)) sum_{w} prod_i [cos(beta) * delta_{w_i, z_i} - i*sin(beta) * delta_{w_i, 1-z_i}] |w>
    
    Actually, let me think of it differently. For each qubit i, the transformation is:
    |0> -> cos(beta)|0> - i*sin(beta)|1>
    |1> -> -i*sin(beta)|0> + cos(beta)|1>
    
    So for a product state |z_1 z_2 ... z_n>, the result is:
    prod_i [cos(beta)|z_i> - i*sin(beta)|1-z_i>]
    
    Expanding this, we get a superposition over all bitstrings w, where the amplitude for |w> is:
    prod_i [cos(beta) if w_i = z_i, else -i*sin(beta)]
    
    So the amplitude for |w> given input |z> is:
    prod_i [cos(beta)^{[w_i=z_i]} * (-i*sin(beta))^{[w_i!=z_i]}]
    
    Let d(z,w) = Hamming distance between z and w.
    Amplitude = cos(beta)^{n-d} * (-i*sin(beta))^d
    
    Now let's redo the full circuit:
    
    Start: |psi_0> = (1/sqrt(2^n)) sum_z |z>
    
    After first cost: |psi_1> = (1/sqrt(2^n)) sum_z phi_1(z) |z>
    where phi_1(z) = exp(-i*gamma_1 * C(z)) and C(z) = sum_k (1 - Z_k(z)) [or with /2 factor]
    
    After first mixer: |psi_1'> = (1/sqrt(2^n)) sum_z phi_1(z) * prod_i [cos(beta_1)|z_i> - i*sin(beta_1)|1-z_i>]
    = (1/sqrt(2^n)) sum_w [sum_z phi_1(z) * prod_i [cos(beta_1)^{[w_i=z_i]} * (-i*sin(beta_1))^{[w_i!=z_i]}]] |w>
    
    Let A_1(w) = sum_z phi_1(z) * prod_i [cos(beta_1)^{[w_i=z_i]} * (-i*sin(beta_1))^{[w_i!=z_i]}]
    
    This is a convolution-like operation. For n=6, 8, 10, we can compute this directly.
    
    After second cost: |psi_2> = (1/sqrt(2^n)) sum_w phi_2(w) * A_1(w) |w>
    where phi_2(w) = exp(-i*gamma_2 * C(w))
    
    After second mixer: |psi_final> = (1/sqrt(2^n)) sum_v A_2(v) |v>
    where A_2(v) = sum_w phi_2(w) * A_1(w) * prod_i [cos(beta_2)^{[v_i=w_i]} * (-i*sin(beta_2))^{[v_i!=w_i]}]
    
    The expected cut is then:
    E[cut] = sum_{(u,v) in E} (0.5 - 0.5 * <Z_u Z_v>)
    <Z_u Z_v> = sum_v |A_2(v)/sqrt(2^n)|^2 * Z_u(v) * Z_v(v)
    = (1/2^n) sum_v |A_2(v)|^2 * Z_u(v) * Z_v(v)
    
    This is computable in O(n * 2^n) time for each parameter evaluation, which is fast for n <= 10.
    
    Let me implement this properly.
    """
    dim = 1 << n
    all_ones = dim - 1
    
    # Precompute Z values for each qubit
    Z = np.zeros((n, dim), dtype=float)
    for i in range(n):
        for j in range(dim):
            Z[i, j] = 1.0 if (j >> i) & 1 == 0 else -1.0
    
    # Precompute cost C(z) for each basis state
    m = len(edges)
    C_vals = np.zeros(dim, dtype=float)
    for k, (u, v) in enumerate(edges):
        C_vals += (1.0 - Z[u] * Z[v])
    # C(z) = sum_k (1 - Z_u Z_v) for edge k
    
    # Compute phi_1(z) = exp(-i * gamma_1 * C(z))
    phi_1 = np.exp(-1j * gammas[0] * C_vals)
    
    # Apply first mixer: A_1(w) = sum_z phi_1(z) * prod_i [cos(b1)^{[w_i=z_i]} * (-i*sin(b1))^{[w_i!=z_i]}]
    # This is equivalent to: for each z, add phi_1(z) * amp(z->w) to A_1(w)
    # where amp(z->w) = prod_i [cos(b1) if w_i=z_i else -i*sin(b1)]
    
    # Efficient computation: the mixer is a tensor product of 2x2 matrices
    # M = [[cos(b), -i*sin(b)], [-i*sin(b), cos(b)]]
    # A_1 = M^{⊗n} @ phi_1 (as a vector)
    
    # For n up to 10, dim up to 1024, we can do this with iterative tensor contractions
    # or just compute it directly using the structure.
    
    # Actually, for n <= 10, dim <= 1024, we can just do the full matrix-vector product
    # by constructing the mixer matrix, but that's 1024x1024 which is fine.
    
    # Better: use the fact that M^{⊗n} can be applied iteratively.
    # Start with vector v = phi_1 (shape: 2^n)
    # For each qubit i, apply M to the i-th index.
    
    def _apply_mixer(state, beta, n, dim):
        """Apply exp(-i*beta*sum X_i) to state vector."""
        c = np.cos(beta)
        s = np.sin(beta)
        # M = [[c, -i*s], [-i*s, c]]
        # For each qubit i, new_state[..., w_i, ...] = c*state[..., w_i, ...] - i*s*state[..., 1-w_i, ...]
        
        state = state.copy()
        for i in range(n):
            stride = 1 << i
            period = stride * 2
            new_state = np.zeros_like(state)
            for base in range(0, dim, period):
                for offset in range(stride):
                    idx0 = base + offset
                    idx1 = base + stride + offset
                    new_state[idx0] = c * state[idx0] - 1j * s * state[idx1]
                    new_state[idx1] = -1j * s * state[idx0] + c * state[idx1]
            state = new_state
        return state
    
    # Step 1: phi_1
    # Step 2: apply mixer with beta_1
    A_1 = _apply_mixer(phi_1, betas[0], n, dim)
    
    # Step 3: apply second cost: multiply by phi_2(w) = exp(-i * gamma_2 * C(w))
    phi_2 = np.exp(-1j * gammas[1] * C_vals)
    A_2_pre = A_1 * phi_2
    
    # Step 4: apply mixer with beta_2
    A_2 = _apply_mixer(A_2_pre, betas[1], n, dim)
    
    # Compute expected cut
    # |A_2(v)|^2 / 2^n is the probability of state v
    probs = np.abs(A_2) ** 2 / dim
    
    cut_val = 0.0
    for (u, v) in edges:
        corr = np.sum(probs * Z[u] * Z[v])
        cut_val += 0.5 - 0.5 * corr
    
    return cut_val

def _objective(params, n, edges, optimum):
    """Negative ratio: minimize -expected_cut/optimum."""
    if _remaining_time() < 1.0:
        return 1e10
    
    gammas = params[0::2]
    betas = params[1::2]
    
    try:
        ec = _expected_cut_fast(n, edges, gammas, betas)
        if optimum <= 0:
            return 0.0
        return -ec / optimum
    except:
        return 1.0

def _optimize_angles(n, edges, p=2):
    """Find optimal QAOA angles for a given graph using multi-start Nelder-Mead."""
    if _remaining_time() < 5.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    optimum = _exact_maxcut(n, edges)
    if optimum <= 0:
        return np.zeros((p, 2))
    
    n_params = 2 * p  # 4 parameters for p=2
    best_params = None
    best_val = 1e10
    
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # Generate diverse initializations
    inits = []
    
    # Known good starting points
    base_inits = [
        [0.5, 0.3, 0.3, 0.2],
        [0.3, 0.5, 0.5, 0.3],
        [np.pi/4, np.pi/6, np.pi/6, np.pi/8],
        [0.2, 0.4, 0.4, 0.2],
        [0.4, 0.2, 0.2, 0.4],
        [0.1, 0.6, 0.6, 0.1],
        [0.6, 0.1, 0.1, 0.6],
        [0.25, 0.35, 0.35, 0.25],
        [0.4, 0.4, 0.4, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.7, 0.1, 0.1, 0.7],
        [0.1, 0.7, 0.7, 0.1],
        [0.35, 0.25, 0.25, 0.35],
        [0.45, 0.35, 0.35, 0.45],
    ]
    for b in base_inits:
        inits.append(np.array(b, dtype=float))
    
    # Density-dependent initializations
    if density < 0.4:
        inits.append(np.array([0.3, 0.4, 0.4, 0.3]))
        inits.append(np.array([0.2, 0.5, 0.5, 0.2]))
        inits.append(np.array([0.15, 0.45, 0.45, 0.15]))
    else:
        inits.append(np.array([0.4, 0.3, 0.3, 0.4]))
        inits.append(np.array([0.5, 0.2, 0.2, 0.5]))
        inits.append(np.array([0.45, 0.25, 0.25, 0.45]))
    
    # Random initializations with fixed seed for reproducibility
    rng = np.random.RandomState(12345)
    for _ in range(10):
        inits.append(rng.uniform(0.05, 0.8, size=n_params))
    
    # Evaluate all initializations first to find the best starting point
    init_vals = []
    for init in inits:
        if _remaining_time() < 5.0:
            break
        try:
            val = _objective(init, n, edges, optimum)
            init_vals.append((val, init.copy()))
        except:
            continue
    
    if not init_vals:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    # Sort by objective value (ascending, since we're minimizing)
    init_vals.sort(key=lambda x: x[0])
    
    # Use top initializations for optimization
    top_inits = [iv[1] for iv in init_vals[:min(8, len(init_vals))]]
    
    # Also include the absolute best initialization
    if init_vals[0][1] not in top_inits:
        top_inits.append(init_vals[0][1])
    
    for init in top_inits:
        if _remaining_time() < 5.0:
            break
        
        try:
            result = minimize(_objective, init, args=(n, edges, optimum),
                             method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-8,
                                      'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            continue
    
    if best_params is None:
        # Use the best initialization as fallback
        best_params = init_vals[0][1]
    
    # Post-process: refine from the best point
    if _remaining_time() > 10.0:
        for _ in range(3):
            if _remaining_time() < 5.0:
                break
            try:
                result = minimize(_objective, best_params, args=(n, edges, optimum),
                                 method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-9,
                                          'adaptive': True})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                break
    
    gammas = best_params[0::2]
    betas = best_params[1::2]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """Main entry point: return QAOA angles for the given graph."""
    if _remaining_time() < 2.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    try:
        result = _optimize_angles(n, edges, p)
        # Ensure output shape is (p, 2)
        if result.shape != (p, 2):
            result = np.tile([0.35, 0.25], (p, 1))
        return result
    except:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
# EVOLVE-BLOCK-END
