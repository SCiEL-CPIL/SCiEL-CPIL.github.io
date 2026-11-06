"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product


def _exhaustive_maxcut(n, edges):
    """Compute the exact MaxCut value via exhaustive search for small n."""
    if n <= 1:
        return 0
    best_cut = 0
    # For n up to 10, 2^10 = 1024 assignments is trivial
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            # Check if u and v are in different partitions
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return best_cut


def _expected_cut_value(n, edges, gammas, betas):
    """
    Compute the expected cut size for a given QAOA circuit.
    The circuit is: |+>^n -> exp(-i g1 C) -> exp(-i b1 sum X) -> exp(-i g2 C) -> exp(-i b2 sum X)
    Wait, the standard QAOA is:
    U = exp(-i beta_p H_B) ... exp(-i beta_1 H_B) exp(-i gamma_p H_C) ... exp(-i gamma_1 H_C)
    But the task says: "applies exp(-i gamma C), then exp(-i beta sum X), for each row"
    This implies the sequence is:
    Row 0: exp(-i gamma_0 C) then exp(-i beta_0 X)
    Row 1: exp(-i gamma_1 C) then exp(-i beta_1 X)
    So the total unitary is:
    U = exp(-i beta_1 X) exp(-i gamma_1 C) exp(-i beta_0 X) exp(-i gamma_0 C)
    
    We need the expected value of the MaxCut objective function C in the final state.
    C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2 = |E|/2 - 1/2 sum_{(u,v) in E} Z_u Z_v
    
    Since we start in |+>^n, and the Hamiltonians are diagonal or X-only,
    this is a small quantum circuit. We can simulate it exactly for n=6,8,10.
    The state vector has size 2^n. For n=10, that's 1024 complex numbers.
    Applying exponentials of sums of commuting operators (like sum X or C) is easy.
    
    exp(-i beta sum X) = tensor product of exp(-i beta X) for each qubit.
    exp(-i gamma C) = exp(-i gamma * sum_{(u,v)} (1 - Z_u Z_v)/2)
    Since C is a sum of commuting terms (Z_u Z_v commute), we can apply it by
    multiplying each basis state |b> by exp(-i gamma * c(b)), where c(b) is the cut size
    of assignment b.
    """
    dim = 1 << n
    
    # Initialize state to |+>^n
    # |+> = (1/sqrt(2)) * (|0> + |1>)
    # So the state vector is 1/sqrt(2^n) * [1, 1, ..., 1]
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cut sizes for all basis states?
    # Or just compute on the fly?
    # For n=10, 1024 states. Precomputing is fine.
    cut_sizes = np.zeros(dim, dtype=np.int32)
    for mask in range(dim):
        cut = 0
        for u, v in edges:
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut += 1
        cut_sizes[mask] = cut
    
    # Apply layers
    # The order in the task description:
    # "applies exp(-i gamma C), then exp(-i beta sum X), for each row"
    # This usually means:
    # Start with |+>
    # For row i in 0..p-1:
    #   Apply exp(-i gamma_i C)
    #   Apply exp(-i beta_i sum X)
    
    # Let's verify the standard QAOA convention.
    # Usually: U = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1)
    # If we start with |+> and apply U_C(gamma_1) first, then U_B(beta_1), etc.
    # The task says "for each row", implying a sequence.
    # Row 0: gamma_0, beta_0.
    # So: State -> exp(-i gamma_0 C) -> exp(-i beta_0 X) -> exp(-i gamma_1 C) -> exp(-i beta_1 X) ...
    
    for i in range(p):
        g = gammas[i]
        b = betas[i]
        
        # 1. Apply exp(-i g C)
        # C |b> = cut_size(b) |b>
        # So state[b] *= exp(-i g * cut_sizes[b])
        phases = np.exp(-1j * g * cut_sizes)
        state *= phases
        
        # 2. Apply exp(-i b sum X)
        # This is a tensor product of single-qubit rotations.
        # exp(-i b X) = [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
        # We can apply this by reshaping the state into a tensor of shape (2,)*n
        # and applying the rotation to each qubit axis.
        
        # Reshape state to (2, 2, ..., 2)
        tensor = state.reshape([2] * n)
        
        # Apply rotation to each qubit
        c = np.cos(b)
        s = np.sin(b)
        # The matrix is [[c, -is], [-is, c]]
        # For a qubit state vector [a, b], the new state is [c*a - is*b, -is*a + c*b]
        
        # We need to do this for each axis.
        # It's easier to use np.einsum or reshape operations.
        # Let's iterate over qubits.
        for q in range(n):
            # Move axis q to the front
            tensor = np.moveaxis(tensor, q, 0)
            # tensor shape: (2, 2, ..., 2)
            # Apply matrix M = [[c, -is], [-is, c]]
            # new_tensor[0, ...] = c * tensor[0, ...] - 1j*s * tensor[1, ...]
            # new_tensor[1, ...] = -1j*s * tensor[0, ...] + c * tensor[1, ...]
            t0 = tensor[0].copy()
            t1 = tensor[1].copy()
            tensor[0] = c * t0 - 1j * s * t1
            tensor[1] = -1j * s * t0 + c * t1
            # Move axis back
            tensor = np.moveaxis(tensor, 0, q)
        
        state = tensor.reshape(dim)
        
        # Normalize to prevent numerical drift
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
    
    # Calculate expected cut
    # <C> = sum_b |state[b]|^2 * cut_sizes[b]
    probs = np.abs(state) ** 2
    expected_cut = np.sum(probs * cut_sizes)
    
    return expected_cut


def angles(n, edges, p):
    # Compute exact optimum
    opt_cut = _exhaustive_maxcut(n, edges)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Search space for angles
    # We want to maximize expected_cut / opt_cut
    # Let's do a coarse grid search first, then a local refinement.
    
    # Typical QAOA angles are in [0, pi/2] or [0, pi]
    # Let's try a grid.
    # For p=2, we have 4 parameters.
    
    # Grid resolution
    # Let's try 10 steps in [0, pi/2] for each parameter.
    # 10^4 = 10000 evaluations.
    # Each evaluation is O(2^n * n * p). For n=10, p=2, this is very fast.
    
    # However, the function is called for 6 graphs.
    # We can cache the results for specific (n, edges) tuples?
    # The edges are a list of tuples. We can hash them.
    
    # Let's do a simple grid search.
    # To be safe with time, let's use a smaller grid initially or optimize the search.
    
    # Actually, since n is small (<=10), we can afford a slightly finer grid.
    # Let's try 15 steps. 15^4 = 50625 evaluations.
    # 50k * (1024 * 10 * 2) ops ~ 10^9 ops. Might be too slow in Python if not vectorized well.
    # The state vector simulation is vectorized in NumPy, so it's fast.
    # Let's try 10 steps first.
    
    steps = 10
    vals = np.linspace(0, np.pi/2, steps)
    
    best_val = -1
    best_angles = [0.35, 0.25, 0.35, 0.25] # Fallback
    
    # Grid search
    for g1 in vals:
        for b1 in vals:
            for g2 in vals:
                for b2 in vals:
                    gammas = np.array([g1, g2])
                    betas = np.array([b1, b2])
                    val = _expected_cut_value(n, edges, gammas, betas)
                    if val > best_val:
                        best_val = val
                        best_angles = [g1, b1, g2, b2]
    
    # Local refinement using gradient-free optimization (Nelder-Mead or similar)
    # We can use scipy.optimize.minimize if available, or implement a simple local search.
    # Let's try a simple local search around the best point.
    
    # Define objective to minimize (negative of ratio)
    def objective(params):
        g1, b1, g2, b2 = params
        gammas = np.array([g1, g2])
        betas = np.array([b1, b2])
        val = _expected_cut_value(n, edges, gammas, betas)
        return -val
    
    # Try a few local optimizations with different starting points near the best
    # We'll use a simple coordinate descent or just trust the grid if it's fine.
    # Given the small dimension, let's try a few random restarts with local refinement.
    
    # Actually, let's just use the grid result for now. It's robust.
    # If we have time, we can add scipy.optimize.
    
    # Let's try to use scipy if available to do a quick polish.
    try:
        from scipy.optimize import minimize
        # Start from the best grid point
        x0 = np.array(best_angles)
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.success and -res.fun > best_val:
            best_angles = res.x.tolist()
            best_val = -res.fun
    except Exception:
        pass
    
    # Construct the output array (p, 2)
    # Rows are (gamma, beta)
    # best_angles is [g1, b1, g2, b2]
    result = np.array([
        [best_angles[0], best_angles[1]],
        [best_angles[2], best_angles[3]]
    ])
    
    return result
# EVOLVE-BLOCK-END
