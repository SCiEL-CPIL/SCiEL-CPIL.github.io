"""Rewritten program for QAOA parameter design task."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Cache for graph properties and optimal values
_graph_cache = {}
_angle_cache = {}

def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exact_maxcut(n, adj):
    """Compute exact MaxCut value via brute force for small n."""
    best = 0.0
    for mask in range(1 << n):
        cut = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    if ((mask >> i) & 1) != ((mask >> j) & 1):
                        cut += 1.0
        if cut > best:
            best = cut
    return best

def _expected_cut_value(n, adj, gammas, betas):
    """
    Compute expected cut value for QAOA with given angles.
    Uses the analytical formula for p=2 QAOA on unweighted graphs.
    
    The expected cut is: sum_{(i,j) in E} (1 - cos(gamma) * (cos(beta)^2 * (A_ij - 2*deg_i*deg_j/n^2 + ...) ))
    Actually, let's use the standard formula:
    For p=2, the expected cut can be computed via:
    E[cut] = sum_{(i,j) in E} [1/2 - 1/2 * <sigma_i^z sigma_j^z>]
    
    For unweighted QAOA with p=2, the correlation <sigma_i^z sigma_j^z> can be computed
    using the state after the ansatz.
    
    A more efficient approach: use the fact that for p=2, we can compute the expectation
    by tracking the state or using known analytical results.
    
    Actually, the simplest correct approach for small n is to use the full state vector
    or to use the formula based on the two-layer ansatz.
    
    For p=2 QAOA on a graph, the expected cut value is:
    E[cut] = sum_{(i,j) in E} (1/2) * (1 - cos(2*beta) * cos(gamma) + ... )
    
    Let me use a cleaner formulation. The standard result for p=2 QAOA:
    
    After the first layer (gamma1, beta1) and second layer (gamma2, beta2),
    the probability of a cut edge being cut can be computed.
    
    For a single edge (i,j), the probability that sigma_i^z and sigma_j^z have opposite signs
    is what we need.
    
    For the full p=2 case, this is complex. Let me use a different approach:
    simulate the quantum state for small n (n <= 12 is feasible with 2^n states).
    """
    # For n up to 12, we can use the full state vector approach
    # But for speed, let's use the analytical formula for p=2
    
    # Actually, for n=6,8,10, the full state has 2^6=64, 2^8=256, 2^10=1024 states
    # This is very feasible. Let's use the state vector approach.
    
    dim = 1 << n
    # Start in |+>^n state: uniform superposition
    # In computational basis, this is 1/sqrt(2^n) for all basis states
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply the QAOA layers
    # Each layer: exp(-i*gamma*C) then exp(-i*beta*sum X)
    # C = sum_{(i,j) in E} (I - sigma_i^z sigma_j^z)/2 for MaxCut
    # Actually, for MaxCut, the cost Hamiltonian is typically:
    # C = sum_{(i,j) in E} (1 - sigma_i^z sigma_j^z)/2
    # So exp(-i*gamma*C) = exp(-i*gamma/2 * sum (1 - sigma_i^z sigma_j^z))
    # = exp(-i*gamma/2 * |E|) * exp(i*gamma/2 * sum sigma_i^z sigma_j^z)
    # The global phase doesn't matter, so we can use exp(i*gamma/2 * sum sigma_i^z sigma_j^z)
    
    # For efficiency, let's build the operators
    # Actually, for small n, let's just apply the gates directly
    
    # Build the cost operator as a matrix
    # C_matrix = sum_{(i,j) in E} (I - Z_i Z_j)/2
    # We need to apply exp(-i*gamma*C)
    
    # For small n, we can construct the full matrix
    # Z_i Z_j is diagonal in the computational basis
    # So C is also diagonal (since it's a sum of Z_i Z_j terms)
    
    # Z_i Z_j on basis state |x> gives (-1)^(x_i + x_j)
    # So C |x> = sum_{(i,j) in E} (1 - (-1)^(x_i + x_j))/2 |x>
    # = (number of cut edges in x) |x>
    
    # So C is diagonal with entries = cut_value(x) for each basis state x
    
    # Therefore exp(-i*gamma*C) is also diagonal
    # exp(-i*gamma*cut_value(x))
    
    # For the mixer: exp(-i*beta*sum X_i)
    # sum X_i = X_1 + X_2 + ... + X_n
    # This is not diagonal, so we need to apply it as a matrix or use a trick
    
    # For n <= 12, let's build the full mixer matrix
    # X_i flips bit i
    # sum X_i applied to |x> gives sum_i |x with bit i flipped>
    
    # Let's build the mixer matrix M where M[x, y] = <x| exp(-i*beta*sum X) |y>
    # This is expensive to build for each beta, but for n <= 10 it's manageable
    
    # Actually, let's use a smarter approach. For p=2, we can use the fact that
    # the mixer exp(-i*beta*sum X) can be applied efficiently.
    
    # For now, let's use the full matrix approach for correctness, and optimize later
    
    # Precompute cut values for all basis states
    cut_vals = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        cut = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    if ((x >> i) & 1) != ((x >> j) & 1):
                        cut += 1.0
        cut_vals[x] = cut
    
    # Build the mixer matrix for given beta
    # M = exp(-i*beta * sum X_i)
    # sum X_i is a sum of n Pauli-X operators
    # For small n, we can build it as a dense matrix
    
    # Actually, let's use a different approach: use the fact that for p=2,
    # we can compute the expectation value without full state evolution
    
    # Let me use the standard QAOA expectation formula for p=2
    
    # For p=2 QAOA, the expected cut is:
    # E[cut] = sum_{(i,j) in E} P(i,j cut)
    # where P(i,j cut) = (1 - <sigma_i^z sigma_j^z>)/2
    
    # For the two-layer ansatz, <sigma_i^z sigma_j^z> can be computed
    # using the known results from QAOA literature.
    
    # Actually, let me just use the full state vector approach since n is small.
    
    # Build the full mixer matrix
    # sum X_i: for each basis state x, (sum X_i)|x> = sum_i |x^e_i>
    # where x^e_i is x with bit i flipped
    
    # The matrix for sum X_i in computational basis:
    # A[x, y] = 1 if y = x^e_i for some i, else 0
    # Actually, A[x, y] = number of i such that y = x with bit i flipped
    # But X_i is a unitary, so sum X_i is not unitary in general
    
    # Let me just build the matrix numerically
    A = np.zeros((dim, dim), dtype=np.float64)
    for x in range(dim):
        for i in range(n):
            y = x ^ (1 << i)
            A[x, y] += 1.0
    
    # Now we need exp(-i*beta*A)
    # Use eigendecomposition or matrix exponential
    from scipy.linalg import expm
    M = expm(-1j * betas[0] * A)  # Wait, this is for p=1
    
    # For p=2, we have two layers
    # Layer 1: exp(-i*gamma1*C) then exp(-i*beta1*sum X)
    # Layer 2: exp(-i*gamma2*C) then exp(-i*beta2*sum X)
    
    # C is diagonal, so exp(-i*gamma*C) is diagonal
    # Let's compute the full evolution
    
    # Start state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Layer 1: apply exp(-i*gamma1*C)
    psi = psi * np.exp(-1j * gammas[0] * cut_vals)
    
    # Apply exp(-i*beta1*sum X)
    M1 = expm(-1j * betas[0] * A)
    psi = M1 @ psi
    
    # Layer 2: apply exp(-i*gamma2*C)
    psi = psi * np.exp(-1j * gammas[1] * cut_vals)
    
    # Apply exp(-i*beta2*sum X)
    M2 = expm(-1j * betas[1] * A)
    psi = M2 @ psi
    
    # Compute expected cut
    # E[cut] = sum_{(i,j) in E} (1 - <sigma_i^z sigma_j^z>)/2
    # <sigma_i^z sigma_j^z> = sum_x (-1)^(x_i + x_j) |psi_x|^2
    
    probs = np.abs(psi) ** 2
    
    expected_cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                # Compute <sigma_i^z sigma_j^z>
                corr = 0.0
                for x in range(dim):
                    if ((x >> i) & 1) != ((x >> j) & 1):
                        corr -= probs[x]
                    else:
                        corr += probs[x]
                expected_cut += (1.0 - corr) / 2.0
    
    return expected_cut


def _expected_cut_fast(n, adj, gammas, betas, cut_vals, A_matrix):
    """Faster version with precomputed matrices."""
    from scipy.linalg import expm
    
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Layer 1
    psi = psi * np.exp(-1j * gammas[0] * cut_vals)
    M1 = expm(-1j * betas[0] * A_matrix)
    psi = M1 @ psi
    
    # Layer 2
    psi = psi * np.exp(-1j * gammas[1] * cut_vals)
    M2 = expm(-1j * betas[1] * A_matrix)
    psi = M2 @ psi
    
    probs = np.abs(psi) ** 2
    
    expected_cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                corr = 0.0
                for x in range(dim):
                    if ((x >> i) & 1) != ((x >> j) & 1):
                        corr -= probs[x]
                    else:
                        corr += probs[x]
                expected_cut += (1.0 - corr) / 2.0
    
    return expected_cut


def _objective(params, n, adj, cut_vals, A_matrix, exact_opt):
    """Objective function to minimize (negative ratio)."""
    gammas = params[0:2]
    betas = params[2:4]
    try:
        exp_cut = _expected_cut_fast(n, adj, gammas, betas, cut_vals, A_matrix)
        ratio = exp_cut / exact_opt
        return -ratio
    except:
        return 1e10


def _spectral_init(n, adj):
    """Initialize angles based on graph spectral properties."""
    # Use eigenvalues of the adjacency matrix for initialization
    eigvals = np.linalg.eigvalsh(adj)
    max_eig = np.max(np.abs(eigvals))
    
    # Heuristic initialization based on graph properties
    num_edges = np.sum(adj) / 2
    density = num_edges / (n * (n - 1) / 2) if n > 1 else 0
    
    gamma1 = 0.5 / (1 + max_eig)
    gamma2 = 0.3 / (1 + max_eig)
    beta1 = 0.3
    beta2 = 0.2
    
    return np.array([gamma1, gamma2, beta1, beta2])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of edges (u, v)
        p: number of QAOA layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build adjacency matrix
    adj = _build_adjacency(n, edges)
    
    # Compute exact MaxCut
    exact_opt = _exact_maxcut(n, adj)
    if exact_opt == 0:
        # No edges, return zeros
        return np.zeros((p, 2))
    
    # Precompute cut values and mixer matrix
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        cut = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    if ((x >> i) & 1) != ((x >> j) & 1):
                        cut += 1.0
        cut_vals[x] = cut
    
    # Build mixer matrix (sum of X_i)
    A_matrix = np.zeros((dim, dim), dtype=np.float64)
    for x in range(dim):
        for i in range(n):
            y = x ^ (1 << i)
            A_matrix[x, y] += 1.0
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for optimization, return spectral init
        init = _spectral_init(n, adj)
        return np.array([[init[0], init[2]], [init[1], init[3]]])
    
    # Spectral initialization
    init_params = _spectral_init(n, adj)
    
    # Try BFGS from spectral init
    best_params = init_params.copy()
    best_score = -_objective(best_params, n, adj, cut_vals, A_matrix, exact_opt)
    
    try:
        result = minimize(
            _objective,
            init_params,
            args=(n, adj, cut_vals, A_matrix, exact_opt),
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        if result.fun < -best_score:
            best_params = result.x
            best_score = -result.fun
    except:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        return np.array([[best_params[0], best_params[2]], [best_params[1], best_params[3]]])
    
    # Try a few random restarts with BFGS
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        
        random_params = np.array([
            np.random.uniform(0, np.pi),
            np.random.uniform(0, np.pi),
            np.random.uniform(0, np.pi),
            np.random.uniform(0, np.pi)
        ])
        
        try:
            result = minimize(
                _objective,
                random_params,
                args=(n, adj, cut_vals, A_matrix, exact_opt),
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if result.fun < -best_score:
                best_params = result.x
                best_score = -result.fun
        except:
            pass
    
    # Extract angles
    gamma1, gamma2, beta1, beta2 = best_params
    
    # Ensure angles are in reasonable range
    gamma1 = np.mod(gamma1, 2 * np.pi)
    gamma2 = np.mod(gamma2, 2 * np.pi)
    beta1 = np.mod(beta1, 2 * np.pi)
    beta2 = np.mod(beta2, 2 * np.pi)
    
    return np.array([[gamma1, beta1], [gamma2, beta2]])
