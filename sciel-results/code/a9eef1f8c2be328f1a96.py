"""Program for spectral graph lifts task."""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    def build_signed_adj(signs):
        """Build signed adjacency matrix from edge signs."""
        A = np.zeros((n, n))
        for (u, v), s in zip(edges, signs):
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(A):
        """Compute spectral radius (max abs eigenvalue)."""
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, None

    # Phase-based optimization
    # Objective: Minimize spectral radius of the "cosine" adjacency matrix.
    # This acts as a smooth proxy for the signed adjacency spectral radius.
    
    def phase_objective(theta):
        # theta is a vector of length n
        C = np.zeros((n, n))
        for (u, v) in edges:
            val = np.cos(theta[u] - theta[v])
            C[u, v] = val
            C[v, u] = val
        eigvals = np.linalg.eigvalsh(C)
        return np.max(np.abs(eigvals))

    # Initialize phases randomly
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Try multiple random initializations for phase optimization
    for restart in range(5):
        theta0 = rng.uniform(0, 2 * np.pi, n)
        
        # Use gradient descent (L-BFGS-B) to minimize the proxy objective
        try:
            result = minimize(phase_objective, theta0, method='L-BFGS-B', options={'maxiter': 100})
            theta_opt = result.x
        except Exception:
            theta_opt = theta0

        # Convert phases to signs
        signs = np.ones(len(edges), dtype=int)
        for i, (u, v) in enumerate(edges):
            val = np.cos(theta_opt[u] - theta_opt[v])
            signs[i] = 1 if val >= 0 else -1
        
        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # If no valid lift found yet, fallback to random search + local search
    if best_signs is None:
        # Seed random samples
        for _ in range(budget):
            signs = rng.choice([-1, 1], size=len(edges))
            valid, radius = is_valid_lift(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        if best_signs is None:
            # Last resort: all +1
            best_signs = np.ones(len(edges), dtype=int)
            valid, radius = is_valid_lift(best_signs)
            if valid:
                best_radius = radius
            else:
                # Try to fix connectivity? Or just return best effort.
                # The task says "or the whole result is invalid".
                # We must return a valid one if possible.
                pass

    # Local search refinement from the best found signs
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Local search: try flipping each edge
        for iteration in range(100):
            improved = False
            for i in range(len(edges)):
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                valid, radius = is_valid_lift(current_signs)
                
                if valid and radius < current_radius:
                    current_radius = radius
                    improved = True
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
            
            if not improved:
                break

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
