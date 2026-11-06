import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    """
    Finds 8 points maximizing the product of pairwise distances normalized by diameter.
    Uses a parameterization based on the coefficients of a monic polynomial of degree 8
    with real coefficients. The roots of this polynomial are the 8 points.
    This allows for any configuration with conjugate symmetry (which is required for real coefficients).
    """
    
    def roots_from_coeffs(coeffs):
        """
        Compute roots of a monic polynomial from its coefficients [c7, c6, ..., c0].
        coeffs[0] is c7, ..., coeffs[7] is c0.
        """
        # np.roots expects [a_n, ..., a_0]
        # We assume a_n = 1
        poly_coeffs = np.concatenate(([1.0], coeffs))
        try:
            roots = np.roots(poly_coeffs)
        except Exception:
            return None
        return roots

    def log_objective(coeffs):
        """
        Minimizes -log(J) where J = prod_{i!=j} d_ij / D^56.
        coeffs: array of 8 floats [c7, c6, c5, c4, c3, c2, c1, c0]
        """
        roots = roots_from_coeffs(coeffs)
        if roots is None:
            return 1e10
        
        # Check for numerical issues: roots should be finite
        if not np.all(np.isfinite(roots)):
            return 1e10
            
        # Check for distinct points (roots should be distinct)
        # If roots are too close, the product of distances becomes 0, log -> -inf, -log -> +inf
        # We can check the minimum distance between distinct roots.
        n = len(roots)
        if n != 8:
            return 1e10
            
        # Calculate pairwise distances
        # Use complex arithmetic
        diff = roots[:, np.newaxis] - roots[np.newaxis, :]
        dists = np.abs(diff)
        
        # Exclude self-distances
        mask = ~np.eye(8, dtype=bool)
        valid_dists = dists[mask]
        
        # If any distance is very small, penalty
        min_dist = np.min(valid_dists)
        if min_dist < 1e-8:
            return 1e10
            
        log_prod = np.sum(np.log(valid_dists))
        D = np.max(dists)
        
        if D < 1e-8:
            return 1e10
            
        log_J = log_prod - 56.0 * np.log(D)
        
        return -log_J

    def get_points_from_coeffs(coeffs):
        roots = roots_from_coeffs(coeffs)
        if roots is None:
            return None
        return roots

    # Initial guesses
    # 1. Regular Octagon
    # Roots of z^8 - 1 = 0? No, regular octagon centered at 0.
    # Vertices at $e^{i \pi k / 4}$.
    # Polynomial: $z^8 - 1 = 0$.
    # Coeffs: [0, 0, 0, 0, 0, 0, 0, -1]
    init_reg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    
    # 2. Two squares?
    # Roots of $(z^4 - 1)(z^4 - a) = 0$?
    # Or maybe a configuration with 4 real roots and 2 complex pairs?
    
    # 3. Perturbations of the regular octagon
    starts = [init_reg]
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Small perturbations
        pert = rng.normal(0, 0.01, size=8)
        starts.append(init_reg + pert)
        
    # Add some structured starts
    # e.g., roots of $z^8 - a z^4 + b = 0$
    # This has symmetry $z \to -z$ and $z \to 1/z$?
    # Let's try $z^8 - c z^4 + 1 = 0$. Roots are on unit circle?
    # If $z^4 = w$, $w^2 - c w + 1 = 0$.
    # $w = (c \pm \sqrt{c^2-4})/2$.
    # If $|c| < 2$, $w$ is complex unit modulus.
    # Then $z$ are 8th roots of $w$.
    # This generates a set of 8 points on the unit circle, forming two squares?
    # Let's try c = 1.5.
    # w1, w2 are conjugate complex numbers on unit circle.
    # Roots of $z^4 - w1 = 0$ and $z^4 - w2 = 0$.
    # This is a subset of the regular octagon?
    # Regular octagon is $z^8 - 1 = 0$. Here $c=0$.
    # Let's add a start for c=1.0.
    init_sq = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]) # z^8 + z^4 + 1 = 0
    # Roots of $z^8 + z^4 + 1 = 0$ are the 12th roots of unity excluding 4th roots?
    # $(z^{12}-1)/(z^4-1) = z^8+z^4+1$.
    # So these are points on the unit circle.
    starts.append(init_sq)
    
    # Another start: $z^8 - 2 z^4 + 1 = 0 \implies (z^4-1)^2=0$. Bad (repeated roots).
    # $z^8 - 0.5 z^4 + 1 = 0$.
    init_sq2 = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0])
    starts.append(init_sq2)

    best_score = 1e10
    best_coeffs = init_reg
    
    for start in starts:
        try:
            # Use Nelder-Mead for robustness without gradients
            res = minimize(log_objective, start, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # Get the points from the best coefficients
    pts = get_points_from_coeffs(best_coeffs)
    
    if pts is None or len(pts) != 8:
        # Fallback to regular octagon
        angles = np.arange(8) * np.pi / 4
        pts = np.exp(1j * angles)
        
    # Check for duplicates/numerical issues in the final set
    # If optimization resulted in bad points, fallback
    diff = pts[:, np.newaxis] - pts[np.newaxis, :]
    dists = np.abs(diff)
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    if np.min(valid_dists) < 1e-6:
        angles = np.arange(8) * np.pi / 4
        pts = np.exp(1j * angles)

    # Scale to diameter 2
    D = np.max(dists)
    if D > 1e-12:
        scale_factor = 2.0 / D
        pts = pts * scale_factor
    else:
        # If diameter is 0 (all same point?), fallback
        angles = np.arange(8) * np.pi / 4
        pts = np.exp(1j * angles) * 1.0 # Diameter 2 for unit circle radius 1/sqrt(2)? No.
        # Regular octagon on unit circle has diameter 2 * cos(pi/8) approx 1.84.
        # We want diameter 2.
        # Radius R such that 2 * R * cos(pi/8) = 2 => R = 1/cos(pi/8).
        R = 1.0 / np.cos(np.pi/8)
        pts = R * np.exp(1j * angles)

    # Convert to list of [x, y]
    result = [[float(p.real), float(p.imag)] for p in pts]
    
    return result

if __name__ == "__main__":
    construct()
