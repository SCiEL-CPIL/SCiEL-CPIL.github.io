"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

# Pre-compute indices for all triples
# C(14, 3) = 364 triples
_IDX_I = []
_IDX_J = []
_IDX_K = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _IDX_I.append(i)
            _IDX_J.append(j)
            _IDX_K.append(k)
_IDX_I = np.array(_IDX_I)
_IDX_J = np.array(_IDX_J)
_IDX_K = np.array(_IDX_K)

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0.0 if invalid.
    """
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
        
        # Compute triangle areas
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _angles_to_points(angles):
    """Convert sorted angles on unit circle to points."""
    angles = np.sort(angles) % (2 * np.pi)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _circular_objective(angles):
    """
    Objective for circular parameterization: minimize -score
    angles: array of N angles (will be sorted and wrapped)
    """
    try:
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            return 1e6
        points = _angles_to_points(angles)
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def _generate_circular_guesses():
    """
    Generate diverse initial angle configurations for circular parameterization.
    Returns list of angle arrays.
    """
    rng = np.random.default_rng(42)
    guesses = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    guesses.append(angles.copy())
    
    # 2. Perturbed regular 14-gon
    for _ in range(10):
        a = angles + rng.normal(0, 0.3, N)
        guesses.append(a.copy())
    
    # 3. Two-cluster: 7 points near 0, 7 points near pi
    for spread in [0.1, 0.2, 0.3, 0.4]:
        a1 = rng.uniform(-spread, spread, 7)
        a2 = np.pi + rng.uniform(-spread, spread, 7)
        guesses.append(np.concatenate([a1, a2]))
    
    # 4. Three-cluster: 5, 5, 4 points
    for spread in [0.1, 0.2, 0.3]:
        a1 = rng.uniform(-spread, spread, 5)
        a2 = 2*np.pi/3 + rng.uniform(-spread, spread, 5)
        a3 = 4*np.pi/3 + rng.uniform(-spread, spread, 4)
        guesses.append(np.concatenate([a1, a2, a3]))
    
    # 5. Four-cluster: 4, 4, 3, 3 points
    for spread in [0.1, 0.2]:
        a1 = rng.uniform(-spread, spread, 4)
        a2 = np.pi/2 + rng.uniform(-spread, spread, 4)
        a3 = np.pi + rng.uniform(-spread, spread, 3)
        a4 = 3*np.pi/2 + rng.uniform(-spread, spread, 3)
        guesses.append(np.concatenate([a1, a2, a3, a4]))
    
    # 6. Random sorted angles with minimum separation
    for _ in range(10):
        # Generate random angles with some separation
        base = np.sort(rng.uniform(0, 2*np.pi, N))
        # Add small perturbations
        base = base + rng.normal(0, 0.1, N)
        guesses.append(base)
    
    # 7. Alternating pattern (even/odd indices spread)
    for offset in [0, np.pi/14, np.pi/7]:
        a = np.linspace(offset, 2*np.pi + offset, N, endpoint=False)
        guesses.append(a)
    
    return guesses

def _refine_angles(angles, deadline):
    """
    Refine angles using Powell optimization.
    Returns (best_angles, best_score).
    """
    best_angles = angles.copy()
    best_score = -_circular_objective(angles)
    
    # Try multiple Powell runs with different settings
    for maxiter in [200, 500, 1000]:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                _circular_objective,
                angles,
                method='Powell',
                options={'maxiter': maxiter, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_angles = res.x
        except Exception:
            pass
    
    # Coordinate ascent polish
    if time.time() < deadline - 5:
        cur_angles = best_angles.copy()
        cur_score = best_score
        step_sizes = [0.1, 0.01, 0.001, 0.0001]
        
        for ss in step_sizes:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(N):
                    for delta in [ss, -ss]:
                        trial = cur_angles.copy()
                        trial[i] += delta
                        s = -_circular_objective(trial)
                        if s > cur_score + 1e-12:
                            cur_angles = trial
                            cur_score = s
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_angles = cur_angles.copy()
    
    return best_angles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    best_points = None
    best_score = -np.inf
    best_angles = None
    
    guesses = _generate_circular_guesses()
    
    # Score all initial guesses
    scored_guesses = []
    for g in guesses:
        points = _angles_to_points(g)
        s = _score(points)
        scored_guesses.append((s, g))
        if s > best_score:
            best_score = s
            best_points = points.copy()
            best_angles = g.copy()
            
    # Sort by score descending
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 1: Refine top candidates using Powell
    num_refines = 15
    
    for idx in range(min(num_refines, len(scored_guesses))):
        if time.time() > deadline - 20:  # Leave 20s for final polish
            break
            
        s_init, angles = scored_guesses[idx]
        
        try:
            refined_angles, refined_score = _refine_angles(angles, deadline)
            if refined_score > best_score:
                best_score = refined_score
                best_points = _angles_to_points(refined_angles)
                best_angles = refined_angles
        except Exception:
            pass
    
    # Phase 2: Final polish on the best angles
    if best_angles is not None and time.time() < deadline - 5:
        polished_angles, polished_score = _refine_angles(best_angles, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = _angles_to_points(polished_angles)
            best_angles = polished_angles
    
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = _angles_to_points(angles)
        
    return best_points
# EVOLVE-BLOCK-END
