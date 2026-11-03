import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLES)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    Normalized area = 0.5 * |det(bary_matrix)|
    """
    # Extract coordinates for all triples
    # p[_TRIPLES[:, 0]] gives shape (165, 3) for first vertex of each triple
    # We need to compute det for each triple.
    # det of matrix with rows r1, r2, r3 is r1 dot (r2 cross r3)
    
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # Cross product r2 x r3
    # (r2[1]*r3[2] - r2[2]*r3[1], r2[2]*r3[0] - r2[0]*r3[2], r2[0]*r3[1] - r2[1]*r3[0])
    cross_r2_r3 = np.cross(r2, r3)
    
    # Dot product r1 . (r2 x r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # P = u*A + v*B + w*C
    # x = 0*u + 1*v + 0.5*w = v + 0.5*w
    # y = 0*u + 0*v + sqrt(3)/2*w = sqrt(3)/2*w
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to be valid (non-negative, sum to 1).
    This is a projection onto the simplex.
    """
    b = bary.copy()
    # Normalize first to handle scale
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    
    # Clip negative values to 0
    b = np.clip(b, 0, 1)
    
    # Re-normalize
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _objective_flat(x_flat):
    """
    Objective function for optimization.
    x_flat: shape (N*3,)
    Returns negative of min area (since we want to maximize, minimize negative).
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    area = _min_area_bary(bary)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _is_time_up():
        return time.time() > deadline - 2.0
    
    best_area = 0.0
    best_bary = None
    
    # Strategy:
    # 1. Run Differential Evolution for global search.
    #    DE is robust and good at escaping local minima.
    #    We use a moderate population and max iterations.
    # 2. Take the best result from DE and refine with Nelder-Mead.
    #    NM is deterministic and high precision.
    
    # Define bounds for barycentric coordinates [0, 1]
    # Since we project, we can allow slightly outside or just [0,1].
    # Using [0,1] is safe.
    bounds = [(0.0, 1.0)] * (N * 3)
    
    # Initial population for DE
    # We can seed it with a few structured guesses to help DE
    # But standard DE random init is usually fine.
    # Let's try a mix: standard DE, then refine.
    
    try:
        # DE Parameters
        # popsize=20, maxiter=100 is a reasonable start.
        # tol=1e-6, mutation=(0.5, 1.0), recombination=0.9
        res = differential_evolution(
            _objective_flat,
            bounds,
            maxiter=150,
            popsize=30,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=42,
            init='sobol',  # Better initial distribution
            polish=False,  # We will do our own polish with NM
            workers=1
        )
        
        if _is_time_up():
            return _bary_to_cart(best_bary) if best_bary is not None else _fallback()
            
        current_bary = _project_bary(res.x.reshape(N, 3))
        current_area = _min_area_bary(current_bary)
        
        if current_area > best_area:
            best_area = current_area
            best_bary = current_bary.copy()
            
        # Refine with Nelder-Mead
        # NM works on the flat vector
        x0 = current_bary.flatten()
        
        # NM options: maxiter, xatol, fatol
        # We want high precision.
        res_nm = minimize(
            _objective_flat,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if not _is_time_up():
            refined_bary = _project_bary(res_nm.x.reshape(N, 3))
            refined_area = _min_area_bary(refined_bary)
            
            if refined_area > best_area:
                best_area = refined_area
                best_bary = refined_bary.copy()
                
        # If DE didn't give a good enough start or if we have time, try a second round
        # with a different seed or initial guess?
        # DE is stochastic, so running it again might find a different basin.
        # However, NM refinement is key.
        
        # Let's try one more DE run if time permits, with a different seed
        if not _is_time_up():
            res2 = differential_evolution(
                _objective_flat,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.9,
                seed=123,
                init='sobol',
                polish=False,
                workers=1
            )
            if not _is_time_up():
                current_bary2 = _project_bary(res2.x.reshape(N, 3))
                current_area2 = _min_area_bary(current_bary2)
                
                if current_area2 > best_area:
                    best_area = current_area2
                    best_bary = current_bary2.copy()
                    
                # Refine this one too
                x02 = current_bary2.flatten()
                res_nm2 = minimize(
                    _objective_flat,
                    x02,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                if not _is_time_up():
                    refined_bary2 = _project_bary(res_nm2.x.reshape(N, 3))
                    refined_area2 = _min_area_bary(refined_bary2)
                    
                    if refined_area2 > best_area:
                        best_area = refined_area2
                        best_bary = refined_bary2.copy()
                        
    except Exception as e:
        # Fallback if optimization fails
        pass

    if best_bary is None:
        # Simple fallback: random points
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)

def _fallback():
    """Fallback if no valid result found."""
    rng = np.random.default_rng(0)
    bary = rng.dirichlet([1, 1, 1], size=N)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)
