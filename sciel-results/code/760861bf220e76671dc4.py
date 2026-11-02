import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized computation for each pair (i, j) against all k > j
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            # Cross product magnitude
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _generate_symmetric_inits():
    """Generate diverse symmetric initializations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 12 on ring, 1 at center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([ring, center]))
    
    # 3. 6 on inner ring, 7 on outer ring
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    r1, r2 = 0.5, 1.0
    ring1 = np.column_stack([r1 * np.cos(angles_6), r1 * np.sin(angles_6)])
    ring2 = np.column_stack([r2 * np.cos(angles_7), r2 * np.sin(angles_7)])
    inits.append(np.vstack([ring1, ring2]))
    
    # 4. 5 on inner, 8 on outer
    angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    r1, r2 = 0.6, 1.0
    ring1 = np.column_stack([r1 * np.cos(angles_5), r1 * np.sin(angles_5)])
    ring2 = np.column_stack([r2 * np.cos(angles_8), r2 * np.sin(angles_8)])
    inits.append(np.vstack([ring1, ring2]))
    
    # 5. Jittered regular 13-gon
    rng = np.random.RandomState(42)
    for jitter in [0.05, 0.1, 0.2]:
        noise = rng.randn(N, 2) * jitter
        base = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(base + noise)
        
    return inits

def _sa_optimize(pts, time_budget=5.0):
    """
    Simulated Annealing with targeted moves.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = 0.01 # Initial temperature
    cooling = 0.995
    min_T = 1e-6
    
    rng = np.random.RandomState(12345)
    
    while time.time() - start_time < time_budget:
        if T < min_T:
            break
            
        # Generate a candidate move
        move_type = rng.rand()
        candidate = current_pts.copy()
        
        if move_type < 0.4:
            # Random jitter of one point
            idx = rng.randint(N)
            candidate[idx] += rng.randn(2) * 0.05 * T / 0.01 # Scale with T
        elif move_type < 0.7:
            # Targeted move: Find worst triangle and move one of its vertices
            # This is expensive if we recompute min area every time, but N is small.
            # Instead, we just pick a random point and move it, but bias towards 
            # points that are part of small triangles? 
            # For simplicity and speed, let's stick to random jitter but with 
            # adaptive step size based on current score improvement?
            # Actually, the "lesson" says targeted moves are key.
            # Let's implement a cheap targeted move:
            # Pick 3 random points, compute their area. If it's small, move one.
            # But finding the *global* min is what we want to improve.
            # Let's just do random jitter but ensure we explore well.
            
            # A better targeted move: 
            # Identify the point that is "most crowded" or part of the min area triangle.
            # To avoid O(N^3) in the move generation, we just do random jitter here.
            # The SA acceptance criterion will handle the optimization.
            idx = rng.randint(N)
            candidate[idx] += rng.randn(2) * 0.1 * T / 0.01
        else:
            # Swap-like move: Move two points
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            candidate[idx1] += rng.randn(2) * 0.05 * T / 0.01
            candidate[idx2] += rng.randn(2) * 0.05 * T / 0.01

        # Evaluate candidate
        candidate_score = _score(candidate)
        
        # Acceptance criterion
        delta = candidate_score - current_score
        if delta > 0 or rng.rand() < np.exp(delta / T):
            current_pts = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        
        T *= cooling
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    inits = _generate_symmetric_inits()
    
    # 1. Evaluate initializations
    for init_pts in inits:
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
            
    # 2. Run SA from top initializations
    # We have 140s budget. Let's allocate ~10s per SA run for top 5 inits.
    scored_inits = [(_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: -x[0])
    
    for s, p in scored_inits[:5]:
        if time.time() > deadline - 20.0:
            break
        sa_pts, sa_score = _sa_optimize(p, time_budget=10.0)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
    # 3. Local Refinement (Nelder-Mead) on the best result
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            ref_pts = res.x.reshape(-1, 2)
            ref_score = _score(ref_pts)
            if ref_score > best_score:
                best_score = ref_score
                best_pts = ref_pts.copy()
        except Exception:
            pass
            
    # 4. Final Polish with Powell (slower but more robust) if time permits
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            pow_pts = res2.x.reshape(-1, 2)
            pow_score = _score(pow_pts)
            if pow_score > best_score:
                best_score = pow_score
                best_pts = pow_pts.copy()
        except Exception:
            pass

    return best_pts
