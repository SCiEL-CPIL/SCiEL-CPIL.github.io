"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Edge normal from h1
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = np.linalg.norm(n1)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if np.max(proj1) < np.min(proj2) - 1e-10 or np.max(proj2) < np.min(proj1) - 1e-10:
                    return True
            
            # Edge normal from h2
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = np.linalg.norm(n2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if np.max(proj1) < np.min(proj2) - 1e-10 or np.max(proj2) < np.min(proj1) - 1e-10:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def honeycomb_init():
        """Create a proper honeycomb lattice initialization - much tighter than square grid."""
        params = np.zeros(3*N + 3)
        s = 1.0
        sqrt3 = np.sqrt(3)
        
        # Honeycomb lattice for flat-top hexagons:
        # Row spacing: sqrt(3) * s (vertical distance between rows)
        # Column spacing: 1.5 * s (horizontal distance between columns)
        # Odd columns offset by sqrt(3)/2 * s vertically
        
        positions = []
        # 4 columns, 3 rows = 12 hexagons
        for col in range(4):
            for row in range(3):
                x = col * 1.5 * s
                y = row * sqrt3 * s
                if col % 2 == 1:
                    y += sqrt3 / 2 * s
                positions.append((x, y, 0.0))
        
        # Center the configuration
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        for i in range(N):
            params[3*i] = positions[i][0] - cx
            params[3*i+1] = positions[i][1] - cy
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def differential_evolution(f, bounds, pop_size=40, max_iter=200, F=0.8, CR=0.5, seed=42):
        """Differential Evolution optimizer."""
        rng = np.random.RandomState(seed)
        n = len(bounds)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, n))
        fvals = np.array([f(pop[i]) for i in range(pop_size)])
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        for gen in range(max_iter):
            # Check time
            if time.time() > deadline - 15.0:
                break
            
            for i in range(pop_size):
                # Select three distinct random individuals
                while True:
                    a, b, c = rng.randint(0, pop_size, size=3)
                    if a != b and b != c and a != c and a != i and b != i and c != i:
                        break
                
                # Mutation
                trial = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                jrand = rng.randint(0, n)
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        trial[j] = pop[a][j] + F * (pop[b][j] - pop[c][j])
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                
                # Selection
                f_trial = f(trial)
                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Elitism: ensure best is preserved
            if fvals[np.argmin(fvals)] < best_f:
                best_f = fvals[np.argmin(fvals)]
                best_x = pop[np.argmin(fvals)].copy()
        
        return best_x, best_f

    def sa_polish(params, f, duration=10.0):
        """Simulated annealing polish with adaptive step size."""
        current = params.copy()
        current_f = f(current)
        best = current.copy()
        best_f = current_f
        
        T = 0.3
        T_min = 0.0005
        cooling = 0.9998
        base_step = 0.15
        start = time.time()
        
        while time.time() - start < duration:
            step = base_step * (0.3 + 0.7 * T / 0.3)
            new = current.copy()
            idx = random.randint(0, len(new) - 1)
            new[idx] += random.gauss(0, step)
            
            new_f = f(new)
            
            if new_f < current_f or random.random() < np.exp(-(new_f - current_f) / max(T, 1e-10)):
                current = new
                current_f = new_f
                if new_f < best_f:
                    best_f = new_f
                    best = new.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best, best_f

    # Initialize with honeycomb lattice
    best_params = honeycomb_init()
    best_score = objective(best_params)
    
    # Define bounds for differential evolution
    bounds = [(-5, 5)] * (3 * N) + [(-2, 2), (-2, 2), (-180, 180)]
    
    # Run differential evolution
    if time.time() < deadline - 20.0:
        de_params, de_score = differential_evolution(
            objective, bounds, pop_size=30, max_iter=150, F=0.8, CR=0.5
        )
        if de_score < best_score:
            best_params = de_params
            best_score = de_score
    
    # Polish with simulated annealing
    if time.time() < deadline - 5.0:
        sa_params, sa_score = sa_polish(best_params, objective, duration=12.0)
        if sa_score < best_score:
            best_params = sa_params
            best_score = sa_score
    
    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to honeycomb init
        fallback = honeycomb_init()
        inner = []
        for i in range(N):
            inner.append([float(fallback[3*i]), float(fallback[3*i+1]), float(fallback[3*i+2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
