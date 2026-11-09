"""Rewritten program: genetic algorithm for packing 11 unit hexagons."""
import numpy as np
from numpy import pi, cos, sin, sqrt, clip
import random

N = 11

# Precompute vertices of a unit regular hexagon (side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees, at distance 1 from center
_HEX_VERTS = np.array([[cos(pi/3 * i), sin(pi/3 * i)] for i in range(6)])

def _hex_verts(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon centered at (cx, cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    rot = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])
    return _HEX_VERTS @ rot.T + np.array([cx, cy])

def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-15:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _min_dist_hex(h1, h2):
    """Minimum distance between two hexagons (negative if overlapping)."""
    v1 = _hex_verts(h1[0], h1[1], h1[2])
    v2 = _hex_verts(h2[0], h2[1], h2[2])
    
    min_d = float('inf')
    
    # Vertex of v1 to edges of v2
    for i in range(6):
        p = v1[i]
        for j in range(6):
            d = _point_to_segment_dist(p, v2[j], v2[(j+1) % 6])
            if d < min_d:
                min_d = d
    
    # Vertex of v2 to edges of v1
    for i in range(6):
        p = v2[i]
        for j in range(6):
            d = _point_to_segment_dist(p, v1[j], v1[(j+1) % 6])
            if d < min_d:
                min_d = d
    
    return min_d

def _compute_outer_side(inner_hexes, center):
    """Find minimum side length of outer hexagon containing all inner hexagons."""
    all_verts = []
    for h in inner_hexes:
        all_verts.extend(_hex_verts(h[0], h[1], h[2]))
    all_verts = np.array(all_verts)
    
    cx, cy = center
    rel = all_verts - np.array([cx, cy])
    
    best_s = float('inf')
    for phi_deg in np.linspace(0, 60, 121):
        phi = np.radians(phi_deg)
        s_val = 0.0
        for k in range(6):
            theta = phi + np.radians(30 + 60 * k)
            dir_vec = np.array([cos(theta), sin(theta)])
            max_dot = np.max(rel @ dir_vec)
            s_needed = max_dot / (sqrt(3) / 2)
            if s_needed > s_val:
                s_val = s_needed
        if s_val < best_s:
            best_s = s_val
    
    return best_s

def _fitness(inner_hexes, center):
    """Return outer side length, or inf if overlapping."""
    for i in range(len(inner_hexes)):
        for j in range(i+1, len(inner_hexes)):
            d = _min_dist_hex(inner_hexes[i], inner_hexes[j])
            if d < -1e-9:
                return float('inf')
    return _compute_outer_side(inner_hexes, center)

# EVOLVE-BLOCK-START
def construct():
    random.seed(42)
    np.random.seed(42)
    
    def _initial_config():
        hexes = []
        hexes.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = pi/3 * k
            x = 2.0 * cos(angle)
            y = 2.0 * sin(angle)
            hexes.append((x, y, 0.0))
        for k in range(4):
            angle = pi/3 * (2*k + 1) + pi/6
            x = 3.464 * cos(angle)
            y = 3.464 * sin(angle)
            hexes.append((x, y, 0.0))
        return hexes
    
    current = _initial_config()
    current_center = (0.0, 0.0)
    current_score = _fitness(current, current_center)
    
    population_size = 40
    max_generations = 150
    mutation_rate = 0.4
    elite_size = 5
    
    population = [(current, current_center, current_score)]
    
    for _ in range(population_size - 1):
        hexes = _initial_config()
        for i in range(len(hexes)):
            hexes[i] = (hexes[i][0] + random.gauss(0, 0.5),
                        hexes[i][1] + random.gauss(0, 0.5),
                        hexes[i][2] + random.gauss(0, 30))
        center = (random.gauss(0, 0.5), random.gauss(0, 0.5))
        score = _fitness(hexes, center)
        population.append((hexes, center, score))
    
    for gen in range(max_generations):
        population.sort(key=lambda x: x[2] if x[2] != float('inf') else 1e9)
        
        new_population = [tuple(p) for p in population[:elite_size]]
        
        while len(new_population) < population_size:
            def _select():
                a = random.choice(population)
                b = random.choice(population)
                return a if a[2] < b[2] else b
            
            p1 = _select()
            p2 = _select()
            
            child_hexes = []
            for i in range(N):
                if random.random() < 0.5:
                    child_hexes.append(p1[0][i])
                else:
                    child_hexes.append(p2[0][i])
            
            child_center = ((p1[1][0] + p2[1][0]) / 2, (p1[1][1] + p2[1][1]) / 2)
            
            for i in range(N):
                if random.random() < mutation_rate:
                    child_hexes[i] = (child_hexes[i][0] + random.gauss(0, 0.3),
                                      child_hexes[i][1] + random.gauss(0, 0.3),
                                      (child_hexes[i][2] + random.gauss(0, 20)) % 60)
            if random.random() < mutation_rate:
                child_center = (child_center[0] + random.gauss(0, 0.2),
                                child_center[1] + random.gauss(0, 0.2))
            
            score = _fitness(child_hexes, child_center)
            new_population.append((child_hexes, child_center, score))
        
        population = new_population
    
    population.sort(key=lambda x: x[2] if x[2] != float('inf') else 1e9)
    best_hexes, best_center, best_score = population[0]
    
    inner = [[h[0], h[1], h[2]] for h in best_hexes]
    
    return inner, (best_center[0], best_center[1]), best_score, 0.0
# EVOLVE-BLOCK-END
