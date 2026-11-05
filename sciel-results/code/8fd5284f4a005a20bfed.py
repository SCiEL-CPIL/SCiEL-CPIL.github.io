"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    # Define several cluster topologies as axial coordinates (q, r)
    topologies = [
        # 3-4-3-2
        [(0, -1), (1, -1), (2, -1),
         (-1, 0), (0, 0), (1, 0), (2, 0),
         (-1, 1), (0, 1), (1, 1),
         (-1, 2), (0, 2)],
        # 2-3-4-3
        [(-1, 0), (0, 0),
         (-1, 1), (0, 1), (1, 1),
         (-2, 2), (-1, 2), (0, 2), (1, 2),
         (-1, 3), (0, 3), (1, 3)],
        # 2-3-4-3 shifted
        [(0, 0), (1, 0),
         (-1, 1), (0, 1), (1, 1),
         (-1, 2), (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
        # 1-3-4-4
        [(0, 0),
         (-1, 1), (0, 1), (1, 1),
         (-1, 2), (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3), (3, 3)],
        # 4-4-4
        [(-1, 0), (0, 0), (1, 0), (2, 0),
         (-1, 1), (0, 1), (1, 1), (2, 1),
         (-1, 2), (0, 2), (1, 2), (2, 2)],
    ]
    
    base_angles = np.arange(6) * np.pi / 3
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def compute_min_S(theta, vertices):
        normals = np.array([[np.cos(theta + i * np.pi / 3 + np.pi / 6),
                             np.sin(theta + i * np.pi / 3 + np.pi / 6)]
                            for i in range(6)])
        projections = vertices @ normals.T
        max_proj = np.max(projections, axis=0)
        return (2.0 / np.sqrt(3)) * np.max(max_proj)
    
    def golden_search(theta_lo, theta_hi, vertices, max_iters=80):
        phi = (np.sqrt(5) - 1) / 2
        lo, hi = theta_lo, theta_hi
        c1 = hi - phi * (hi - lo)
        c2 = lo + phi * (hi - lo)
        f1 = compute_min_S(c1, vertices)
        f2 = compute_min_S(c2, vertices)
        best_theta = c1 if f1 < f2 else c2
        best_S = min(f1, f2)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if f1 < f2:
                hi = c2; c2 = c1; f2 = f1
                c1 = hi - phi * (hi - lo)
                f1 = compute_min_S(c1, vertices)
            else:
                lo = c1; c1 = c2; f1 = f2
                c2 = lo + phi * (hi - lo)
                f2 = compute_min_S(c2, vertices)
            cur_best = c1 if f1 < f2 else c2
            cur_S = min(f1, f2)
            if cur_S < best_S:
                best_S = cur_S
                best_theta = cur_best
        return best_S, best_theta
    
    best_overall_S = float('inf')
    best_result = None
    
    for topo in topologies:
        if time.time() > deadline:
            break
        centers = np.array([q * a1 + r * a2 for q, r in topo])
        centers -= np.mean(centers, axis=0)
        all_vertices = np.array([base_vertices + c for c in centers])
        S, theta = golden_search(0.0, np.pi / 3, all_vertices)
        if S < best_overall_S:
            best_overall_S = S
            best_result = (centers, theta)
    
    if best_result is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], [0.0, 0.0], 40.0, 0.0
    
    centers, theta = best_result
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    return inner, [0.0, 0.0], float(best_overall_S), float(np.degrees(theta))
# EVOLVE-BLOCK-END
