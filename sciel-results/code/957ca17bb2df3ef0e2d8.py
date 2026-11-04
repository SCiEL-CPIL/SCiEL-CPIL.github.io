"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = 0.0

    # Strategy: Topological Search
    # 1. Generate initial contact graphs (random or heuristic)
    # 2. For each graph, solve for the "tight" packing using Newton's method
    #    on the constraint system: distance(i,j) = r_i + r_j for all contacts
    #    and wall constraints if touching.
    # 3. If the solution is valid (no overlaps, inside square), evaluate score.
    # 4. Mutate the graph (add/remove edges) and repeat.
    
    def _eval_packing(xy, r):
        """
        Evaluate a packing: returns (total_radius, is_valid).
        """
        n = len(xy)
        if n != N:
            return 0.0, False
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if ri < 1e-9:
                return 0.0, False
            if x - ri < -1e-8 or 1.0 - x - ri < -1e-8 or y - ri < -1e-8 or 1.0 - y - ri < -1e-8:
                return 0.0, False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-8:
                    return 0.0, False
        return float(np.sum(r)), True

    def _generate_initial_graph(xy, r):
        """
        Generate a contact graph based on current positions.
        Edges exist if circles are close (within 1.1 * (r_i + r_j)) or touching walls.
        We return a set of edges (i, j) for circle-circle and (i, 'wall_x') etc.
        For simplicity in the solver, we will focus on circle-circle contacts primarily,
        but allow wall contacts to be handled by fixing position or adjusting radius.
        
        Actually, a more robust approach for the solver is to treat the problem as:
        Maximize sum(r) subject to:
        1. d(i,j) >= r_i + r_j
        2. 0 <= x_i <= 1, 0 <= y_i <= 1
        3. r_i >= 0
        
        The "tight" packing on a specific graph G means:
        d(i,j) = r_i + r_j for all (i,j) in G.
        This is a system of non-linear equations.
        """
        edges = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Heuristic: if they are close, assume contact
                if dist < 1.05 * (r[i] + r[j]):
                    edges.add((i, j))
        return edges

    def _solve_tight_packing(edges, initial_xy, initial_r, max_iters=500, tol=1e-8):
        """
        Solve for x, y, r such that d(i,j) = r_i + r_j for all (i,j) in edges.
        We use a simple iterative relaxation (Gauss-Seidel style) or Newton.
        Given the complexity, we will use a penalty-based local search on the 
        "tightness" of the graph, but constrained to the graph structure.
        
        Actually, let's use a different approach: 
        Fix the topology. Use a local optimizer to maximize sum(r) 
        subject to the constraints that the circles in the graph are touching.
        
        However, implementing a full constrained NLP with changing topology is hard.
        Let's use a simpler "force" model that respects the graph:
        - Forces only act between connected nodes in the graph.
        - The goal is to expand the graph until it hits walls or becomes over-constrained.
        
        Better idea: 
        Use the existing force-directed method but bias the forces to maintain the graph.
        If (i,j) is in the graph, force d(i,j) -> r_i + r_j.
        If not, force d(i,j) > r_i + r_j (weak repulsion).
        """
        xy = initial_xy.copy()
        r = initial_r.copy()
        
        # We will run a specialized relaxation that tries to make the graph "tight"
        # and expand the radii.
        
        dt = 0.01
        damping = 0.8
        vel_xy = np.zeros_like(xy)
        vel_r = np.zeros_like(r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            f_xy = np.zeros_like(xy)
            f_r = np.zeros_like(r)
            
            # 1. Forces from graph edges: try to make them touch
            for (i, j) in edges:
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-9:
                    continue
                target_dist = r[i] + r[j]
                diff = dist - target_dist
                # If dist > target, pull together. If dist < target, push apart (hard constraint).
                force = 10.0 * diff
                ux = dx / dist
                uy = dy / dist
                f_xy[i, 0] += force * ux
                f_xy[i, 1] += force * uy
                f_xy[j, 0] -= force * ux
                f_xy[j, 1] -= force * uy
                # Radii should grow to meet if possible, but here we fix r? 
                # No, we want to maximize r. So if they are touching, we can grow them?
                # Actually, if we fix the graph, growing r requires moving x,y to maintain contact.
                # Let's just grow r slowly and let the position forces adjust.
            
            # 2. Weak repulsion for non-edges to prevent overlap
            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) in edges or (j, i) in edges:
                        continue
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-9:
                        continue
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        force = 50.0 * overlap
                        ux = dx / dist
                        uy = dy / dist
                        f_xy[i, 0] += force * ux
                        f_xy[i, 1] += force * uy
                        f_xy[j, 0] -= force * ux
                        f_xy[j, 1] -= force * uy
                        f_r[i] -= 5.0 * overlap
                        f_r[j] -= 5.0 * overlap

            # 3. Wall constraints
            for i in range(N):
                if xy[i, 0] < r[i]:
                    f_xy[i, 0] += 10.0 * (r[i] - xy[i, 0])
                if 1.0 - xy[i, 0] < r[i]:
                    f_xy[i, 0] -= 10.0 * (r[i] - (1.0 - xy[i, 0]))
                if xy[i, 1] < r[i]:
                    f_xy[i, 1] += 10.0 * (r[i] - xy[i, 1])
                if 1.0 - xy[i, 1] < r[i]:
                    f_xy[i, 1] -= 10.0 * (r[i] - (1.0 - xy[i, 1]))
                
                # Growth force: try to grow radius
                # Max radius is limited by walls and neighbors
                max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                # Also limited by neighbors
                for j in range(N):
                    if i == j:
                        continue
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    avail = dist - r[j]
                    if avail < max_r:
                        max_r = avail
                if r[i] < max_r:
                    f_r[i] += 1.0 * (max_r - r[i])
                else:
                    f_r[i] -= 1.0 * (r[i] - max_r)

            # Update
            vel_xy = damping * vel_xy + dt * f_xy
            vel_r = damping * vel_r + dt * f_r
            xy += vel_xy
            r += vel_r
            
            # Clamp
            for i in range(N):
                r[i] = max(r[i], 0.001)
                r_clip = min(r[i], 0.49)
                xy[i, 0] = np.clip(xy[i, 0], r_clip, 1.0 - r_clip)
                xy[i, 1] = np.clip(xy[i, 1], r_clip, 1.0 - r_clip)
                r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                r[i] = max(r[i], 0.001)
            
            dt *= 0.999

        return xy, r

    def _mutate_graph(edges, xy, r):
        """
        Randomly add or remove an edge from the graph.
        """
        edges = set(edges)
        if random.random() < 0.5:
            # Add edge
            # Find a pair that is close but not connected
            candidates = []
            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) not in edges:
                        dx = xy[i, 0] - xy[j, 0]
                        dy = xy[i, 1] - xy[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < 1.2 * (r[i] + r[j]):
                            candidates.append((i, j))
            if candidates:
                e = random.choice(candidates)
                edges.add(e)
        else:
            # Remove edge
            if edges:
                e = random.choice(list(edges))
                edges.remove(e)
        
        return edges

    # --- Main Loop ---
    
    # 1. Initial Setup: Create a reasonable starting packing
    # Use a grid or random initialization
    rng = np.random.RandomState(42)
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # Simple grid init
    side = int(np.ceil(np.sqrt(N)))
    count = 0
    for i in range(side):
        for j in range(side):
            if count < N:
                xy[count, 0] = (i + 0.5) / side
                xy[count, 1] = (j + 0.5) / side
                r[count] = 0.4 / side
                count += 1
    
    # Initial relaxation to get a valid packing
    # We'll use a standard force-directed relax for the first few seconds to get a valid base
    deadline_init = start_time + 10.0
    dt = 0.01
    damping = 0.8
    vel_xy = np.zeros_like(xy)
    vel_r = np.zeros_like(r)
    
    while time.time() < deadline_init:
        f_xy = np.zeros_like(xy)
        f_r = np.zeros_like(r)
        for i in range(N):
            for j in range(i + 1, N):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-9:
                    continue
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    force = 50.0 * overlap
                    ux = dx / dist
                    uy = dy / dist
                    f_xy[i, 0] += force * ux
                    f_xy[i, 1] += force * uy
                    f_xy[j, 0] -= force * ux
                    f_xy[j, 1] -= force * uy
                    f_r[i] -= 5.0 * overlap
                    f_r[j] -= 5.0 * overlap
            
            if xy[i, 0] < r[i]:
                f_xy[i, 0] += 10.0 * (r[i] - xy[i, 0])
            if 1.0 - xy[i, 0] < r[i]:
                f_xy[i, 0] -= 10.0 * (r[i] - (1.0 - xy[i, 0]))
            if xy[i, 1] < r[i]:
                f_xy[i, 1] += 10.0 * (r[i] - xy[i, 1])
            if 1.0 - xy[i, 1] < r[i]:
                f_xy[i, 1] -= 10.0 * (r[i] - (1.0 - xy[i, 1]))
            
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                avail = dist - r[j]
                if avail < max_r:
                    max_r = avail
            if r[i] < max_r:
                f_r[i] += 1.0 * (max_r - r[i])
            else:
                f_r[i] -= 1.0 * (r[i] - max_r)

        vel_xy = damping * vel_xy + dt * f_xy
        vel_r = damping * vel_r + dt * f_r
        xy += vel_xy
        r += vel_r
        
        for i in range(N):
            r[i] = max(r[i], 0.001)
            r_clip = min(r[i], 0.49)
            xy[i, 0] = np.clip(xy[i, 0], r_clip, 1.0 - r_clip)
            xy[i, 1] = np.clip(xy[i, 1], r_clip, 1.0 - r_clip)
            r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = max(r[i], 0.001)
        
        dt *= 0.9999

    # Check initial validity
    score, valid = _eval_packing(xy, r)
    if valid:
        best_score = score
        best_result = np.column_stack([xy, r])
    
    # 2. Topological Search Loop
    current_edges = _generate_initial_graph(xy, r)
    
    while time.time() < deadline:
        # Solve for tight packing on current graph
        new_xy, new_r = _solve_tight_packing(current_edges, xy, r, max_iters=200)
        
        # Check validity
        new_score, new_valid = _eval_packing(new_xy, new_r)
        
        if new_valid and new_score > best_score:
            best_score = new_score
            best_result = np.column_stack([new_xy, new_r])
            xy = new_xy
            r = new_r
            # Regenerate graph based on new positions
            current_edges = _generate_initial_graph(xy, r)
        
        # Mutate graph
        current_edges = _mutate_graph(current_edges, xy, r)
        
        # If we get stuck or invalid, try to recover by relaxing without graph constraints
        if not new_valid:
             # Do a few steps of standard relaxation to fix overlaps
             for _ in range(50):
                 f_xy = np.zeros_like(xy)
                 f_r = np.zeros_like(r)
                 for i in range(N):
                     for j in range(i + 1, N):
                         dx = xy[i, 0] - xy[j, 0]
                         dy = xy[i, 1] - xy[j, 1]
                         dist = np.sqrt(dx * dx + dy * dy)
                         if dist < 1e-9:
                             continue
                         overlap = r[i] + r[j] - dist
                         if overlap > 0:
                             force = 50.0 * overlap
                             ux = dx / dist
                             uy = dy / dist
                             f_xy[i, 0] += force * ux
                             f_xy[i, 1] += force * uy
                             f_xy[j, 0] -= force * ux
                             f_xy[j, 1] -= force * uy
                             f_r[i] -= 5.0 * overlap
                             f_r[j] -= 5.0 * overlap
                 
                 if xy[i, 0] < r[i]: f_xy[i, 0] += 10.0 * (r[i] - xy[i, 0])
                 if 1.0 - xy[i, 0] < r[i]: f_xy[i, 0] -= 10.0 * (r[i] - (1.0 - xy[i, 0]))
                 if xy[i, 1] < r[i]: f_xy[i, 1] += 10.0 * (r[i] - xy[i, 1])
                 if 1.0 - xy[i, 1] < r[i]: f_xy[i, 1] -= 10.0 * (r[i] - (1.0 - xy[i, 1]))
                 
                 # No growth force during recovery, just fix overlaps
                 
                 xy += 0.01 * f_xy
                 r += 0.01 * f_r
                 
                 for i in range(N):
                     r[i] = max(r[i], 0.001)
                     r_clip = min(r[i], 0.49)
                     xy[i, 0] = np.clip(xy[i, 0], r_clip, 1.0 - r_clip)
                     xy[i, 1] = np.clip(xy[i, 1], r_clip, 1.0 - r_clip)
                     r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                     r[i] = max(r[i], 0.001)
                 
                 score, valid = _eval_packing(xy, r)
                 if valid:
                     break
             
             if valid:
                 current_edges = _generate_initial_graph(xy, r)
             else:
                 # If still invalid, reset to last best
                 if best_result is not None:
                     xy = best_result[:, :2].copy()
                     r = best_result[:, 2].copy()
                     current_edges = _generate_initial_graph(xy, r)

    # Final Result
    if best_result is None:
        # Fallback
        result = np.zeros((N, 3))
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    return best_result

# EVOLVE-BLOCK-END
