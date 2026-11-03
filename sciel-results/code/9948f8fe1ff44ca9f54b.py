import numpy as np
import time
import math

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def cdist(a, b):
        """Compute pairwise distances."""
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def generate_hex_lattice(target_count=210, margin=0.002):
        """
        Generate a hexagonal lattice of candidate sites inside [-0.5+margin, 0.5-margin]^2.
        We want slightly more than N sites so we can choose a subset.
        """
        # Target spacing: if we pack ~210 circles in a unit square, spacing ~ 1/sqrt(210) ≈ 0.069
        # Hex spacing d: area per circle in hex = d^2 * sqrt(3)/2
        # For 210 circles: 210 * d^2 * sqrt(3)/2 ≈ 1 => d ≈ sqrt(2/(210*sqrt(3))) ≈ 0.068
        d = 1.0 / math.sqrt(target_count * math.sqrt(3) / 2)
        
        row_spacing = d * math.sqrt(3) / 2
        sites = []
        
        # Generate rows
        y = -0.5 + margin + d / 2
        row_idx = 0
        while y <= 0.5 - margin - d / 2 + 1e-9:
            if row_idx % 2 == 0:
                x_start = -0.5 + margin + d / 2
                n_in_row = int((0.5 - margin - d / 2 - x_start) / d) + 1
            else:
                x_start = -0.5 + margin + d
                n_in_row = int((0.5 - margin - d / 2 - x_start) / d) + 1
            
            for col in range(max(0, n_in_row)):
                x = x_start + col * d
                if -0.5 + margin - 1e-9 <= x <= 0.5 - margin + 1e-9:
                    sites.append([x, y])
            
            y += row_spacing
            row_idx += 1
        
        sites = np.array(sites)
        return sites, d

    def initial_selection(sites, n_select=N):
        """Select n_select sites from sites, preferring central positions for better wall distances."""
        n_sites = len(sites)
        # Score by distance from center (prefer center) but also by local density
        centers_dist = np.sqrt(np.sum(sites * sites, axis=1))
        # Random shuffle with slight bias toward center
        order = np.random.permutation(n_sites)
        # Sort by a mix of centrality and randomness
        scores = centers_dist[order] + np.random.exponential(0.05, n_sites)
        selected = order[np.argsort(scores)[:n_select]]
        return sites[selected].copy()

    def pair_swap_search(sites, selected_mask, time_limit=10.0):
        """
        Stochastic local search: try swapping a selected site with an unselected site.
        Accept if it improves the minimum radius (or with some probability for exploration).
        """
        n_sites = len(sites)
        n_selected = np.sum(selected_mask)
        
        # Precompute which sites are selected
        selected_indices = np.where(selected_mask)[0]
        unselected_indices = np.where(~selected_mask)[0]
        
        current_sites = sites[selected_mask].copy()
        current_r = compute_radius(current_sites)
        
        best_sites = current_sites.copy()
        best_r = current_r
        
        start = time.time()
        temp = 0.005  # initial temperature for acceptance
        cooling = 0.995
        
        # Build adjacency: for each site, which other sites are "close" (within 3*d)
        # This limits swap candidates to nearby sites for efficiency
        d_est = 0.07  # approximate spacing
        adj_radius = 3 * d_est
        
        # Precompute distances between all sites (for neighbor lookup)
        # This is n_sites^2 which for ~210 sites is ~44k, fine
        all_dists = cdist(sites, sites)
        
        # For each site, find neighbors within adj_radius
        neighbors = []
        for i in range(n_sites):
            mask = all_dists[i] < adj_radius
            mask[i] = False
            neighbors.append(np.where(mask)[0])
        
        iteration = 0
        stall_count = 0
        
        while time.time() < start + time_limit:
            iteration += 1
            
            # Pick a random selected site to remove
            sel_idx = np.random.randint(0, len(selected_indices))
            remove_site = selected_indices[sel_idx]
            
            # Find candidate replacement sites: neighbors of remove_site that are unselected
            cand_neighbors = neighbors[remove_site]
            cand_mask = np.isin(cand_neighbors, unselected_indices)
            candidates = cand_neighbors[cand_mask]
            
            if len(candidates) == 0:
                # Try any unselected site (less likely to help but keeps search going)
                if len(unselected_indices) > 0:
                    candidates = np.random.choice(unselected_indices, size=1)
                else:
                    continue
            
            add_site = np.random.choice(candidates)
            
            # Perform the swap
            new_selected = selected_mask.copy()
            new_selected[remove_site] = False
            new_selected[add_site] = True
            new_sites = sites[new_selected]
            
            new_r = compute_radius(new_sites)
            
            if new_r > current_r + 1e-10:
                # Accept improvement
                selected_mask = new_selected
                current_sites = new_sites
                current_r = new_r
                selected_indices = np.where(selected_mask)[0]
                unselected_indices = np.where(~selected_mask)[0]
                stall_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_sites = current_sites.copy()
            else:
                # Accept with probability based on temperature (simulated annealing)
                if np.random.random() < math.exp((new_r - current_r) / (temp + 1e-12)):
                    selected_mask = new_selected
                    current_sites = new_sites
                    current_r = new_r
                    selected_indices = np.where(selected_mask)[0]
                    unselected_indices = np.where(~selected_mask)[0]
                else:
                    stall_count += 1
            
            # Cool down
            temp *= cooling
            if temp < 1e-6:
                temp = 1e-6
            
            # Periodically re-seed temperature if stalled
            if stall_count > 500:
                temp = max(temp, 0.002)
                stall_count = 0
        
        return best_sites, best_r

    def local_refine(centers, time_limit=5.0):
        """
        After discrete selection, do a small continuous refinement:
        nudge each point slightly to maximize the minimum distance.
        """
        centers = centers.copy()
        start = time.time()
        current_r = compute_radius(centers)
        
        # Simple iterative: for each point, try small perturbations in 8 directions
        step_size = 0.002
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
        ])
        
        improved = True
        while improved and time.time() < start + time_limit:
            improved = False
            for i in range(N):
                if time.time() > start + time_limit - 0.5:
                    break
                for d in directions:
                    new_c = centers.copy()
                    new_c[i] += d * step_size
                    # Clamp to square
                    new_c[i, 0] = np.clip(new_c[i, 0], -0.5, 0.5)
                    new_c[i, 1] = np.clip(new_c[i, 1], -0.5, 0.5)
                    new_r = compute_radius(new_c)
                    if new_r > current_r + 1e-10:
                        centers = new_c
                        current_r = new_r
                        improved = True
                        break
                # Reduce step size occasionally
            step_size *= 0.95
            if step_size < 0.0001:
                break
        
        return centers, current_r

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    # Generate hex lattice with various parameters
    lattice_configs = [
        (210, 0.002),
        (215, 0.003),
        (220, 0.002),
        (210, 0.005),
    ]
    
    for target_count, margin in lattice_configs:
        if time.time() > deadline - 30:
            break
        
        sites, d = generate_hex_lattice(target_count, margin)
        
        if len(sites) < N:
            continue
        
        # Multiple random initial selections
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            selected_mask = np.zeros(len(sites), dtype=bool)
            # Initial selection: random subset
            indices = np.random.choice(len(sites), size=N, replace=False)
            selected_mask[indices] = True
            
            # Run pair-swap search
            remaining = deadline - time.time() - 10.0
            t_limit = min(remaining, 20.0)
            if t_limit < 5.0:
                t_limit = 5.0
            
            result_sites, result_r = pair_swap_search(sites, selected_mask, time_limit=t_limit)
            
            if result_r > best_radius:
                best_radius = result_r
                best_centers = result_sites.copy()
            
            # Local continuous refinement
            if time.time() < deadline - 5:
                refined, refined_r = local_refine(result_sites, time_limit=min(5.0, deadline - time.time() - 3))
                if refined_r > best_radius:
                    best_radius = refined_r
                    best_centers = refined.copy()
    
    # If we still have time, do one more pass with the best result
    if best_centers is not None and time.time() < deadline - 3:
        refined, refined_r = local_refine(best_centers, time_limit=min(5.0, deadline - time.time() - 2))
        if refined_r > best_radius:
            best_radius = refined_r
            best_centers = refined.copy()
    
    # Fallback: simple grid
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Final safety check
    final_r = compute_radius(best_centers)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
