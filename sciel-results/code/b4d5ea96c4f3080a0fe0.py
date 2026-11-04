import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from collections import defaultdict

N = 300

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit

    def remaining_time():
        return deadline - (time.time() - start_time)

    def is_finished():
        return (time.time() - start_time) > deadline

    def _compute_radius_fast(p):
        """Compute the minimum radius for a given set of points."""
        p = np.asarray(p)
        if p.shape != (N, 2):
            raise ValueError("Wrong shape")
        norms = np.linalg.norm(p, axis=1)
        min_bound = 1.0 - np.min(norms)
        if min_bound <= 0:
            return 0.0
        # Use cdist for speed
        dists = cdist(p, p)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        return min(min_pair, min_bound)

    def _init_hex():
        """Initialize with hexagonal lattice."""
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points, radius = points[mask], radius[mask]
            if len(points) >= N:
                points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
        return points

    def _build_spatial_hash(points, cell_size):
        """Build a spatial hash for fast neighbor lookups."""
        cell_map = defaultdict(list)
        for i, (x, y) in enumerate(points):
            cx = int(np.floor(x / cell_size))
            cy = int(np.floor(y / cell_size))
            cell_map[(cx, cy)].append(i)
        return cell_map

    def _get_neighbors_spatial(points, cell_map, cell_size, query_idx):
        """Get potential neighbors using spatial hash."""
        x, y = points[query_idx]
        cx = int(np.floor(x / cell_size))
        cy = int(np.floor(y / cell_size))
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in cell_map:
                    neighbors.extend(cell_map[key])
        return neighbors

    def _compute_delaunay_edges(points):
        """Compute Delaunay triangulation edges."""
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            return list(edges)
        except Exception:
            return []

    def _spectral_regularizer(points, beta_spec=1.0):
        """
        Compute spectral regularizer: variance of Delaunay edge lengths.
        This encourages uniform spacing across the lattice.
        """
        edges = _compute_delaunay_edges(points)
        if len(edges) < 2:
            return 0.0, np.zeros_like(points)
        
        edge_lengths = np.array([np.linalg.norm(points[a] - points[b]) for a, b in edges])
        if len(edge_lengths) == 0:
            return 0.0, np.zeros_like(points)
        
        mean_len = np.mean(edge_lengths)
        var = np.var(edge_lengths)
        
        # Gradient of variance w.r.t. points
        # d(var)/d(points) = (2/N_edges) * sum_edges (L_e - mean) * (p_a - p_b) / L_e
        grad = np.zeros_like(points)
        for (a, b), L in zip(edges, edge_lengths):
            if L < 1e-10:
                continue
            coeff = (2.0 / len(edges)) * (L - mean_len) / L
            diff = points[a] - points[b]
            grad[a] += coeff * diff
            grad[b] -= coeff * diff
        
        return beta_spec * var, beta_spec * grad

    def _soft_min_optimize(x0, beta=1000, max_iter=500, use_spectral=False, beta_spec=0.0):
        """Optimize using soft-min formulation with optional spectral regularizer."""
        
        def get_points(x):
            return x.reshape(N, 2)

        def objective_and_grad(x):
            points = get_points(x)
            # Pairwise distances using cdist
            dists = cdist(points, points)
            iu = np.triu_indices(N, k=1)
            d_ij = dists[iu]
            # Boundary distances
            norms = np.linalg.norm(points, axis=1)
            d_b = 1.0 - norms
            v = np.concatenate([d_ij / 2.0, d_b])
            # Soft-min
            v_min = np.min(v)
            exp_v = np.exp(-beta * (v - v_min))
            S = np.sum(exp_v)
            weights = exp_v / S
            softmin_val = v_min - (1.0 / beta) * np.log(S)
            # Gradient
            w_pairs = weights[:len(d_ij)]
            w_bound = weights[len(d_ij):]
            w_mat = np.zeros((N, N))
            w_mat[iu] = w_pairs
            w_sym = w_mat + w_mat.T
            d_safe = np.where(dists < 1e-9, 1e-9, dists)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :]
            factor = w_sym / (2.0 * d_safe)
            np.fill_diagonal(factor, 0.0)
            grad_points = np.einsum('ij,ijk->ik', factor, diff)
            norms_safe = np.where(norms < 1e-9, 1e-9, norms)
            grad_bound_points = - (w_bound[:, np.newaxis] * points) / norms_safe[:, np.newaxis]
            total_grad = grad_points + grad_bound_points
            
            # Add spectral regularizer if enabled
            if use_spectral and beta_spec > 0:
                spec_val, spec_grad = _spectral_regularizer(points, beta_spec)
                total_grad = total_grad + spec_grad
                obj_val = -softmin_val + spec_val
            else:
                obj_val = -softmin_val
            
            grad_val = -total_grad
            return obj_val, grad_val

        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return x0

    def _polish(x0):
        """Final polish with high beta."""
        return _soft_min_optimize(x0, beta=50000, max_iter=300, use_spectral=True, beta_spec=0.5)

    def _random_perturb(x0, magnitude=0.01):
        """Random perturbation for restarts."""
        return x0 + np.random.randn(*x0.shape) * magnitude

    def _local_search(points, n_iter=200):
        """
        Local search: move points to increase minimum distance.
        Uses a greedy approach: for each point, try small moves in random directions.
        Uses spatial hashing for efficiency.
        """
        p = points.copy()
        current_score = _compute_radius_fast(p)
        
        # Estimate cell size based on average spacing
        avg_spacing = np.sqrt(np.pi / N)
        cell_size = avg_spacing * 1.5
        
        for i in range(n_iter):
            if is_finished():
                break
            
            # Pick a random point
            idx = np.random.randint(N)
            original = p[idx].copy()
            
            # Try small perturbations
            best_move = None
            best_local_score = current_score
            
            for _ in range(20):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                for step in [0.005, 0.01, 0.02, 0.05]:
                    p[idx] = original + step * direction
                    # Check if still inside unit circle
                    if np.linalg.norm(p[idx]) > 1.0:
                        p[idx] = original
                        continue
                    local_score = _compute_radius_fast(p)
                    if local_score > best_local_score:
                        best_local_score = local_score
                        best_move = p[idx].copy()
            
            if best_move is not None:
                p[idx] = best_move
                current_score = best_local_score
            else:
                p[idx] = original
        
        return p, current_score

    def _topological_swap(points, max_swaps=5):
        """
        Attempt topological swaps: identify pairs of points that can be swapped
        to improve local structure while preserving Delaunay connectivity.
        """
        p = points.copy()
        current_score = _compute_radius_fast(p)
        
        # Compute Delaunay edges
        edges = _compute_delaunay_edges(p)
        if len(edges) == 0:
            return p, current_score
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Try to find beneficial swaps
        for _ in range(max_swaps):
            if is_finished():
                break
            
            # Pick a random point
            idx = np.random.randint(N)
            original = p[idx].copy()
            
            # Try moving to position of a neighbor
            if idx in adj and len(adj[idx]) > 0:
                neighbors = adj[idx]
                for nb in neighbors:
                    if nb == idx:
                        continue
                    # Try swapping positions
                    p[idx], p[nb] = p[nb].copy(), p[idx].copy()
                    
                    # Check validity
                    valid = True
                    for j in range(N):
                        if np.linalg.norm(p[j]) > 1.0:
                            valid = False
                            break
                    
                    if valid:
                        new_score = _compute_radius_fast(p)
                        if new_score > current_score:
                            current_score = new_score
                        else:
                            # Swap back
                            p[idx], p[nb] = p[nb].copy(), p[idx].copy()
                    else:
                        # Swap back
                        p[idx], p[nb] = p[nb].copy(), p[idx].copy()
        
        return p, current_score

    # Main optimization loop
    best_x = None
    best_score = 0.0

    # Step 1: Hexagonal initialization
    hex_points = _init_hex()
    score = _compute_radius_fast(hex_points)
    if score > best_score:
        best_score = score
        best_x = hex_points.copy()

    if is_finished():
        return best_x

    # Step 2: Soft-min optimization with annealing
    betas = [10, 50, 200, 1000, 5000]
    current_x = best_x.flatten()

    for beta in betas:
        if is_finished() or remaining_time() < 15:
            break
        max_iter = min(800, int(remaining_time() * 4))
        if max_iter < 50:
            break
        new_x = _soft_min_optimize(current_x, beta=beta, max_iter=max_iter)
        score = _compute_radius_fast(new_x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = new_x.reshape(N, 2).copy()
            current_x = new_x
        elif is_finished():
            break

    if is_finished():
        return best_x

    # Step 3: Local search to fine-tune
    if remaining_time() > 20:
        local_result, local_score = _local_search(best_x, n_iter=300)
        if local_score > best_score:
            best_score = local_score
            best_x = local_result.copy()

    if is_finished():
        return best_x

    # Step 4: Topological swaps to escape local minima
    if remaining_time() > 25:
        swapped, swap_score = _topological_swap(best_x, max_swaps=10)
        if swap_score > best_score:
            best_score = swap_score
            best_x = swapped.copy()

    if is_finished():
        return best_x

    # Step 5: Random restarts with spectral regularization
    n_restarts = 4
    for i in range(n_restarts):
        if is_finished() or remaining_time() < 25:
            break
        perturbed = _random_perturb(best_x, magnitude=0.03)
        new_x = _soft_min_optimize(perturbed.flatten(), beta=1000, max_iter=300, use_spectral=True, beta_spec=0.3)
        score = _compute_radius_fast(new_x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = new_x.reshape(N, 2).copy()

    # Step 6: Final polish with spectral regularizer
    if remaining_time() > 15:
        polished = _polish(best_x.flatten())
        score = _compute_radius_fast(polished.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = polished.reshape(N, 2).copy()

    return best_x.reshape(N, 2)
