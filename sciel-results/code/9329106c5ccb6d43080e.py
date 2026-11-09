import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
import math

def _extract_coords(dm):
    """
    Reconstruct approximate coordinates from the distance matrix.
    Uses MDS (Multidimensional Scaling) or a simple geometric approach.
    Since the problem states "50 uniform random points in the unit square",
    we can recover coordinates up to rotation/translation/reflection.
    """
    n = dm.shape[0]
    # Use Classical MDS
    # 1. Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dm ** 2) @ J
    
    # 2. Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # 3. Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 4. Take the first 2 components
    # Ensure positive eigenvalues for square root
    eigenvalues = np.maximum(eigenvalues, 0)
    coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    
    # 5. Normalize to unit square [0,1]x[0,1]
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    if range_x < 1e-10: range_x = 1.0
    if range_y < 1e-10: range_y = 1.0
    
    coords[:, 0] = (coords[:, 0] - min_x) / range_x
    coords[:, 1] = (coords[:, 1] - min_y) / range_y
    
    return coords

def _nearest_neighbor_tour(dm, start=0):
    """Generate a tour using the nearest neighbor heuristic."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dm[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total

def _two_opt(tour, dm, max_iter=500):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
        if not improved:
            break
    return tour, best_len

def _multi_start_two_opt(dm, n_starts=10, max_iter_per_start=100):
    """Run multiple 2-opt starts to find good reference tours."""
    n = dm.shape[0]
    best_tours = []
    
    for s in range(n_starts):
        start_node = s % n
        tour = _nearest_neighbor_tour(dm, start=start_node)
        tour, length = _two_opt(tour, dm, max_iter=max_iter_per_start)
        best_tours.append((tour, length))
        
    # Return the best few tours for consensus
    best_tours.sort(key=lambda x: x[1])
    return best_tours[:5]

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Strategy: 
    1. Reconstruct coordinates.
    2. Compute Voronoi-based topological heuristic (favor edges within/near Voronoi cells).
    3. Compute multi-start 2-opt consensus heuristic (favor edges common to good local tours).
    4. Combine them.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # 1. Base heuristic: inverse distance
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # 2. Reconstruct coordinates for spatial awareness
    try:
        coords = _extract_coords(dm)
    except Exception:
        coords = None
        
    heuristic = base_heuristic.copy()
    
    # 3. Voronoi Topological Heuristic
    if coords is not None:
        try:
            # Compute Voronoi diagram
            vor = Voronoi(coords)
            
            # For each point, find its Voronoi cell neighbors (points sharing an edge in the Voronoi diagram)
            # This is a strong signal for "good" edges in TSP on random points.
            # We will boost heuristic for these neighbor pairs.
            
            # Get Voronoi regions
            # Note: Voronoi regions are defined by half-planes. 
            # A simpler proxy for "Voronoi neighbors" is Delaunay neighbors.
            # Delaunay triangulation edges are a subset of "good" TSP edges.
            delaunay = Delaunay(coords)
            
            # Initialize a zero matrix for topological boost
            topo_boost = np.zeros((n, n))
            
            # Iterate over Delaunay simplices (triangles in 2D)
            for simplex in delaunay.simplices:
                i, j, k = simplex
                # Boost edges (i,j), (j,k), (k,i)
                topo_boost[i, j] += 1.0
                topo_boost[j, i] += 1.0
                topo_boost[j, k] += 1.0
                topo_boost[k, j] += 1.0
                topo_boost[k, i] += 1.0
                topo_boost[i, k] += 1.0
                
            # Also consider the convex hull edges which are definitely in Delaunay
            # Delaunay already includes them.
            
            # Normalize topo_boost
            max_topo = np.max(topo_boost)
            if max_topo > 0:
                topo_boost = topo_boost / max_topo
            
            # Combine: Add a weighted topological boost
            # Edges that are Delaunay neighbors get a significant boost
            heuristic += 5.0 * topo_boost * base_heuristic
            
            if time.time() > deadline:
                heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
                return heuristic
                
        except Exception:
            pass # Fallback to just base + consensus if Voronoi/Delaunay fails

    # 4. Multi-Start 2-Opt Consensus Heuristic
    try:
        # Run multi-start local search
        # Use time budget to decide how many starts
        # For n=50, 2-opt is fast. Let's do 10 starts.
        tours = _multi_start_two_opt(dm, n_starts=10, max_iter_per_start=100)
        
        # Create a consensus matrix
        consensus_boost = np.zeros((n, n))
        
        for tour, length in tours:
            # Boost edges in the tour
            # Weight by quality (1/length)
            weight = 1.0 / length
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                consensus_boost[u, v] += weight
                # Since TSP is symmetric, we can boost both directions or just one.
                # The ant chooses i->j. If the tour has i->j, boost i->j.
                # If the tour has j->i, boost j->i.
                # In a symmetric TSP, the set of edges is the same.
                # So boosting both directions is safe and often beneficial for convergence.
                consensus_boost[v, u] += weight
        
        # Normalize consensus boost
        max_consensus = np.max(consensus_boost)
        if max_consensus > 0:
            consensus_boost = consensus_boost / max_consensus
            
        # Combine: Add weighted consensus boost
        # This reinforces edges that appear in multiple good local solutions
        heuristic += 10.0 * consensus_boost * base_heuristic
        
    except Exception:
        pass
    
    # Final cleanup
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Optional: Normalize the whole heuristic to a reasonable range to avoid numerical issues
    # The ACO probability is proportional to pheromone * heuristic.
    # Pheromone starts at 1.
    # If heuristic is very large, it dominates. If very small, pheromone dominates.
    # Let's normalize so the max heuristic is around 10-100.
    max_h = np.max(heuristic)
    if max_h > 0:
        heuristic = heuristic / max_h * 50.0 # Target max value ~50
        
    return heuristic
