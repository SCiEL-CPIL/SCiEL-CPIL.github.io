"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 83.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for fast matrix construction
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Categorize edges by type for structured initialization
    top_cycle = []
    bottom_cycle = []
    matching = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle.append(idx)
        elif u >= m and v >= m:
            bottom_cycle.append(idx)
        else:
            matching.append(idx)
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Check if the 2-lift is connected using BFS
        # Build adjacency for the lift graph with 2n vertices
        lift_adj = [[] for _ in range(2 * n)]
        for i in range(num_edges):
            u, v = edges[i]
            s = signs[i]
            for l in [0, 1]:
                l2 = l if s == 1 else 1 - l
                lift_adj[u * 2 + l].append(v * 2 + l2)
                lift_adj[v * 2 + l2].append(u * 2 + l)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return all(visited)
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    def make_valid(signs, rng):
        """Try to make signs valid by flipping edges until connected"""
        signs = signs.copy()
        if is_valid(signs):
            return signs
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            flip_idx = rng.integers(num_edges)
            signs[flip_idx] *= -1
            if is_valid(signs):
                return signs
        return None
    
    def structured_init(rng):
        """Create structured initial sign assignments"""
        strategy = rng.integers(4)
        signs = np.ones(num_edges, dtype=int)
        
        if strategy == 0:
            # Alternating on top cycle
            for i, idx in enumerate(top_cycle):
                signs[idx] = 1 if i % 2 == 0 else -1
            # Alternating on bottom cycle
            for i, idx in enumerate(bottom_cycle):
                signs[idx] = 1 if i % 2 == 0 else -1
            # Random matching
            for idx in matching:
                signs[idx] = rng.choice([-1, 1])
        elif strategy == 1:
            # All +1 on cycles, alternating matching
            for i, idx in enumerate(matching):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif strategy == 2:
            # Random with bias toward -1 on matching
            signs = rng.choice([-1, 1], size=num_edges)
            for idx in matching:
                signs[idx] = -1 if rng.random() < 0.6 else 1
        else:
            # Fully random
            signs = rng.choice([-1, 1], size=num_edges)
        
        return signs
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase 1: Exhaustive search for small m (m <= 5)
    if m <= 5:
        num_signs = 3 * m
        for trial in range(1 << num_signs):
            if time.time() > deadline - 5:
                break
            
            signs = np.array([(1 if (trial >> i) & 1 else -1) for i in range(num_signs)])
            
            if not is_valid(signs):
                continue
            
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # Phase 2: Simulated annealing with multiple restarts and structured inits
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Use structured initialization
        signs = structured_init(rng)
        
        # Ensure valid start
        if not is_valid(signs):
            valid_signs = make_valid(signs, rng)
            if valid_signs is None:
                continue
            signs = valid_signs
        
        cur_radius = spectral_radius(signs)
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = signs.copy()
        
        # Simulated annealing with adaptive parameters
        T_start = 0.05
        T_end = 0.0005
        num_steps = 3000
        step = 0
        
        for step in range(num_steps):
            if time.time() > deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (step / num_steps)
            
            # Pick a random edge to flip
            idx = rng.integers(num_edges)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            if is_valid(signs):
                new_radius = spectral_radius(signs)
                delta = new_radius - cur_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_radius = new_radius
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
            else:
                signs[idx] = old_sign
    
    # Phase 3: Local search polish - try flipping each edge individually
    if best_signs is not None and time.time() < deadline - 3:
        current_signs = best_signs.copy()
        current_radius = best_radius
        improved = True
        
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(num_edges):
                if time.time() > deadline - 3:
                    break
                
                old_sign = current_signs[idx]
                current_signs[idx] = -old_sign
                
                if is_valid(current_signs):
                    new_radius = spectral_radius(current_signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = current_signs.copy()
                    else:
                        current_signs[idx] = old_sign
                else:
                    current_signs[idx] = old_sign
    
    # Phase 4: Final high-precision polish with smaller steps
    if best_signs is not None and time.time() < deadline - 1:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Try pairs of flips
        for i in range(num_edges):
            if time.time() > deadline - 1:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline - 1:
                    break
                
                old_si = current_signs[i]
                old_sj = current_signs[j]
                current_signs[i] = -old_si
                current_signs[j] = -old_sj
                
                if is_valid(current_signs):
                    new_radius = spectral_radius(current_signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = current_signs.copy()
                    else:
                        current_signs[i] = old_si
                        current_signs[j] = old_sj
                else:
                    current_signs[i] = old_si
                    current_signs[j] = old_sj
    
    # If no valid solution found, return all +1
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(2000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
