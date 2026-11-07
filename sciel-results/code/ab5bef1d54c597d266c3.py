import time
import numpy as np

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Simulated Annealing to minimize the spectral radius of the signed adjacency matrix.
    This approach uses stochastic acceptance of worse solutions to escape local minima,
    which is a different optimization paradigm from the deterministic local search in the seed.
    """
    deadline = time.time() + 55.0  # Safety margin: 55s out of 90s limit
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute edge indices for fast matrix construction
    # We will build the matrix from scratch each time, but we can optimize 
    # by only updating the changed entries if we had a persistent matrix, 
    # but rebuilding a small NxN matrix is very fast in NumPy.
    
    def build_signed_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            val = float(signs[i])
            S[u, v] = val
            S[v, u] = val
        return S

    def get_spectral_radius(signs):
        S = build_signed_matrix(signs)
        try:
            eigvals = np.linalg.eigvalsh(S)
            return float(np.max(np.abs(eigvals)))
        except np.linalg.LinAlgError:
            return np.inf

    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Generate a set of diverse initializations
    # 1. All +1
    # 2. All -1
    # 3. Alternating on cycles (structured)
    # 4. Random
    
    initializations = []
    
    # Structured init: Alternating signs on the two cycles
    # Edges are: top cycle (0..m-1), bottom cycle (m..2m-1), matching (i, i+m)
    # We need to know which edge index corresponds to which type.
    # Based on the problem description: "two m-cycles... joined by matching"
    # The order in 'edges' list is not explicitly defined as grouped, 
    # but usually, it's generated systematically. 
    # To be safe, we assume the edges are just a list. 
    # However, for structured init, we can just use random or all-ones.
    # Let's add a few random ones.
    
    initializations.append(np.ones(num_edges, dtype=np.int8))
    initializations.append(-np.ones(num_edges, dtype=np.int8))
    
    for _ in range(20):
        initializations.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))
        
    def simulated_annealing(start_signs, start_temp=1.0, alpha=0.99, steps_per_temp=50):
        current_signs = start_signs.copy()
        current_radius = get_spectral_radius(current_signs)
        
        best_local_signs = current_signs.copy()
        best_local_radius = current_radius
        
        temp = start_temp
        step = 0
        
        while temp > 1e-4 and time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to flip
                idx = rng.integers(0, num_edges)
                old_sign = current_signs[idx]
                current_signs[idx] = -old_sign
                
                new_radius = get_spectral_radius(current_signs)
                delta = new_radius - current_radius
                
                # Acceptance criterion
                if delta < 0 or rng.random() < np.exp(-delta / temp):
                    current_radius = new_radius
                    if new_radius < best_local_radius:
                        best_local_radius = new_radius
                        best_local_signs = current_signs.copy()
                else:
                    # Revert
                    current_signs[idx] = old_sign
                    
            temp *= alpha
            step += 1
            
        return best_local_signs, best_local_radius

    # Run SA for each initialization
    for init in initializations:
        if time.time() > deadline:
            break
            
        # Adjust parameters based on time remaining? 
        # Keep it simple: fixed schedule per run, but limit total runs.
        
        res_signs, res_radius = simulated_annealing(init)
        
        if res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()
            
        # If we found a very good solution (close to theoretical min for signed graphs), 
        # we might stop early. The theoretical min for 3-regular signed graphs 
        # can be as low as 2 (if bipartite structure allows). 
        # For prism graphs, we expect values around 2.0 - 3.0.
        if best_radius < 2.05:
             break

    # If time remains, do more random restarts
    while time.time() < deadline - 2.0:
        random_init = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        res_signs, res_radius = simulated_annealing(random_init)
        if res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()
            
        if best_radius < 2.05:
            break

    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        
    return best_signs
