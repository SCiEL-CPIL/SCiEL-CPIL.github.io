import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Exhaustive search for small m (m <= 5) to guarantee global optimum.
    2. Multi-Start Local Search (Hill Climbing) for larger m.
       - Diverse initializations (structured + random).
       - Strict hill climbing: flip edge if it improves spectral radius and maintains connectivity.
       - Perturbation (Shake): Randomly flip a few edges to escape local minima, then hill climb again.
    3. Efficient connectivity check using BFS on the signed graph (coboundary check).
    """
    
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints for fast matrix construction
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Precompute adjacency list for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        Equivalent to the signing NOT being a coboundary.
        """
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        # Contradiction found: No coboundary exists.
                        # Therefore, the lift IS connected.
                        return True
        
        # No contradiction found: A coboundary exists.
        # Therefore, the lift is DISCONNECTED.
        return False

    def get_radius(signs):
        """
        Computes the spectral radius of the signed adjacency matrix.
        """
        S = np.zeros((n, n), dtype=np.float64)
        # Vectorized assignment
        S[u_arr, v_arr] = signs.astype(float)
        S[v_arr, u_arr] = signs.astype(float)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    best_signs = None
    best_radius = np.inf
    
    # --- Strategy 1: Exhaustive Search for small m ---
    # m=4: 12 edges -> 4096 combos
    # m=5: 15 edges -> 32768 combos
    # m=6: 18 edges -> 262144 combos (Might be slow, check time)
    
    if num_edges <= 15:
        max_combo = 1 << num_edges
        # Use a simple loop. For 32k iterations, this is very fast.
        for k in range(max_combo):
            if time.time() > deadline - 2.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            
            if not is_lift_connected(signs):
                continue
                
            radius = get_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        if best_signs is not None:
            return best_signs
            
    # If m=6 (18 edges), exhaustive is 262k. 
    # 262k * (connectivity check + eigvalsh) might take a few seconds.
    # Let's try it if we have plenty of time.
    elif num_edges <= 18 and time.time() < deadline - 10.0:
        max_combo = 1 << num_edges
        # We can't easily vectorize the connectivity check for all 2^18 at once.
        # But we can iterate.
        # To speed up, we can skip connectivity check for many? No, it's fast.
        # Let's just run it.
        for k in range(max_combo):
            if time.time() > deadline - 2.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            
            if not is_lift_connected(signs):
                continue
                
            radius = get_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                
        if best_signs is not None:
            return best_signs

    # --- Strategy 2: Multi-Start Local Search for larger m ---
    
    def local_search(start_signs):
        """
        Performs hill climbing. Returns best signs and radius found.
        """
        current_signs = start_signs.copy()
        
        # Ensure start is connected. If not, try to fix or return inf.
        if not is_lift_connected(current_signs):
            # Try to fix by flipping random edges until connected
            for _ in range(100):
                idx = np.random.randint(num_edges)
                current_signs[idx] *= -1
                if is_lift_connected(current_signs):
                    break
            if not is_lift_connected(current_signs):
                return None, np.inf

        current_radius = get_radius(current_signs)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 5.0:
                    break
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                if not is_lift_connected(current_signs):
                    current_signs[i] = old_sign
                    continue
                
                new_radius = get_radius(current_signs)
                
                if new_radius < current_radius - 1e-9:
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old_sign
                    
        return current_signs, current_radius

    # Generate diverse initializations
    rng = np.random.default_rng(42)
    initializations = []
    
    # 1. All +1
    init_all_plus = np.ones(num_edges, dtype=np.int8)
    initializations.append(init_all_plus)
    
    # 2. All -1
    init_all_minus = -np.ones(num_edges, dtype=np.int8)
    initializations.append(init_all_minus)
    
    # 3. Structured: Alternating on cycles
    # Prism graph structure:
    # Edges 0..m-1: Cycle 1
    # Edges m..2m-1: Cycle 2
    # Edges 2m..3m-1: Matching
    
    # Pattern A: + on Cycle 1, - on Cycle 2, + on Matching
    init_struct1 = np.ones(num_edges, dtype=np.int8)
    init_struct1[m:2*m] = -1
    initializations.append(init_struct1)
    
    # Pattern B: - on Cycle 1, + on Cycle 2, - on Matching
    init_struct2 = -np.ones(num_edges, dtype=np.int8)
    init_struct2[m:2*m] = 1
    initializations.append(init_struct2)
    
    # Pattern C: Checkerboard on base graph (if bipartite, but prism is not bipartite for odd m)
    # Just random is good too.
    
    # 4. Random initializations
    for _ in range(20):
        init_rand = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        initializations.append(init_rand)

    # Run local search from each initialization
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
            
        res_signs, res_radius = local_search(init)
        
        if res_signs is not None and res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()
            
    # If we still have time, do a "Shake and Local Search" to escape local minima
    if best_signs is not None and time.time() < deadline - 5.0:
        # Perform several rounds of perturbation
        max_shakes = 50
        for _ in range(max_shakes):
            if time.time() > deadline - 5.0:
                break
            
            # Start from current best
            perturbed = best_signs.copy()
            
            # Randomly flip 1-3 edges
            num_flips = rng.integers(1, 4)
            for _ in range(num_flips):
                idx = rng.integers(0, num_edges)
                perturbed[idx] *= -1
            
            # Run local search from perturbed state
            res_signs, res_radius = local_search(perturbed)
            
            if res_signs is not None and res_radius < best_radius:
                best_radius = res_radius
                best_signs = res_signs.copy()

    # Fallback
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            # Try to find any valid connected signing
            for _ in range(1000):
                idx = np.random.randint(num_edges)
                best_signs[idx] *= -1
                if is_lift_connected(best_signs):
                    break

    return best_signs
