import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Exhaustive search for small m (m <= 6) to guarantee global optimum.
    2. Simulated Annealing for larger m to escape local minima.
       - Uses a geometric cooling schedule.
       - Rejects moves that violate connectivity.
       - Accepts worse moves with probability exp(-delta / T).
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
    # m=6: 18 edges -> 262144 combos
    
    if num_edges <= 18:
        max_combo = 1 << num_edges
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

    # --- Strategy 2: Simulated Annealing for larger m ---
    
    rng = np.random.default_rng(3)
    
    # Generate diverse initializations
    # 1. Random
    # 2. Structured: Alternating signs on cycles to induce cancellation
    initializations = []
    
    # Random init
    init_rand = rng.choice([-1, 1], size=num_edges).astype(np.int8)
    # Ensure connectivity for start
    while not is_lift_connected(init_rand):
        idx = rng.integers(0, num_edges)
        init_rand[idx] *= -1
        if rng.integers(0, 100) == 0: # Safety reset
             init_rand = rng.choice([-1, 1], size=num_edges).astype(np.int8)
    initializations.append(init_rand)
    
    # Structured init: 
    # Prism graph edges:
    # 0..m-1: Cycle 1
    # m..2m-1: Cycle 2
    # 2m..3m-1: Matching
    # Try alternating signs on the cycles
    init_struct = np.ones(num_edges, dtype=np.int8)
    # Cycle 1: alternating
    for i in range(m):
        if i % 2 == 0:
            init_struct[i] = -1
    # Cycle 2: alternating
    for i in range(m):
        if i % 2 == 0:
            init_struct[m+i] = -1
    # Matching: all +1
    # Check connectivity
    if is_lift_connected(init_struct):
        initializations.append(init_struct)
    
    # Another structured: All -1 on matching, +1 on cycles
    init_struct2 = np.ones(num_edges, dtype=np.int8)
    init_struct2[2*m:] = -1
    if is_lift_connected(init_struct2):
        initializations.append(init_struct2)

    def simulated_annealing(start_signs):
        """
        Performs Simulated Annealing. Returns best signs and radius found.
        """
        current_signs = start_signs.copy()
        current_radius = get_radius(current_signs)
        
        best_signs_local = current_signs.copy()
        best_radius_local = current_radius
        
        # SA Parameters
        T0 = 1.0
        T_end = 0.001
        cooling_rate = 0.999
        # Number of steps per temperature level
        # We have ~60s. Let's estimate steps.
        # Each step: flip, check conn (O(N+E)), check radius (O(N^3)).
        # For N=18, eigvalsh is very fast (~10-20us). Conn check ~5us.
        # ~25us per step. 60s / 25us = 2.4M steps.
        # Let's do 100 steps per temp level.
        steps_per_temp = 100
        
        T = T0
        while T > T_end and time.time() < deadline - 5.0:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5.0:
                    break
                    
                # Pick random edge to flip
                idx = rng.integers(0, num_edges)
                old_sign = current_signs[idx]
                current_signs[idx] = -old_sign
                
                # Check connectivity
                if not is_lift_connected(current_signs):
                    # Reject move
                    current_signs[idx] = old_sign
                else:
                    new_radius = get_radius(current_signs)
                    delta = new_radius - current_radius
                    
                    # Accept if better, or with probability exp(-delta/T)
                    if delta < 0:
                        # Better
                        current_signs[idx] = -old_sign # Already flipped, so this restores? No.
                        # Wait, current_signs[idx] is already -old_sign.
                        # We want to keep it.
                        current_radius = new_radius
                        if new_radius < best_radius_local:
                            best_radius_local = new_radius
                            best_signs_local = current_signs.copy()
                    else:
                        # Worse
                        prob = np.exp(-delta / T)
                        if rng.random() < prob:
                            # Accept
                            current_radius = new_radius
                            # Keep the flip
                        else:
                            # Reject
                            current_signs[idx] = old_sign
            
            T *= cooling_rate
        
        return best_signs_local, best_radius_local

    # Run SA from each initialization
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
            
        res_signs, res_radius = simulated_annealing(init)
        
        if res_signs is not None and res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()

    # If we still have time, run SA again from the best found so far
    if best_signs is not None and time.time() < deadline - 5.0:
        res_signs, res_radius = simulated_annealing(best_signs)
        if res_signs is not None and res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()

    # Fallback
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            for _ in range(1000):
                idx = np.random.randint(num_edges)
                best_signs[idx] *= -1
                if is_lift_connected(best_signs):
                    break

    return best_signs
