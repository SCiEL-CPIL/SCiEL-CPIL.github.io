import time
import numpy as np
import sys

def sign_edges(graph, budget):
    """
    Optimizes edge signs on a prism graph to minimize the spectral radius of the signed adjacency matrix.
    Uses Simulated Annealing with vectorized spectral evaluation.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # 1. Precompute graph structure for fast spectral radius calculation
    # The signed adjacency matrix S is symmetric.
    # We can represent the sign assignment as a vector s of length num_edges.
    # S is determined by s.
    
    # To speed up eigvalsh, we note that S is sparse but small (n=8..18).
    # However, constructing S from scratch every iteration is slow.
    # We can precompute the indices where S is non-zero.
    
    # Let's build the base unsigned adjacency structure.
    # S[u, v] = sign(edge_index) if (u,v) is an edge.
    
    # We will maintain the sign vector `s` (values +/-1).
    # We need to compute max(|eigvalsh(S)|).
    
    # Precompute edge indices for the matrix
    # Since the graph is undirected, S[u,v] = S[v,u].
    # We need to map edge indices to (u, v) pairs.
    
    edge_u = np.array([e[0] for e in edges])
    edge_v = np.array([e[1] for e in edges])
    
    # We will construct S in each iteration.
    # Optimization: Use a flat array for S and reshape? Or just build the matrix.
    # n is small (max 18), so n^2 is 324. Building an 18x18 matrix is fast.
    # eigvalsh on 18x18 is also very fast (~microseconds).
    # The bottleneck is Python overhead.
    
    # Let's try a simpler approach first: 
    # Represent signs as a vector of 0/1 for internal computation, map to -1/+1.
    # Or just keep -1/+1.
    
    # Initialize with random signs
    rng = np.random.default_rng(42)
    s = rng.choice([-1, 1], size=num_edges)
    
    def get_spectral_radius(s):
        """
        Compute the spectral radius (max abs eigenvalue) of the signed adjacency matrix.
        Returns (radius, is_valid).
        Validity: The lift must be connected and 3-regular.
        The problem states: "The lift must be connected and 3-regular, or the whole result is invalid."
        The base graph is 3-regular (prism graph). The lift is always 3-regular if the base is.
        Connectivity of the lift depends on the signs.
        A 2-lift of a connected graph is connected iff the signs do not define a bipartition of the base graph
        such that all edges cross the bipartition in a specific way? 
        Actually, for a 2-lift, the lift is disconnected if and only if there exists a function f: V -> {0,1}
        such that for every edge (u,v) with sign s_e, f(v) = f(u) if s_e=1 and f(v) != f(u) if s_e=-1.
        This is equivalent to the system of equations x_v = s_e * x_u having a solution in {1, -1}.
        If such a solution exists, the lift splits into two components.
        If no such solution exists (i.e., the product of signs around some cycle is -1? No, that's for Z2 gauge field),
        let's check the condition.
        
        The lift graph has vertices (v, 0), (v, 1).
        Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
        Edge (u,v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
        
        The lift is disconnected iff there is a labeling sigma: V -> {0,1} such that
        for every edge (u,v) with sign s_e:
        if s_e = 1: sigma(u) == sigma(v)
        if s_e = -1: sigma(u) != sigma(v)
        
        This is equivalent to checking if the signed graph is "balanced" in the sense of Z2.
        We can check this by BFS/DFS on the base graph.
        """
        # Build S
        S = np.zeros((n, n))
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            val = s[i]
            S[u, v] = val
            S[v, u] = val
        
        # Check validity (connectivity of lift)
        # Try to assign sigma
        sigma = np.full(n, -1, dtype=int)
        sigma[0] = 0
        queue = [0]
        valid = True
        while queue:
            u = queue.pop(0)
            for i in range(num_edges):
                if edge_u[i] == u:
                    v = edge_v[i]
                    s_e = s[i]
                    if s_e == 1:
                        expected = sigma[u]
                    else:
                        expected = 1 - sigma[u]
                    if sigma[v] == -1:
                        sigma[v] = expected
                        queue.append(v)
                    else:
                        if sigma[v] != expected:
                            valid = False
                            break
            if not valid:
                break
        
        if not valid:
            # If invalid, return a large penalty
            return 100.0, False
            
        # Compute spectral radius
        # S is symmetric
        eigvals = np.linalg.eigvalsh(S)
        radius = np.max(np.abs(eigvals))
        return radius, True

    # Initial evaluation
    best_s = s.copy()
    best_radius, valid = get_spectral_radius(best_s)
    if not valid:
        # If initial is invalid, try to fix or just keep searching
        # Simulated annealing will handle it via penalties
        pass
    current_radius = best_radius
    current_s = s.copy()
    
    # Simulated Annealing
    # T0 = 1.0, cooling rate 0.99, etc.
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    # Precompute neighbors for faster update? 
    # Flipping sign of edge i changes S[i_u, i_v] and S[i_v, i_u].
    # We can update eigenvalues incrementally? No, eigvalsh is cheap enough for n=18.
    
    # Optimization: Vectorize the validity check?
    # The validity check is a BFS. It's O(N+E).
    # For n=18, E=3*18/2 = 27. Very fast.
    
    # Let's optimize the loop.
    # Instead of calling a function, inline the logic if possible?
    # Or just rely on the small size.
    
    while time.time() < deadline:
        # Pick a random edge to flip
        i = rng.integers(0, num_edges)
        current_s[i] *= -1
        
        new_radius, new_valid = get_spectral_radius(current_s)
        
        if not new_valid:
            new_radius = 100.0 + new_radius # Penalty
            
        delta = new_radius - current_radius
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_s = current_s.copy()
        else:
            current_s[i] *= -1 # Revert
        
        T *= cooling
        if T < min_T:
            T = min_T
            # Optional: restart with lower T or break if converged
            # Let's just continue with min_T for a bit, then maybe restart?
            # For now, just keep going.
            
    return best_s.astype(int)

