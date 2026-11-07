import time
import numpy as np
from itertools import product as iproduct

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge types: cycle edges (top, bottom) and matching edges
    # Top cycle: (i, i+1) for i in 0..m-1 (with wrap)
    # Bottom cycle: (m+i, m+i+1) for i in 0..m-1 (with wrap)
    # Matching: (i, m+i) for i in 0..m-1
    
    def get_edge_types():
        """Categorize edges into top cycle, bottom cycle, and matching"""
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
        
        return top_cycle, bottom_cycle, matching
    
    top_cycle, bottom_cycle, matching = get_edge_types()
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    
    def is_connected_lift(signs):
        """
        A 2-lift of a prism graph is connected iff the product of signs
        around any cycle that uses matching edges is -1.
        
        For prism graphs, we can check: the product of signs around the
        "ladder" cycles should be -1 for connectivity.
        
        Specifically, for each i, consider the cycle:
        i -> i+1 -> m+i+1 -> m+i -> i
        The product of signs on this cycle must be -1 for the lift to be connected.
        
        Actually, the lift is connected iff there exists at least one cycle
        with odd number of -1 signs (product = -1). For prism graphs, we can
        check the product around the "rungs" cycles.
        """
        # The lift is connected iff the product of signs around the outer cycle
        # (going around one full ladder) is -1, OR we can check individual cycles.
        
        # More precisely: the 2-lift is connected iff the signing is not "balanced"
        # on all cycles. For a prism, we can check: product of signs on top cycle *
        # product of signs on bottom cycle * product of signs on matching edges should
        # give us info.
        
        # Actually, the standard result: a 2-lift is connected iff there's a cycle
        # with odd number of -1 edges. For prism graphs, the key cycles are the
        # "ladder rungs": for each i, the 4-cycle i-(i+1)-(m+i+1)-(m+i)-i.
        
        # Let's check if any 4-cycle has product -1
        # 4-cycle for index i: edges (i, i+1), (i+1, m+i+1), (m+i+1, m+i), (m+i, i)
        # But we need to map these to edge indices.
        
        # Simpler: use the fact that for a prism, the lift is connected iff
        # the product of all top cycle signs * product of all bottom cycle signs *
        # (product of matching signs)^2 ... no, that's not right.
        
        # Let's just do BFS on the lift graph to check connectivity.
        # Build adjacency for the lift
        lift_adj = {v: set() for v in range(2*n)}
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # +1: join equal sides
                lift_adj[u].add(v)
                lift_adj[v].add(u)
                lift_adj[u + n].add(v + n)
                lift_adj[v + n].add(u + n)
            else:
                # -1: join opposite sides
                lift_adj[u].add(v + n)
                lift_adj[v + n].add(u)
                lift_adj[u + n].add(v)
                lift_adj[v].add(u + n)
        
        # BFS from node 0
        visited = set([0])
        queue = [0]
        while queue:
            curr = queue.pop(0)
            for neighbor in lift_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == 2 * n
    
    def is_valid(signs):
        if not is_connected_lift(signs):
            return False
        # Check 3-regular: each vertex in lift should have degree 3
        # This is automatically satisfied by the construction if the base is 3-regular
        # and the lift is well-defined, so we just need connectivity.
        return True
    
    # Strategy: Use the structure of the prism graph
    # The key insight: for the spectral radius, we want to minimize the max eigenvalue.
    # 
    # For a prism graph, the signed adjacency matrix has a specific structure.
    # Let's parameterize by:
    # - signs on matching edges (m of them)
    # - signs on top cycle (m of them)  
    # - signs on bottom cycle (m of them)
    #
    # Total: 3m signs. For m=4: 12 signs, m=5: 15, m=7: 21, m=9: 27.
    #
    # For small m (4, 5), we can do exhaustive search.
    # For larger m, we need smarter search.
    
    best_radius = float('inf')
    best_signs = None
    
    # For m <= 6, try exhaustive search over a subset
    if m <= 6:
        # Exhaustive search over all 2^(3m) sign assignments
        # For m=4: 2^12 = 4096, m=5: 2^15 = 32768, m=6: 2^18 = 262144
        num_signs = 3 * m
        
        for trial in range(1 << num_signs):
            if time.time() > deadline - 5:
                break
            
            # Convert trial to signs
            signs = np.array([(1 if (trial >> i) & 1 else -1) for i in range(num_signs)])
            
            if not is_valid(signs):
                continue
            
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # For larger m, use local search / simulated annealing
    else:
        # Start with random valid signing
        rng = np.random.default_rng(42)
        
        def random_valid_signing():
            while True:
                signs = rng.choice([-1, 1], size=num_edges)
                if is_valid(signs):
                    return signs
        
        best_signs = random_valid_signing()
        best_radius = spectral_radius(best_signs)
        
        # Simulated annealing
        current_signs = best_signs.copy()
        current_radius = best_radius
        T = 1.0
        cooling = 0.999
        min_T = 1e-6
        
        while time.time() < deadline - 5 and T > min_T:
            # Pick a random edge to flip
            idx = rng.integers(0, num_edges)
            new_signs = current_signs.copy()
            new_signs[idx] *= -1
            
            if not is_valid(new_signs):
                T *= cooling
                continue
            
            new_radius = spectral_radius(new_signs)
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_signs = new_signs
                current_radius = new_radius
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
            
            T *= cooling
    
    # Final polish: try flipping each edge individually
    if best_signs is not None and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(num_edges):
                if time.time() > deadline - 2:
                    break
                new_signs = best_signs.copy()
                new_signs[idx] *= -1
                if not is_valid(new_signs):
                    continue
                r = spectral_radius(new_signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = new_signs
                    improved = True
    
    if best_signs is None:
        # Fallback: return a valid signing
        best_signs = random_valid_signing()
    
    return best_signs
# EVOLVE-BLOCK-END
