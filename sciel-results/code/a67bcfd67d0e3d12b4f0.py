import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    def verify(n, edges):
        """
        Verify the graph is C4-free and has min degree >= n-39.
        Returns True if valid.
        """
        if n < 1 or n > 47:
            return False
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Check min degree
        degrees = adj.sum(axis=1)
        if np.min(degrees) < target_deg:
            return False
            
        # Check C4-free
        # A C4 exists if any two vertices have >= 2 common neighbors
        # We can check this via matrix multiplication or pair counting.
        # For n=47, n^3 is small enough for a few checks, but let's be efficient.
        # A2 = adj @ adj. A2[i,j] is count of common neighbors.
        # If any A2[i,j] >= 2 for i != j, there is a C4 (or a triangle with a chord? No, C4 is 4-cycle).
        # Wait, A2[i,j] >= 2 means i and j share 2 neighbors. 
        # If i and j are adjacent, this forms a triangle plus a chord? No.
        # i - k1 - j - k2 - i is a C4.
        # If i and j are NOT adjacent, i-k1-j-k2-i is a C4.
        # In both cases, the existence of 2 common neighbors creates a C4.
        # Proof: Let N(i) and N(j) be neighbors. If |N(i) \cap N(j)| >= 2, let k1, k2 be distinct common neighbors.
        # The cycle i-k1-j-k2-i is a simple cycle of length 4.
        # So checking for pairs with >= 2 common neighbors is sufficient and necessary for C4.
        
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
            
        return True

    def simulated_annealing(n, target_deg, time_budget):
        """
        Simulated Annealing to find a C4-free graph with min degree >= target_deg.
        State: adjacency matrix (upper triangle bits).
        Move: Flip a random edge.
        Energy: Number of C4s (pairs with >=2 common neighbors) * large_weight 
                + Number of vertices with degree < target_deg * degree_weight.
        We want to minimize Energy.
        """
        if time.time() - start_time > deadline - 5:
            return None
            
        # Initialize with a random graph of moderate density
        # We want to start somewhat close to the solution.
        # Expected degree in G(n,p) is (n-1)p.
        # Target degree is small (e.g., 7 for n=46).
        # But C4-free graphs are sparse.
        # Let's start with p such that expected degree is slightly above target.
        p = max(0.05, target_deg / (n - 1) * 1.5)
        
        adj = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    
        # Precompute neighbor lists for fast common neighbor counting?
        # Or just use matrix operations for energy calculation?
        # n=47, matrix mult is 47^3 ~ 100k ops. Very fast.
        
        def get_energy(adj):
            A = adj.astype(np.int32)
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            
            # Count C4s: sum of max(0, A2[i,j] - 1) for all i<j?
            # Actually, each C4 is counted by its 4 pairs of opposite vertices?
            # No, a C4 (v1,v2,v3,v4) has pairs (v1,v3) and (v2,v4) sharing 2 neighbors.
            # So each C4 contributes 2 to the sum of (A2[i,j] - 1) over i<j?
            # Let's just count the number of pairs with A2 >= 2.
            c4_count = np.sum(A2 >= 2)
            
            # Degree deficit
            degrees = A.sum(axis=1)
            deficit = np.sum(np.maximum(0, target_deg - degrees))
            
            # Weighting: C4s are "hard" constraints, deficits are "soft" but we need them to be 0.
            # If we have any C4, it's invalid.
            # Let's use a large penalty for C4s.
            energy = c4_count * 1000 + deficit
            return energy, c4_count, deficit

        current_energy, cur_c4, cur_def = get_energy(adj)
        best_energy = current_energy
        best_adj = adj.copy()
        
        T = 100.0
        cooling_rate = 0.995
        min_T = 0.01
        
        # Track time
        last_time = time.time()
        
        while T > min_T:
            if time.time() - start_time > deadline - 5:
                break
                
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            # Flip edge (i,j)
            old_val = adj[i, j]
            adj[i, j] = 1 - old_val
            adj[j, i] = adj[i, j]
            
            new_energy, new_c4, new_def = get_energy(adj)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj.copy()
            else:
                # Revert
                adj[i, j] = old_val
                adj[j, i] = old_val
                
            T *= cooling_rate
            
            # Optional: Restart if stuck for too long? 
            # Simple cooling usually works well enough for small n.
            
        # Final check on best found
        if verify(n, [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]):
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]
            return {'n': n, 'edges': edges}
            
        return None

    def greedy_with_backtracking(n, target_deg, time_budget):
        """
        A more structured approach: 
        1. Start with empty graph.
        2. Greedily add edges that don't create C4s, prioritizing low-degree vertices.
        3. If stuck, backtrack or restart.
        This is similar to the current program but we'll try a different ordering:
        Process vertices in a fixed random order, but for each vertex, try to connect 
        to as many other vertices as possible without creating C4s.
        """
        if time.time() - start_time > deadline - 5:
            return None
            
        # Try multiple random orderings
        for attempt in range(100):
            if time.time() - start_time > deadline - 5:
                break
                
            adj = np.zeros((n, n), dtype=np.int32)
            degrees = np.zeros(n, dtype=int)
            # common[u][v] is not efficient to maintain as a dict of sets for n=47? 
            # Let's just use the matrix multiplication check for each candidate edge.
            # n=47, checking one edge: O(n) to find common neighbors.
            # Total edges ~ 1000. 1000 * 47 = 47k ops. Very fast.
            
            # Shuffle vertices
            vertices = list(range(n))
            random.shuffle(vertices)
            
            # We want to maximize min degree.
            # Strategy: For each vertex u in vertices, try to add edges to all v 
            # such that adding (u,v) doesn't create a C4.
            # To help min degree, we should probably process vertices that currently have low degree first?
            # But if we process in random order, we might create a graph where early vertices get high degree
            # and later vertices are constrained.
            
            # Better: Iterate multiple passes.
            # Pass 1: Add all "safe" edges.
            # Pass 2: Check degrees. If min deg < target, try to add more edges to deficient vertices.
            
            # Let's just do a single pass but with a specific heuristic:
            # Sort vertices by index (randomized).
            # For each u, try to connect to v > u.
            
            # Actually, let's use the "greedy independent set" style logic on the complement?
            # No, let's stick to direct C4 avoidance.
            
            # Heuristic: 
            # 1. Start with a random 7-regular graph (if possible) or similar.
            # 2. Check for C4s.
            # 3. If C4s exist, remove edges to break them.
            # 4. If degrees drop, try to add edges back.
            
            # Let's try constructing a random regular graph first.
            # n=46, target=7. 7-regular graph on 46 vertices exists (46*7 is even).
            # Generate random 7-regular graph.
            
            if n % 2 == 0 and target_deg % 2 == 0:
                 # Can use configuration model or simple random regular graph generator
                 pass
                 
            # Simpler: Start with empty, add edges greedily.
            # To ensure high min degree, we should ensure every vertex gets connected to 'target_deg' others.
            
            # Let's try a specific construction:
            # Partition vertices into groups? No, C4-free is global.
            
            # Let's use the SA result if available, otherwise this greedy.
            # Since SA is already implemented above, this function is a fallback.
            # Let's make this greedy slightly different: 
            # "Conflict-directed" backtracking is too complex.
            # Let's just do:
            # 1. Random graph G(n, p) with p chosen such that E[deg] = target_deg.
            # 2. While C4s exist: remove random edge from a C4.
            # 3. While min_deg < target: add random edge that doesn't create C4.
            
            p = target_deg / (n - 1)
            adj = np.zeros((n, n), dtype=np.int32)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        
            # Phase 1: Remove C4s
            for _ in range(1000):
                if time.time() - start_time > deadline - 5:
                    break
                A = adj.astype(np.int32)
                A2 = A @ A
                np.fill_diagonal(A2, 0)
                if not np.any(A2 >= 2):
                    break
                # Find a C4
                pairs = np.argwhere(A2 >= 2)
                if len(pairs) == 0:
                    break
                u, v = pairs[0]
                # Find common neighbors
                common = np.where(adj[u] & adj[v])[0]
                if len(common) >= 2:
                    # Pick two common neighbors
                    c1, c2 = random.sample(common.tolist(), 2)
                    # Remove one of the edges: (u,c1), (u,c2), (v,c1), (v,c2)
                    # Prefer removing an edge incident to the vertex with higher degree?
                    # Or just random.
                    ru, rv = random.choice([(u, c1), (u, c2), (v, c1), (v, c2)])
                    adj[ru, rv] = 0
                    adj[rv, ru] = 0
                    
            # Phase 2: Boost degrees
            for _ in range(2000):
                if time.time() - start_time > deadline - 5:
                    break
                degrees = adj.sum(axis=1)
                if np.min(degrees) >= target_deg:
                    break
                # Find deficient vertex
                min_deg = np.min(degrees)
                deficient = np.where(degrees == min_deg)[0]
                u = random.choice(deficient)
                
                # Try to add an edge
                added = False
                # Candidates: v not adjacent to u
                candidates = [v for v in range(n) if v != u and not adj[u, v]]
                random.shuffle(candidates)
                
                for v in candidates:
                    # Check if adding (u,v) creates C4
                    # Common neighbors of u and v
                    common = np.sum(adj[u] & adj[v])
                    if common < 2:
                        # Safe
                        adj[u, v] = 1
                        adj[v, u] = 1
                        added = True
                        break
                        
                if not added:
                    # Stuck? Try a different deficient vertex or break
                    continue
            
            if verify(n, [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]):
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]
                return {'n': n, 'edges': edges}
                
        return None

    # Main logic
    # Try n=47, then 46, etc.
    # SA is powerful, so give it the most time for the highest n.
    
    # Allocation: 
    # n=47: 60s
    # n=46: 40s
    # n=45: 20s
    # n=44: 10s
    # n=43: 5s
    # Rest: 0s (fallback)
    
    time_alloc = {
        47: 60,
        46: 40,
        45: 20,
        44: 10,
        43: 5,
        42: 2,
        41: 1,
        40: 1,
        39: 1
    }
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() - start_time > deadline - 2:
            break
            
        target_deg = max(0, n - 39)
        
        # Method 1: Simulated Annealing
        res = simulated_annealing(n, target_deg, time_alloc.get(n, 5))
        if res:
            return res
            
        # Method 2: Greedy with Repair
        res = greedy_with_backtracking(n, target_deg, 5)
        if res:
            return res
            
    # Fallback: n=39, empty graph
    return {'n': 39, 'edges': []}
