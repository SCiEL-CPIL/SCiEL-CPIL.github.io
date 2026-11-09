import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # Precompute C5 lookup table for 5-vertex graphs
    # There are 10 possible edges in K5.
    # Edge indices: (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3, (1,2)=4, (1,3)=5, (1,4)=6, (2,3)=7, (2,4)=8, (3,4)=9
    c5_table = [False] * 1024
    
    def get_edge_mask(v1, v2, v3, v4, v5):
        # Map vertices 0..4 to edge indices
        # 0-1: 0, 0-2: 1, 0-3: 2, 0-4: 3, 1-2: 4, 1-3: 5, 1-4: 6, 2-3: 7, 2-4: 8, 3-4: 9
        mask = 0
        if True: mask |= (1 << 0) # 0-1
        if True: mask |= (1 << 1) # 0-2
        # This is static for the local 5 vertices, but we need to map global to local.
        # Actually, we just need to check if the induced subgraph on 5 specific vertices contains a C5.
        # We can precompute which of the 1024 masks contain a C5.
        pass

    # Precompute C5 masks
    for mask in range(1024):
        # Build adjacency for 5 vertices
        adj = [[False]*5 for _ in range(5)]
        edges = [
            (0,1,0), (0,2,1), (0,3,2), (0,4,3),
            (1,2,4), (1,3,5), (1,4,6),
            (2,3,7), (2,4,8), (3,4,9)
        ]
        for u, v, idx in edges:
            if mask & (1 << idx):
                adj[u][v] = True
                adj[v][u] = True
        
        # Check for C5
        found = False
        # Check all permutations of 0,1,2,3,4
        # To avoid duplicates, fix 0 at start, check 4! = 24 perms
        for p in range(24):
            # Generate permutation
            # This is slow in Python. Let's hardcode or use a faster check.
            # A graph on 5 vertices has a C5 if and only if it contains the cycle 0-1-2-3-4-0 (or any rotation/reflection).
            # There are 12 distinct cycles.
            # We can just check if the subgraph is isomorphic to C5 or contains C5.
            # Simpler: Check if there exists a Hamiltonian cycle.
            # For 5 vertices, we can just check the 12 cycles explicitly.
            cycles = [
                (0,1,2,3,4),
                (0,1,2,4,3),
                (0,1,3,2,4),
                (0,1,3,4,2),
                (0,1,4,2,3),
                (0,1,4,3,2),
                (0,2,1,3,4),
                (0,2,1,4,3),
                (0,2,3,1,4),
                (0,2,3,4,1),
                (0,2,4,1,3),
                (0,2,4,3,1)
            ]
            for c in cycles:
                if (adj[c[0]][c[1]] and adj[c[1]][c[2]] and adj[c[2]][c[3]] and adj[c[3]][c[4]] and adj[c[4]][c[0]]):
                    found = True
                    break
            if found:
                break
        c5_table[mask] = found

    def is_c5_free(adj_masks, n):
        """
        adj_masks: list of integers, adj_masks[i] is bitmask of neighbors of i.
        """
        for combo in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = combo
            # Extract mask for these 5 vertices
            # Edges: 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4
            mask = 0
            if adj_masks[v0] & (1 << v1): mask |= 1 << 0
            if adj_masks[v0] & (1 << v2): mask |= 1 << 1
            if adj_masks[v0] & (1 << v3): mask |= 1 << 2
            if adj_masks[v0] & (1 << v4): mask |= 1 << 3
            if adj_masks[v1] & (1 << v2): mask |= 1 << 4
            if adj_masks[v1] & (1 << v3): mask |= 1 << 5
            if adj_masks[v1] & (1 << v4): mask |= 1 << 6
            if adj_masks[v2] & (1 << v3): mask |= 1 << 7
            if adj_masks[v2] & (1 << v4): mask |= 1 << 8
            if adj_masks[v3] & (1 << v4): mask |= 1 << 9
            
            if c5_table[mask]:
                return False
        return True

    def has_independent_set_9(adj_masks, n):
        """
        Returns True if there is an independent set of size 9.
        Uses backtracking.
        """
        # Heuristic: Order vertices by degree (descending) to prune faster?
        # Or just use indices 0..n-1.
        # We need to find a set of 9 vertices with no edges between them.
        
        # Precompute non-neighbor masks
        # non_adj[i] = bitmask of vertices NOT adjacent to i (and i != j)
        non_adj = [0] * n
        for i in range(n):
            # All bits set except i and neighbors
            all_bits = (1 << n) - 1
            non_adj[i] = all_bits & ~(adj_masks[i] | (1 << i))
        
        def backtrack(start, current_set, size):
            if size == 9:
                return True
            # Pruning: if not enough vertices left
            if n - start < 9 - size:
                return False
            
            # Try adding vertex 'start'
            # If we add 'start', the next candidate must be in non_adj[start]
            # and > start
            
            # Option 1: Include 'start'
            # Find next vertex > start that is non-adjacent to all in current_set
            # Actually, we can just iterate.
            
            # To speed up, we can maintain a candidate set.
            # But for n=35, simple backtracking might be ok if we prune well.
            
            # Let's try a simpler approach:
            # Iterate through vertices from 'start' to n-1
            for i in range(start, n):
                # Check if i is non-adjacent to all in current_set
                ok = True
                for v in current_set:
                    if adj_masks[v] & (1 << i):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Include i
                if backtrack(i + 1, current_set + [i], size + 1):
                    return True
                
                # Exclude i (implicit in loop)
                
            return False
        
        return backtrack(0, [], 0)

    def verify(adj_masks, n):
        if not is_c5_free(adj_masks, n):
            return False
        if has_independent_set_9(adj_masks, n):
            return False
        return True

    def to_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Search for n=35, 34, 33
    for n in [35, 34, 33]:
        if time.time() - start_time > deadline - 20:
            break
            
        # Initialize
        # Start with 8 K4s + extra vertices connected densely
        adj_masks = [0] * n
        
        # Add 8 K4s (32 vertices)
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    adj_masks[base + i] |= (1 << (base + j))
                    adj_masks[base + j] |= (1 << (base + i))
        
        # Connect extra vertices (32, 33, ...) to many others
        for v in range(32, n):
            # Connect to all in first 7 K4s (28 vertices)
            for u in range(28):
                adj_masks[v] |= (1 << u)
                adj_masks[u] |= (1 << v)
            # Connect to 3 random in 8th K4
            block8 = [32, 33, 34, 35] if n > 35 else [32, 33, 34, 35] # Wait, 8th K4 is 28..31
            # Block 7 is 28,29,30,31
            for u in random.sample([28, 29, 30, 31], 3):
                adj_masks[v] |= (1 << u)
                adj_masks[u] |= (1 << v)
        
        # Simulated Annealing
        best_adj = adj_masks[:]
        best_score = 999999
        
        # Score function
        def score(adj):
            # Count C5s
            c5_count = 0
            for combo in combinations(range(n), 5):
                v0, v1, v2, v3, v4 = combo
                mask = 0
                if adj[v0] & (1 << v1): mask |= 1 << 0
                if adj[v0] & (1 << v2): mask |= 1 << 1
                if adj[v0] & (1 << v3): mask |= 1 << 2
                if adj[v0] & (1 << v4): mask |= 1 << 3
                if adj[v1] & (1 << v2): mask |= 1 << 4
                if adj[v1] & (1 << v3): mask |= 1 << 5
                if adj[v1] & (1 << v4): mask |= 1 << 6
                if adj[v2] & (1 << v3): mask |= 1 << 7
                if adj[v2] & (1 << v4): mask |= 1 << 8
                if adj[v3] & (1 << v4): mask |= 1 << 9
                if c5_table[mask]:
                    c5_count += 1
            
            # Check alpha
            alpha_penalty = 0
            if has_independent_set_9(adj, n):
                alpha_penalty = 10000
            
            return c5_count + alpha_penalty
        
        current_adj = adj_masks[:]
        current_score = score(current_adj)
        
        T = 100.0
        T_min = 1.0
        cooling = 0.9995
        
        steps = 0
        max_steps = 50000
        while T > T_min and steps < max_steps:
            if time.time() - start_time > deadline - 5:
                break
            
            # Random move: flip a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            new_score = score(current_adj)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
                    if best_score == 0:
                        break
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            
            T *= cooling
            steps += 1
        
        if best_score == 0:
            if verify(best_adj, n):
                return {'n': n, 'edges': to_edges(best_adj, n)}

    # Fallback: n=32
    n = 32
    adj_masks = [0] * n
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj_masks[base + i] |= (1 << (base + j))
                adj_masks[base + j] |= (1 << (base + i))
    
    return {'n': n, 'edges': to_edges(adj_masks, n)}

