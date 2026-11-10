import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Simulated Annealing to minimize the number of regular induced 
    subgraphs of size 6. Once zero are found, verify for all sizes >= 6.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute subsets for n=20 (max target)
    # We will run for n=20, 19, etc.
    # But precomputing for 20 covers 19 if we mask? No, vertices are 0..n-1.
    # We'll precompute for n=20 and use a subset of vertices for smaller n?
    # No, it's cleaner to precompute for each n.
    
    def get_subsets(n, size):
        """Yields (mask, vertices) for all subsets of given size."""
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            yield mask, comb

    def build_edge_subset_index(n, size):
        """
        Returns a list of lists.
        index[u][v] is a list of indices into the global subsets list 
        for subsets of 'size' that contain edge (u, v).
        """
        subsets = list(get_subsets(n, size))
        # Map subset index to mask and vertices
        # We need to know which subsets contain edge (u, v)
        # Initialize
        edge_idx = [[[] for _ in range(n)] for _ in range(n)]
        
        for i, (mask, verts) in enumerate(subsets):
            # Add this subset index to all pairs in verts
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    u = verts[a]
                    v = verts[b]
                    edge_idx[u][v].append(i)
                    edge_idx[v][u].append(i)
                    
        return subsets, edge_idx

    def is_regular_mask(adj, mask, verts):
        """Check if induced subgraph on verts is regular."""
        if not verts:
            return False
        first_deg = (adj[verts[0]] & mask).bit_count()
        for u in verts[1:]:
            if (adj[u] & mask).bit_count() != first_deg:
                return False
        return True

    def try_find_graph(n, time_budget):
        """
        Attempts to find a valid graph on n vertices using SA.
        Returns edges list or None.
        """
        if time.time() > time_budget:
            return None
            
        # Precompute for this n
        subsets6, edge_idx6 = build_edge_subset_index(n, 6)
        
        # We also need to check larger sizes for final verification
        # We'll generate them on the fly or precompute if n is small enough
        # For n=20, precomputing all subsets >= 6 is heavy but doable?
        # C(20,6)=38k, C(20,7)=77k, C(20,8)=125k... Total ~600k.
        # Storing 600k masks is fine.
        # But we only need them for verification.
        
        # SA State
        adj = [0] * n
        # Initialize random graph
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Initial cost: count of regular size-6 subgraphs
        is_reg6 = [False] * len(subsets6)
        cost = 0
        for i, (mask, verts) in enumerate(subsets6):
            if is_regular_mask(adj, mask, verts):
                is_reg6[i] = True
                cost += 1
        
        best_cost = cost
        best_adj = adj[:]
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        max_steps = 50000 # Limit steps to avoid infinite loop if stuck
        
        while time.time() < time_budget and T > min_T and steps < max_steps:
            steps += 1
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            if adj[u] & (1 << v):
                # Remove
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                delta_flip = -1
            else:
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                delta_flip = 1
            
            # Update cost
            old_cost = cost
            affected_indices = edge_idx6[u][v]
            
            for idx in affected_indices:
                mask, verts = subsets6[idx]
                new_reg = is_regular_mask(adj, mask, verts)
                old_reg = is_reg6[idx]
                
                if new_reg != old_reg:
                    is_reg6[idx] = new_reg
                    if new_reg:
                        cost += 1
                    else:
                        cost -= 1
            
            delta = cost - old_cost
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                if delta == 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / T)
            
            if not accept:
                # Revert
                if delta_flip == -1:
                    # Was removed, add back
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Was added, remove
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                
                # Revert cost changes
                for idx in affected_indices:
                    mask, verts = subsets6[idx]
                    new_reg = is_regular_mask(adj, mask, verts)
                    old_reg = is_reg6[idx]
                    
                    if new_reg != old_reg:
                        is_reg6[idx] = new_reg
                        if new_reg:
                            cost += 1
                        else:
                            cost -= 1
            
            if cost < best_cost:
                best_cost = cost
                best_adj = adj[:]
                # If we reach 0, we can stop SA phase 1
                if best_cost == 0:
                    break
            
            T *= cooling
        
        # Use best_adj found
        adj = best_adj
        
        # Final Verification: Check all subsets of size >= 6
        # Generate all subsets of size k for k=6..n
        # This is expensive, so we only do it if best_cost == 0
        if best_cost == 0:
            # Verify
            valid = True
            for k in range(6, n + 1):
                for mask, verts in get_subsets(n, k):
                    if is_regular_mask(adj, mask, verts):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return edges
        
        return None

    # Try n=20, then 19, etc.
    for n in range(20, 4, -1):
        edges = try_find_graph(n, deadline)
        if edges is not None:
            return {'n': n, 'edges': edges}
        
        # If we spent too much time on n, break
        if time.time() > deadline - 10:
            break
            
    # Fallback: Return a trivial graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
