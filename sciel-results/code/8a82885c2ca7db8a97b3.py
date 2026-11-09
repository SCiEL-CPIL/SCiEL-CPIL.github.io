"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses Branch and Bound on the compatibility graph with bitmask adjacency
    and greedy coloring for tight upper bounds.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Build adjacency bitmasks: adj[i] has bit j set if lcm(i,j) <= N
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Compute degrees for ordering
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = adj[i].bit_count()
    
    # Sort nodes by degree descending
    nodes_sorted = sorted(nodes, key=lambda x: degrees[x], reverse=True)
    
    # Greedy initial solution
    best_clique = []
    candidates_mask = (1 << (N + 1)) - 2  # bits 1..N set
    for u in nodes_sorted:
        if candidates_mask & (1 << u):
            best_clique.append(u)
            candidates_mask &= adj[u]
    
    best_size = len(best_clique)
    
    # Greedy coloring of subgraph induced by a bitmask
    # Returns list of (vertex, color) sorted by color descending, then degree
    def color_sort(candidates_mask):
        if candidates_mask == 0:
            return []
        
        color_of = {}
        color_sets = []
        remaining = candidates_mask
        
        # Process vertices in order of decreasing degree in the candidate subgraph
        # First, get all vertices in candidates
        verts = []
        temp = candidates_mask
        while temp:
            v = temp & (-temp).bit_length() - 1
            if v >= 1 and v <= N and (candidates_mask >> v) & 1:
                verts.append(v)
            temp &= temp - 1
        
        # Sort by degree in candidate subgraph
        verts.sort(key=lambda x: (adj[x] & candidates_mask).bit_count(), reverse=True)
        
        for u in verts:
            used_colors = set()
            # Check neighbors of u that are in candidates and already colored
            neighbors = adj[u] & candidates_mask
            temp = neighbors
            while temp:
                v = temp & (-temp).bit_length() - 1
                if v >= 1 and v <= N and v in color_of:
                    used_colors.add(color_of[v])
                temp &= temp - 1
            
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[u] = c
            if c >= len(color_sets):
                color_sets.append([])
            color_sets[c].append(u)
        
        # Build result: vertices ordered by color descending
        result = []
        for c in range(len(color_sets) - 1, -1, -1):
            for v in color_sets[c]:
                result.append((v, c))
        
        return result
    
    # Branch and Bound
    def bnb(clique, candidates_mask):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
        
        # Upper bound: len(clique) + max possible clique in candidates
        # Use coloring for tighter bound
        colored = color_sort(candidates_mask)
        if not colored:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
        
        max_color = colored[0][1]
        if len(clique) + max_color + 1 <= best_size:
            return
        
        # Process vertices in color order (higher color first)
        # Group by color for efficient processing
        i = 0
        while i < len(colored):
            u, c = colored[i]
            
            # If even with this color we can't beat best, break
            if len(clique) + c + 1 <= best_size:
                break
            
            # Check if u is still in candidates (it should be)
            if not (candidates_mask & (1 << u)):
                i += 1
                continue
            
            # Branch: include u
            clique.append(u)
            new_candidates = (candidates_mask & ~(1 << u)) & adj[u]
            bnb(clique, new_candidates)
            clique.pop()
            
            # Remove u from candidates for subsequent branches
            candidates_mask &= ~(1 << u)
            
            i += 1
    
    # Start B&B
    bnb([], (1 << (N + 1)) - 2)
    
    # If we timed out, return the best we found
    return sorted(best_clique)

# EVOLVE-BLOCK-END
