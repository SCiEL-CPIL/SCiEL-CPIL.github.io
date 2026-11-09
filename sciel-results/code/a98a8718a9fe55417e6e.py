import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    N = 512
    
    # 1. Build adjacency matrix using numpy for speed
    # adj[i,j] = True if lcm(i+1, j+1) <= 512
    # Use 0-indexed: node i represents value i+1
    values = np.arange(1, N + 1)
    # Compute lcm matrix
    # lcm(a,b) = a*b/gcd(a,b)
    # We need a*b/gcd(a,b) <= 512
    
    # Build adjacency as boolean matrix
    adj_mat = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            a, b = i + 1, j + 1
            g = gcd(a, b)
            if (a // g) * b <= N:
                adj_mat[i, j] = True
                adj_mat[j, i] = True
    
    # Convert to bitmasks for fast operations
    # adj_mask[i] is a Python int bitmask of neighbors of node i
    adj_mask = [0] * N
    for i in range(N):
        mask = 0
        for j in range(N):
            if adj_mat[i, j]:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # 2. Greedy initial solution
    # Sort by degree descending
    degrees = [adj_mask[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    best_clique = []
    candidates = (1 << N) - 1  # all bits set
    for u in order:
        if candidates & (1 << u):
            best_clique.append(u)
            candidates &= adj_mask[u]
    
    best_size = len(best_clique)
    
    # 3. Tomita B&B with coloring
    def get_color_sorted_list(cand_mask):
        """Greedy coloring for Tomita B&B. Returns (P, colors) where P is list of vertices
        and colors[i] is upper bound on clique size in subgraph induced by P[0..i]."""
        if cand_mask == 0:
            return [], []
        
        # Extract vertices
        verts = []
        m = cand_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Sort by degree in induced subgraph (descending)
        verts.sort(key=lambda v: (adj_mask[v] & cand_mask).bit_count(), reverse=True)
        
        P = []
        colors = []
        # used_colors[c] = mask of vertices assigned color c
        used_colors = []
        
        for v in verts:
            # Find smallest color c such that v is not adjacent to any vertex in used_colors[c]
            c = 0
            while c < len(used_colors):
                if (adj_mask[v] & used_colors[c]) == 0:
                    break
                c += 1
            
            if c == len(used_colors):
                used_colors.append(0)
            used_colors[c] |= (1 << v)
            
            P.append(v)
            colors.append(c + 1)
        
        return P, colors
    
    def expand(clique, cand_mask):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
        
        if cand_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Pruning: even if we add all candidates, can't beat best
        if len(clique) + cand_mask.bit_count() <= best_size:
            return
        
        P, colors = get_color_sorted_list(cand_mask)
        
        # Iterate in reverse order (highest index first)
        for i in range(len(P) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = P[i]
            # Bound: colors[i] is upper bound on clique size in P[0..i]
            if len(clique) + colors[i] <= best_size:
                # Can't beat best with this vertex or any earlier ones
                # Since we're going backwards and colors are non-decreasing in the Tomita sense,
                # we can break
                break
            
            # Branch: include v
            clique.append(v)
            
            # New candidates: vertices in P[0..i-1] that are adjacent to v
            new_cand = 0
            for j in range(i):
                u = P[j]
                if adj_mask[v] & (1 << u):
                    new_cand |= (1 << u)
            
            expand(clique, new_cand)
            clique.pop()
    
    # Start B&B
    expand([], (1 << N) - 1)
    
    # 4. Local improvement: try to add vertices or swap
    # Try adding any vertex that's compatible with all in best_clique
    clique_set = set(best_clique)
    for v in range(N):
        if v not in clique_set:
            compatible = True
            for u in clique_set:
                if not adj_mat[v, u]:
                    compatible = False
                    break
            if compatible:
                best_clique.append(v)
                best_size += 1
                clique_set.add(v)
    
    # Try swap: remove one, add two
    for i in range(len(best_clique)):
        if time.time() > deadline:
            break
        removed = best_clique[i]
        temp_clique = [x for j, x in enumerate(best_clique) if j != i]
        temp_set = set(temp_clique)
        
        added = 0
        for v in range(N):
            if v not in temp_set and v != removed:
                compatible = True
                for u in temp_set:
                    if not adj_mat[v, u]:
                        compatible = False
                        break
                if compatible:
                    best_clique.append(v)
                    temp_set.add(v)
                    added += 1
                    if added >= 2:
                        break
        
        if added > 1:
            best_size = len(best_clique)
    
    # Convert back to 1-indexed values
    result = sorted([v + 1 for v in best_clique])
    
    # Final verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) // gcd(a, b) > N:
                # Invalid - fallback to simple seed
                return list(range(1, 23))
    
    return result

# EVOLVE-BLOCK-END
