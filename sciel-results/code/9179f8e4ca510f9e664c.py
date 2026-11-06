import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    best_edges = 0
    best_graph = None
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_k33(neighbors):
        # Check if any triple has >= 3 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common_ij = neighbors[i] & neighbors[j]
                if popcount(common_ij) < 2:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & neighbors[k]
                    if popcount(common_ijk) >= 3:
                        return True, (i, j, k)
        return False, None
        
    def find_k33_edge(neighbors, i, j, k):
        # Find an edge to remove to break the K3,3 involving i, j, k
        # The K3,3 is between {i, j, k} and some 3 vertices in N(i) & N(j) & N(k)
        common = neighbors[i] & neighbors[j] & neighbors[k]
        # Pick 3 vertices from common
        verts = []
        for v in range(n):
            if common & (1 << v):
                verts.append(v)
                if len(verts) == 3:
                    break
        # The subgraph induced by {i, j, k} U verts is a K3,3
        # We can remove any edge in this subgraph.
        # Let's remove a random one.
        # Edges are between i,j,k and verts.
        # There are 9 edges.
        # e.g., (i, verts[0])
        return (i, verts[0])
        
    def can_add_edge(neighbors, u, v):
        # Check if adding (u, v) creates a K3,3
        # A new K3,3 must use (u, v).
        # So u is in one part, v in the other.
        # Part 1: {u, a, b}, Part 2: {v, c, d}
        # Conditions:
        # a, b in N(v) \ {u}
        # c, d in N(u) \ {v}
        # a~c, a~d, b~c, b~d
        # This is a C4 in the bipartite graph between N(v)\{u} and N(u)\{v}
        
        Cv = neighbors[v] & ~(1 << u)
        Cu = neighbors[u] & ~(1 << v)
        
        # Check for C4
        # Iterate over a in Cv
        a_mask = Cv
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            # Candidates for c: must be in Cu and neighbor of a
            # But we also need to ensure we can find distinct b, d
            # Let's just check if there exist a != b in Cv and c != d in Cu such that all 4 edges exist.
            # For fixed a, c:
            # If a~c:
            #   We need b in (Cv \ {a}) & N(c)
            #   And d in (Cu \ {c}) & N(a)
            #   And b~d
            
            c_mask = Cu & neighbors[a]
            while c_mask:
                c = (c_mask & -c_mask).bit_length() - 1
                c_mask &= c_mask - 1
                
                # We have a~c.
                # Need b in Cv, b != a, b~c
                # Need d in Cu, d != c, d~a
                # And b~d
                
                # Candidates for b:
                b_cands = (Cv & ~(1 << a)) & neighbors[c]
                # Candidates for d:
                d_cands = (Cu & ~(1 << c)) & neighbors[a]
                
                # Check if any b in b_cands is connected to any d in d_cands
                b_mask = b_cands
                while b_mask:
                    b = (b_mask & -b_mask).bit_length() - 1
                    b_mask &= b_mask - 1
                    if neighbors[b] & d_cands:
                        return False # C4 found, cannot add
        
        return True

    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        neighbors = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        # Repair phase
        changed = True
        while changed:
            if time.time() > deadline:
                break
            has, triple = has_k33(neighbors)
            if not has:
                break
            u, v = find_k33_edge(neighbors, *triple)
            neighbors[u] &= ~(1 << v)
            neighbors[v] &= ~(1 << u)
        
        # Optimization phase
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (neighbors[u] & (1 << v)):
                        if can_add_edge(neighbors, u, v):
                            neighbors[u] |= (1 << v)
                            neighbors[v] |= (1 << u)
                            improved = True
        
        # Count edges
        edges = sum(popcount(neighbors[i]) for i in range(n)) // 2
        
        if edges > best_edges:
            best_edges = edges
            best_graph = [ [i, j] for i in range(n) for j in range(i+1, n) if neighbors[i] & (1 << j) ]
            
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_graph
