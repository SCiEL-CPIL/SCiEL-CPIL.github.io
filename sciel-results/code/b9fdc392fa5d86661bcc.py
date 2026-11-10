import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin
    
    # Fast C6 check using bitmasks for adjacency
    # adj_bits[i] is a bitmask of neighbors of vertex i
    def can_add(adj_bits, u, v):
        # Find all paths of length 4 from u to some node d, where d is adjacent to v
        # Path: u - a - b - c - d, then check if d-v edge exists
        # We need simple paths, so track visited with bitmask
        
        vu = 1 << u
        vv = 1 << v
        
        # Level 1: neighbors of u (excluding v)
        level1 = adj_bits[u] & ~vv & ~vu
        
        # For each node a in level1, find neighbors b (excluding u, v)
        # Then for each b, find neighbors c (excluding u, v, a, b)
        # Then for each c, find neighbors d (excluding u, v, a, b, c)
        # Check if d is in adj_bits[v]
        
        # Use iterative approach with pruning
        # Represent frontier as list of (node, path_mask)
        frontier = []
        for a in range(N):
            if (level1 >> a) & 1:
                frontier.append((a, vu | (1 << a)))
        
        # Level 2: extend to b
        frontier2 = []
        for a, mask_a in frontier:
            # neighbors of a, excluding u, v, a
            nbs_a = adj_bits[a] & ~vu & ~vv & ~(1 << a)
            for b in range(N):
                if (nbs_a >> b) & 1:
                    frontier2.append((b, mask_a | (1 << b)))
        
        # Level 3: extend to c
        frontier3 = []
        for b, mask_b in frontier2:
            nbs_b = adj_bits[b] & ~vu & ~vv & ~(1 << b)
            for c in range(N):
                if (nbs_b >> c) & 1:
                    frontier3.append((c, mask_b | (1 << c)))
        
        # Level 4: extend to d, check if d is adjacent to v
        adj_v = adj_bits[v]
        for c, mask_c in frontier3:
            nbs_c = adj_bits[c] & ~vu & ~vv & ~(1 << c)
            # Check if any neighbor d of c (not in path, not u, not v) is adjacent to v
            candidates = nbs_c & adj_v & ~mask_c
            if candidates:
                return False  # C6 found
        
        return True
    
    # Compute spectral gap heuristic score for an edge (u, v)
    # Higher score = more likely to be C6-free
    def spectral_score(adj_matrix, u, v):
        # Quick approximation: use degree-based proxy
        # Vertices with lower common neighborhood are less likely to create C6
        du = adj_matrix[u].sum()
        dv = adj_matrix[v].sum()
        common = (adj_matrix[u] & adj_matrix[v]).sum()
        # Lower common neighborhood = safer to add
        # Also prefer edges between lower-degree vertices
        return -common - 0.1 * (du + dv)
    
    # Compute spectral gap of current graph (largest - second largest eigenvalue)
    def spectral_gap(adj_matrix):
        if adj_matrix.sum() == 0:
            return 0.0
        try:
            eigvals = np.linalg.eigvalsh(adj_matrix)
            # eigvalsh returns in ascending order
            return eigvals[-1] - eigvals[-2]
        except:
            return 0.0

    # Initial Graph: 8 disjoint K5s
    adj_list = [[] for _ in range(N)]
    adj_set = [set() for _ in range(N)]
    adj_bits = [0] * N
    adj_matrix = np.zeros((N, N), dtype=np.int8)
    edges = []
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1
                edges.append([u, v])
                
    best_edges = len(edges)
    best_adj_list = [lst[:] for lst in adj_list]
    best_adj_set = [set(s) for s in adj_set]
    best_adj_bits = adj_bits[:]
    best_adj_matrix = adj_matrix.copy()
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local Search with spectral guidance
    while time.time() < deadline:
        # Create list of missing edges
        missing = []
        for u, v in all_pairs:
            if v not in adj_set[u]:
                missing.append((u, v))
                
        if not missing:
            break
        
        # Score missing edges using spectral heuristic
        # Use a quick proxy: common neighbors + degree
        scored_missing = []
        for u, v in missing:
            common = (adj_bits[u] & adj_bits[v]).bit_count()
            du = adj_bits[u].bit_count()
            dv = adj_bits[v].bit_count()
            # Lower common and lower degrees = safer
            score = -common - 0.05 * (du + dv)
            scored_missing.append((score, u, v))
        
        # Sort by score (highest first = safest to add)
        scored_missing.sort(key=lambda x: x[0], reverse=True)
        
        # Take top candidates, add some randomness
        n_candidates = min(len(scored_missing), 50)
        top_candidates = [item[1:] for item in scored_missing[:n_candidates]]
        random.shuffle(top_candidates)
        
        added_in_pass = 0
        for u, v in top_candidates:
            if time.time() > deadline:
                break
            if v in adj_set[u]:
                continue
            if can_add(adj_bits, u, v):
                # Add edge
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1
                edges.append([u, v])
                added_in_pass += 1
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_adj_list = [lst[:] for lst in adj_list]
                    best_adj_set = [set(s) for s in adj_set]
                    best_adj_bits = adj_bits[:]
                    best_adj_matrix = adj_matrix.copy()
        
        if added_in_pass == 0:
            # Stuck. Perform a kick: remove a few random edges
            if len(edges) > 0:
                # Remove 1 to 3 random edges
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    # Remove from structures
                    if v in adj_set[u]:
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        adj_matrix[u, v] = 0
                        adj_matrix[v, u] = 0
                    edges.pop(idx)
                    # Update best if current is better (shouldn't be, but safe)
                    if len(edges) > best_edges:
                        best_edges = len(edges)
                        best_adj_list = [lst[:] for lst in adj_list]
                        best_adj_set = [set(s) for s in adj_set]
                        best_adj_bits = adj_bits[:]
                        best_adj_matrix = adj_matrix.copy()

    # Return the best graph found
    final_edges = []
    for u in range(N):
        for v in best_adj_list[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
