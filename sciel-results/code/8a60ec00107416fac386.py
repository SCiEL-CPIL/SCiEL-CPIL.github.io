"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _build_bitmasks(n, adj):
    """Build bitmask representation of adjacency for fast clique checking."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks


def _has_k5_fast(n, adj, masks):
    """Check if graph has a clique of size 5 using bitmask operations."""
    for v in range(n):
        neigh_mask = masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 4:
            continue
        
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i+1, len(neigh)):
                nj = neigh[j]
                if not (masks[ni] & (1 << nj)):
                    continue
                for k in range(j+1, len(neigh)):
                    nk = neigh[k]
                    if not (masks[ni] & (1 << nk)) or not (masks[nj] & (1 << nk)):
                        continue
                    common = neigh_mask & masks[ni] & masks[nj] & masks[nk]
                    if common:
                        return True
    return False


def _has_ind5_fast(n, adj, masks):
    """Check if graph has an independent set of size 5."""
    all_mask = (1 << n) - 1
    comp_masks = [all_mask ^ masks[i] ^ (1 << i) for i in range(n)]
    
    for v in range(n):
        neigh_mask = comp_masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 4:
            continue
        
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i+1, len(neigh)):
                nj = neigh[j]
                if not (comp_masks[ni] & (1 << nj)):
                    continue
                for k in range(j+1, len(neigh)):
                    nk = neigh[k]
                    if not (comp_masks[ni] & (1 << nk)) or not (comp_masks[nj] & (1 << nk)):
                        continue
                    common = neigh_mask & comp_masks[ni] & comp_masks[nj] & comp_masks[nk]
                    if common:
                        return True
    return False


def _is_valid_fast(n, adj, masks):
    """Check if graph is (5,5)-free using fast bitmask methods."""
    if _has_k5_fast(n, adj, masks):
        return False
    if _has_ind5_fast(n, adj, masks):
        return False
    return True


def _try_add_vertex(current_n, current_adj, current_masks, deadline):
    """
    Try to add a single vertex with various connection patterns.
    Returns (success, new_n, new_adj, new_masks) if successful.
    """
    new_vertex = current_n
    
    # Strategy 1: Try specific densities with multiple random samples
    densities = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for density in densities:
        for _ in range(20):
            if time.time() > deadline:
                return False, current_n, current_adj, current_masks
            
            new_adj = [s.copy() for s in current_adj]
            new_adj.append(set())
            
            new_adj[new_vertex] = set()
            for i in range(current_n):
                if random.random() < density:
                    new_adj[new_vertex].add(i)
                    new_adj[i].add(new_vertex)
            
            new_masks = _build_bitmasks(current_n + 1, new_adj)
            if _is_valid_fast(current_n + 1, new_adj, new_masks):
                return True, current_n + 1, new_adj, new_masks
    
    # Strategy 2: Greedy - start with no edges, add edges one by one
    # while maintaining validity
    new_adj = [s.copy() for s in current_adj]
    new_adj.append(set())
    
    # Get vertices sorted by degree (prefer connecting to high-degree vertices)
    vertices_by_degree = sorted(range(current_n), key=lambda v: len(current_adj[v]), reverse=True)
    
    # Try adding edges in random order
    shuffled_vertices = vertices_by_degree.copy()
    random.shuffle(shuffled_vertices)
    
    for v in shuffled_vertices:
        if time.time() > deadline:
            break
        if random.random() < 0.5:
            new_adj[new_vertex].add(v)
            new_adj[v].add(new_vertex)
            new_masks = _build_bitmasks(current_n + 1, new_adj)
            if not _is_valid_fast(current_n + 1, new_adj, new_masks):
                new_adj[new_vertex].remove(v)
                new_adj[v].remove(new_vertex)
    
    if len(new_adj[new_vertex]) > 0:
        new_masks = _build_bitmasks(current_n + 1, new_adj)
        if _is_valid_fast(current_n + 1, new_adj, new_masks):
            return True, current_n + 1, new_adj, new_masks
    
    # Strategy 3: Try to mirror a random existing vertex's neighborhood
    for _ in range(25):
        if time.time() > deadline:
            break
        template = random.randint(0, current_n - 1)
        new_adj = [s.copy() for s in current_adj]
        new_adj.append(set())
        
        # Connect new vertex to neighbors of template (and possibly non-neighbors)
        for i in range(current_n):
            if i == template:
                continue
            if i in current_adj[template]:
                if random.random() < 0.7:
                    new_adj[new_vertex].add(i)
                    new_adj[i].add(new_vertex)
            else:
                if random.random() < 0.3:
                    new_adj[new_vertex].add(i)
                    new_adj[i].add(new_vertex)
        
        new_masks = _build_bitmasks(current_n + 1, new_adj)
        if _is_valid_fast(current_n + 1, new_adj, new_masks):
            return True, current_n + 1, new_adj, new_masks
    
    # Strategy 4: Try complement of a random vertex's neighborhood
    for _ in range(15):
        if time.time() > deadline:
            break
        template = random.randint(0, current_n - 1)
        new_adj = [s.copy() for s in current_adj]
        new_adj.append(set())
        
        # Connect new vertex to non-neighbors of template (and possibly neighbors)
        for i in range(current_n):
            if i == template:
                continue
            if i in current_adj[template]:
                if random.random() < 0.3:
                    new_adj[new_vertex].add(i)
                    new_adj[i].add(new_vertex)
            else:
                if random.random() < 0.7:
                    new_adj[new_vertex].add(i)
                    new_adj[i].add(new_vertex)
        
        new_masks = _build_bitmasks(current_n + 1, new_adj)
        if _is_valid_fast(current_n + 1, new_adj, new_masks):
            return True, current_n + 1, new_adj, new_masks
    
    return False, current_n, current_adj, current_masks


def _try_flip_edge(current_n, current_adj, current_masks, deadline):
    """
    Try flipping a random edge. Returns (success, new_adj, new_masks).
    """
    i = random.randint(0, current_n - 1)
    j = random.randint(0, current_n - 1)
    if i == j:
        return False, current_adj, current_masks
    
    new_adj = [s.copy() for s in current_adj]
    
    if j in new_adj[i]:
        new_adj[i].remove(j)
        new_adj[j].remove(i)
    else:
        new_adj[i].add(j)
        new_adj[j].add(i)
    
    new_masks = _build_bitmasks(current_n, new_adj)
    if _is_valid_fast(current_n, new_adj, new_masks):
        return True, new_adj, new_masks
    
    return False, current_adj, current_masks


def _try_expand(n, adj, max_n, deadline):
    """
    Try to expand the graph by adding vertices and flipping edges.
    Uses multiple restarts with different random seeds.
    """
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_n = n
        current_adj = [s.copy() for s in adj]
        current_masks = _build_bitmasks(n, adj)
        
        add_streak = 0
        flip_streak = 0
        max_streak = 20
        
        while current_n < max_n:
            if time.time() > deadline:
                break
            
            # Decide whether to try adding or flipping
            if add_streak < max_streak and (flip_streak > 5 or random.random() < 0.7):
                # Try to add a vertex
                success, new_n, new_adj, new_masks = _try_add_vertex(
                    current_n, current_adj, current_masks, deadline
                )
                if success:
                    current_n = new_n
                    current_adj = new_adj
                    current_masks = new_masks
                    if current_n > best_n:
                        best_n = current_n
                        best_adj = [s.copy() for s in current_adj]
                    add_streak += 1
                    flip_streak = 0
                else:
                    add_streak += 1
                    if add_streak > max_streak:
                        add_streak = 0
                        flip_streak = 0
            else:
                # Try edge flipping
                success, new_adj, new_masks = _try_flip_edge(
                    current_n, current_adj, current_masks, deadline
                )
                if success:
                    current_adj = new_adj
                    current_masks = new_masks
                    flip_streak += 1
                    add_streak = 0
                else:
                    flip_streak += 1
                    if flip_streak > max_streak:
                        flip_streak = 0
                        add_streak = 0
            
            # Occasionally reset streaks to explore
            if (add_streak + flip_streak) % 30 == 0:
                add_streak = 0
                flip_streak = 0
    
    return best_n, best_adj


def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s limit
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_37_edges()
    adj = _build_adjacency(n, edges)
    
    # Verify it's valid
    masks = _build_bitmasks(n, adj)
    if not _is_valid_fast(n, adj, masks):
        # Should not happen, but just in case
        return n, edges
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to expand
    expanded_n, expanded_adj = _try_expand(n, adj, 46, deadline)
    if expanded_n > best_n:
        best_n = expanded_n
        best_adj = expanded_adj
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if j in best_adj[i]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
