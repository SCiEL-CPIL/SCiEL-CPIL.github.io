"""
Improved program for Ramsey R(5,5) graph construction.
Uses bitmask-based validity checking and a more aggressive local search
to expand beyond the Paley 37-vertex seed graph.
"""
import time
import random
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
            for j in range(i+1, len(neigh)):
                for k in range(j+1, len(neigh)):
                    if not (masks[neigh[i]] & (1 << neigh[j]) and 
                            masks[neigh[i]] & (1 << neigh[k]) and
                            masks[neigh[j]] & (1 << neigh[k])):
                        continue
                    common = neigh_mask & masks[neigh[i]] & masks[neigh[j]] & masks[neigh[k]]
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
            for j in range(i+1, len(neigh)):
                for k in range(j+1, len(neigh)):
                    if not (comp_masks[neigh[i]] & (1 << neigh[j]) and 
                            comp_masks[neigh[i]] & (1 << neigh[k]) and
                            comp_masks[neigh[j]] & (1 << neigh[k])):
                        continue
                    common = neigh_mask & comp_masks[neigh[i]] & comp_masks[neigh[j]] & comp_masks[neigh[k]]
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

def _try_expand(n, adj, max_n, deadline):
    """
    Improved local search to expand the graph.
    Uses a combination of adding vertices and flipping edges with better heuristics.
    """
    start_time = time.time()
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    current_n = n
    current_adj = [s.copy() for s in adj]
    current_masks = _build_bitmasks(n, adj)
    
    no_improve_count = 0
    max_no_improve = 100
    
    while current_n < max_n and time.time() < deadline:
        # 1. Try to add a new vertex with multiple strategies
        new_vertex = current_n
        found = False
        
        # Strategy 1: Random connection patterns
        for attempt in range(80):
            if time.time() > deadline:
                break
            
            candidate_adj = [s.copy() for s in current_adj]
            candidate_adj.append(set())
            
            # Vary density based on attempt
            base_density = 0.5
            density = base_density + 0.2 * (attempt % 5 - 2) / 2
            
            for i in range(current_n):
                if random.random() < density:
                    candidate_adj[new_vertex].add(i)
                    candidate_adj[i].add(new_vertex)
            
            candidate_masks = _build_bitmasks(current_n + 1, candidate_adj)
            if _is_valid_fast(current_n + 1, candidate_adj, candidate_masks):
                current_n += 1
                current_adj = candidate_adj
                current_masks = candidate_masks
                best_n = current_n
                best_adj = [s.copy() for s in current_adj]
                no_improve_count = 0
                found = True
                break
        
        # Strategy 2: Greedy connection - try each possible subset of neighbors
        if not found and current_n < max_n and time.time() < deadline:
            new_vertex = current_n
            # Try connecting to random small sets
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                candidate_adj = [s.copy() for s in current_adj]
                candidate_adj.append(set())
                
                # Pick 3-6 random neighbors
                num_neighbors = random.randint(3, 6)
                neighbors = random.sample(range(current_n), min(num_neighbors, current_n))
                for i in neighbors:
                    candidate_adj[new_vertex].add(i)
                    candidate_adj[i].add(new_vertex)
                
                candidate_masks = _build_bitmasks(current_n + 1, candidate_adj)
                if _is_valid_fast(current_n + 1, candidate_adj, candidate_masks):
                    current_n += 1
                    current_adj = candidate_adj
                    current_masks = candidate_masks
                    best_n = current_n
                    best_adj = [s.copy() for s in current_adj]
                    no_improve_count = 0
                    found = True
                    break
        
        if not found:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
            # 2. Try edge flipping (local search)
            # Flip a random edge
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j:
                continue
            
            old_state = j in current_adj[i]
            if old_state:
                current_adj[i].remove(j)
                current_adj[j].remove(i)
            else:
                current_adj[i].add(j)
                current_adj[j].add(i)
            
            new_masks = _build_bitmasks(current_n, current_adj)
            if _is_valid_fast(current_n, current_adj, new_masks):
                current_masks = new_masks
                no_improve_count = 0
            else:
                # Revert
                if old_state:
                    current_adj[i].add(j)
                    current_adj[j].add(i)
                else:
                    current_adj[i].remove(j)
                    current_adj[j].remove(i)
    
    return best_n, best_adj

def _try_add_vertices_aggressive(n, adj, max_n, deadline):
    """
    More aggressive vertex addition using iterative deepening.
    Tries to add vertices one at a time with increasing connectivity constraints.
    """
    current_n = n
    current_adj = [s.copy() for s in adj]
    current_masks = _build_bitmasks(n, adj)
    
    while current_n < max_n and time.time() < deadline:
        new_vertex = current_n
        found = False
        
        # Try different numbers of edges to the new vertex
        for num_edges in range(0, min(current_n + 1, 12)):
            if time.time() > deadline:
                break
            
            # Try multiple random subsets of size num_edges
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                if num_edges == 0:
                    neighbor_set = set()
                elif num_edges >= current_n:
                    neighbor_set = set(range(current_n))
                else:
                    neighbor_set = set(random.sample(range(current_n), num_edges))
                
                candidate_adj = [s.copy() for s in current_adj]
                candidate_adj.append(set())
                
                for i in neighbor_set:
                    candidate_adj[new_vertex].add(i)
                    candidate_adj[i].add(new_vertex)
                
                candidate_masks = _build_bitmasks(current_n + 1, candidate_adj)
                if _is_valid_fast(current_n + 1, candidate_adj, candidate_masks):
                    current_n += 1
                    current_adj = candidate_adj
                    current_masks = candidate_masks
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            # Try edge flips to unlock vertex addition
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j:
                    continue
                
                old_state = j in current_adj[i]
                if old_state:
                    current_adj[i].remove(j)
                    current_adj[j].remove(i)
                else:
                    current_adj[i].add(j)
                    current_adj[j].add(i)
                
                new_masks = _build_bitmasks(current_n, current_adj)
                if _is_valid_fast(current_n, current_adj, new_masks):
                    current_masks = new_masks
    
    return current_n, current_adj

def construct():
    """
    Construct a (5,5)-free graph with as many vertices as possible.
    1. Start with Paley 37.
    2. Use aggressive local search to expand.
    """
    start_time = time.time()
    deadline = start_time + 295  # Safety margin
    
    n, edges = _paley_37_edges()
    adj = _build_adjacency(n, edges)
    
    # Verify it's valid
    masks = _build_bitmasks(n, adj)
    if not _is_valid_fast(n, adj, masks):
        return n, edges
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try multiple expansion strategies
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        
        # Restart with some randomization
        current_adj = [s.copy() for s in best_adj]
        # Randomly flip some edges for diversity
        for _ in range(10):
            i = random.randint(0, best_n - 1)
            j = random.randint(0, best_n - 1)
            if i == j:
                continue
            if j in current_adj[i]:
                current_adj[i].remove(j)
                current_adj[j].remove(i)
            else:
                current_adj[i].add(j)
                current_adj[j].add(i)
        
        current_masks = _build_bitmasks(best_n, current_adj)
        if not _is_valid_fast(best_n, current_adj, current_masks):
            continue
        
        expanded_n, expanded_adj = _try_expand(best_n, current_adj, 46, deadline)
        if expanded_n > best_n:
            best_n = expanded_n
            best_adj = expanded_adj
    
    # Final attempt with aggressive strategy
    if best_n < 46 and time.time() < deadline - 15:
        aggressive_n, aggressive_adj = _try_add_vertices_aggressive(best_n, best_adj, 46, deadline)
        if aggressive_n > best_n:
            best_n = aggressive_n
            best_adj = aggressive_adj
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if j in best_adj[i]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
