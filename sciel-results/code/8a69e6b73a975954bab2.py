import time
import random
import sys

def construct():
    """
    Constructs a maximum C4-free edge set in Q7 using randomized greedy with restarts.
    """
    d = 7
    n_vertices = 1 << d
    total_edges = n_vertices * d // 2
    
    # Precompute all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Map edge tuple to index
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Precompute for each edge index, the list of "conflict triples"
    # An edge e is blocked if any of its conflict triples are fully in S.
    # Conflict triple for e: For each square containing e, the other 3 edges.
    # We store these as lists of edge indices.
    conflict_triples = [[] for _ in range(total_edges)]
    
    for idx, (u, v) in enumerate(edges):
        # Find the dimension i of this edge
        diff = u ^ v
        # diff is a power of 2
        i = diff.bit_length() - 1
        
        # Find all squares containing this edge
        # A square is defined by the edge (u,v) and another dimension j != i
        # The other two edges are (u ^ (1<<j), v ^ (1<<j)) and (u, u ^ (1<<j)) and (v, v ^ (1<<j))?
        # No.
        # Square vertices: u, v, u^k, v^k where k = 1<<j.
        # Edges: (u,v) [idx], (u^k, v^k), (u, u^k), (v, v^k).
        # The other 3 edges are:
        # e1 = (u ^ k, v ^ k)
        # e2 = (u, u ^ k)
        # e3 = (v, v ^ k)
        
        for j in range(d):
            if j == i:
                continue
            k = 1 << j
            u2 = u ^ k
            v2 = v ^ k
            
            # Get indices of the other 3 edges
            # They must exist in the edge list
            e1 = (min(u2, v2), max(u2, v2))
            e2 = (min(u, u2), max(u, u2))
            e3 = (min(v, v2), max(v, v2))
            
            idx1 = edge_to_idx[e1]
            idx2 = edge_to_idx[e2]
            idx3 = edge_to_idx[e3]
            
            conflict_triples[idx].append((idx1, idx2, idx3))

    best_count = 0
    best_set = set()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    while time.time() < deadline:
        random.shuffle(edges)
        current_set = set()
        
        for idx in range(total_edges):
            u, v = edges[idx]
            
            # Check if adding idx creates a C4
            # It creates a C4 if any conflict triple for idx is fully in current_set
            is_valid = True
            for t1, t2, t3 in conflict_triples[idx]:
                if t1 in current_set and t2 in current_set and t3 in current_set:
                    is_valid = False
                    break
            
            if is_valid:
                current_set.add(idx)
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = set(current_set)
            
            # Early termination if we hit the theoretical max (308) or close
            if best_count >= 308:
                break
        
        # Optional: Check time inside loop if shuffle is fast
        if time.time() > deadline:
            break

    # Convert set of indices back to list of edge tuples
    result = [list(edges[i]) for i in best_set]
    return result
