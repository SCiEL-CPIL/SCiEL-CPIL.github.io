"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges and face adjacencies
    all_edges = []
    edge_set = set()
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
                edge_set.add((u, v))
    
    m = len(all_edges)  # 448 edges
    
    # Precompute for each edge, the other 3 edges in each 4-cycle it belongs to
    # An edge (u,v) with direction i is in a 4-cycle with direction j (j!=i):
    #   u, v, u^j, v^j
    # The other 3 edges are: (u, u^j), (v, v^j), (u^j, v^j)
    # Precompute: for each edge index, list of (other_edge1, other_edge2, other_edge3)
    # where each other_edge is (a,b) with a<b
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, find all 4-cycles containing it
    # face_conflicts[edge_idx] = list of tuples (e1, e2, e3) where e1,e2,e3 are edge indices
    face_conflicts = [[] for _ in range(m)]
    
    for idx, (u, v) in enumerate(all_edges):
        # Determine which bit differs
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            # Other 3 edges in this 4-cycle
            e1 = (min(u, c), max(u, c))
            e2 = (min(v, e), max(v, e))
            e3 = (min(c, e), max(c, e))
            e1_idx = edge_to_idx[e1]
            e2_idx = edge_to_idx[e2]
            e3_idx = edge_to_idx[e3]
            face_conflicts[idx].append((e1_idx, e2_idx, e3_idx))
    
    def try_greedy(seed):
        rng = random.Random(seed)
        edge_order = list(range(m))
        rng.shuffle(edge_order)
        
        chosen = np.zeros(m, dtype=bool)
        count = 0
        
        for idx in edge_order:
            # Check if adding this edge completes any 4-cycle
            can_add = True
            for (e1, e2, e3) in face_conflicts[idx]:
                if chosen[e1] and chosen[e2] and chosen[e3]:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
        
        return chosen, count
    
    def local_improve(chosen, count):
        """Try to add more edges by removing one and re-adding several."""
        best_chosen = chosen.copy()
        best_count = count
        m_local = m
        
        # Try single-edge swaps: remove edge i, try to add back all previously blocked edges
        # Simpler: just try adding each non-chosen edge if possible
        for idx in range(m_local):
            if chosen[idx]:
                continue
            can_add = True
            for (e1, e2, e3) in face_conflicts[idx]:
                if chosen[e1] and chosen[e2] and chosen[e3]:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                best_chosen = chosen.copy()
                best_count = count
        
        # Try 1-for-1 swaps: remove one edge, see if we can add two
        for idx in range(m_local):
            if not chosen[idx]:
                continue
            # Temporarily remove
            chosen[idx] = False
            count -= 1
            
            # Try to add edges
            gained = 0
            added = []
            for jdx in range(m_local):
                if chosen[jdx]:
                    continue
                can_add = True
                for (e1, e2, e3) in face_conflicts[jdx]:
                    if chosen[e1] and chosen[e2] and chosen[e3]:
                        can_add = False
                        break
                if can_add:
                    chosen[jdx] = True
                    count += 1
                    gained += 1
                    added.append(jdx)
            
            if gained > 1:
                best_chosen = chosen.copy()
                best_count = count
            else:
                # Revert
                for jdx in added:
                    chosen[jdx] = False
                    count -= 1
                chosen[idx] = True
                count += 1
        
        return best_chosen, best_count
    
    # Run multiple greedy restarts with different seeds
    best_count = 0
    best_chosen = None
    
    num_restarts = 0
    for seed in range(2000):
        if time.time() > deadline - 5.0:
            break
        chosen, count = try_greedy(seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        num_restarts += 1
    
    # Local improvement on best result
    if best_chosen is not None and time.time() < deadline - 10.0:
        best_chosen, best_count = local_improve(best_chosen, best_count)
    
    # Convert back to list of edges
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            out.append(list(all_edges[idx]))
    
    return out
# EVOLVE-BLOCK-END
