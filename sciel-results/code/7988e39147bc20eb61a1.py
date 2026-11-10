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
    
    # Build a "face neighbors" structure for fast conflict tracking
    # For each edge, list of (face_id, position_in_face) where face_id identifies the 4-cycle
    # and position is which of the 4 edges this is
    # Actually, let's use a different approach:
    # conflict_count[edge_idx] = number of 4-cycles where the other 3 edges are all chosen
    # When we add/remove an edge, we update conflict_count for all edges sharing a face with it
    
    # For each edge, precompute the set of other edges that share a face with it
    # edge_face_neighbors[edge_idx] = set of edge indices that share at least one 4-cycle
    edge_face_neighbors = [set() for _ in range(m)]
    
    for idx in range(m):
        for (e1, e2, e3) in face_conflicts[idx]:
            edge_face_neighbors[idx].add(e1)
            edge_face_neighbors[idx].add(e2)
            edge_face_neighbors[idx].add(e3)
    
    def try_greedy(seed):
        rng = random.Random(seed)
        edge_order = list(range(m))
        rng.shuffle(edge_order)
        
        chosen = np.zeros(m, dtype=np.int8)  # 0 or 1
        # conflict_count[i] = number of 4-cycles for edge i where the other 3 edges are chosen
        conflict_count = np.zeros(m, dtype=np.int8)
        count = 0
        
        for idx in edge_order:
            if conflict_count[idx] == 0:
                # Add this edge
                chosen[idx] = 1
                count += 1
                # Update conflict_count for all edges sharing a face with idx
                for neighbor in edge_face_neighbors[idx]:
                    if chosen[neighbor] == 0:
                        # Check if adding idx completes any face for neighbor
                        # For each face containing both idx and neighbor, check if other 2 are chosen
                        # This is expensive; let's use a simpler approach
                        pass
        
        # The above is getting complex. Let's stick with the direct check but optimize it.
        # Alternative: use a bitmap approach
        return chosen, count
    
    def try_greedy_v2(seed):
        """Optimized greedy using direct face checks but with early termination."""
        rng = random.Random(seed)
        edge_order = list(range(m))
        rng.shuffle(edge_order)
        
        chosen = np.zeros(m, dtype=np.bool_)
        count = 0
        
        for idx in edge_order:
            # Quick check: only check faces where at least 2 other edges are already chosen
            can_add = True
            for (e1, e2, e3) in face_conflicts[idx]:
                # Need all three to be chosen to block
                if chosen[e1] and chosen[e2] and chosen[e3]:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
        
        return chosen, count
    
    def local_improve_v2(chosen, count):
        """Improved local improvement."""
        best_chosen = chosen.copy()
        best_count = count
        
        # First, try to add any edge that can be added
        changed = True
        while changed:
            changed = False
            for idx in range(m):
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
                    changed = True
                    best_chosen = chosen.copy()
                    best_count = count
        
        # Try 1-for-2 swaps: remove one edge, see if we can add at least 2
        for idx in range(m):
            if time.time() > deadline - 5.0:
                break
            if not chosen[idx]:
                continue
            # Temporarily remove
            chosen[idx] = False
            count -= 1
            
            # Try to add edges
            added = []
            for jdx in range(m):
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
                    added.append(jdx)
            
            if len(added) > 1:
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
    for seed in range(5000):
        if time.time() > deadline - 5.0:
            break
        chosen, count = try_greedy_v2(seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        num_restarts += 1
    
    # Local improvement on best result
    if best_chosen is not None and time.time() < deadline - 10.0:
        best_chosen, best_count = local_improve_v2(best_chosen, best_count)
    
    # Convert back to list of edges
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            out.append(list(all_edges[idx]))
    
    return out
# EVOLVE-BLOCK-END
