import random
import time
import numpy as np

def construct():
    """
    Constructs a C4-free edge set of Q7 maximizing the number of edges.
    Uses randomized local search with swap moves and numpy-accelerated checks.
    """
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Precompute edge representation
    # Each edge is (u, v) with u < v, and we know its axis
    # For fast lookup, use a set of (u, v) tuples
    
    # Precompute for each vertex and axis, the neighbor
    # neighbor[v][i] = v ^ (1 << i)
    neighbor = np.zeros((n_vertices, d), dtype=np.int32)
    for v in range(n_vertices):
        for i in range(d):
            neighbor[v, i] = v ^ (1 << i)
    
    # Generate all edges with their axes
    all_edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = neighbor[u, i]
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
    
    n_total_edges = len(all_edges)
    
    # For each edge, precompute the other edges in each face containing it
    # edge_faces[edge_idx][j] = (edge_idx_j1, edge_idx_j2, edge_idx_j3) for axis j != axis
    # where these are the other 3 edges in the face with axis j
    
    edge_faces = [dict() for _ in range(n_total_edges)]
    for idx, (u, v, axis) in enumerate(all_edges):
        for j in range(d):
            if j == axis:
                continue
            u_j = neighbor[u, j]
            v_j = neighbor[v, j]
            # The other 3 edges in this face
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            edge_faces[idx][j] = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3])
    
    def is_valid_add(selected, edge_idx):
        """Check if adding edge edge_idx creates a C4."""
        u, v, axis = all_edges[edge_idx]
        for j, (e1, e2, e3) in edge_faces[edge_idx].items():
            if selected[e1] and selected[e2] and selected[e3]:
                return False
        return True
    
    def count_violations(selected):
        """Count the number of C4 faces that are fully selected."""
        # For each face, check if all 4 edges are selected
        # A face is defined by two vertices differing in 2 bits
        # Actually, easier: for each edge and each other axis, check if the face is complete
        # But that's O(n_edges * d)
        # Alternative: iterate over all faces
        # A face is determined by a vertex v and two axes i < j
        # Vertices: v, v^i, v^j, v^(i|j)
        # Edges: (v, v^i), (v, v^j), (v^i, v^j), (v^j, v^(i|j)), (v^i, v^(i|j)), (v, v^(i|j))
        # Wait, a 2D face in Q7 has 4 vertices and 4 edges
        # Vertices: v, v^i, v^j, v^(i^j)
        # Edges: (v, v^i), (v^i, v^(i^j)), (v^(i^j), v^j), (v^j, v)
        # That's 4 edges
        count = 0
        for v in range(n_vertices):
            for i in range(d):
                vi = neighbor[v, i]
                if vi <= v:
                    continue
                for j in range(i + 1, d):
                    vj = neighbor[v, j]
                    vij = neighbor[vi, j]  # = v ^ (1<<i) ^ (1<<j)
                    # Edges: (v, vi), (vi, vij), (vij, vj), (vj, v)
                    e1 = (min(v, vi), max(v, vi))
                    e2 = (min(vi, vij), max(vi, vij))
                    e3 = (min(vij, vj), max(vij, vj))
                    e4 = (min(vj, v), max(vj, v))
                    if selected[edge_to_idx[e1]] and selected[edge_to_idx[e2]] and \
                       selected[edge_to_idx[e3]] and selected[edge_to_idx[e4]]:
                        count += 1
        return count
    
    def greedy_initial():
        """Greedy construction with random edge ordering."""
        selected = np.zeros(n_total_edges, dtype=bool)
        order = list(range(n_total_edges))
        random.shuffle(order)
        for idx in order:
            if is_valid_add(selected, idx):
                selected[idx] = True
        return selected
    
    def local_search(selected, time_limit):
        """Local search with swap moves."""
        start_time = time.time()
        best_selected = selected.copy()
        best_size = selected.sum()
        stuck_counter = 0
        max_stuck = 500
        
        while time.time() - start_time < time_limit:
            # Try a swap: remove one edge, try to add others
            # Pick a random edge to remove
            if best_size == 0:
                break
            remove_idx = random.randint(0, n_total_edges - 1)
            if not selected[remove_idx]:
                # Try adding a random edge instead
                add_idx = random.randint(0, n_total_edges - 1)
                if not selected[add_idx] and is_valid_add(selected, add_idx):
                    selected[add_idx] = True
                    stuck_counter = 0
                    if selected.sum() > best_size:
                        best_size = selected.sum()
                        best_selected = selected.copy()
                else:
                    stuck_counter += 1
                continue
            
            # Remove the edge
            selected[remove_idx] = False
            removed_edge = all_edges[remove_idx]
            
            # Try to add edges that were blocked by this edge
            # Find edges that share a face with the removed edge
            # For each axis j != axis of removed edge, the face has 3 other edges
            # Those edges might now be addable
            candidates = set()
            for j, (e1, e2, e3) in edge_faces[remove_idx].items():
                for e in (e1, e2, e3):
                    if not selected[e]:
                        candidates.add(e)
            
            # Also try some random edges
            for _ in range(3):
                rand_idx = random.randint(0, n_total_edges - 1)
                if not selected[rand_idx]:
                    candidates.add(rand_idx)
            
            added_count = 0
            for cand in candidates:
                if is_valid_add(selected, cand):
                    selected[cand] = True
                    added_count += 1
            
            if added_count > 1:
                # Net gain
                stuck_counter = 0
                if selected.sum() > best_size:
                    best_size = selected.sum()
                    best_selected = selected.copy()
            elif added_count == 1:
                stuck_counter += 1
            else:
                # Revert
                selected[remove_idx] = True
                stuck_counter += 2
            
            if stuck_counter > max_stuck:
                # Restart with a new greedy initial
                selected = greedy_initial()
                stuck_counter = 0
                if selected.sum() > best_size:
                    best_size = selected.sum()
                    best_selected = selected.copy()
        
        return best_selected
    
    # Main search loop
    start_time = time.time()
    time_limit = 140  # Leave 10s margin
    
    best_selected = greedy_initial()
    best_size = best_selected.sum()
    
    # Run local search
    current_selected = best_selected.copy()
    while time.time() - start_time < time_limit:
        current_selected = local_search(current_selected, time_limit - (time.time() - start_time))
        if current_selected.sum() > best_size:
            best_size = current_selected.sum()
            best_selected = current_selected.copy()
        
        # Check if we should restart
        if time.time() - start_time < time_limit * 0.5:
            # Do another greedy initial and local search
            new_initial = greedy_initial()
            if new_initial.sum() > best_size:
                best_size = new_initial.sum()
                best_selected = new_initial.copy()
                current_selected = best_selected.copy()
    
    # Convert to list of edges
    result = []
    for idx in range(n_total_edges):
        if best_selected[idx]:
            u, v, axis = all_edges[idx]
            result.append([u, v])
    
    return result
