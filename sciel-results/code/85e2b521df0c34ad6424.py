import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute edges
    idx_to_edge = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Precompute faces and conflict graph
    # A conflict exists between two edges if they belong to the same face.
    # We build the adjacency matrix of the conflict graph.
    # Size: 448x448. Very small.
    # conflict[e1, e2] = 1 if e1 and e2 are in a common face.
    
    # First, map edge -> list of faces it belongs to.
    # Then, for each face, add conflicts between all pairs of edges in it.
    
    edge_faces = [[] for _ in range(num_edges)]
    
    # Iterate over all faces
    # Face defined by (i, j, u) where u has 0 in bits i and j? 
    # Actually, any 4 vertices forming a square.
    # Unique face: min vertex u, dims i, j.
    # To ensure uniqueness, require u has 0 in bits i and j? No, just iterate all u, i, j and use set of edges.
    
    seen_faces = set()
    face_edge_list = [] # List of lists of edge indices
    
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                # Check if this is a canonical face to avoid duplicates?
                # Or just use the set of edge indices as key.
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edge_list.append(idxs)
                    
    # Build conflict graph
    # conflict_matrix is a numpy array of uint8, 448x448
    conflict_matrix = np.zeros((num_edges, num_edges), dtype=np.uint8)
    
    for face in face_edge_list:
        e0, e1, e2, e3 = face
        # All pairs are in conflict
        pairs = [(e0, e1), (e0, e2), (e0, e3), (e1, e2), (e1, e3), (e2, e3)]
        for a, b in pairs:
            conflict_matrix[a, b] = 1
            conflict_matrix[b, a] = 1
            
    # Degrees in conflict graph
    # deg[e] = number of other edges that share a face with e
    deg = np.sum(conflict_matrix, axis=1)
    
    # 3. Strategy: Greedy Independent Set on Conflict Graph
    # We want to select a set of edges S such that no two edges in S are in conflict?
    # NO. The constraint is: No face has ALL 4 edges selected.
    # This is NOT an Independent Set problem on the graph where edges connect faces.
    # It is a "4-uniform hypergraph independent set" problem, or equivalently,
    # we want to remove a minimum set of edges R such that no face is fully contained in E\R.
    # This is the Minimum Hitting Set problem on the hypergraph of faces.
    
    # However, the prompt asks to try: "Formulate ... as a maximum weight independent set on the conflict graph 
    # where vertices are hypercube edges and edges connect conflicts within C4s"
    # Wait. If we define conflict as "sharing a face", then an Independent Set in this graph 
    # would be a set of edges where NO two share a face. 
    # This is much stricter than the requirement (which only forbids 4 edges forming a face).
    # An Independent Set in the "share a face" graph is a valid solution, but likely very small.
    # Let's estimate: Max degree in this graph?
    # An edge belongs to (d-1) faces? No.
    # Edge (u, u^2^i). It belongs to faces involving dimension i and any other dimension j.
    # For each j != i, there is a face containing this edge.
    # So each edge is in d-1 = 6 faces.
    # Each face has 4 edges.
    # Total conflicts per edge: 6 faces * 3 other edges = 18 conflicts.
    # So the conflict graph is 18-regular (approximately).
    # An independent set in a 18-regular graph on 448 vertices is roughly 448/19 ~ 23 edges.
    # This is way below 304.
    
    # So the "Independent Set on conflict graph" interpretation in the prompt is likely a misdirection 
    # or refers to a different graph structure.
    # Actually, if we view the problem as: Select edges E' such that no C4 is monochromatic.
    # This is the "Maximum C4-free subgraph" problem.
    
    # Let's re-read the directive: "Formulate the problem as a maximum weight independent set on the conflict graph 
    # where vertices are hypercube edges and edges connect conflicts within C4s"
    # If "conflict" means "cannot both be selected", then the constraint "No 2D face may have all four edges selected" 
    # is a 4-clique constraint, not a pairwise one.
    # Standard MIS handles pairwise constraints.
    
    # Perhaps the intended "conflict graph" is the **Line Graph of the Hypercube**?
    # In the line graph, vertices are edges of Q7. Two vertices are connected if the edges are adjacent in Q7.
    # A C4 in Q7 corresponds to a 4-cycle in the line graph?
    # Yes. The four edges of a face form a 4-cycle in the line graph.
    # So the problem is: Find a large subset of vertices in the Line Graph of Q7 such that no 4-cycle is fully selected.
    # This is still not a standard MIS.
    
    # However, there is a known reduction:
    # If we color the edges with 2 colors, and take one color class, does it avoid C4s?
    # No.
    
    # Let's look at the "Inspiration 1" which scored 1.12. It used numpy, random, time.
    # The current program uses SA.
    
    # Let's try a **Greedy Hitting Set** approach on the faces, which is a different design family from SA.
    # We want to REMOVE edges to break all C4s.
    # Minimize removed edges.
    # Greedy: While there is a C4 (face with 4 edges), pick an edge in it that is in the most other C4s, and remove it.
    # This is the standard heuristic for Set Cover / Hitting Set.
    
    # We can implement this efficiently using the face_counts.
    
    # State:
    # active_edges: boolean array of length 448. Initially all True.
    # face_counts: array of length num_faces. Initially all 4.
    # 
    # Loop:
    # Find all faces with count 4.
    # If none, stop.
    # For each edge e, count how many "active" faces (count 4) it belongs to.
    # Pick e with max count.
    # Remove e:
    #   active_edges[e] = False
    #   For each face f containing e:
    #       face_counts[f] -= 1
    #       If face_counts[f] becomes 3, it is no longer "active" (no longer a C4).
    
    # This is a deterministic heuristic.
    # To improve, we can randomize the tie-breaking or do multiple restarts with different initial "weights" or random edge orders.
    # Or, we can use this as the initialization for a local search that tries to re-add edges.
    
    # Let's implement the Greedy Hitting Set first.
    
    best_removed = num_edges # Start with worst case
    best_state = None
    
    num_faces = len(face_edge_list)
    
    # Precompute edge -> faces mapping for fast updates
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)
            
    rng = np.random.default_rng(42)
    
    # Run greedy multiple times with random perturbations in tie-breaking
    # Or just run once? The greedy hitting set is usually quite good.
    # Let's run a few variants:
    # 1. Pure Greedy (max degree in active hypergraph)
    # 2. Randomized Greedy (pick randomly among top K)
    
    runs = 50
    time_per_run = (deadline - start_time) / runs
    
    for run_idx in range(runs):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        active = np.ones(num_edges, dtype=np.uint8)
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        
        # Active faces are those with count 4.
        # We maintain a set/list of active face indices?
        # Or just scan? 672 faces is small. Scanning is fine.
        
        removed_count = 0
        
        # Greedy Loop
        # To speed up, we can maintain a priority queue or just scan.
        # Scanning 448 edges * 672 faces is fast enough.
        
        while True:
            # Find active faces
            active_faces = np.where(face_counts == 4)[0]
            if len(active_faces) == 0:
                break
                
            # Calculate score for each edge: number of active faces it belongs to
            scores = np.zeros(num_edges, dtype=np.int32)
            for f_idx in active_faces:
                for e in face_edge_list[f_idx]:
                    scores[e] += 1
            
            # Find max score
            max_score = np.max(scores)
            if max_score == 0:
                break # Should not happen if active_faces is not empty
                
            # Get all edges with max score
            candidates = np.where(scores == max_score)[0]
            
            # Randomly pick one
            e_to_remove = candidates[rng.integers(0, len(candidates))]
            
            # Remove it
            active[e_to_remove] = 0
            removed_count += 1
            
            # Update face counts
            for f_idx in edge_to_faces[e_to_remove]:
                face_counts[f_idx] -= 1
        
        selected_count = num_edges - removed_count
        
        if selected_count > (num_edges - best_removed):
            best_removed = removed_count
            best_state = active.copy()
            
    if best_state is None:
        # Fallback: return all edges? No, must be valid.
        # If greedy failed completely (unlikely), return empty or a known valid set.
        # Let's just return the result of the last run if it's valid.
        # Actually, the loop above ensures validity.
        # If best_state is None, it means we didn't run any loop?
        # Initialize best_state to the result of the first run.
        pass
        
    # If best_state is still None (e.g. 0 runs executed due to time), do a quick deterministic greedy
    if best_state is None:
        active = np.ones(num_edges, dtype=np.uint8)
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        removed_count = 0
        while True:
            active_faces = np.where(face_counts == 4)[0]
            if len(active_faces) == 0:
                break
            scores = np.zeros(num_edges, dtype=np.int32)
            for f_idx in active_faces:
                for e in face_edge_list[f_idx]:
                    scores[e] += 1
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            e_to_remove = candidates[0] # Deterministic
            active[e_to_remove] = 0
            removed_count += 1
            for f_idx in edge_to_faces[e_to_remove]:
                face_counts[f_idx] -= 1
        best_state = active
        best_removed = removed_count

    # Convert to list of edges
    result = []
    for i in range(num_edges):
        if best_state[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
