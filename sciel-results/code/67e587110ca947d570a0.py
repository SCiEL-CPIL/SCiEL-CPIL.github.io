import time
import random
from collections import defaultdict

def construct():
    """
    Construct a C4-free edge set of Q7 maximizing the number of edges.
    Uses a local search approach with multiple restarts and edge flips.
    """
    d = 7
    n = 1 << d
    deadline = time.time() + 130  # 130s safety margin before 150s kill
    
    # Pre-compute edges
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_idx[e] = len(edges)
                edges.append(e)
    m = len(edges)  # 448
    
    # For each edge, find all 4-cycles (faces) it belongs to
    # A face is defined by two non-adjacent vertices that differ in exactly 2 bits
    # Face: {x, x^bi, x^bj, x^bi^bj} where bi != bj
    # Edges of face: (x, x^bi), (x, x^bj), (x^bi, x^bi^bj), (x^bj, x^bi^bj)
    
    # For each pair of bits (i,j), and each x, the face is x, x^(1<<i), x^(1<<j), x^(1<<(i|j))
    # But we need to be careful: a face is the same regardless of which vertex we start from.
    # A face is uniquely identified by its 4 vertices. Let's use a canonical representation.
    
    # Actually, let's think differently. Each face is a set of 4 edges.
    # Let's precompute for each edge, which faces it's in, and for each face, which edges it contains.
    
    # A face in Q7 is determined by: a base vertex x (7 bits), and two distinct bit positions (i, j).
    # The 4 vertices are: x, x^(1<<i), x^(1<<j), x^(1<<(i|j))  [where (1<<i)|(1<<j) = (1<<i)^(1<<j) since i!=j]
    # The 4 edges are:
    #   (min(x, x^(1<<i)), max(x, x^(1<<i)))
    #   (min(x, x^(1<<j)), max(x, x^(1<<j)))
    #   (min(x^(1<<i), x^(1<<(i|j))), max(x^(1<<i), x^(1<<(i|j))))
    #   (min(x^(1<<j), x^(1<<(i|j))), max(x^(1<<j), x^(1<<(i|j))))
    
    # To avoid duplicates: a face is the same regardless of which of its 4 vertices is "x".
    # We can canonicalize by taking the min vertex as x, and i < j.
    # But different x's can give the same face. 
    # Actually, each face has exactly 4 vertices. If we pick the smallest vertex as x, and the two bit positions
    # that differ from x to get the other two vertices... but we need to be careful.
    
    # Simpler approach: enumerate all faces by choosing a canonical form.
    # A face is a set of 4 vertices {a, b, c, d} where each pair differs in at most 2 bits,
    # and they form a 4-cycle.
    
    # Let's just enumerate: for each x (0 to 127), for each pair (i,j) with i<j,
    # the face has vertices x, x^(1<<i), x^(1<<j), x^((1<<i)^(1<<j)).
    # To avoid duplicates, we can require that x is the minimum of the 4 vertices,
    # and i < j.
    
    face_edges = {}  # face_id -> list of 4 edge indices
    edge_faces = defaultdict(list)  # edge_idx -> list of face_ids
    
    face_count = 0
    for x in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                verts = [v0, v1, v2, v3]
                if min(verts) != x:
                    continue
                # Build the 4 edges
                e0 = (min(v0, v1), max(v0, v1))
                e1 = (min(v0, v2), max(v0, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v3), max(v2, v3))
                edge_indices = []
                for e in [e0, e1, e2, e3]:
                    if e not in edge_to_idx:
                        # This shouldn't happen
                        continue
                    edge_indices.append(edge_to_idx[e])
                if len(edge_indices) != 4:
                    continue
                fid = face_count
                face_count += 1
                face_edges[fid] = edge_indices
                for ei in edge_indices:
                    edge_faces[ei].append(fid)
    
    total_faces = face_count  # Should be 127 * C(7,2) / 4 = 127 * 21 / 4... let's not worry about exact count
    
    # Now, local search:
    # State: a set of edges (using a boolean array or set)
    # We track for each face, how many of its 4 edges are present.
    # A face is "bad" if all 4 edges are present.
    
    def count_bad_faces(edge_set, face_count_counts):
        """Count faces with all 4 edges present."""
        bad = 0
        for fid in range(total_faces):
            if face_count_counts[fid] == 4:
                bad += 1
        return bad
    
    def update_face_counts(edge_idx, add, edge_set, face_counts):
        """Update face counts when adding/removing an edge."""
        for fid in edge_faces[edge_idx]:
            if add:
                face_counts[fid] += 1
            else:
                face_counts[fid] -= 1
            # We only care about whether face is bad (count == 4)
    
    def run_local_search(initial_edges, deadline):
        """
        Run local search starting from initial_edges.
        Returns the best edge set found.
        """
        edge_set = set(initial_edges)
        face_counts = [0] * total_faces
        
        # Initialize face counts
        for ei in edge_set:
            for fid in edge_faces[ei]:
                face_counts[fid] += 1
        
        # Count initial bad faces
        bad_count = sum(1 for fc in face_counts if fc == 4)
        
        best_edge_set = set(edge_set)
        best_count = len(edge_set)
        
        # If there are bad faces, we need to remove edges first
        # Phase 1: Remove edges to eliminate all bad faces
        while bad_count > 0 and time.time() < deadline:
            # Find a bad face
            bad_face = None
            for fid in range(total_faces):
                if face_counts[fid] == 4:
                    bad_face = fid
                    break
            if bad_face is None:
                bad_count = 0
                break
            
            # Find the edge in this face that appears in the most other bad faces
            # Or just pick a random edge from the face
            face_eis = face_edges[bad_face]
            
            # Score each edge by how many other bad faces it's in
            best_ei = -1
            best_score = -1
            for ei in face_eis:
                score = 0
                for fid2 in edge_faces[ei]:
                    if face_counts[fid2] == 4:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_ei = ei
            
            if best_ei < 0:
                best_ei = face_eis[0]
            
            # Remove the edge
            edge_set.remove(best_ei)
            update_face_counts(best_ei, False, edge_set, face_counts)
            bad_count -= 1  # This face is no longer bad
            # But other faces might become non-bad too, so we should recount
            # For efficiency, let's just recount periodically or accept some error
            # Actually, removing an edge can only decrease face counts, so no new bad faces
            # But other faces that were bad might become non-bad if they shared this edge
            # Let's be safe and recount
            bad_count = sum(1 for fc in face_counts if fc == 4)
        
        # Phase 2: Try to add edges to maximize count
        # We can try adding edges in a greedy or random order
        remaining = [ei for ei in range(m) if ei not in edge_set]
        random.shuffle(remaining)
        
        for ei in remaining:
            if time.time() > deadline:
                break
            # Check if adding this edge creates a bad face
            ok = True
            for fid in edge_faces[ei]:
                if face_counts[fid] == 3:  # Adding would make it 4
                    ok = False
                    break
            if ok:
                edge_set.add(ei)
                update_face_counts(ei, True, edge_set, face_counts)
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_edge_set = set(edge_set)
        
        # Phase 3: Try flips - remove an edge and see if we can add more
        # Or try random restarts within the local search
        # For now, let's try a few rounds of random edge removal and re-adding
        
        for round_num in range(20):
            if time.time() > deadline:
                break
            
            # Try to improve by removing some edges and re-adding
            # Pick a random subset of edges to remove
            remove_count = min(len(edge_set), random.randint(1, 5))
            to_remove = random.sample(list(edge_set), remove_count)
            
            for ei in to_remove:
                edge_set.remove(ei)
                update_face_counts(ei, False, edge_set, face_counts)
            
            # Re-add greedily
            remaining2 = [ei for ei in range(m) if ei not in edge_set]
            random.shuffle(remaining2)
            for ei in remaining2:
                if time.time() > deadline:
                    break
                ok = True
                for fid in edge_faces[ei]:
                    if face_counts[fid] == 3:
                        ok = False
                        break
                if ok:
                    edge_set.add(ei)
                    update_face_counts(ei, True, edge_set, face_counts)
            
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edge_set = set(edge_set)
        
        return list(best_edge_set)
    
    # Main loop: multiple restarts
    best_edge_set = []
    best_count = 0
    
    # Start with a random initial set
    # Try different densities
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Choose a random initial density
        density = random.uniform(0.5, 0.9)
        initial = [ei for ei in range(m) if random.random() < density]
        
        result = run_local_search(initial, deadline)
        
        if len(result) > best_count:
            best_count = len(result)
            best_edge_set = result
            # If we hit a very high count, we might stop early
            if best_count >= 308:
                break
    
    # If we didn't find anything good, try a deterministic greedy
    if best_count == 0:
        # Greedy: scan edges and add if no face would become bad
        edge_set = set()
        face_counts = [0] * total_faces
        for ei in range(m):
            ok = True
            for fid in edge_faces[ei]:
                if face_counts[fid] == 3:
                    ok = False
                    break
            if ok:
                edge_set.add(ei)
                for fid in edge_faces[ei]:
                    face_counts[fid] += 1
        best_edge_set = list(edge_set)
        best_count = len(best_edge_set)
    
    # Convert to list of [u, v]
    result = []
    for ei in best_edge_set:
        u, v = edges[ei]
        result.append([u, v])
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Edges found: {len(res)}")
