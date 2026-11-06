import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Precompute faces
    face_edges = []
    # face_mask[f] will be an int with bits set for the 4 edges in face f
    face_mask = []
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                m = (1 << e1) | (1 << e2) | (1 << e3) | (1 << e4)
                face_mask.append(m)
                
    n_faces = len(face_mask)
    all_edges_mask = (1 << n_edges) - 1
    
    # Greedy initialization to get a good upper bound (lower bound on removed edges)
    # We want to minimize removed edges.
    # Greedy: Pick edge hitting most unsatisfied faces.
    
    def get_greedy_solution():
        removed_mask = 0
        kept_mask = 0
        undecided_mask = all_edges_mask
        unsatisfied = set(range(n_faces))
        
        # Precompute face hit counts for speed? 
        # Just check mask intersection.
        
        while unsatisfied:
            best_e = -1
            best_gain = -1
            # Iterate through undecided edges? 
            # Or iterate through unsatisfied faces and pick an edge in one?
            # Greedy Set Cover: Pick edge with max gain.
            # Gain = number of unsatisfied faces it hits.
            
            # To speed up, we can just iterate all undecided edges.
            # 448 edges * 672 faces check? 
            # We can precompute edge_faces list.
            
            # Let's precompute edge_faces
            # This is done outside or inside?
            
            # For simplicity in the inner loop, we might just scan.
            # But we are in a helper function.
            
            # Actually, let's just use a simple heuristic:
            # Pick a random unsatisfied face, pick a random edge in it?
            # No, max gain is better.
            
            # We need edge_faces.
            # Let's compute it once globally.
            
            pass # Placeholder
            
    # Compute edge_faces
    edge_faces = [[] for _ in range(n_edges)]
    for f in range(n_faces):
        m = face_mask[f]
        # Extract bits
        temp = m
        while temp:
            lsb = temp & -temp
            e = lsb.bit_length() - 1
            edge_faces[e].append(f)
            temp ^= lsb
            
    def get_greedy_solution():
        removed_mask = 0
        kept_mask = 0
        undecided_mask = all_edges_mask
        face_hit = [0] * n_faces
        unsatisfied_count = n_faces
        
        def add_edge(e):
            nonlocal undecided_mask, unsatisfied_count
            undecided_mask &= ~(1 << e)
            removed_mask |= (1 << e)
            for f in edge_faces[e]:
                if face_hit[f] == 0:
                    unsatisfied_count -= 1
                face_hit[f] += 1
            return removed_mask, kept_mask, undecided_mask, face_hit, unsatisfied_count

        # We need to return the masks
        # Let's just do it inline in the main scope or return state
        
        while unsatisfied_count > 0:
            best_gain = -1
            best_e = -1
            # Iterate undecided edges
            temp = undecided_mask
            while temp:
                lsb = temp & -temp
                e = lsb.bit_length() - 1
                gain = 0
                for f in edge_faces[e]:
                    if face_hit[f] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
                temp ^= lsb
                
            if best_e == -1:
                break
                
            # Update state
            removed_mask |= (1 << best_e)
            undecided_mask &= ~(1 << best_e)
            for f in edge_faces[best_e]:
                if face_hit[f] == 0:
                    unsatisfied_count -= 1
                face_hit[f] += 1
                
        return removed_mask, kept_mask, undecided_mask, face_hit, unsatisfied_count

    init_removed, init_kept, init_undec, init_hit, init_unsat = get_greedy_solution()
    best_removed_count = bin(init_removed).count('1')
    
    # Branch and Bound
    start_time = time.time()
    timeout = 115
    
    # State: removed_mask, kept_mask, undecided_mask, face_hit (list), unsatisfied_count, current_count
    # We can optimize face_hit by just checking masks?
    # Checking masks for all 672 faces is 672 ops.
    # Updating face_hit is 6 ops per edge.
    # Using face_hit is faster.
    
    # To avoid copying large lists, we use recursion with backtracking.
    # Python recursion limit might be an issue? Depth is up to 448.
    # Default limit is 1000. Should be fine.
    
    def solve(removed_mask, kept_mask, undecided_mask, face_hit, unsatisfied_count, current_count):
        nonlocal best_removed_count, timeout, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if current_count >= best_removed_count:
            return
            
        # Check if any face is invalid (all kept)
        # We can check this during backtracking?
        # If we keep an edge, we must ensure no face becomes fully kept.
        # In the branching step, we only branch on REMOVING an edge from an unsatisfied face.
        # We never explicitly "keep" an edge in a way that violates constraints immediately,
        # unless we decide to keep it by not removing it?
        # No, the logic is:
        # We pick an unsatisfied face F.
        # We MUST remove at least one edge from F.
        # We branch on each edge e in F: "Remove e".
        # The other edges in F are NOT kept explicitly. They remain undecided.
        # So we never create a state where a face is "fully kept" because we only force removals.
        # Wait, if we don't remove an edge, is it kept?
        # In this model, "undecided" means we haven't decided yet.
        # At the end, all undecided edges are KEPT.
        # So if a face has no removed edges, and we finish, it is fully kept -> INVALID.
        # So the condition "unsatisfied_count == 0" means all faces have at least one removed edge.
        # This is the SUCCESS condition.
        
        if unsatisfied_count == 0:
            best_removed_count = current_count
            return
            
        # Pick an unsatisfied face
        # Find one with fewest undecided edges to minimize branching
        # face_hit[f] == 0 means unsatisfied.
        # undecided_in_f = face_mask[f] & undecided_mask
        # count bits in undecided_in_f
        
        best_face = -1
        min_undec = 100
        for f in range(n_faces):
            if face_hit[f] == 0:
                undec_bits = (face_mask[f] & undecided_mask).bit_count()
                if undec_bits < min_undec:
                    min_undec = undec_bits
                    best_face = f
                    if min_undec == 1:
                        break
        
        if best_face == -1:
            # Should not happen if unsatisfied_count > 0
            return
            
        # Branch on edges in best_face
        undec_in_face = face_mask[best_face] & undecided_mask
        
        # Order edges: maybe random or by degree?
        # Randomize to break symmetry
        edge_list = []
        temp = undec_in_face
        while temp:
            lsb = temp & -temp
            e = lsb.bit_length() - 1
            edge_list.append(e)
            temp ^= lsb
            
        random.shuffle(edge_list)
        
        for e in edge_list:
            # Remove e
            new_removed = removed_mask | (1 << e)
            new_undec = undecided_mask & ~(1 << e)
            new_kept = kept_mask # kept doesn't change explicitly, undecided shrinks
            
            # Update face_hit
            new_face_hit = face_hit[:] # Copy
            new_unsat = unsatisfied_count
            for f in edge_faces[e]:
                if new_face_hit[f] == 0:
                    new_unsat -= 1
                new_face_hit[f] += 1
                
            solve(new_removed, new_kept, new_undec, new_face_hit, new_unsat, current_count + 1)
            
            if time.time() - start_time > timeout:
                return

    solve(init_removed, init_kept, init_undec, init_hit, init_unsat, best_removed_count)
    
    # Construct result
    # Edges NOT in best_removed_mask are kept.
    # We need to recover the best_removed_mask.
    # The solve function updates best_removed_count but not the mask itself in the global scope easily without passing it up.
    # Let's modify solve to store the best mask.
    
    # Re-implementing to store mask
    best_mask = init_removed
    
    def solve2(removed_mask, kept_mask, undecided_mask, face_hit, unsatisfied_count, current_count):
        nonlocal best_removed_count, best_mask, timeout, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if current_count >= best_removed_count:
            return
            
        if unsatisfied_count == 0:
            best_removed_count = current_count
            best_mask = removed_mask
            return
            
        best_face = -1
        min_undec = 100
        for f in range(n_faces):
            if face_hit[f] == 0:
                undec_bits = (face_mask[f] & undecided_mask).bit_count()
                if undec_bits < min_undec:
                    min_undec = undec_bits
                    best_face = f
                    if min_undec == 1:
                        break
        
        if best_face == -1:
            return
            
        undec_in_face = face_mask[best_face] & undecided_mask
        
        edge_list = []
        temp = undec_in_face
        while temp:
            lsb = temp & -temp
            e = lsb.bit_length() - 1
            edge_list.append(e)
            temp ^= lsb
            
        random.shuffle(edge_list)
        
        for e in edge_list:
            new_removed = removed_mask | (1 << e)
            new_undec = undecided_mask & ~(1 << e)
            
            new_face_hit = face_hit[:]
            new_unsat = unsatisfied_count
            for f in edge_faces[e]:
                if new_face_hit[f] == 0:
                    new_unsat -= 1
                new_face_hit[f] += 1
                
            solve2(new_removed, new_kept, new_undec, new_face_hit, new_unsat, current_count + 1)
            
            if time.time() - start_time > timeout:
                return

    solve2(init_removed, init_kept, init_undec, init_hit, init_unsat, best_removed_count)
    
    result = []
    for i, (u, v) in enumerate(edges):
        if not (best_mask & (1 << i)):
            result.append([u, v])
            
    return result
