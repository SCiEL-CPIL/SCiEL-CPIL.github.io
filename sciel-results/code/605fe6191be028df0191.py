import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448

    # 1. Precompute Edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx

    # 2. Precompute Faces
    # A face is a set of 4 edges forming a C4.
    # Unique representation: sorted tuple of edge indices.
    faces = []
    face_to_edges = []  # list of lists of edge indices
    edge_to_faces = [[] for _ in range(n_edges)]
    
    seen_faces = set()
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n_vertices):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_idx[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_idx[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_idx[(min(v2, v4), max(v2, v4))]
                
                face_tuple = tuple(sorted([e1, e2, e3, e4]))
                
                if face_tuple not in seen_faces:
                    seen_faces.add(face_tuple)
                    f_idx = len(faces)
                    faces.append(face_tuple)
                    face_to_edges.append(list(face_tuple))
                    for e in face_tuple:
                        edge_to_faces[e].append(f_idx)
    
    n_faces = len(faces)
    
    # Convert face_to_edges to numpy arrays for faster iteration if needed
    face_edges_np = [np.array(fe, dtype=np.int32) for fe in face_to_edges]
    edge_faces_np = [np.array(e_f, dtype=np.int32) for e_f in edge_to_faces]

    def greedy_hitting_set(randomize_ties=False):
        R = set()
        unhit_faces = set(range(n_faces))
        
        while unhit_faces:
            if randomize_ties and unhit_faces:
                f_idx = random.choice(list(unhit_faces))
            else:
                f_idx = next(iter(unhit_faces))
                
            face = faces[f_idx]
            
            best_edge = -1
            max_count = -1
            counts = {}
            for e in face:
                count = 0
                for f_idx2 in edge_to_faces[e]:
                    if f_idx2 in unhit_faces:
                        count += 1
                counts[e] = count
                if count > max_count:
                    max_count = count
                    best_edge = e
            
            if randomize_ties and max_count > 0:
                candidates = [e for e in face if counts[e] == max_count]
                best_edge = random.choice(candidates)
            
            R.add(best_edge)
            
            for f_idx2 in edge_to_faces[best_edge]:
                if f_idx2 in unhit_faces:
                    unhit_faces.remove(f_idx2)
                    
        return R

    def cleanup(R):
        R = set(R)
        changed = True
        while changed:
            changed = False
            for e in list(R):
                is_redundant = True
                for f_idx in edge_to_faces[e]:
                    face = faces[f_idx]
                    has_other = False
                    for e2 in face:
                        if e2 != e and e2 in R:
                            has_other = True
                            break
                    if not has_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    R.remove(e)
                    changed = True
        return R

    def local_search_hs(R, time_limit):
        """
        Local search on the Hitting Set R.
        Tries to find 1-2 exchanges: remove e1, e2 from R, add e3 not in R,
        such that all faces uncovered by removing e1,e2 are covered by e3.
        This reduces |R| by 1.
        Also handles 2-0 exchanges (remove e1,e2, add nothing) if faces are covered by others.
        """
        R = set(R)
        # Precompute face counts
        face_counts = np.zeros(n_faces, dtype=np.int32)
        for e in R:
            for f_idx in edge_to_faces[e]:
                face_counts[f_idx] += 1

        improved = True
        t_end = time.time() + time_limit
        
        while improved and time.time() < t_end:
            improved = False
            R_list = list(R)
            random.shuffle(R_list)
            
            # Try pairs
            for i in range(len(R_list)):
                if time.time() >= t_end:
                    break
                e1 = R_list[i]
                for j in range(i + 1, len(R_list)):
                    if time.time() >= t_end:
                        break
                    e2 = R_list[j]
                    
                    # Temporarily remove e1, e2
                    for f_idx in edge_to_faces[e1]:
                        face_counts[f_idx] -= 1
                    for f_idx in edge_to_faces[e2]:
                        face_counts[f_idx] -= 1
                    
                    # Find uncovered faces
                    uncovered = []
                    for f_idx in edge_to_faces[e1]:
                        if face_counts[f_idx] == 0:
                            uncovered.append(f_idx)
                    for f_idx in edge_to_faces[e2]:
                        if face_counts[f_idx] == 0 and f_idx not in uncovered:
                            uncovered.append(f_idx)
                    
                    if not uncovered:
                        # We can just remove e1 and e2
                        R.remove(e1)
                        R.remove(e2)
                        improved = True
                        # No need to restore counts as we actually removed them
                        break
                    else:
                        # Need to find e3 not in R that covers all uncovered faces
                        # Intersection of edges in all uncovered faces
                        # Start with edges of first uncovered face
                        candidate_edges = set(faces[uncovered[0]])
                        for f_idx in uncovered[1:]:
                            candidate_edges &= set(faces[f_idx])
                        
                        # Filter out edges already in R
                        candidate_edges -= R
                        
                        found_e3 = -1
                        for e3 in candidate_edges:
                            # Verify e3 covers all uncovered (should be true by intersection)
                            # And e3 is not e1 or e2 (they are in R, so excluded)
                            found_e3 = e3
                            break
                        
                        if found_e3 != -1:
                            # Perform swap: remove e1, e2, add e3
                            R.remove(e1)
                            R.remove(e2)
                            R.add(found_e3)
                            # Update counts
                            for f_idx in edge_to_faces[found_e3]:
                                face_counts[f_idx] += 1
                            improved = True
                            break
                        else:
                            # Restore counts
                            for f_idx in edge_to_faces[e1]:
                                face_counts[f_idx] += 1
                            for f_idx in edge_to_faces[e2]:
                                face_counts[f_idx] += 1
                
                if improved:
                    # Restart loop to re-evaluate with new R
                    break

        return R

    best_R_size = float('inf')
    best_R = set()
    
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        use_random = (i % 5 != 0)
        R = greedy_hitting_set(randomize_ties=use_random)
        R = cleanup(R)
        
        # Local search with a small budget per restart
        # If R is large, local search might take longer.
        # Cap local search time to 0.1s per restart to allow many restarts.
        # Or adapt: if R size is close to best, spend more time.
        ls_time = 0.05
        if len(R) <= best_R_size:
            ls_time = 0.1
        R = local_search_hs(R, ls_time)
        
        if len(R) < best_R_size:
            best_R_size = len(R)
            best_R = R.copy()
            
        if best_R_size == 144:
            break

    # Final polish on the best R with a larger time budget
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 2
        best_R = local_search_hs(best_R, remaining)
        best_R = cleanup(best_R)

    selected_edges = []
    for idx, (u, v) in enumerate(edges):
        if idx not in best_R:
            selected_edges.append([u, v])
            
    return selected_edges
