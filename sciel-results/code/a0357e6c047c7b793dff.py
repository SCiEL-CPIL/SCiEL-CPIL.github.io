    def local_search(chosen, face_counts, count, max_time=None):
        if max_time is None:
            max_time = deadline - time.time() - 5
        
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
        
        no_improve = 0
        max_no_improve = 20000
        
        while time.time() < max_time:
            # Find a tight edge
            tight_edges = []
            for idx in range(m):
                if chosen[idx]:
                    # Check if in any face with count >= 3
                    is_tight = False
                    for f_idx in edge_faces[idx]:
                        if face_counts[f_idx] >= 3:
                            is_tight = True
                            break
                    if is_tight:
                        tight_edges.append(idx)
            
            if tight_edges:
                rem_idx = random.choice(tight_edges)
            else:
                # If no tight edges, we are in a "loose" state.
                # Greedy polish should have handled this?
                # If we are here, maybe we should break or do random.
                rem_idx = random.randint(0, m - 1)
                # If we pick a random edge and it's not tight, we will revert.
                # This is a waste.
                # If no tight edges, we can probably break out of local search
                # because we are likely at a local optimum where no single removal helps.
                # (Unless a 2-removal helps, but we don't do that).
                # So break?
                # Let's just break if no tight edges found.
                break

            # Remove rem_idx
            chosen[rem_idx] = False
            count -= 1
            for f_idx in edge_faces[rem_idx]:
                face_counts[f_idx] -= 1
            
            # Try to add neighbors
            added_count = 0
            added_edges = []
            
            neighbors = conflict_adj[rem_idx][:]
            random.shuffle(neighbors)
            
            for add_idx in neighbors:
                if not chosen[add_idx]:
                    can_add = True
                    for f_idx in edge_faces[add_idx]:
                        if face_counts[f_idx] >= 3:
                            can_add = False
                            break
                    if can_add:
                        chosen[add_idx] = True
                        count += 1
                        added_count += 1
                        added_edges.append(add_idx)
                        for f_idx in edge_faces[add_idx]:
                            face_counts[f_idx] += 1
            
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
                best_face_counts = face_counts.copy()
                no_improve = 0
            else:
                # Revert
                for add_idx in added_edges:
                    chosen[add_idx] = False
                    count -= 1
                    for f_idx in edge_faces[add_idx]:
                        face_counts[f_idx] -= 1
                
                chosen[rem_idx] = True
                count += 1
                for f_idx in edge_faces[rem_idx]:
                    face_counts[f_idx] += 1
                
                no_improve += 1
            
            if no_improve > max_no_improve:
                chosen = best_chosen.copy()
                face_counts = best_face_counts.copy()
                count = best_count
                no_improve = 0
                
        return best_chosen, best_face_counts, best_count
