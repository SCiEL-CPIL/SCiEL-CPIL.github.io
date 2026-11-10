<<<<<<< SEARCH
        else:
            # Rebuild lines greedily from current points
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            seen = set()
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in seen:
                    seen.add(tuple(nl))
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    current_line_set = set(tuple(l) for l in current_lines)
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
=======
        else:
            # Rebuild lines greedily from current points
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            seen = set()
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in seen:
                    seen.add(tuple(nl))
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    current_line_set = set(tuple(l) for l in current_lines)
                    # Recompute incidence matrix
                    P = current_points.astype(np.int64)
                    px = P[:, 0].reshape(-1, 1)
                    py = P[:, 1].reshape(-1, 1)
                    L = current_lines.astype(np.int64)
                    A = L[:, 0].reshape(1, -1)
                    B = L[:, 1].reshape(1, -1)
                    C = L[:, 2].reshape(1, -1)
                    incidence_matrix = (A * px + B * py + C == 0).astype(np.int64)
                    point_counts = incidence_matrix.sum(axis=1)
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
>>>>>>> REPLACE
