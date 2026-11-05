def get_walks(max_len):
    walks = []
    # Start from each vertex
    for start in range(6):
        # State: (curr_vertex, prev_vertex, length_remaining, path_edges, path_dirs, prev_edge_idx, prev_dir)
        # We need to know the previous edge to prevent backtracking.
        
        def dfs(curr, prev, length, path_edges, path_dirs, prev_edge, prev_dir):
            if length == 0:
                if curr == start:
                    # Found a closed non-backtracking walk
                    walks.append((list(path_edges), list(path_dirs)))
                return
            
            for next_v in range(6):
                if next_v == prev:
                    continue
                # Check if edge exists
                if (curr < 3 and next_v >= 3) or (curr >= 3 and next_v < 3):
                    if curr < 3:
                        idx = curr * 3 + (next_v - 3)
                        dir = 1
                    else:
                        idx = next_v * 3 + (curr - 3)
                        dir = -1
                    
                    # Check for backtracking
                    if prev_edge is not None and idx == prev_edge and dir == -prev_dir:
                        continue
                        
                    path_edges.append(idx)
                    path_dirs.append(dir)
                    dfs(next_v, curr, length - 1, path_edges, path_dirs, idx, dir)
                    path_edges.pop()
                    path_dirs.pop()
            
        # Initial step
        for first in range(6):
            if first == start: continue
            if (start < 3 and first >= 3) or (start >= 3 and first < 3):
                if start < 3:
                    idx = start * 3 + (first - 3)
                    dir = 1
                else:
                    idx = first * 3 + (start - 3)
                    dir = -1
                dfs(first, start, max_len - 1, [idx], [dir], idx, dir)
    return walks
