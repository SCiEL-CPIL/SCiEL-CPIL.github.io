import time
import random
import sys

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency sets for each color
    color_adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            color_adj[c][i].add(j)
            color_adj[c][j].add(i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = color_adj[c][u] & color_adj[c][v]
                if len(common) >= 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    best_n = 1
    best_colors = []
    
    # We will try to build up to n=29
    # Strategy: Greedy construction with restarts
    
    def greedy_build(target_n):
        """
        Try to build a valid 5-coloring of K_target_n using greedy edge assignment.
        Returns (colors, success)
        """
        n = 1
        cm = [[-1] * target_n for _ in range(target_n)]
        
        # We build the graph vertex by vertex.
        # For each new vertex v, we assign colors to edges (v, u) for u < v.
        # We maintain the state of common neighbors.
        
        # State: common_count[c][u][v] = number of common neighbors of u and v in color c
        # This is O(5 * n^2) memory, which is fine for n=29.
        common_count = [[[0] * target_n for _ in range(target_n)] for _ in range(5)]
        
        # adj[c][u] = set of neighbors of u in color c
        adj = [[set() for _ in range(target_n)] for _ in range(5)]
        
        # To speed up, we can just store the sets and compute intersections on the fly
        # or maintain the count. Maintaining count is faster for the heuristic.
        
        for v in range(1, target_n):
            # We need to assign colors to edges (v, u) for u in 0..v-1
            # Let's do it in random order to break symmetry and allow different paths
            us = list(range(v))
            random.shuffle(us)
            
            assigned_colors = {}
            
            # Heuristic: Assign one by one, choosing the color that minimizes
            # the increase in "forbidden pairs" (pairs with exactly 1 common neighbor)
            # and strictly avoids creating C4s (pairs with >= 2 common neighbors).
            
            # We need to be careful: when we assign color c to (v, u), 
            # it affects common_count for pairs (u, w) and (v, w) where w is connected to v or u in color c.
            # Actually, it's simpler to just check validity and use a simple heuristic for ordering.
            
            # Let's use a backtracking-lite approach: 
            # Try to assign all edges for vertex v.
            # If we get stuck, we can try a different order or random restart.
            
            # For simplicity and speed, let's try a pure greedy with random tie-breaking.
            # If it fails, we return failure and the outer loop restarts.
            
            # Pre-calculate potential conflicts?
            # For each u in us, and each color c in 0..4:
            #   If we assign c to (v, u):
            #     For each w in adj[c][u]:
            #       If w < v: # w is already in the graph
            #         common_count[c][v][w] would increase by 1.
            #         If common_count[c][v][w] was 1, it becomes 2 -> C4! Forbidden.
            #         If common_count[c][v][w] was 0, it becomes 1 -> New Forbidden Pair.
            #     For each w in adj[c][v] (currently assigned neighbors of v in color c):
            #       common_count[c][u][w] would increase by 1.
            #       If common_count[c][u][w] was 1, it becomes 2 -> C4! Forbidden.
            #       If common_count[c][u][w] was 0, it becomes 1 -> New Forbidden Pair.
            
            # This logic depends on the current state of adj[c][v] and common_count.
            # Since we assign sequentially, adj[c][v] grows.
            
            # Let's iterate through us.
            success = True
            for u in us:
                best_color = -1
                best_score = float('inf')
                best_new_forbidden = float('inf')
                
                for c in range(5):
                    # Check if assigning c to (v, u) is valid
                    valid = True
                    new_forbidden = 0
                    
                    # Check neighbors of u in color c
                    for w in adj[c][u]:
                        if w == v: continue # v is not in adj yet
                        # w is an existing vertex (w < v)
                        # Adding edge (v, u) with color c means v and u share w in color c.
                        # So common_count[c][v][w] increases by 1.
                        # We need to know the current common_count[c][v][w].
                        # Note: common_count is only updated when we finalize an edge.
                        # So we must simulate the update or maintain a temporary state.
                        
                        # Since we are doing greedy, we can just check the sets.
                        # Common neighbors of v and w in color c currently:
                        # adj[c][v] & adj[c][w]
                        # If len(adj[c][v] & adj[c][w]) >= 1, then adding u (which is in adj[c][w])
                        # makes it >= 2?
                        # Wait. u is in adj[c][w]. v is NOT in adj[c][w] yet (we are adding it).
                        # So currently, common neighbors of v and w in color c are adj[c][v] & adj[c][w].
                        # After adding (v, u) with color c:
                        # adj[c][v] will contain u.
                        # New common neighbors of v and w: (adj[c][v] U {u}) & adj[c][w]
                        # = (adj[c][v] & adj[c][w]) U ({u} & adj[c][w])
                        # Since u in adj[c][w], the size increases by 1 if u was not already in adj[c][v] (which it isn't).
                        # So if len(adj[c][v] & adj[c][w]) == 1, it becomes 2 -> C4.
                        # If len(adj[c][v] & adj[c][w]) == 0, it becomes 1 -> New Forbidden Pair.
                        
                        common_vw = len(adj[c][v] & adj[c][w])
                        if common_vw >= 1:
                            valid = False
                            break
                        else: # common_vw == 0
                            new_forbidden += 1
                    
                    if not valid:
                        continue
                    
                    # Check neighbors of v in color c (already assigned)
                    for w in adj[c][v]:
                        # w is an existing vertex (w < v)
                        # Adding edge (v, u) with color c means u and v share w in color c? No.
                        # It means u and v are connected.
                        # Does it affect common_count[c][u][w]?
                        # u and w share neighbors in color c.
                        # Currently: adj[c][u] & adj[c][w].
                        # After adding (v, u) with color c:
                        # adj[c][u] will contain v.
                        # New common neighbors of u and w: (adj[c][u] U {v}) & adj[c][w]
                        # = (adj[c][u] & adj[c][w]) U ({v} & adj[c][w])
                        # Since v in adj[c][w] (because w in adj[c][v] and graph is undirected),
                        # the size increases by 1 if v was not already in adj[c][u].
                        
                        common_uw = len(adj[c][u] & adj[c][w])
                        if common_uw >= 1:
                            valid = False
                            break
                        else: # common_uw == 0
                            new_forbidden += 1
                    
                    if valid:
                        # Score: minimize new_forbidden
                        # Tie-break: random
                        score = new_forbidden
                        if score < best_score or (score == best_score and random.random() < 0.5):
                            best_score = score
                            best_color = c
                
                if best_color == -1:
                    return None, False
                
                # Assign best_color to (v, u)
                assigned_colors[u] = best_color
                c = best_color
                adj[c][v].add(u)
                adj[c][u].add(v)
            
            # If we reached here, we successfully assigned all edges for vertex v
            # Update best_n if this is a complete graph
            if v == target_n - 1:
                # Flatten to list
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        colors.append(cm[i][j] if cm[i][j] != -1 else -1)
                # Wait, we didn't update cm. Let's just return the adj sets or reconstruct.
                # For efficiency, we can just return success and the final n.
                # The caller can reconstruct if needed, or we can return the colors.
                # Let's return the colors to be safe.
                # We need to fill cm.
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        # Find color
                        # This is slow. Let's maintain cm in the loop.
                        pass
                
                # Let's refactor to maintain cm.
                return None, True # Placeholder
        
        return None, False

    # The above greedy_build is a bit messy with state management.
    # Let's write a cleaner version inside construct.
    
    def try_construct_n(n_target):
        cm = [[-1] * n_target for _ in range(n_target)]
        adj = [[set() for _ in range(n_target)] for _ in range(5)]
        
        for v in range(1, n_target):
            us = list(range(v))
            random.shuffle(us)
            
            for u in us:
                best_color = -1
                best_new_forbidden = float('inf')
                
                for c in range(5):
                    valid = True
                    new_forbidden = 0
                    
                    # Check conflicts with neighbors of u in color c
                    for w in adj[c][u]:
                        if w >= v: continue # Should not happen
                        # Check if v and w already have a common neighbor in color c
                        # If yes, adding u (which is common to v and w via color c edge) creates C4
                        # Common neighbors of v and w in color c: adj[c][v] & adj[c][w]
                        # Note: u is in adj[c][w]. Is u in adj[c][v]? No, not yet.
                        # So if len(adj[c][v] & adj[c][w]) >= 1, then u becomes the 2nd common neighbor.
                        common_vw = len(adj[c][v] & adj[c][w])
                        if common_vw >= 1:
                            valid = False
                            break
                        else:
                            new_forbidden += 1
                    
                    if not valid:
                        continue
                    
                    # Check conflicts with neighbors of v in color c
                    for w in adj[c][v]:
                        if w >= v: continue # Should not happen
                        # Check if u and w already have a common neighbor in color c
                        # Common neighbors of u and w in color c: adj[c][u] & adj[c][w]
                        # v is in adj[c][w]. Is v in adj[c][u]? No, not yet.
                        common_uw = len(adj[c][u] & adj[c][w])
                        if common_uw >= 1:
                            valid = False
                            break
                        else:
                            new_forbidden += 1
                    
                    if valid:
                        if new_forbidden < best_new_forbidden or (new_forbidden == best_new_forbidden and random.random() < 0.5):
                            best_new_forbidden = new_forbidden
                            best_color = c
                
                if best_color == -1:
                    return None
                
                c = best_color
                cm[v][u] = c
                cm[u][v] = c
                adj[c][v].add(u)
                adj[c][u].add(v)
        
        # Success
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(cm[i][j])
        return colors

    # Main loop
    # Start from a known good small n, or just 1.
    # We know n=26 is possible. Let's try to jump to 26 quickly?
    # Greedy might get stuck early. 
    # Let's try to build up to 29.
    
    # To improve success rate, we can use a "warm start" if we have one.
    # But let's stick to pure greedy with restarts.
    
    # We'll try to build n=29. If it fails, we try n=28, etc.
    # Or we build incrementally: try to add 1 vertex. If fail, restart the whole process for that n.
    
    # Let's try to reach 29.
    for n_target in range(29, 1, -1):
        if time.time() > deadline:
            break
        
        success = False
        for attempt in range(50): # 50 restarts per n
            if time.time() > deadline:
                break
            colors = try_construct_n(n_target)
            if colors is not None:
                # Verify just in case
                if _is_valid_coloring(n_target, colors):
                    best_n = n_target
                    best_colors = colors
                    success = True
                    break
        
        if success:
            # If we found a valid coloring for n_target, we can try to go higher?
            # No, we are iterating downwards. If we found 29, we break.
            # If we found 28, we break the loop? No, we want max n.
            # So if we found a valid n, we break the outer loop because we are going downwards.
            break
    else:
        # If we didn't find any n > 1, we should have at least n=2 or something.
        # The loop goes 29 down to 2. If none worked, best_n is 1.
        pass
    
    # If best_n is still 1, try to find a small valid one (e.g., n=2 is always valid)
    if best_n < 2:
        best_n = 2
        best_colors = [0]
        
    return {'n': best_n, 'colors': best_colors}
