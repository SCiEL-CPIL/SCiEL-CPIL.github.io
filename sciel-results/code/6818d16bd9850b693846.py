import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    N = 20
    C = 20
    
    # graph[u][v] = colour, -1 if no edge
    # We will maintain a list of edges to return
    
    def is_valid_addition(graph, u, v, c):
        """
        Check if adding edge (u, v) with colour c is valid.
        Assumption: edge (u, v) does not currently exist.
        """
        # A 4-cycle containing (u, v) is u-a-b-v-u.
        # Neighbors of u: a
        # Neighbors of v: b
        # Must have edge (a, b).
        # Colours: c, graph[u][a], graph[a][b], graph[b][v]
        # Must be all distinct.
        
        # Optimization: Pre-compute neighbors? 
        # With N=20, we can just iterate.
        
        # Get neighbors of u and v
        # Note: graph is symmetric
        
        for a in range(N):
            if a == u or a == v:
                continue
            if graph[u][a] == -1:
                continue
            
            for b in range(N):
                if b == u or b == v:
                    continue
                if b == a:
                    continue
                if graph[v][b] == -1:
                    continue
                if graph[a][b] == -1:
                    continue
                
                # Found a potential 4-cycle u-a-b-v-u
                # Check colours
                c1 = c
                c2 = graph[u][a]
                c3 = graph[a][b]
                c4 = graph[v][b]
                
                # Distinctness check
                if len({c1, c2, c3, c4}) < 4:
                    return False
        
        return True

    def get_all_edges(graph):
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if graph[u][v] != -1:
                    edges.append([u, v, graph[u][v]])
        return edges

    best_edge_count = 0
    best_graph = None
    best_edges = None
    
    # We will run several iterations of randomized construction
    # Since N is small, we can afford to be thorough.
    
    for iteration in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        graph = [[-1] * N for _ in range(N)]
        
        # List of all possible edges
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        
        # Phase 1: Greedy Addition
        # We iterate through shuffled edges and try to add them.
        # We might need multiple passes as adding one edge might block another,
        # but the order matters.
        
        progress = True
        # Limit passes to avoid infinite loops if no progress
        max_passes = 10
        passes = 0
        while progress and passes < max_passes:
            progress = False
            passes += 1
            for u, v in all_edges:
                if graph[u][v] != -1:
                    continue
                
                # Try to find a valid colour
                # Shuffle colours to randomize choice
                colours = list(range(C))
                random.shuffle(colours)
                
                added = False
                for c in colours:
                    if is_valid_addition(graph, u, v, c):
                        graph[u][v] = c
                        graph[v][u] = c
                        progress = True
                        added = True
                        break
                
                if not added:
                    continue
                
                # Check time
                if time.time() > deadline:
                    return get_all_edges(best_graph) if best_graph else []

        # Phase 2: Local Optimization (Colour Swapping)
        # Try to change colours of existing edges to allow new edges.
        # We can try to swap colours of two edges, or just change one.
        # Changing one edge colour is simpler.
        
        current_edge_count = sum(1 for u in range(N) for v in range(u+1, N) if graph[u][v] != -1)
        
        opt_progress = True
        opt_passes = 0
        max_opt_passes = 5
        while opt_progress and opt_passes < max_opt_passes and time.time() < deadline:
            opt_progress = False
            opt_passes += 1
            
            # Try to change colour of a random existing edge
            existing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if graph[u][v] != -1]
            if not existing_edges:
                break
            
            random.shuffle(existing_edges)
            
            for u, v in existing_edges:
                old_c = graph[u][v]
                # Try other colours
                for new_c in range(C):
                    if new_c == old_c:
                        continue
                    
                    # Temporarily change
                    graph[u][v] = new_c
                    graph[v][u] = new_c
                    
                    # Check if this change is valid for all 4-cycles containing (u, v)
                    valid_change = True
                    for a in range(N):
                        if a == u or a == v or graph[u][a] == -1:
                            continue
                        for b in range(N):
                            if b == u or b == v or b == a or graph[v][b] == -1 or graph[a][b] == -1:
                                continue
                            
                            c1 = new_c
                            c2 = graph[u][a]
                            c3 = graph[a][b]
                            c4 = graph[v][b]
                            
                            if len({c1, c2, c3, c4}) < 4:
                                valid_change = False
                                break
                        if not valid_change:
                            break
                    
                    if valid_change:
                        # Keep the change
                        opt_progress = True
                        # After changing, check if we can add any new edges
                        # This might be expensive, so we do a quick check
                        # Try to add any missing edge with any colour
                        added_new = False
                        for u2, v2 in all_edges:
                            if graph[u2][v2] != -1:
                                continue
                            for c2 in range(C):
                                if is_valid_addition(graph, u2, v2, c2):
                                    graph[u2][v2] = c2
                                    graph[v2][u2] = c2
                                    added_new = True
                                    break
                            if added_new:
                                break
                        if added_new:
                            # Update count
                            current_edge_count += 1
                    else:
                        # Revert
                        graph[u][v] = old_c
                        graph[v][u] = old_c

        # Update best
        current_edges = get_all_edges(graph)
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_graph = [row[:] for row in graph]
            best_edges = current_edges
            
        # If we reached a very high number, we can stop early
        # Max edges is 190. If we get close, it's likely optimal or near-optimal.
        if best_edge_count > 150:
            break

    if best_edges:
        return best_edges
    else:
        # Fallback: return a valid sparse graph if nothing else worked
        # A star is always valid
        return [[0, v, 0] for v in range(1, N)]
