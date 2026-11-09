"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    n = P['n']
    vertices = list(range(n))
    total_edges = n * (n - 1) // 2  # 276
    
    # Build position map
    pos = {}
    for i, edge in enumerate(P['edge_order']):
        u, v = edge
        if u > v:
            u, v = v, u
        pos[(u, v)] = i + 1  # 1-based index
    
    # Precompute edge positions in a 2D array for fast access
    edge_pos = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            edge_pos[i][j] = edge_pos[j][i] = pos[(i, j)]
    
    def score_cycle(cycle):
        """Compute the max edge position in a cycle."""
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            p = edge_pos[u][v]
            if p > max_pos:
                max_pos = p
        return max_pos
    
    def find_ham_cycle_iterative(adj, K, time_limit):
        """
        Iterative DFS with aggressive pruning to find Hamilton cycle
        in subgraph with edges having position <= K.
        Returns the cycle as a list, or None.
        """
        # Quick degree check
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        # Check if 0 can connect to complete the cycle
        # We fix vertex 0 as start and end
        
        # Build neighbor lists sorted by degree (MRV heuristic)
        neighbor_order = [list(adj[v]) for v in range(n)]
        for v in range(n):
            neighbor_order[v].sort(key=lambda x: len(adj[x]))
        
        # Iterative DFS
        path = [0]
        used = [False] * n
        used[0] = True
        # Stack of (current_vertex, next_neighbor_index)
        stack = [(0, 0)]
        
        while stack:
            if time.time() > time_limit:
                return None
            
            cur, ni = stack[-1]
            
            if len(path) == n:
                # Check if we can close the cycle
                if 0 in adj[cur]:
                    return path[:]
                # Can't close, backtrack
                path.pop()
                used[cur] = False
                stack.pop()
                continue
            
            # Find next unvisited neighbor
            found = False
            while ni < len(neighbor_order[cur]):
                nxt = neighbor_order[cur][ni]
                ni += 1
                if not used[nxt]:
                    # Pruning checks
                    if len(path) == n - 1:
                        # This will be the last vertex, must connect to 0
                        if 0 not in adj[nxt]:
                            continue
                        # Also check that the remaining unvisited vertex (if any)
                        # can connect properly - but at this point only nxt is unvisited
                        # among those not in path
                        break
                    
                    # Connectivity pruning: check unvisited vertices remain connected
                    # and at least one connects to the path
                    unvisited = [v for v in range(n) if not used[v] and v != nxt]
                    if unvisited:
                        # BFS/DFS to check connectivity of unvisited subgraph
                        visited_u = set()
                        stack_u = [unvisited[0]]
                        visited_u.add(unvisited[0])
                        while stack_u:
                            u = stack_u.pop()
                            for w in adj[u]:
                                if w in unvisited and w not in visited_u:
                                    visited_u.add(w)
                                    stack_u.append(w)
                        if len(visited_u) != len(unvisited):
                            continue  # Disconnected, skip this branch
                        
                        # Check at least one unvisited vertex connects to path
                        path_set = set(path)
                        has_conn = False
                        for v in unvisited:
                            for w in adj[v]:
                                if w in path_set:
                                    has_conn = True
                                    break
                            if has_conn:
                                break
                        if not has_conn:
                            continue
                    
                    # Accept this neighbor
                    found = True
                    break
            
            if not found:
                # No more neighbors, backtrack
                path.pop()
                used[cur] = False
                stack.pop()
                if stack:
                    parent, _ = stack[-1]
                    stack[-1] = (parent, ni)  # Update parent's next index
                continue
            
            # Move to next vertex
            used[nxt] = True
            path.append(nxt)
            stack[-1] = (cur, ni)  # Update current's next index
            stack.append((nxt, 0))
        
        return None
    
    def ils_refine(initial_cycle, time_limit, rng):
        """
        Iterated Local Search with Or-Opt moves to refine a Hamilton cycle.
        Returns the best cycle found.
        """
        best = initial_cycle[:]
        best_score = score_cycle(best)
        current = best[:]
        current_score = best_score
        
        # Or-Opt: move a segment of length 1, 2, or 3
        def or_opt_move(cycle, rng):
            """Apply a random Or-Opt move."""
            seg_len = rng.choice([1, 2, 3])
            if seg_len >= len(cycle):
                seg_len = 1
            i = rng.randint(0, len(cycle) - 1)
            j = (i + seg_len) % len(cycle)
            
            # Extract segment
            if j > i:
                segment = cycle[i:j]
                remaining = cycle[:i] + cycle[j:]
            else:
                # Wraps around
                segment = cycle[i:] + cycle[:j]
                remaining = cycle[j:i]
            
            # Insert at random position
            if not remaining:
                return segment[:]
            
            insert_pos = rng.randint(0, len(remaining))
            new_cycle = remaining[:insert_pos] + segment + remaining[insert_pos:]
            return new_cycle
        
        # 2-opt move
        def two_opt_move(cycle, rng):
            """Apply a random 2-opt move."""
            i, j = rng.sample(range(len(cycle)), 2)
            if j < i:
                i, j = j, i
            return cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
        
        # Or-Opt + 2-opt combined
        def random_move(cycle, rng):
            if rng.random() < 0.5:
                return or_opt_move(cycle, rng)
            else:
                return two_opt_move(cycle, rng)
        
        # Local search: hill climbing
        def local_search(cycle, time_limit):
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                score = score_cycle(cycle)
                for _ in range(100):
                    if time.time() > time_limit:
                        break
                    new_cycle = random_move(cycle, rng)
                    new_score = score_cycle(new_cycle)
                    if new_score < score:
                        cycle = new_cycle
                        score = new_score
                        improved = True
            return cycle, score
        
        # Main ILS loop
        while time.time() < time_limit:
            # Local search
            current, current_score = local_search(current, time_limit)
            
            if current_score < best_score:
                best = current[:]
                best_score = current_score
            
            # Perturbation
            if time.time() > time_limit:
                break
            current = random_move(current, rng)
            current = random_move(current, rng)  # Double perturbation
            current_score = score_cycle(current)
        
        return best, best_score
    
    # Phase 1: Binary search for the minimal K where a Hamilton cycle exists
    # Lower bound: for each vertex, the 2nd smallest incident edge position
    min_deg2 = 0
    for v in vertices:
        incident = []
        for u in vertices:
            if u != v:
                incident.append(edge_pos[v][u])
        incident.sort()
        if len(incident) >= 2:
            if incident[1] > min_deg2:
                min_deg2 = incident[1]
    
    best_cycle = None
    best_K = total_edges + 1
    
    # Binary search
    lo = min_deg2
    hi = total_edges
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build adjacency list for edges with pos <= mid
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if edge_pos[u][v] <= mid:
                    adj[u].append(v)
                    adj[v].append(u)
        
        # Check for Hamilton cycle with time limit
        cycle = find_ham_cycle_iterative(adj, mid, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Phase 2: If we found a cycle, try to improve it with ILS
    if best_cycle is not None:
        # Try to find a better cycle using ILS starting from our solution
        rng = random.Random(42)
        improved_cycle, improved_score = ils_refine(best_cycle, deadline, rng)
        if improved_score < best_K:
            best_cycle = improved_cycle
            best_K = improved_score
        return best_cycle
    
    # Phase 3: Fallback - use ILS from a random starting point
    rng = random.Random(123)
    random_cycle = list(range(n))
    random.shuffle(random_cycle)
    
    improved_cycle, improved_score = ils_refine(random_cycle, deadline, rng)
    return improved_cycle

# EVOLVE-BLOCK-END
