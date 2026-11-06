import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin before 150s limit

    # Hard-coded known valid 4-coloring of K_17 (cyclic construction)
    # Distance assignment for n=17: distances 1..8
    # We use a known valid cyclic coloring
    def get_known_17():
        n = 17
        # Cyclic coloring with specific distance assignments
        # dist_assignment[d-1] = color for distance d
        # A known valid assignment for C6-free 4-coloring of K_17
        # Let's use the one from literature: 
        # Actually, let me construct one that works.
        # For n=17, we need 4 colors for distances 1-8.
        # Let's try: [0,1,2,3,0,1,2,3] - might have C6s
        # Let's try: [0,1,2,3,1,0,3,2]
        # We need to verify. Let's just use a search to find one quickly.
        # But for safety, let's embed a verified one.
        
        # Actually, the most reliable approach: 
        # The known R_4(C6) > 17 coloring exists. 
        # Let me use a specific cyclic one that is known to work.
        # Distance assignment: [0, 1, 2, 3, 0, 2, 1, 3]
        # Let's verify this in code at startup if time permits, 
        # but for the hard-coded fallback, I'll use a construction 
        # that I can verify is C6-free.
        
        # Actually, let me just use the backtracking solver to find n=17
        # quickly, and if it fails, return n=9.
        # But to guarantee score >= 17, let me embed a known valid one.
        
        # Known valid cyclic 4-coloring of K_17 without C6:
        # This is a specific assignment that works:
        dist_assign = [0, 1, 2, 3, 1, 0, 3, 2]
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d_min = min(d, n - d)
                c = dist_assign[d_min - 1]
                colors.append(c)
        
        # Verify it
        if verify_coloring_fast(n, colors):
            return {'n': n, 'colors': colors}
        
        # If the hard-coded one doesn't work, try another
        dist_assign2 = [0, 1, 2, 3, 2, 0, 3, 1]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d_min = min(d, n - d)
                c = dist_assign2[d_min - 1]
                colors.append(c)
        if verify_coloring_fast(n, colors):
            return {'n': n, 'colors': colors}
        
        # Fallback to n=9
        n = 9
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = j - i
                d_min = min(d, n - d)
                colors.append((d_min - 1) % 4)
        return {'n': n, 'colors': colors}

    def verify_coloring_fast(n, colors):
        """Fast verification using bitset C6 detection."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        for c in range(4):
            # Build adjacency as bitmasks
            adj = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    idx += 1
            
            if has_c6_bitset(adj, n):
                return False
        return True

    def has_c6_bitset(adj, n):
        """Check if graph with bitmask adjacency contains a C6.
        For each edge (u,v), check if there's a path of length 4 from u to v
        through 4 distinct internal vertices.
        """
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    continue
                # Find neighbors of u (excluding v) and neighbors of v (excluding u)
                neighbors_u = adj[u] & ~(1 << v) & ~(1 << u)
                neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
                
                # We need a path u - a - b - c - d - v where a,b,c,d are distinct
                # and not u,v
                # a in neighbors_u, d in neighbors_v, b adjacent to a, c adjacent to d, b-c edge
                # Let's iterate over possible a and d
                a_list = [i for i in range(n) if (neighbors_u >> i) & 1]
                d_list = [i for i in range(n) if (neighbors_v >> i) & 1]
                
                for a in a_list:
                    for d in d_list:
                        if a == d:
                            continue
                        # Need b adjacent to a (b != u, v, a, d) and c adjacent to d (c != u, v, a, d)
                        # and b-c edge, b != c
                        neighbors_a = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                        neighbors_d = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                        
                        b_list = [i for i in range(n) if (neighbors_a >> i) & 1]
                        for b in b_list:
                            if b == d:
                                continue
                            # c must be in neighbors_d and adjacent to b
                            c_mask = neighbors_d & adj[b]
                            if c_mask:
                                # There exists a c
                                return True
        return False

    def solve_color_class(n, forbidden_mask, max_time, seed=0):
        """
        Backtracking solver to find a C6-free graph on n vertices.
        forbidden_mask: bitmask of edges that must NOT be in this color class
                       (i.e., edges already assigned to other colors)
        Returns adjacency bitmask list or None.
        """
        random.seed(seed)
        deadline = time.time() + max_time
        
        # Edge list in lex order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        num_edges = len(edges)
        
        # Build initial adjacency
        adj = [0] * n
        # Start with empty graph, add edges one by one
        
        # For efficiency, process edges in a smart order
        # Let's use a randomized order for diversity
        edge_indices = list(range(num_edges))
        random.shuffle(edge_indices)
        
        # But for determinism in finding a solution, let's use lex order
        # Actually, let's try both: first lex, then random
        
        def backtrack(edge_idx, adj_state):
            if time.time() > deadline:
                return None
            if edge_idx == num_edges:
                return adj_state[:]
            
            i, j = edges[edge_idx]
            # Try adding edge (i,j)
            # Check if adding (i,j) creates a C6
            adj_state[i] |= (1 << j)
            adj_state[j] |= (1 << i)
            
            # Check if this creates a C6
            if not creates_c6(adj_state, i, j, n):
                result = backtrack(edge_idx + 1, adj_state)
                if result is not None:
                    return result
            
            # Backtrack
            adj_state[i] &= ~(1 << j)
            adj_state[j] &= ~(1 << i)
            
            # Try not adding edge (i,j)
            result = backtrack(edge_idx + 1, adj_state)
            if result is not None:
                return result
            
            return None
        
        def creates_c6(adj, u, v, n):
            """Check if adding edge (u,v) to adj creates a C6 containing (u,v)."""
            # A C6 containing edge (u,v) has form u-a-b-c-d-v-u
            # where a,b,c,d are distinct and not u,v
            neighbors_u = adj[u] & ~(1 << v) & ~(1 << u)
            neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
            
            a_list = [i for i in range(n) if (neighbors_u >> i) & 1]
            d_list = [i for i in range(n) if (neighbors_v >> i) & 1]
            
            for a in a_list:
                for d in d_list:
                    if a == d:
                        continue
                    # Need b adjacent to a, c adjacent to d, b-c edge
                    # b,c distinct from u,v,a,d and from each other
                    neighbors_a = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                    neighbors_d = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                    
                    b_list = [i for i in range(n) if (neighbors_a >> i) & 1]
                    for b in b_list:
                        if b == d:
                            continue
                        c_mask = neighbors_d & adj[b]
                        if c_mask:
                            return True
            return False
        
        # Start backtracking
        adj_state = [0] * n
        result = backtrack(0, adj_state)
        return result

    def solve_full_coloring(n, max_time, seed=0):
        """
        Solve for a complete 4-coloring of K_n where no color class has a C6.
        Uses sequential color class construction.
        """
        random.seed(seed)
        deadline = time.time() + max_time
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        num_edges = len(edges)
        
        # colors[i] = color of edge i, -1 for unassigned
        colors = [-1] * num_edges
        
        # Process color classes one by one: 0, 1, 2, 3
        # For each color class, find a C6-free subgraph of the remaining edges
        
        for c in range(4):
            if time.time() > deadline:
                return None
            
            # Find all unassigned edges
            unassigned = [i for i in range(num_edges) if colors[i] == -1]
            
            if not unassigned:
                break
            
            # Build forbidden mask: edges already assigned to other colors
            # We need to find a C6-free subset of unassigned edges for color c
            
            # Use backtracking on the unassigned edges
            # For efficiency, we want a "dense" C6-free graph to leave fewer edges for later colors
            
            # Let's use a greedy approach first: try to add edges in lex order, 
            # skip if it creates a C6
            adj = [0] * n
            color_edges = []
            
            for edge_idx in unassigned:
                i, j = edges[edge_idx]
                # Try adding this edge
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
                if not creates_c6_global(adj, i, j, n):
                    color_edges.append(edge_idx)
                    colors[edge_idx] = c
                else:
                    # Backtrack
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            
            # Check if all edges are assigned
            remaining = [i for i in range(num_edges) if colors[i] == -1]
            if remaining:
                # Greedy didn't work, need to be smarter
                # For now, let's try a different order
                # This is a simplified version; a full solver would be more complex
                pass
        
        # Check if all edges are colored
        if any(c == -1 for c in colors):
            return None
        
        return colors
    
    def creates_c6_global(adj, u, v, n):
        """Check if adding edge (u,v) to adj creates any C6 (not just one containing u,v)."""
        # Simplified: just check C6s containing edge (u,v)
        # This is not fully correct but works as a heuristic
        neighbors_u = adj[u] & ~(1 << v) & ~(1 << u)
        neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
        
        a_list = [i for i in range(n) if (neighbors_u >> i) & 1]
        d_list = [i for i in range(n) if (neighbors_v >> i) & 1]
        
        for a in a_list:
            for d in d_list:
                if a == d:
                    continue
                neighbors_a = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                neighbors_d = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                
                b_list = [i for i in range(n) if (neighbors_a >> i) & 1]
                for b in b_list:
                    if b == d:
                        continue
                    c_mask = neighbors_d & adj[b]
                    if c_mask:
                        return True
        return False

    # Strategy:
    # 1. Try to find n=18 with a quick search
    # 2. Fall back to known n=17
    # 3. If that fails, return n=9
    
    # First, try the hard-coded n=17
    result = get_known_17()
    if result['n'] >= 17:
        return result
    
    # Try to find n=18 with a time-limited search
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Try a simple randomized approach for n=18
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            colors = solve_full_coloring(18, min(remaining_time - 10, 20), seed=attempt)
            if colors is not None and verify_coloring_fast(18, colors):
                return {'n': 18, 'colors': colors}
    
    # Final fallback
    n = 9
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            d_min = min(d, n - d)
            colors.append((d_min - 1) % 4)
    return {'n': n, 'colors': colors}
