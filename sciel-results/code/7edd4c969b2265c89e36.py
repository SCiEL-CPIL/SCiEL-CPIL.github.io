"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    n = 15
    
    # Helper: check if a graph (given as adjacency bitmask) is valid
    # Valid means: no simple 3-edge path s-u-v-t has deg(s) == deg(t)
    def is_valid(adj):
        """
        adj: list of n ints, where adj[i] is a bitmask of neighbors of vertex i
        Returns True if the graph satisfies the constraint.
        """
        # Compute degrees
        deg = [bin(adj[i]).count('1') for i in range(n)]
        
        # For every edge (u, v), and for every s in N(u) \ {v}, t in N(v) \ {u},
        # with s != t, we need deg[s] != deg[t].
        # Actually the path is s-u-v-t where edges are (s,u), (u,v), (v,t).
        # s and t must be distinct from each other and from u, v (simple path).
        # s is a neighbor of u (s != v since edge (u,v) exists and s-u is an edge, 
        #   but wait: s is in N(u). If s==v, then the path is v-u-v-t which is not simple.
        #   So s must be in N(u) \ {v}.
        # t is in N(v) \ {u}.
        # Also s != t.
        
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                if not (adj[u] >> v) & 1:
                    continue
                # (u,v) is an edge
                # s must be in N(u) \ {v}
                # t must be in N(v) \ {u}
                # s != t
                # deg[s] != deg[t]
                
                # Get neighbors of u excluding v
                nu = adj[u] & ~(1 << v)
                # Get neighbors of v excluding u
                nv = adj[v] & ~(1 << u)
                
                # Iterate over s in nu
                s_mask = nu
                while s_mask:
                    s = (s_mask & -s_mask).bit_length() - 1
                    s_mask &= s_mask - 1
                    # Iterate over t in nv
                    t_mask = nv & ~(1 << s)  # t != s
                    while t_mask:
                        t = (t_mask & -t_mask).bit_length() - 1
                        t_mask &= t_mask - 1
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Helper: count edges from adj
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # Helper: convert adj list to edge list
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask &= mask - 1
        return edges
    
    # Start with K_{7,8}
    adj = [0] * n
    for a in range(7):
        for b in range(7, 15):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    best_adj = adj[:]
    best_edges = count_edges(adj)
    
    # Local search: try to add edges, remove edges, or swap
    # Strategy: greedy fill - try adding each missing edge if it keeps validity
    # Then try removing an edge and re-greedy-filling to escape local optima
    
    def greedy_fill(adj):
        """Try to add every possible missing edge if valid."""
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        # Try adding edge (u, v)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_valid(adj):
                            changed = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
        return adj
    
    # Initial greedy fill (should be no-op since K_{7,8} is already max in its family,
    # but let's verify)
    adj = greedy_fill(adj)
    if count_edges(adj) > best_edges:
        best_edges = count_edges(adj)
        best_adj = adj[:]
    
    # Local search with random restarts and edge removal
    random.seed(42)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Start from best_adj or a perturbed version
        if iteration == 0:
            cur_adj = best_adj[:]
        else:
            # Perturb: remove 1-3 random edges
            cur_adj = best_adj[:]
            num_removals = random.randint(1, 3)
            for _ in range(num_removals):
                # Pick a random edge
                edges_list = adj_to_edges(cur_adj)
                if not edges_list:
                    break
                e = random.choice(edges_list)
                u, v = e
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
        
        # Greedy fill
        cur_adj = greedy_fill(cur_adj)
        cur_edges = count_edges(cur_adj)
        
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = cur_adj[:]
    
    # Final check and return
    if not is_valid(best_adj):
        # Fallback to K_{7,8}
        adj = [0] * n
        for a in range(7):
            for b in range(7, 15):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        best_adj = adj
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
