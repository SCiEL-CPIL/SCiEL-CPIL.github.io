"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 40
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Bitset representation of adjacency
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    adj = [0] * n
    
    def get_neighbors_mask(i):
        return adj[i]
    
    def expand_mask(mask, depth=1):
        """
        Compute the set of vertices reachable from 'mask' in exactly 'depth' steps.
        mask is an integer bitmask of starting vertices.
        Returns an integer bitmask of reachable vertices.
        """
        current_mask = mask
        for _ in range(depth):
            next_mask = 0
            # Iterate through set bits in current_mask
            # Since n=40, we can just loop 0..39
            # Optimization: use bit extraction
            m = current_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                # Index of this bit
                idx = (lsb.bit_length() - 1)
                # OR with neighbors of idx
                next_mask |= adj[idx]
                # Clear the bit
                m ^= lsb
            current_mask = next_mask
        return current_mask

    def is_c6_free(u, v):
        """
        Check if adding edge (u, v) creates a C6.
        This is true if there is a path of length 5 between u and v in the current graph.
        Path length 5: u -> x1 -> x2 -> x3 -> x4 -> v
        So v must be in the set of vertices reachable from u in 5 steps.
        Note: The path must be simple? 
        The definition of C6 requires a simple cycle. 
        If there is a walk of length 5, does it imply a simple path of length 5?
        In a simple graph, if there is a walk of length 5 from u to v, 
        and u, v are not adjacent, 
        we need to be careful. 
        However, for the purpose of "does adding (u,v) create a C6", 
        we are looking for a cycle of length 6.
        A cycle of length 6 corresponds to a path of length 5 in the graph *without* the edge (u,v).
        If the walk of length 5 is not simple, it contains a smaller cycle.
        If the graph is C6-free, can it have a walk of length 5 that doesn't correspond to a C6?
        Yes, e.g. u-a-b-a-c-v. This is a walk of length 5.
        Adding (u,v) creates a cycle u-a-b-a-c-v-u? No, that's not a simple cycle.
        The simple cycle would be u-a-c-v-u (length 4) if a-c and c-v exist?
        Actually, if a walk of length 5 exists, there is a path of length <= 5.
        If there is a path of length 5, adding the edge creates a cycle of length 6.
        Is it always a *simple* cycle?
        If the path is not simple, it repeats a vertex.
        Then there is a shorter path.
        If there is a path of length 5, there is a simple path of length 5?
        Not necessarily. You might have a path of length 5 that loops.
        But if there is *any* path of length 5, does it guarantee a C6?
        Consider u-x-y-z-w-v.
        If x=z, then u-x-y-x-w-v. Cycle u-x-y-x... no.
        If the walk is not simple, it contains a cycle.
        If the graph is C6-free, it has no cycles of length 6.
        But it might have cycles of length 3, 4, 5.
        If there is a walk of length 5, it might be composed of a triangle and an edge.
        Example: u-a-b-a-c-v. 
        Adding (u,v) creates u-a-b-a-c-v-u.
        This is not a simple cycle.
        However, if such a walk exists, does a simple path of length 5 exist?
        No.
        So `expand_mask` (which finds *any* walk) might give false positives.
        
        We need to check for a *simple* path of length 5.
        Or, more accurately, we need to check if adding (u,v) creates a *simple* cycle of length 6.
        
        Let's re-evaluate.
        If we use `expand_mask`, we check for walks.
        If a walk of length 5 exists, it implies a path of length <= 5.
        If the shortest path length is 5, then adding the edge creates a C6.
        If the shortest path length is < 5, adding the edge creates a smaller cycle (C3, C4, C5).
        But the graph is already C6-free. It might not be C4-free or C5-free.
        The task says "containing no six-cycle". It allows C3, C4, C5.
        So if adding (u,v) creates a C4, it is ALLOWED.
        We only reject if it creates a C6.
        
        So we need to check: Is there a path of length 5 between u and v?
        If yes, adding (u,v) creates a cycle of length 6.
        Is that cycle necessarily simple?
        If there is a walk of length 5, there is a path of length <= 5.
        If the path length is exactly 5, the cycle is length 6.
        If the path length is 4, the cycle is length 5 (allowed).
        If the path length is 3, the cycle is length 4 (allowed).
        If the path length is 2, the cycle is length 3 (allowed).
        
        So we reject (u,v) ONLY IF the shortest path distance between u and v is exactly 5.
        Wait. If there is a path of length 5 AND a path of length 3, adding (u,v) creates a C4 and a C6.
        The C6 exists. So we must reject.
        So we reject if there is *any* simple path of length 5.
        
        Does `expand_mask` detect simple paths?
        `expand_mask` finds vertices reachable by *any* walk.
        If a vertex is reachable by a walk of length 5, is it reachable by a simple path of length 5?
        No.
        However, if it is reachable by a walk of length 5, it is reachable by a path of length $d \le 5$.
        If $d=5$, we have a simple path of length 5 (the shortest path is simple).
        If $d<5$, we have a shorter path.
        So, if $v$ is in `expand_mask(N(u), 5)`, it means there is a walk of length 5.
        This implies there is a path of length $d \le 5$.
        If $d=5$, we have a C6.
        If $d<5$, we have a C$d+1$.
        We only care if $d=5$.
        
        How to check if $d=5$?
        We can compute the sets of vertices reachable in *exactly* $k$ steps?
        No, "exactly" with walks is tricky because you can stay put? No, simple graph, no self-loops.
        But you can go back and forth.
        
        Let's compute $S_k$ = set of vertices reachable from $u$ by a path of length *at most* $k$.
        $S_0 = \{u\}$
        $S_1 = N(u)$
        $S_2 = N(S_1)$
        ...
        $S_5 = N(S_4)$
        
        If $v \in S_5$ and $v \notin S_4$, then the shortest path is 5.
        If $v \in S_4$, the shortest path is $\le 4$.
        
        So:
        1. Compute $S_4$.
        2. If $v \in S_4$, return True (Allowed, creates C<=5).
        3. Compute $S_5$.
        4. If $v \in S_5$, return False (Forbidden, creates C6).
        5. Else return True.
        
        Wait, if $v \in S_5$ but not $S_4$, the shortest path is 5.
        Adding (u,v) creates a cycle of length 6.
        Is it simple? Yes, the shortest path is simple.
        
        So the check is:
        `S_4 = expand_from_set({u}, 4)`
        `if v in S_4: return True`
        `S_5 = expand_from_set({u}, 5)`
        `if v in S_5: return False`
        `return True`
        
        We can optimize `expand_from_set` to take a mask.
        """
        # Start with mask for u
        mask_u = 1 << u
        
        # Compute S_4
        s4 = mask_u
        for _ in range(4):
            next_s = 0
            m = s4
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                next_s |= adj[idx]
                m ^= lsb
            s4 = next_s
        
        # Check if v is in S_4
        if s4 & (1 << v):
            return True
        
        # Compute S_5 (one more step from S_4)
        s5 = 0
        m = s4
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s5 |= adj[idx]
            m ^= lsb
        
        # Check if v is in S_5
        if s5 & (1 << v):
            return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Phase 1: Randomized Greedy Initialization
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer

    for restart in range(50):
        random.seed(restart * 100 + 42)
        
        # Reset
        adj = [0] * n
        
        # Greedy
        candidates = list(all_edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if is_c6_free(u, v):
                add_edge(u, v)
        
        current_count = count_edges()
        
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges_list()
        
        # Phase 2: Local Search / ILS
        # Try to improve by swapping or adding if possible
        # Since greedy is maximal, we can't add. We must swap.
        # ILS: Perturb and Repair.
        
        for _ in range(200): # 200 ILS iterations per restart
            if time.time() - start_time > time_limit:
                break
            
            # Save state
            adj_backup = adj[:]
            
            # Perturb: Remove k random edges
            k = 3
            current_edges = get_edges_list()
            if len(current_edges) < k:
                break
            random.shuffle(current_edges)
            for i in range(k):
                u, v = current_edges[i]
                remove_edge(u, v)
            
            # Repair: Greedily add edges until stuck
            # Use a random order for non-edges? Or all edges?
            # To be fast, try adding random non-edges first, then scan all?
            # Scanning all 780 edges with fast check is fast enough.
            # 780 * ~100 ops = 78k ops. 
            # 200 * 50 restarts * 78k = 780M ops. Might be tight in Python.
            # Let's limit repair to a subset or optimize.
            
            # Optimized Repair: 
            # Maintain a list of non-edges? 
            # Or just iterate all edges and try to add if not present and valid.
            
            for u, v in all_edges:
                if not (adj[u] & (1 << v)):
                    if is_c6_free(u, v):
                        add_edge(u, v)
            
            new_count = count_edges()
            
            if new_count > best_count:
                best_count = new_count
                best_edges = get_edges_list()
            
            # Accept move if better or equal (to explore)?
            # Or always revert if not better?
            # Standard ILS: accept if better, or with probability p.
            # Here, we want to find the global max.
            # If new_count > current_count (before perturbation), we keep it.
            # But we lost current_count.
            # Let's keep a state variable.
            
            # Actually, the loop above overwrites adj.
            # We need to compare new_count with the count before perturbation.
            # Let's restructure.

    # Refine the ILS loop to track counts properly
    
    # Reset for final run
    adj = [0] * n
    current_edges = []
    
    # Initial Greedy
    random.seed(42)
    candidates = list(all_edges)
    random.shuffle(candidates)
    for u, v in candidates:
        if is_c6_free(u, v):
            add_edge(u, v)
            current_edges.append((u, v))
    
    current_count = count_edges()
    best_count = current_count
    best_edges = get_edges_list()
    
    # ILS
    for i in range(1000): # Up to 1000 ILS iterations
        if time.time() - start_time > time_limit:
            break
        
        adj_backup = adj[:]
        edges_backup = current_edges[:]
        
        # Perturb
        k = 5
        if len(current_edges) < k:
            break
        random.shuffle(current_edges)
        for j in range(k):
            u, v = current_edges[j]
            remove_edge(u, v)
            current_edges.remove((u, v))
        
        # Repair
        # Try to add edges
        # Use a random order of non-edges? 
        # We can just iterate all_edges and add if valid.
        # This is deterministic repair.
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                if is_c6_free(u, v):
                    add_edge(u, v)
                    current_edges.append((u, v))
        
        new_count = count_edges()
        
        if new_count > best_count:
            best_count = new_count
            best_edges = get_edges_list()
        
        # Accept if new_count >= current_count (before perturb)?
        # If new_count < current_count, revert.
        # If new_count == current_count, accept with prob 0.5 to escape?
        if new_count < current_count:
            # Revert
            adj = adj_backup
            current_edges = edges_backup
        else:
            # Keep new state
            current_count = new_count
            # adj and current_edges are already updated

    return best_edges
# EVOLVE-BLOCK-END
