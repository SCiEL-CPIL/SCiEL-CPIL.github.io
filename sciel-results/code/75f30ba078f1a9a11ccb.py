import random
import time
import math

def construct():
    deadline = time.time() + 115  # Safety margin
    
    def count_k4(adj, n):
        """Count number of K4s in the graph."""
        count = 0
        # A K4 is a set of 4 vertices {a,b,c,d} where all 6 edges exist.
        # Iterate over all 4-subsets. For n=35, C(35,4) = 52360.
        # This is fast enough to call occasionally, but for every step it might be slow.
        # We can optimize by only checking K4s involving the changed edge.
        # But for simplicity and robustness in the inner loop of SA, 
        # let's use a faster incremental check or just recompute if n is small.
        # Actually, for n=35, 52360 checks * 6 edge lookups = ~300k ops.
        # If we do 10k steps, that's 3 billion ops. Too slow in Python.
        
        # Better approach: Precompute triangles.
        # A K4 consists of 4 triangles.
        # Or: Iterate over pairs of vertices (u,v) that are connected.
        # Their common neighbors form a graph. Any edge in the common neighbor graph is a K4.
        # Common neighbors of (u,v): cn(u,v).
        # For each edge (w,x) in the subgraph induced by cn(u,v), we have a K4 {u,v,w,x}.
        
        # Let's implement a function that counts K4s efficiently enough for ~1000 steps/sec.
        # Or even better, use the penalty function incrementally?
        # No, let's stick to a reasonably fast full count first.
        
        # Optimization:
        # For each pair (u,v) with edge:
        #   Find common neighbors cn.
        #   Count edges in G[cn].
        # Sum these counts. Note that each K4 is counted 3 times?
        # K4 {a,b,c,d}:
        # Pairs: (a,b) -> cn={c,d} -> edge (c,d) exists. Count 1.
        # (a,c) -> cn={b,d} -> edge (b,d) exists. Count 1.
        # (a,d) -> cn={b,c} -> edge (b,c) exists. Count 1.
        # (b,c) -> cn={a,d} -> edge (a,d) exists. Count 1.
        # (b,d) -> cn={a,c} -> edge (a,c) exists. Count 1.
        # (c,d) -> cn={a,b} -> edge (a,b) exists. Count 1.
        # Total count = 6. So divide by 6.
        
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    # Find common neighbors
                    cn = []
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            cn.append(w)
                    # Count edges in cn
                    ec = 0
                    for i in range(len(cn)):
                        for j in range(i+1, len(cn)):
                            if adj[cn[i]][cn[j]]:
                                ec += 1
                    total += ec
        return total // 6

    def count_i6(adj, n):
        """Count number of independent 6-sets."""
        # Similar logic. An independent set is a set of vertices with no edges between them.
        # This is the clique number of the complement graph.
        # We can count cliques of size 6 in the complement graph.
        # Complement adj: c_adj[u][v] = 1 - adj[u][v] for u!=v, 0 for u==v.
        
        # Same algorithm as count_k4 but on complement.
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: # Edge in complement
                    # Find common non-neighbors in original (common neighbors in complement)
                    cn = []
                    for w in range(n):
                        if w != u and w != v and not adj[u][w] and not adj[v][w]:
                            cn.append(w)
                    # Count non-edges in original among cn (edges in complement)
                    ec = 0
                    for i in range(len(cn)):
                        for j in range(i+1, len(cn)):
                            if not adj[cn[i]][cn[j]]:
                                ec += 1
                    total += ec
        return total // 6

    def penalty(adj, n):
        """Return (k4_count, i6_count, total_penalty)."""
        k4 = count_k4(adj, n)
        i6 = count_i6(adj, n)
        # We want both to be 0.
        # Weight them equally? Or prioritize one?
        # Usually K4 is easier to break? Not necessarily.
        # Let's use sum.
        return k4 + i6

    def try_sa(n_target):
        """
        Run Simulated Annealing to find a valid graph on n_target vertices.
        """
        if n_target < 1:
            return None
            
        # Initialize graph
        # Start with a random graph with p=0.5? Or a known valid smaller graph extended?
        # Random p=0.5 is a good start.
        adj = [[0]*n_target for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        current_penalty = penalty(adj, n_target)
        
        # If we start lucky, return immediately
        if current_penalty == 0:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i][j]:
                        edges.append([i, j])
            return edges

        # SA Parameters
        # T_start should be high enough to accept bad moves.
        # Max penalty is bounded. For n=35, max K4 is C(35,4)=52360. Max I6 is C(35,6)=1.6M.
        # Let's set T_start = 1000.
        T = 1000.0
        T_min = 0.01
        alpha = 0.995 # Cooling rate
        # Steps per temperature? Or just run until time limit?
        # Let's run for a fixed number of steps or until time.
        
        best_adj = [row[:] for row in adj]
        best_penalty = current_penalty
        
        steps = 0
        max_steps = 50000 # Cap for this single SA run
        
        while T > T_min and steps < max_steps:
            # Pick a random edge to flip
            u = random.randint(0, n_target-1)
            v = random.randint(0, n_target-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip edge (u,v)
            old_val = adj[u][v]
            adj[u][v] = 1 - old_val
            adj[v][u] = adj[u][v]
            
            new_penalty = penalty(adj, n_target)
            delta = new_penalty - current_penalty
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = [row[:] for row in adj]
                    if best_penalty == 0:
                        edges = []
                        for i in range(n_target):
                            for j in range(i+1, n_target):
                                if best_adj[i][j]:
                                    edges.append([i, j])
                        return edges
            else:
                # Reject: revert
                adj[u][v] = old_val
                adj[v][u] = old_val
                
            T *= alpha
            steps += 1
            
            # Time check inside loop
            if time.time() > deadline - 5:
                break

        # Return best found if valid, else None
        if best_penalty == 0:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if best_adj[i][j]:
                        edges.append([i, j])
            return edges
        return None

    # Fallback: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    p_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_result = {'n': 17, 'edges': p_edges}

    # Try to improve from 17 upwards?
    # Or just try 35 directly?
    # SA on 35 might be hard to converge to 0.
    # Maybe start from a valid 17 and add vertices?
    # The prompt suggests SA on edge set.
    # Let's try multiple restarts of SA on n=35.
    
    # To increase chances, we can also try n=34, 33, etc.
    # But 35 is the target.
    
    # Optimization: The penalty function is expensive.
    # For n=35, count_k4 is ~50k pairs * avg cn size.
    # If p=0.5, cn size is ~17. Edges in cn is C(17,2)=136.
    # 50k * 136 = 6.8M ops per penalty call.
    # Python does ~10-50M simple ops/sec. So ~0.1-0.5s per penalty call.
    # 50k steps * 0.3s = 15000s. WAY TOO SLOW.
    
    # We need a faster penalty or incremental update.
    # Incremental update for edge flip (u,v):
    # 
    # Delta K4:
    # If we add edge (u,v):
    #   New K4s are formed by pairs (w,x) such that w,x are common neighbors of u and v?
    #   No. A K4 containing (u,v) must have two other vertices w,x such that
    #   w,x are connected to u and v, AND w,x are connected to each other.
    #   So, count edges in the common neighborhood of u and v.
    #   Let CN(u,v) = {w | adj[u][w] and adj[v][w]}.
    #   New K4s = number of edges in G[CN(u,v)].
    #   
    #   If we remove edge (u,v):
    #   We destroy K4s that contain (u,v).
    #   Same count: edges in G[CN(u,v)].
    #   
    #   So delta_K4 = +/- count_edges_in_induced_subgraph(CN(u,v)).
    #   Calculating this requires finding CN(u,v) and counting edges.
    #   CN size ~ n/2 = 17. Edges check C(17,2) = 136.
    #   This is much faster than full recount (50k * 136).
    #   136 ops per step. 50k steps = 6.8M ops. ~0.1s. Feasible!
    #
    # Delta I6:
    #   Similar for independent sets.
    #   An I6 containing u and v requires w,x to be non-neighbors of u and v,
    #   and w,x to be non-neighbors of each other.
    #   Let NON_CN(u,v) = {w | not adj[u][w] and not adj[v][w]}.
    #   Delta_I6 = +/- count_non_edges_in_induced_subgraph(NON_CN(u,v)).
    #   Which is count of pairs (w,x) in NON_CN(u,v) such that not adj[w][x].
    #   
    #   This is also fast.
    
    # Let's rewrite try_sa with incremental penalty.
    
    def try_sa_fast(n_target):
        adj = [[0]*n_target for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Initial penalty
        current_k4 = count_k4(adj, n_target)
        current_i6 = count_i6(adj, n_target)
        current_pen = current_k4 + current_i6
        
        if current_pen == 0:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i][j]:
                        edges.append([i, j])
            return edges

        best_adj = [row[:] for row in adj]
        best_pen = current_pen
        
        T = 100.0
        T_min = 0.1
        alpha = 0.999
        steps = 0
        max_steps = 20000
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5:
                break
                
            u = random.randint(0, n_target-1)
            v = random.randint(0, n_target-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_val = adj[u][v]
            
            # Calculate delta
            # Common neighbors
            cn = []
            for w in range(n_target):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    cn.append(w)
            
            # Count edges in cn
            edges_in_cn = 0
            for i in range(len(cn)):
                for j in range(i+1, len(cn)):
                    if adj[cn[i]][cn[j]]:
                        edges_in_cn += 1
            
            # Non-common neighbors (for I6)
            non_cn = []
            for w in range(n_target):
                if w != u and w != v and not adj[u][w] and not adj[v][w]:
                    non_cn.append(w)
            
            # Count non-edges in non_cn
            non_edges_in_non_cn = 0
            for i in range(len(non_cn)):
                for j in range(i+1, len(non_cn)):
                    if not adj[non_cn[i]][non_cn[j]]:
                        non_edges_in_non_cn += 1
            
            if old_val == 0: # Adding edge
                delta_k4 = edges_in_cn
                delta_i6 = -non_edges_in_non_cn
            else: # Removing edge
                delta_k4 = -edges_in_cn
                delta_i6 = non_edges_in_non_cn
                
            delta_pen = delta_k4 + delta_i6
            
            new_pen = current_pen + delta_pen
            
            if delta_pen <= 0 or random.random() < math.exp(-delta_pen / T):
                # Accept
                adj[u][v] = 1 - old_val
                adj[v][u] = adj[u][v]
                current_pen = new_pen
                current_k4 += delta_k4
                current_i6 += delta_i6
                
                if current_pen < best_pen:
                    best_pen = current_pen
                    best_adj = [row[:] for row in adj]
                    if best_pen == 0:
                        edges = []
                        for i in range(n_target):
                            for j in range(i+1, n_target):
                                if best_adj[i][j]:
                                    edges.append([i, j])
                        return edges
            else:
                # Reject: no change to adj
                pass

            T *= alpha
            steps += 1
            
        if best_pen == 0:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if best_adj[i][j]:
                        edges.append([i, j])
            return edges
        return None

    # Strategy: Try to find a graph for n=35, then 34, etc.
    # Since SA can get stuck, we run multiple attempts.
    
    for n_target in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        # Try a few restarts for each n
        attempts = 5 if n_target > 30 else 10
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
            edges = try_sa_fast(n_target)
            if edges is not None:
                return {'n': n_target, 'edges': edges}
                
    return best_result
