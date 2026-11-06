import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_edges = len(all_edges)
    
    def has_q3_with_edge(adj, u, v):
        # Check if adding edge (u,v) creates a Q3 containing (u,v).
        # We assume (u,v) is already in adj when calling this.
        
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # Need at least 2 other neighbors for u and v
        if (nu.bit_count() < 2) or (nv.bit_count() < 2):
            return False
            
        # Get lists of neighbors
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        # Iterate pairs w1, w2 from nu_list
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                # Find y1 in N(w1) ∩ N(w2) \ {u}
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                y1_candidates = [k for k in range(n) if (inter_w1_w2 >> k) & 1]
                
                for y1 in y1_candidates:
                    # Iterate pairs z1, z2 from nv_list
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            # Find y2 in N(z1) ∩ N(z2) \ {v}
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            
                            y2_candidates = [m for m in range(n) if (inter_z1_z2 >> m) & 1]
                            
                            for y2 in y2_candidates:
                                # Check distinctness
                                if y1 == y2: continue
                                if y1 in (u, v, w1, w2, z1, z2): continue
                                if y2 in (u, v, w1, w2, z1, z2): continue
                                if w1 == z1 or w1 == z2 or w2 == z1 or w2 == z2: continue
                                
                                # Check remaining edges:
                                # 1. y1-y2
                                if not ((adj[y1] >> y2) & 1):
                                    continue
                                    
                                # 2. w1-z1 and w2-z2 OR w1-z2 and w2-z1
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                    return True
                                    
        return False

    def run_greedy(seed):
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [0] * n
        edge_set = set()
        
        count = 0
        for u, v in edges:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if (u, v) in edge_set:
                    continue
                # Try add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return count, list(edge_set)

    def run_sa(seed):
        random.seed(seed)
        # Start with a random dense graph
        adj = [0] * n
        edge_set = set()
        
        # Random initialization: keep edges with prob 0.8
        for u, v in all_edges:
            if random.random() < 0.8:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        count = len(edge_set)
        
        # Simulated Annealing
        # We want to maximize edges while keeping Q3 count = 0.
        # Objective: Maximize (edges - penalty * Q3_count)
        # Penalty is huge.
        
        # Helper to count Q3s? That's expensive.
        # Instead, we can use a "repair" strategy or just check validity.
        # But SA needs a score.
        # Let's use a heuristic score:
        # If Q3-free: score = count
        # Else: score = count - 100 * (number of Q3s)
        
        # Counting Q3s is expensive. Let's approximate or use a different move set.
        # Move: Add edge, Remove edge, Swap edge.
        
        # To make SA fast, we only accept moves that are valid (Q3-free) or improve edge count significantly?
        # Actually, the "inspiration" says: "Simulated annealing with a Q3-count penalty allows the search to temporarily accept graphs containing Q3 subgraphs".
        
        # We need a fast way to count Q3s or at least detect them.
        # has_q3_with_edge is fast for *one* edge.
        # To count all Q3s, we'd have to run it for every edge? No, that counts Q3s containing that edge.
        # Summing over all edges / 12 gives Q3 count? (Each Q3 has 12 edges).
        # That's 120 calls to has_q3_with_edge. Might be too slow for SA steps.
        
        # Alternative: Just try to add edges. If it creates a Q3, remove a random edge from the Q3?
        # This is a "repair" heuristic, not pure SA.
        
        # Let's stick to the "greedy + local search" which is proven to work (score 1.46).
        # The prompt asks to combine. The current program (1.30) is slow because of the full check.
        # Inspiration 2 (1.46) uses the fast check.
        # Inspiration 1 (1.10) is a static graph.
        # Inspiration 3 (1.10) is unknown.
        
        # The best strategy is clearly the fast greedy + local search from Inspiration 2.
        # I will use that. I will add multiple restarts.
        
        pass

    best_count = 0
    best_edges = []
    
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_greedy(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
        
        # If we hit a very high count, we can break
        if best_count >= 55:
            # 55 edges on 16 vertices is quite dense (max 120).
            # K_16 has 120. Q3-free max is likely around 50-60.
            # Let's keep going to try for higher.
            pass

    return [list(e) for e in best_edges]
