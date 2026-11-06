import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Precompute all Q3 subgraphs in K16
    # A Q3 is defined by 8 vertices and 12 edges.
    # We need to know for any pair of edges, if they can co-exist in a Q3.
    # Actually, for the incremental check, we just need to check if adding (u,v) creates a Q3.
    # The current program does this efficiently by checking neighbors.
    
    # Let's use the neighbor-based check from the current program as it is fast.
    
    def has_q3_with_edge(adj, u, v):
        # u, v are the endpoints of the new edge.
        # In Q3, u and v are adjacent.
        # N(u) in Q3: {v, w1, w2}
        # N(v) in Q3: {u, z1, z2}
        
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
                # y1 must be connected to w1 and w2.
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
                                # Vertices: u, v, w1, w2, z1, z2, y1, y2
                                # All must be distinct.
                                if y1 == y2: continue
                                if y1 in (u, v, w1, w2, z1, z2): continue
                                if y2 in (u, v, w1, w2, z1, z2): continue
                                if w1 == w2 or z1 == z2: continue # handled by loops
                                if w1 == z1 or w1 == z2 or w2 == z1 or w2 == z2: continue
                                # u, v are distinct from others by construction (nu/nv exclude v/u)
                                
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
        
        # Local search: try to add missing edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate over missing edges
            for u, v in all_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
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
        
        # Local search: try to swap edges?
        # If adding (u,v) fails, maybe we can remove (a,b) and add (u,v)?
        # This is more complex. Let's stick to addition first.
        # If we have time, we can do a second pass with swaps.
        
        return count, list(edge_set)

    # Run multiple restarts
    restart = 0
    # We can run more restarts if the greedy+local search is fast enough.
    # With n=16, the check is very fast.
    while time.time() < deadline:
        count, edges_list = run_greedy(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
        
        # If we reach a high count, we might break early or continue to see if we can improve.
        # The theoretical max for Q3-free on 16 vertices is likely around 55-60.
        # The seed program got 1.46, which is 58.46 edges? No, score is value/seed.
        # If seed is 40 (Shrikhande complement?), then 1.46 * 40 = 58.4.
        # Let's assume the seed value is fixed.
        # We just want to maximize count.

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
