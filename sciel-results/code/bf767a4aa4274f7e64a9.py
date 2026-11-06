import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m_total = len(all_edges)
    
    best_edges = []
    best_count = 0
    best_adj = [0] * n
    
    def has_q3_with_edge(adj, u, v):
        # Check if the graph (with edge u-v already added) contains a Q3 containing (u,v)
        # This is an incremental check assuming (u,v) is the "new" edge.
        # Q3 structure:
        # u is connected to v, w1, w2
        # v is connected to u, z1, z2
        # w1, w2 connected to u, y1
        # z1, z2 connected to v, y2
        # w1, w2 connected to z1, z2 (crossed) and y1
        # z1, z2 connected to w1, w2 (crossed) and y2
        # y1, y2 connected to w1, w2 (or z1, z2) and each other
        
        # More precisely, the Q3 subgraph induced by {u, v, w1, w2, z1, z2, y1, y2}
        # Edges:
        # (u,v), (u,w1), (u,w2)
        # (v,z1), (v,z2)
        # (w1,y1), (w2,y1)
        # (z1,y2), (z2,y2)
        # (w1,z1), (w2,z2)  OR  (w1,z2), (w2,z1)
        # (y1,y2)
        # Note: w1, w2 are not connected to each other. z1, z2 are not connected.
        # w1, w2 are not connected to z2, z1 (in the first case).
        
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False
            
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                # y1 is a common neighbor of w1 and w2, not u
                # In Q3, w1 and w2 are distance 2 from u.
                # They share exactly one common neighbor other than u?
                # In Q3, u=000, w1=010, w2=100. Common neighbors of 010 and 100: 110.
                # 110 is distance 2 from u.
                # So y1 = N(w1) & N(w2) & ~{u}
                # There might be multiple candidates in a general graph.
                
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                y1_candidates = [k for k in range(n) if (inter_w1_w2 >> k) & 1]
                
                for y1 in y1_candidates:
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            # y2 is a common neighbor of z1 and z2, not v
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            y2_candidates = [m for m in range(n) if (inter_z1_z2 >> m) & 1]
                            
                            for y2 in y2_candidates:
                                # Distinctness check
                                # Vertices: u, v, w1, w2, z1, z2, y1, y2
                                # Must be all distinct.
                                # u, v distinct from others by construction (nu/nv exclude v/u)
                                # w1, w2 distinct (loop)
                                # z1, z2 distinct (loop)
                                # w1, w2 not in nv (since nu & nv = 0? No, they can be common neighbors)
                                # Wait, if w1 is in nv, then w1 is a neighbor of v.
                                # In Q3, w1 (010) is not neighbor of v (001).
                                # So we must ensure w1, w2 are not in nv_list?
                                # No, the definition of Q3 requires specific adjacencies.
                                # If w1 is neighbor of v, it might still form a Q3?
                                # Let's check the edges.
                                # If w1 ~ v, then we have an extra edge not in Q3.
                                # But the question is "contains a subgraph isomorphic to Q3".
                                # So extra edges are fine.
                                # However, the vertices must be distinct.
                                # So w1 != z1, w1 != z2, etc.
                                
                                if y1 == y2: continue
                                if y1 in (u, v, w1, w2, z1, z2): continue
                                if y2 in (u, v, w1, w2, z1, z2): continue
                                if w1 in (z1, z2): continue
                                if w2 in (z1, z2): continue
                                
                                # Check edges
                                # 1. y1-y2
                                if not ((adj[y1] >> y2) & 1): continue
                                
                                # 2. Cross edges
                                # Case A: (w1,z1) and (w2,z2)
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                # Case B: (w1,z2) and (w2,z1)
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                    return True
                                    
        return False

    def count_q3s(adj):
        # Count total number of Q3 subgraphs in the graph
        # This is expensive, so we use it sparingly (e.g., for initial penalty or final check)
        # Actually, for SA, we need to evaluate the objective function frequently.
        # The objective is to minimize the number of Q3s while maximizing edges.
        # Or, we can use a penalty method: Score = Edges - Lambda * Q3s.
        # If we use the incremental check `has_q3_with_edge`, we can update the count of Q3s
        # containing a specific edge.
        # But `has_q3_with_edge` returns True/False, not the count.
        # We need to count how many Q3s are formed.
        # Let's modify the check to return the count of Q3s containing (u,v).
        
        pass # Not implemented, using boolean check for now in a different strategy

    def run_sa(seed):
        random.seed(seed)
        
        # Start with a random dense graph?
        # Or start with a greedy solution and perturb it?
        # The prompt suggests: "starting from a dense random graph"
        # But a dense random graph on 16 vertices has ~120 edges and many Q3s.
        # We need to remove edges to eliminate Q3s.
        
        # Let's try starting from a random graph with p=0.8 (approx 96 edges)
        # And then use SA to remove edges that are part of many Q3s?
        # Or add edges?
        # If we start dense, we must remove edges.
        # If we start sparse, we add edges.
        
        # Let's try a hybrid:
        # 1. Greedy construction to get a valid Q3-free graph.
        # 2. SA to improve it by swapping edges (remove one, add another) or just adding.
        
        # Actually, the prompt says: "Use a simulated annealing approach on the full edge set, starting from a dense random graph... allowing the search to accept temporary violations"
        
        # Let's implement that.
        # State: set of edges.
        # Objective: Maximize (Number of Edges - Penalty * Number of Q3s)
        # We need to count Q3s efficiently.
        
        # Efficient Q3 counting for a given edge addition/removal is hard.
        # But we can estimate the change in Q3 count.
        # If we add (u,v), the number of new Q3s is the number of Q3s containing (u,v) in the new graph.
        # This is exactly what `has_q3_with_edge` checks for existence.
        # We can modify it to count.
        
        def count_q3s_with_edge(adj, u, v):
            # Count Q3s containing (u,v) in the graph defined by adj (which includes u-v)
            count = 0
            nu = adj[u] & ~(1 << v)
            nv = adj[v] & ~(1 << u)
            
            if nu.bit_count() < 2 or nv.bit_count() < 2:
                return 0
                
            nu_list = [i for i in range(n) if (nu >> i) & 1]
            nv_list = [i for i in range(n) if (nv >> i) & 1]
            
            for i in range(len(nu_list)):
                w1 = nu_list[i]
                for j in range(i+1, len(nu_list)):
                    w2 = nu_list[j]
                    
                    inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                    if inter_w1_w2 == 0:
                        continue
                    y1_candidates = [k for k in range(n) if (inter_w1_w2 >> k) & 1]
                    
                    for y1 in y1_candidates:
                        for k in range(len(nv_list)):
                            z1 = nv_list[k]
                            for l in range(k+1, len(nv_list)):
                                z2 = nv_list[l]
                                
                                inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                                if inter_z1_z2 == 0:
                                    continue
                                y2_candidates = [m for m in range(n) if (inter_z1_z2 >> m) & 1]
                                
                                for y2 in y2_candidates:
                                    if y1 == y2: continue
                                    if y1 in (u, v, w1, w2, z1, z2): continue
                                    if y2 in (u, v, w1, w2, z1, z2): continue
                                    if w1 in (z1, z2): continue
                                    if w2 in (z1, z2): continue
                                    
                                    if not ((adj[y1] >> y2) & 1): continue
                                    
                                    if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                        count += 1
                                    if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                        count += 1
                                        
            return count

        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        max_steps = 5000
        
        # Initialize with random dense graph
        adj = [0] * n
        edge_set = set()
        for u, v in all_edges:
            if random.random() < 0.8:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                
        current_q3_count = 0
        # Calculate initial Q3 count (expensive, but done once)
        # We can sum count_q3s_with_edge for all edges? No, that overcounts.
        # Each Q3 has 12 edges. So sum(counts)/12 = total Q3s.
        total_q3_sum = 0
        for u, v in edge_set:
            total_q3_sum += count_q3s_with_edge(adj, u, v)
        current_q3_count = total_q3_sum // 12
        
        current_edges = len(edge_set)
        
        best_adj_sa = [0]*n
        best_edges_sa = 0
        best_q3_sa = 10**9
        best_score = -10**9
        
        # We want to maximize: edges - penalty * q3s
        # If q3s > 0, score is low.
        # We want to reach q3s = 0 with max edges.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Choose a random edge to toggle
            u, v = random.choice(all_edges)
            
            if (u, v) in edge_set:
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_set.remove((u, v))
                
                # Calculate change in Q3 count
                # The Q3s that disappear are those containing (u,v) in the NEW graph?
                # No, those containing (u,v) in the OLD graph.
                # We already removed it from adj.
                # So we need to count Q3s containing (u,v) in the graph WITHOUT (u,v)?
                # No, `count_q3s_with_edge` assumes the edge is present.
                # We need a version that counts Q3s containing (u,v) given that (u,v) is NOT present?
                # No, if (u,v) is not present, it can't be part of a Q3.
                # The Q3s that are destroyed are those that contained (u,v).
                # In the old graph (with u,v), we can count them.
                # But we already removed u,v from adj.
                # Let's re-add temporarily to count? Or keep track.
                
                # Let's re-add to count, then remove again.
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                removed_q3s = count_q3s_with_edge(adj, u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                delta_q3 = -removed_q3s
                delta_edges = -1
                
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                added_q3s = count_q3s_with_edge(adj, u, v)
                
                delta_q3 = added_q3s
                delta_edges = 1
                
                # Tentatively accept?
                # We need to decide if we keep it.
                # If we don't keep it, we must remove it.
                # Let's calculate the new score first.
                
            new_q3 = current_q3_count + delta_q3
            new_edges = current_edges + delta_edges
            
            # Score function
            # We want to minimize Q3s first, then maximize edges.
            # Score = -new_q3 * 100 + new_edges
            # If new_q3 == 0, Score = new_edges.
            # If new_q3 > 0, Score is very negative.
            
            old_score = -current_q3_count * 100 + current_edges
            new_score = -new_q3 * 100 + new_edges
            
            delta_score = new_score - old_score
            
            if delta_score >= 0:
                accept = True
            else:
                # Accept with probability exp(delta_score / T)
                if delta_score > -50 * T: # Avoid underflow
                    accept = random.random() < (2.718281828 ** (delta_score / T))
                else:
                    accept = False
            
            if accept:
                current_q3_count = new_q3
                current_edges = new_edges
                
                if (u, v) not in edge_set and delta_edges == 1:
                    edge_set.add((u, v))
                elif (u, v) in edge_set and delta_edges == -1:
                    pass # Already removed
                
                # Update best
                if new_q3 == 0 and new_edges > best_edges_sa:
                    best_edges_sa = new_edges
                    best_adj_sa = adj[:]
                    best_q3_sa = 0
                    best_score = new_edges
                elif new_q3 < best_q3_sa:
                    best_q3_sa = new_q3
                    best_edges_sa = new_edges
                    best_adj_sa = adj[:]
                    best_score = new_score
            
            else:
                # Reject move: revert changes
                if delta_edges == 1: # We tried to add
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                elif delta_edges == -1: # We tried to remove
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        # Final check: is the best solution Q3-free?
        # If best_q3_sa > 0, we might try to clean it up?
        # Or just return the best Q3-free solution found during SA.
        # If none found, return empty or best effort?
        # The task requires a Q3-free graph.
        # So we must return a valid graph.
        # If SA didn't find a Q3-free graph, we should fall back to a greedy one.
        
        if best_q3_sa == 0:
            # Reconstruct edges from best_adj_sa
            edges_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if (best_adj_sa[u] >> v) & 1:
                        edges_list.append((u, v))
            return best_edges_sa, edges_list
        else:
            return 0, []

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_sa(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
        
        # If we find a very good solution, we can stop or continue.
        # 60 edges is likely near optimal.
        if best_count >= 60:
            break

    # If no Q3-free graph found (should not happen with greedy fallback, but SA might fail),
    # we should probably run a simple greedy to ensure validity.
    if best_count == 0:
        # Fallback greedy
        random.seed(0)
        edges = all_edges[:]
        random.shuffle(edges)
        adj = [0] * n
        edge_set = set()
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3_with_edge(adj, u, v):
                edge_set.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        best_count = len(edge_set)
        best_edges = list(edge_set)

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
