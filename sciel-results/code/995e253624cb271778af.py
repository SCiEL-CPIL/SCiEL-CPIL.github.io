import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Fast Q3 detection using bit-level iteration
    def has_q3_with_edge(adj, u, v):
        # Check if adding edge (u,v) creates a Q3 containing (u,v).
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if (nu.bit_count() < 2) or (nv.bit_count() < 2):
            return False
            
        # Collect neighbors as lists (fast enough for n=16)
        nu_list = []
        mask = nu
        while mask:
            lb = mask & (-mask)
            idx = lb.bit_length() - 1
            nu_list.append(idx)
            mask ^= lb
            
        nv_list = []
        mask = nv
        while mask:
            lb = mask & (-mask)
            idx = lb.bit_length() - 1
            nv_list.append(idx)
            mask ^= lb
            
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                # y1 in N(w1) ∩ N(w2) \ {u}
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                y1_mask = inter_w1_w2
                while y1_mask:
                    lb = y1_mask & (-y1_mask)
                    y1 = lb.bit_length() - 1
                    y1_mask ^= lb
                    
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            # Distinctness checks
                            if y1 == z1 or y1 == z2 or y1 == w1 or y1 == w2:
                                continue
                            
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            y2_mask = inter_z1_z2
                            while y2_mask:
                                lb2 = y2_mask & (-y2_mask)
                                y2 = lb2.bit_length() - 1
                                y2_mask ^= lb2
                                
                                if y2 == y1 or y2 == z1 or y2 == z2 or y2 == w1 or y2 == w2:
                                    continue
                                
                                # Check remaining edges
                                if not ((adj[y1] >> y2) & 1):
                                    continue
                                
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                    return True
        return False

    def is_q3_free(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u, v in edge_list:
            if has_q3_with_edge(adj, u, v):
                return False
        return True

    def get_weight2_graph_edges():
        edges = []
        for i in range(16):
            for j in range(i+1, 16):
                x = i ^ j
                if bin(x).count('1') == 2:
                    edges.append((i, j))
        return edges

    def get_k7_k7_k2_edges():
        # K7 on {0..6}, K7 on {7..13}, K2 on {14,15}
        edges = []
        for i in range(7):
            for j in range(i+1, 7):
                edges.append((i, j))
        for i in range(7, 14):
            for j in range(i+1, 14):
                edges.append((i, j))
        edges.append((14, 15))
        return edges

    # Try the weight-2 graph (Shrikhande candidate)
    w2_edges = get_weight2_graph_edges()
    if len(w2_edges) > best_count and is_q3_free(w2_edges):
        best_count = len(w2_edges)
        best_edges = w2_edges

    # Try K7+K7+K2 seed (57 edges)
    k7_edges = get_k7_k7_k2_edges()
    if len(k7_edges) > best_count and is_q3_free(k7_edges):
        best_count = len(k7_edges)
        best_edges = k7_edges

    def run_greedy(seed):
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [0] * n
        edge_set = set()
        
        count = 0
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1
        
        # Local search: try adding edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            local_edges = all_edges[:]
            random.shuffle(local_edges)
            for u, v in local_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # 1-1 swap local search: remove one edge, try to add two
        improved = True
        while improved and time.time() < deadline:
            improved = False
            local_edges = list(edge_set)
            random.shuffle(local_edges)
            for rem_u, rem_v in local_edges:
                if time.time() > deadline:
                    break
                # Remove edge
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                edge_set.discard((rem_u, rem_v))
                count -= 1
                
                # Try to add two new edges
                added = []
                candidate_edges = all_edges[:]
                random.shuffle(candidate_edges)
                for u, v in candidate_edges:
                    if (u, v) in edge_set or (v, u) in edge_set:
                        continue
                    if u == rem_u or u == rem_v or v == rem_u or v == rem_v:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not has_q3_with_edge(adj, u, v):
                        edge_set.add((u, v))
                        count += 1
                        added.append((u, v))
                        if len(added) == 2:
                            break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                if len(added) >= 1:
                    improved = True
                # If we couldn't add enough, the net gain is len(added) - 1
                # If len(added) == 0, we're worse, so keep the state (already removed)
                # Actually we should only accept if we gained at least 1 edge net
                # If added is empty, we lost 1 edge. That's bad. Let's revert.
                if len(added) == 0:
                    # Revert: add back the removed edge
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                    edge_set.add((rem_u, rem_v))
                    count += 1
                    # Don't mark improved since we reverted
                    pass

        return count, list(edge_set)

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_greedy(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
