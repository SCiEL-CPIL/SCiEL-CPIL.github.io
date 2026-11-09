import random
import time

def construct():
    n_vertices = 40
    best_edges_count = 0
    best_result = []
    
    # Helper to check if adding (u,v) creates a C6
    def creates_c6(u, v, adj, edges):
        # C6: u-a-b-c-d-v
        # We check paths of length 2 from u and v
        # n2_u: (b, a) where u-a-b
        # n2_v: (c, d) where v-d-c
        # Conflict if (b,c) in edges and all distinct.
        
        n2_u = []
        for a in adj[u]:
            for b in adj[a]:
                if b != u:
                    n2_u.append((b, a))
                    
        n2_v = []
        for d in adj[v]:
            for c in adj[d]:
                if c != v:
                    n2_v.append((c, d))
                    
        for b, a in n2_u:
            if b == v or a == v:
                continue
            for c, d in n2_v:
                if c == u or d == u:
                    continue
                if b == c:
                    continue
                if b == d or a == c or a == d:
                    continue
                if (b, c) in edges:
                    return True
        return False

    # Greedy construction
    def greedy_construction(seed_val):
        random.seed(seed_val)
        edges = set()
        adj = [[] for _ in range(n_vertices)]
        
        # Start with 8 disjoint K5s
        n_clusters = 8
        cluster_size = 5
        for k in range(n_clusters):
            base = k * cluster_size
            for i in range(cluster_size):
                for j in range(i + 1, cluster_size):
                    u = base + i
                    v = base + j
                    edges.add((u, v))
                    edges.add((v, u))
                    adj[u].append(v)
                    adj[v].append(u)
                    
        # Generate candidates
        candidates = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                base_i = i * cluster_size
                base_j = j * cluster_size
                for a in range(cluster_size):
                    for b in range(cluster_size):
                        u = base_i + a
                        v = base_j + b
                        candidates.append((u, v))
                        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in edges:
                continue
            if not creates_c6(u, v, adj, edges):
                edges.add((u, v))
                edges.add((v, u))
                adj[u].append(v)
                adj[v].append(u)
                
        return edges

    # Run multiple times
    start_time = time.time()
    num_tries = 200 # Adjust based on time
    
    for i in range(num_tries):
        if time.time() - start_time > 110:
            break
        e_set = greedy_construction(i)
        count = len([e for e in e_set if e[0] < e[1]])
        if count > best_edges_count:
            best_edges_count = count
            best_result = [[u, v] for u, v in e_set if u < v]
            
    return best_result
