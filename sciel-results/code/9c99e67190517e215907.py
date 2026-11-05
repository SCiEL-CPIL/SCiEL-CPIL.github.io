import random
import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin

    best_edges = []
    best_count = 0

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute bit masks for pairs? No, just use integers.
    
    def can_add(u, v, adj_mask, common):
        # 1. Check if u,v already connected
        if (adj_mask[u] >> v) & 1:
            return False
        
        # 2. Check common neighbors of u and v
        # common[u][v] is the count of w such that w~u and w~v
        if common[u][v] > 2:
            return False
        
        # 3. Check neighbors of v (excluding u)
        # For each w in N(v) \ {u}, we need common[u][w] <= 1
        # Because adding (u,v) makes v a common neighbor of u and w.
        # So new common[u][w] = old common[u][w] + 1.
        # We need new <= 2, so old <= 1.
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if common[u][w] >= 2:
                return False
            m -= lsb
        
        # 4. Check neighbors of u (excluding v)
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if common[v][w] >= 2:
                return False
            m -= lsb
        
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append((u, v))
        
        # Update common counts
        # For each w in N(v) \ {u}, common[u][w] increases by 1
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
            
        # For each w in N(u) \ {v}, common[v][w] increases by 1
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove((u, v))
        
        # Update common counts
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges):
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def local_search(initial_edges, max_iters=1000):
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in initial_edges:
            add_edge(u, v, adj_mask, common, edges)

        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        # Try to add edges until no more can be added
        changed = True
        while changed:
            changed = False
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(u, v, adj_mask, common):
                    add_edge(u, v, adj_mask, common, edges)
                    changed = True
                    if len(edges) > best_local_count:
                        best_local_count = len(edges)
                        best_local_edges = list(edges)
                        
        return best_local_edges, adj_mask, common, edges

    def verify_edges(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    return False
        return True

    # Main Search: Iterated Local Search with Simulated Annealing acceptance
    current_adj = [0] * n
    current_common = [[0] * n for _ in range(n)]
    current_edges = []
    
    # Initial Greedy
    greedy_fill(current_adj, current_common, current_edges)
    current_edges, current_adj, current_common, _ = local_search(current_edges)
    
    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = [[u, v] for u, v in current_edges]

    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        # Perturb
        kick_size = random.choice([1, 2, 2, 3, 3, 4])
        if len(current_edges) > kick_size:
            to_remove = random.sample(current_edges, kick_size)
            for u, v in to_remove:
                remove_edge(u, v, current_adj, current_common, current_edges)
        
        # Local Search
        new_edges, new_adj, new_common, _ = local_search(current_edges)
        
        # Acceptance
        delta = len(new_edges) - len(current_edges)
        if delta > 0:
            # Accept improvement
            current_edges = new_edges
            current_adj = new_adj
            current_common = new_common
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [[u, v] for u, v in current_edges]
        else:
            # Accept worsening with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current_edges = new_edges
                current_adj = new_adj
                current_common = new_common
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final Polish: Try to add any remaining edges to the best solution
    if best_edges:
        final_adj = [0] * n
        final_common = [[0] * n for _ in range(n)]
        final_edges = []
        for u, v in best_edges:
            add_edge(u, v, final_adj, final_common, final_edges)
        
        # One more pass of greedy fill
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(u, v, final_adj, final_common):
                add_edge(u, v, final_adj, final_common, final_edges)
        
        if len(final_edges) > best_count:
            best_count = len(final_edges)
            best_edges = [[u, v] for u, v in final_edges]

    if not best_edges:
        best_edges = []
        for i in range(5):
            base = i * 4
            for u in range(4):
                for v in range(u + 1, 4):
                    best_edges.append([base + u, base + v])

    if not verify_edges(best_edges):
        return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
