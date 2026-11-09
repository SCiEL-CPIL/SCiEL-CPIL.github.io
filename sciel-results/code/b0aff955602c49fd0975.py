import random
import time

def construct():
    n = 20
    best_edges = None
    best_count = 0
    start_time = time.time()
    deadline = start_time + 115  # Leave some buffer
    
    def check_k33_free(adj):
        """Check if graph is K3,3-free. Returns (is_free, violating_triple_info)."""
        # For each triple, check common neighbors
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Common neighbors of i, j, k
                    common = []
                    for v in range(n):
                        if v != i and v != j and v != k:
                            if adj[i][v] and adj[j][v] and adj[k][v]:
                                common.append(v)
                    if len(common) >= 3:
                        return False, (i, j, k, common[:3])
        return True, None
    
    def count_edges(adj):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
        return count
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with random graph
        density = 0.5 + 0.3 * random.random()
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i][j] = adj[j][i] = True
        
        # Iteratively remove edges to fix K3,3 violations
        max_iter = 1000
        for it in range(max_iter):
            if time.time() > deadline:
                break
            is_free, info = check_k33_free(adj)
            if is_free:
                break
            # info = (i, j, k, [c1, c2, c3])
            i, j, k, commons = info
            # Remove one random edge from the K3,3
            # The K3,3 has edges between {i,j,k} and {c1,c2,c3}
            a_set = [i, j, k]
            b_set = commons
            # Pick a random edge to remove
            a = random.choice(a_set)
            b = random.choice(b_set)
            adj[a][b] = False
            adj[b][a] = False
        
        if time.time() > deadline:
            break
        
        # Final check
        is_free, _ = check_k33_free(adj)
        if is_free:
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            best_edges.append([i, j])
    
    if best_edges is None:
        # Fallback to K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges
