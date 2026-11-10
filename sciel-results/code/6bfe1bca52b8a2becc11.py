import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    vertices = list(range(16))
    subsets = list(itertools.combinations(vertices, 8))
    
    def is_q3(verts, edge_set):
        # Fast degree check
        deg = [0] * 8
        for i in range(8):
            u = verts[i]
            for j in range(8):
                if i == j: continue
                v = verts[j]
                if u < v:
                    if (u, v) in edge_set:
                        deg[i] += 1
                else:
                    if (v, u) in edge_set:
                        deg[i] += 1
                        
        for d in deg:
            if d != 3:
                return False
                
        # Build adjacency
        adj = [[] for _ in range(8)]
        for i in range(8):
            u = verts[i]
            for j in range(i + 1, 8):
                v = verts[j]
                if u < v:
                    if (u, v) in edge_set:
                        adj[i].append(j)
                        adj[j].append(i)
                else:
                    if (v, u) in edge_set:
                        adj[i].append(j)
                        adj[j].append(i)
                        
        # Bipartite check
        color = [-1] * 8
        for start in range(8):
            if color[start] != -1:
                continue
            color[start] = 0
            queue = [start]
            visited_count = 0
            while queue:
                u = queue.pop(0)
                visited_count += 1
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    else:
                        if color[v] == color[u]:
                            return False
            if visited_count != 8:
                return False
        return True

    best_edges = None
    best_count = -1
    
    MAX_SAMPLES = 10
    
    while time.time() < deadline - 2:
        current_edges = set((u, v) for u in range(16) for v in range(u + 1, 16))
        
        while True:
            if time.time() > deadline:
                break
                
            q3_samples = []
            for S in subsets:
                if is_q3(S, current_edges):
                    q3_samples.append(S)
                    if len(q3_samples) >= MAX_SAMPLES:
                        break
            
            if not q3_samples:
                break
            
            # Count edges in samples
            edge_counts = {}
            for S in q3_samples:
                for i in range(8):
                    for j in range(i + 1, 8):
                        u, v = S[i], S[j]
                        if u > v: u, v = v, u
                        if (u, v) in current_edges:
                            edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
            
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            e_to_remove = random.choice(candidates)
            current_edges.remove(e_to_remove)
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    if best_edges is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    return [list(e) for e in best_edges]
