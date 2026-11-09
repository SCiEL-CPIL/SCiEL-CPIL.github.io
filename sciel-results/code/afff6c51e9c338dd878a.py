import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def find_path_5(u, v, neighbors):
        visited = {u}
        
        def dfs(current, depth, path):
            if depth == 5:
                if v in neighbors[current]:
                    return path + [v]
                return None
            for w in neighbors[current]:
                if w == v and depth < 5:
                    continue
                if w in visited:
                    continue
                visited.add(w)
                res = dfs(w, depth + 1, path + [w])
                if res:
                    return res
                visited.remove(w)
            return None
            
        return dfs(u, 0, [u])

    def creates_c6(u, v, neighbors):
        return find_path_5(u, v, neighbors) is not None

    best_edges = []
    best_count = 0
    
    # Try multiple random restarts
    while time.time() < deadline - 5.0:
        G = set()
        neighbors = [set() for _ in range(n)]
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Greedy
        for u, v in shuffled_edges:
            if not creates_c6(u, v, neighbors):
                G.add((u, v))
                neighbors[u].add(v)
                neighbors[v].add(u)
                
        current_count = len(G)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(G)
            
        # Local Search
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            if not G:
                break
                
            # Pick random edge to remove
            rem_edge = random.choice(list(G))
            
            # Snapshot
            G_copy = set(G)
            neighbors_copy = [set(s) for s in neighbors]
            
            G.remove(rem_edge)
            neighbors[rem_edge[0]].remove(rem_edge[1])
            neighbors[rem_edge[1]].remove(rem_edge[0])
            
            # Try to add all possible edges
            for u, v in all_edges:
                if (u, v) not in G:
                    if not creates_c6(u, v, neighbors):
                        G.add((u, v))
                        neighbors[u].add(v)
                        neighbors[v].add(u)
            
            if len(G) > current_count:
                current_count = len(G)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(G)
            else:
                # Revert
                G = G_copy
                neighbors = neighbors_copy
                
    return [list(e) for e in best_edges]
