import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, leaving 20s margin
    
    best_edges = None
    best_count = -1
    
    vertices = list(range(16))
    # Precompute all 8-combinations
    combos = list(itertools.combinations(vertices, 8))
    
    def get_q3_edges(adj, combo):
        """
        Returns a set of edges (as tuples (u,v) with u<v) if the induced subgraph
        on 'combo' is isomorphic to Q3, else returns None.
        """
        # Map combo vertices to 0..7
        indices = {v: i for i, v in enumerate(combo)}
        
        deg = [0] * 8
        edge_count = 0
        edges = []
        
        for i in range(8):
            u = combo[i]
            mask = adj[u]
            for j in range(i+1, 8):
                v = combo[j]
                if (mask >> v) & 1:
                    deg[i] += 1
                    deg[j] += 1
                    edge_count += 1
                    edges.append((u, v))
        
        if edge_count != 12:
            return None
        
        for d in deg:
            if d != 3:
                return None
        
        # Check bipartite
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        
        while queue:
            u_idx = queue.pop()
            u = combo[u_idx]
            mask = adj[u]
            for v_idx in range(8):
                if v_idx == u_idx: continue
                v = combo[v_idx]
                if (mask >> v) & 1:
                    if color[v_idx] == -1:
                        color[v_idx] = 1 - color[u_idx]
                        queue.append(v_idx)
                    elif color[v_idx] == color[u_idx]:
                        return None
        
        return set(edges)

    def find_all_q3s(adj):
        q3s = []
        for combo in combos:
            edges = get_q3_edges(adj, combo)
            if edges is not None:
                q3s.append(edges)
        return q3s

    def count_edges(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2

    def search(adj):
        """
        Greedy removal of edges to eliminate all Q3s.
        Returns the final adjacency list.
        """
        q3_list = find_all_q3s(adj)
        
        while q3_list:
            if time.time() > deadline:
                break
                
            # Count frequency of each edge in the current set of Q3s
            edge_freq = defaultdict(int)
            for edges in q3_list:
                for e in edges:
                    edge_freq[e] += 1
            
            if not edge_freq:
                break
                
            # Find the edge with the maximum frequency
            max_freq = max(edge_freq.values())
            candidates = [e for e, f in edge_freq.items() if f == max_freq]
            
            # Randomly pick one candidate to escape local minima
            e = random.choice(candidates)
            u, v = e
            
            # Remove edge from graph
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Update Q3 list: remove any Q3 that contained the removed edge
            new_q3_list = []
            for edges in q3_list:
                if e not in edges:
                    new_q3_list.append(edges)
            q3_list = new_q3_list
            
        return adj

    # Run multiple strategies
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Decide initial graph
        if attempts % 3 == 0:
            # K16
            adj = [(1 << 16) - 2 - (1 << i) for i in range(16)]
        elif attempts % 3 == 1:
            # Random p=0.9
            adj = [0] * 16
            for i in range(16):
                for j in range(i+1, 16):
                    if random.random() < 0.9:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # Random p=0.8
            adj = [0] * 16
            for i in range(16):
                for j in range(i+1, 16):
                    if random.random() < 0.8:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Run search
        adj_final = search(adj[:])
        
        # Count edges
        edge_count = count_edges(adj_final)
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = []
            for i in range(16):
                for j in range(i+1, 16):
                    if (adj_final[i] >> j) & 1:
                        best_edges.append([i, j])

    # Fallback if no result found (should not happen)
    if best_edges is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    return best_edges
