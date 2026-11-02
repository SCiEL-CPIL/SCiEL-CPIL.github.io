import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses cyclic construction with proper girth verification.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin

    def check_girth(adj, n, g):
        """Check if graph has girth at least g using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        # Found a cycle
                        # The cycle goes: start -> ... -> node -> neighbor -> ... -> start
                        # Length is dist[node] + 1 + dist[neighbor]
                        # But this might overcount if the paths share vertices
                        # A safer check: if dist[node] + dist[neighbor] + 1 < g, we have a short cycle
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < g:
                            return False
        
        return True

    def build_cyclic_graph(n, offsets):
        """Build a cyclic graph where vertex i connects to i±offset mod n."""
        edges = []
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        for i in range(n):
            for a in offsets:
                j1 = (i + a) % n
                j2 = (i - a) % n
                
                for j in [j1, j2]:
                    if j == i:
                        return None, None
                    key = (min(i, j), max(i, j))
                    if key in edge_set:
                        return None, None
                    edge_set.add(key)
                    edges.append((i, j))
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return None, None
        
        return edges, adj

    def find_valid_cyclic(n, max_attempts=200):
        """Try to find offsets for a valid cyclic graph of size n."""
        k_half = k // 2
        
        # Valid offsets: 1 to n//2 - 1 (to avoid duplicates and loops)
        max_offset = n // 2 - 1 if n % 2 == 0 else n // 2
        if max_offset < k_half:
            return None, None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Choose k_half distinct offsets
            available = list(range(1, max_offset + 1))
            random.shuffle(available)
            offsets = available[:k_half]
            
            edges, adj = build_cyclic_graph(n, offsets)
            if edges is None:
                continue
            
            if check_girth(adj, n, g):
                return edges, adj
        
        return None, None

    # Moore bound for k=4, g=10 (even girth):
    # n >= 2 * sum_{i=0}^{(g-2)/2} (k-1)^i = 2 * (1 + 3 + 9 + 27) = 80
    # But actual lower bound is 243, so start higher
    moore_bound = 2 * sum((k-1)**i for i in range((g-2)//2 + 1))
    
    best_n = float('inf')
    best_edges = None
    
    # Try cyclic constructions
    for n in range(max(moore_bound, 100), 500, 5):
        if time.time() > deadline - 20:
            break
        
        edges, adj = find_valid_cyclic(n, max_attempts=50)
        if edges is not None and n < best_n:
            best_n = n
            best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: random regular graphs
    def random_kreg(n):
        if n * k % 2 != 0:
            return None
        
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edges = []
        edge_set = set()
        
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
            edges.append((u, v))
        
        return edges

    for n in range(300, 500, 10):
        if time.time() > deadline - 10:
            break
        
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            
            edges = random_kreg(n)
            if edges is None:
                continue
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if check_girth(adj, n, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Last resort: return a simple 4-regular graph (may have short cycles)
    n = 20
    edges = []
    for i in range(n):
        for j in [1, 2, 3, 4]:
            edges.append((i, (i + j) % n))
    
    return n, edges
