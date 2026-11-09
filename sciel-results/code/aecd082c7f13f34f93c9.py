import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    # Build conflict graph
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
    
    all_nodes = set(range(1, n + 1))
    
    def greedy_mis(nodes):
        """
        Returns a maximal independent set from the given set of nodes.
        Uses greedy min-degree heuristic.
        """
        if not nodes:
            return set()
        
        available = set(nodes)
        selected = set()
        
        while available:
            # Find node with min degree in available subgraph
            min_node = -1
            min_deg = float('inf')
            for u in available:
                deg = len(adj[u] & available)
                if deg < min_deg:
                    min_deg = deg
                    min_node = u
            
            if min_node == -1:
                break
            
            selected.add(min_node)
            available.discard(min_node)
            # Remove neighbors
            for v in adj[min_node]:
                available.discard(v)
                
        return selected

    # Initial Greedy
    S = greedy_mis(all_nodes)
    
    # Local Search: 1-removal + Re-greedy
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        
        # Try removing each node u in S
        # We iterate over a list copy to avoid modification issues if we update S
        for u in list(S):
            if time.time() > deadline:
                break
            
            S_temp = S - {u}
            
            # Find nodes W that can be added to S_temp
            # W = { v in all_nodes \ S_temp | v has no conflict with any node in S_temp }
            W = set()
            for v in all_nodes:
                if v in S_temp:
                    continue
                # Check if v conflicts with any node in S_temp
                conflict = False
                for s in S_temp:
                    if s in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    W.add(v)
            
            # Run greedy on W to find best independent set I
            I = greedy_mis(W)
            
            new_size = len(S_temp) + len(I)
            if new_size > current_size:
                S = S_temp | I
                current_size = new_size
                improved = True
                # Restart the loop with the new S
                break
                
    return list(S)
