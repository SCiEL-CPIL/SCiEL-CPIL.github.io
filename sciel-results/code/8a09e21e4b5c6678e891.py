import time
import random
import itertools

N = 20
DEADLINE = time.time() + 140

def construct():
    start_time = time.time()
    
    best_adj = None
    best_common = None
    best_edges = 0
    
    # Precompute all pairs for iteration
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    for restart in range(1000):
        if time.time() > DEADLINE:
            break
        
        random.seed(restart)
        
        # Initialize
        adj = [0] * N
        common = [[0] * N for _ in range(N)]
        
        # Greedy initialization
        # Shuffle edges
        edges = all_pairs[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if (adj[u] >> v) & 1:
                continue
            
            # Check if adding (u,v) is valid
            # We need to check triples involving u or v
            valid = True
            
            # Check triples {u, b, c}
            # b, c from 0..19 excluding u, v
            # Optimization: iterate b, c
            for b in range(N):
                if b == u or b == v:
                    continue
                for c in range(b+1, N):
                    if c == u or c == v:
                        continue
                    # Check if v becomes a 3rd common neighbor of u, b, c
                    # Current common neighbors of u, b, c is common[u][b] & adj[c]
                    # If count is 2 and v is in N(b) and v is in N(c), then invalid.
                    cnt = bin(common[u][b] & adj[c]).count('1')
                    if cnt == 2:
                        if (adj[b] >> v) & 1 and (adj[c] >> v) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Check triples {v, b, c}
                for b in range(N):
                    if b == u or b == v:
                        continue
                    for c in range(b+1, N):
                        if c == u or c == v:
                            continue
                        cnt = bin(common[v][b] & adj[c]).count('1')
                        if cnt == 2:
                            if (adj[b] >> u) & 1 and (adj[c] >> u) & 1:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Update common neighbors
                # common[u][w] gains v if v in N(w)
                # common[v][w] gains u if u in N(w)
                for w in range(N):
                    if w == u or w == v:
                        continue
                    if (adj[w] >> v) & 1:
                        common[u][w] |= (1 << v)
                        common[w][u] |= (1 << v)
                    if (adj[w] >> u) & 1:
                        common[v][w] |= (1 << u)
                        common[w][v] |= (1 << u)
        
        # Local Search
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            
            # Try adding edges
            for u, v in all_pairs:
                if (adj[u] >> v) & 1:
                    continue
                
                valid = True
                # Check {u, b, c}
                for b in range(N):
                    if b == u or b == v: continue
                    for c in range(b+1, N):
                        if c == u or c == v: continue
                        cnt = bin(common[u][b] & adj[c]).count('1')
                        if cnt == 2:
                            if (adj[b] >> v) & 1 and (adj[c] >> v) & 1:
                                valid = False
                                break
                    if not valid: break
                
                if valid:
                    # Check {v, b, c}
                    for b in range(N):
                        if b == u or b == v: continue
                        for c in range(b+1, N):
                            if c == u or c == v: continue
                            cnt = bin(common[v][b] & adj[c]).count('1')
                            if cnt == 2:
                                if (adj[b] >> u) & 1 and (adj[c] >> u) & 1:
                                    valid = False
                                    break
                        if not valid: break
                
                if valid:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    for w in range(N):
                        if w == u or w == v: continue
                        if (adj[w] >> v) & 1:
                            common[u][w] |= (1 << v)
                            common[w][u] |= (1 << v)
                        if (adj[w] >> u) & 1:
                            common[v][w] |= (1 << u)
                            common[w][v] |= (1 << u)
                    
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove 1, add 2
                # This is complex to implement efficiently with the fast check.
                # Given the speed of the add check, we can try random kicks.
                # Kick: remove 3 random edges, then greedy add.
                if random.random() < 0.1:
                    # Remove 3 random edges
                    for _ in range(3):
                        u, v = random.choice(all_pairs)
                        if (adj[u] >> v) & 1:
                            # Remove edge
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            
                            # Update common
                            for w in range(N):
                                if w == u or w == v: continue
                                if (adj[w] >> v) & 1:
                                    common[u][w] &= ~(1 << v)
                                    common[w][u] &= ~(1 << v)
                                if (adj[w] >> u) & 1:
                                    common[v][w] &= ~(1 << u)
                                    common[w][v] &= ~(1 << u)
                    
                    # Greedy add
                    for u, v in all_pairs:
                        if (adj[u] >> v) & 1: continue
                        # Same validity check
                        valid = True
                        for b in range(N):
                            if b == u or b == v: continue
                            for c in range(b+1, N):
                                if c == u or c == v: continue
                                cnt = bin(common[u][b] & adj[c]).count('1')
                                if cnt == 2:
                                    if (adj[b] >> v) & 1 and (adj[c] >> v) & 1:
                                        valid = False
                                        break
                            if not valid: break
                        if valid:
                            for b in range(N):
                                if b == u or b == v: continue
                                for c in range(b+1, N):
                                    if c == u or c == v: continue
                                    cnt = bin(common[v][b] & adj[c]).count('1')
                                    if cnt == 2:
                                        if (adj[b] >> u) & 1 and (adj[c] >> u) & 1:
                                            valid = False
                                            break
                                if not valid: break
                        if valid:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            for w in range(N):
                                if w == u or w == v: continue
                                if (adj[w] >> v) & 1:
                                    common[u][w] |= (1 << v)
                                    common[w][u] |= (1 << v)
                                if (adj[w] >> u) & 1:
                                    common[v][w] |= (1 << u)
                                    common[w][v] |= (1 << u)
        
        # Update best
        edge_count = sum(bin(adj[i]).count('1') for i in range(N)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]
            best_common = [row[:] for row in common]
            
    # Final check
    if best_adj is None:
        return []
        
    edges_list = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
    return edges_list
