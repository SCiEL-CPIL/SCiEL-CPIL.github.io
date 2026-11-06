import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                
                if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                    if (adj[u] & adj[a] & adj[b]).bit_count() > 1:
                        return False
                        
                if (adj[a] & (1 << u)) and (adj[b] & (1 << u)):
                    if (adj[v] & adj[a] & adj[b]).bit_count() > 1:
                        return False
                        
        return True

    def local_search(adj):
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & (1 << j))]
        
        steps = 0
        max_steps = 20000
        
        while steps < max_steps and time.time() < deadline - 3:
            steps += 1
            if not missing:
                break
            
            improved = False
            
            # Try adding a random missing edge
            if missing:
                idx = random.randint(0, len(missing) - 1)
                u, v = missing[idx]
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    missing.pop(idx)
                    existing.append((u, v))
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
            
            # If we couldn't add, try a 2-opt swap: remove one existing, add one missing
            if not improved and existing and missing:
                # Pick a random missing edge to try to add
                mi = random.randint(0, len(missing) - 1)
                u, v = missing[mi]
                
                # Try removing a random existing edge to see if we can add (u,v)
                ei = random.randint(0, len(existing) - 1)
                a, b = existing[ei]
                
                # Temporarily remove (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                if can_add(adj, u, v):
                    # Accept the swap
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    existing[ei] = (u, v)
                    missing[mi] = (a, b)
                    # No net change in edge count
                else:
                    # Revert
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                
                # Occasionally try a kick: remove 2-3 edges, greedily add back
                if random.random() < 0.05:
                    kick_size = random.randint(2, 3)
                    removed = []
                    for _ in range(kick_size):
                        if not existing:
                            break
                        ei = random.randint(0, len(existing) - 1)
                        a, b = existing[ei]
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        existing.pop(ei)
                        missing.append((a, b))
                        removed.append((a, b))
                    
                    # Greedily add back as many as possible
                    random.shuffle(missing)
                    i = 0
                    while i < len(missing):
                        u, v = missing[i]
                        if can_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            missing.pop(i)
                            existing.append((u, v))
                            current_edges += 1
                        else:
                            i += 1
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
            
            # If stuck, break out and let restart handle it
            if not improved and random.random() < 0.01:
                break
        
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts with different seeds
    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        random.seed(i * 7919 + 13)  # Better seed spacing
        
        adj = [0] * n
        
        # Greedy initial construction with random edge ordering
        pairs = all_edges[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        edges = sum(x.bit_count() for x in adj) // 2
        
        # Local Search
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Final polish: try to add any remaining missing edges
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (best_adj[i] & (1 << j))]
    for u, v in missing:
        if can_add(best_adj, u, v):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            best_edges += 1
    
    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
