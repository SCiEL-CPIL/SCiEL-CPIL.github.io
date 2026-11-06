import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        """
        if adj[u] & (1 << v):
            return False
            
        # Check triples containing u: {u, a, b}
        # v is added to N({u, a, b}) if v is in N(a) and N(b).
        # We need old |N({u, a, b})| <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # u is added to N({v, a, b}) if u is in N(a) and N(b).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_fill(adj):
        """
        Repeatedly scan all pairs and add edges if valid.
        Returns the number of edges added.
        """
        added = 0
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Randomize order to avoid bias
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            random.shuffle(pairs)
            for u, v in pairs:
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added += 1
                    improved = True
        return added

    def local_search(adj):
        """
        Local search with kicks and swaps.
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Initial fill
        greedy_fill(current_adj)
        current_edges = count_edges(current_adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
            
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try 2-in, 2-out swap
            # Pick 2 edges to remove, 2 to add
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if len(edges) >= 2:
                e1, e2 = random.sample(edges, 2)
                # Remove e1, e2
                u1, v1 = e1
                u2, v2 = e2
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                
                # Try to add 2 new edges
                added = []
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
                random.shuffle(missing)
                for a, b in missing:
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added.append((a, b))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    # Net 0 change, but structure changed
                    # Check if we can add more
                    greedy_fill(current_adj)
                    current_edges = count_edges(current_adj)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                else:
                    # Revert
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
            else:
                # Not enough edges for swap, try kick
                break
            
            # Try Kick: remove k random edges, then greedy fill
            if random.random() < 0.3:
                kick_size = random.randint(1, 3)
                if len(edges) >= kick_size:
                    kick_edges = random.sample(edges, kick_size)
                    for u, v in kick_edges:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                    
                    greedy_fill(current_adj)
                    current_edges = count_edges(current_adj)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    
                    # If kick didn't improve, we might want to revert to best?
                    # No, keep current state for further search, but track best.
                    
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Spectral Initialization Heuristic
    # Generate random symmetric matrix, get top eigenvector, threshold
    import numpy as np
    
    def spectral_init():
        # Random symmetric matrix
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        # Top eigenvector
        eigvals, eigvecs = np.linalg.eigh(A)
        v = eigvecs[:, -1]
        # Threshold to create initial graph
        # We want a dense graph, so threshold at mean
        mean_val = np.mean(v * v.T)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if v[i] * v[j] > mean_val:
                    # Check validity before adding? 
                    # Spectral init might violate K3,3.
                    # We must repair it.
                    # For now, just add if valid
                    if is_valid_addition(adj, i, j):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        return adj

    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Alternate between random and spectral init
        if i % 2 == 0:
            # Random sparse init
            adj = [0] * n
            # Add a few random edges
            for _ in range(10):
                u, v = random.sample(range(n), 2)
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            adj = spectral_init()
        
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
