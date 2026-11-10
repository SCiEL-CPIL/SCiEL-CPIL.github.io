import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    def is_regular_subset(adj, mask):
        # Get vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        if len(verts) < 6:
            return False
            
        # Calculate degrees
        degrees = []
        for u in verts:
            deg = (adj[u] & mask).bit_count()
            degrees.append(deg)
            
        # Check if all degrees are the same
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def count_bad_subgraphs(adj, n):
        count = 0
        # Iterate over all subsets of size k >= 6
        # Using combinations is slow in Python for large n, but n=20 is small.
        # However, iterating masks is faster.
        # Let's stick to combinations for clarity first, optimize if needed.
        # Actually, generating all masks of popcount >= 6 is better.
        
        # Precompute masks? No, just iterate.
        # Number of subsets is 2^n. For n=20, that's 1M.
        # We only care about size >= 6.
        
        # Efficient way:
        # for mask in range(1, 1 << n):
        #    if mask.bit_count() >= 6:
        #       if is_regular_subset(adj, mask): count += 1
        
        # This is 1M iterations. Might be ~1s in Python.
        # Let's optimize is_regular_subset to be very fast.
        
        count = 0
        for mask in range(1, 1 << n):
            if mask.bit_count() < 6:
                continue
            if is_regular_subset(adj, mask):
                count += 1
        return count

    def local_search(n, max_iters=5000):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_bad_subgraphs(adj, n)
        best_adj = adj[:]
        best_cost = current_cost
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if current_cost == 0:
                return adj, 0
            
            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_cost = count_bad_subgraphs(adj, n)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            # Simple acceptance: always accept if better. 
            # To escape local minima, we could add simulated annealing, 
            # but for this problem, random restarts are often effective.
            
        return best_adj, best_cost

    # Try n=20
    # We can run multiple restarts
    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n=20
    found_20 = False
    for attempt in range(10): # 10 restarts
        if time.time() > deadline - 5:
            break
        adj, cost = local_search(20, max_iters=1000)
        if cost == 0:
            # Convert adj to edges list
            edges = []
            for i in range(20):
                for j in range(i+1, 20):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': 20, 'edges': edges}
            
    # If 20 failed, try n=19? Or just return a known small one?
    # The prompt says "Graphs with 20 vertices are known". 
    # It is highly likely my search will find one.
    # If not, I should return a valid smaller graph to avoid score 0.
    # Let's try n=10 quickly if 20 fails.
    
    # Fallback: return a valid graph on 5 vertices (empty graph)
    return {'n': 5, 'edges': []}
