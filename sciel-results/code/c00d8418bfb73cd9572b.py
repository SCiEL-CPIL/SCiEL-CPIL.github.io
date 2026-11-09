import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n = 16
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    best_edges = set()
    best_count = 0

    # Precompute all pairs of disjoint 4-sets
    # A is a 4-set, B is a 4-set, A and B are disjoint.
    # We generate A, then B from the remaining 12 vertices.
    # To avoid duplicates (A,B) and (B,A), we can enforce some ordering, 
    # but since the check is symmetric, we can just generate all and maybe halve it?
    # Or just generate all. 450k pairs is fine.
    
    all_4sets = list(itertools.combinations(range(n), 4))
    
    # Map tuple to index for faster lookup? Or just use tuples.
    # We need to check disjointness.
    # Let's precompute the list of (A, B) pairs.
    pairs_AB = []
    for A in all_4sets:
        A_mask = 0
        for v in A:
            A_mask |= (1 << v)
        # B must be a 4-set from the remaining 12 vertices
        remaining = [v for v in range(n) if not (A_mask >> v) & 1]
        for B in itertools.combinations(remaining, 4):
            pairs_AB.append((A, B))
            
    # There are 1820 * 495 = 900900 pairs.
    # This is 2x the previous 450k estimate because (A,B) and (B,A) are distinct in this generation.
    # But the check is symmetric. We can just run it.
    
    def check_q3_fast(adj):
        """Fast check: return True if Q3 exists, else False."""
        for A, B in pairs_AB:
            # Check no edges in A
            a0, a1, a2, a3 = A
            if (adj[a0] >> a1) & 1: continue
            if (adj[a0] >> a2) & 1: continue
            if (adj[a0] >> a3) & 1: continue
            if (adj[a1] >> a2) & 1: continue
            if (adj[a1] >> a3) & 1: continue
            if (adj[a2] >> a3) & 1: continue
            
            # Check no edges in B
            b0, b1, b2, b3 = B
            if (adj[b0] >> b1) & 1: continue
            if (adj[b0] >> b2) & 1: continue
            if (adj[b0] >> b3) & 1: continue
            if (adj[b1] >> b2) & 1: continue
            if (adj[b1] >> b3) & 1: continue
            if (adj[b2] >> b3) & 1: continue
            
            # Check degrees: each a in A must have 3 neighbors in B
            # Each b in B must have 3 neighbors in A
            
            # Build B mask
            B_mask = (1 << b0) | (1 << b1) | (1 << b2) | (1 << b3)
            A_mask = (1 << a0) | (1 << a1) | (1 << a2) | (1 << a3)
            
            if bin(adj[a0] & B_mask).count('1') != 3: continue
            if bin(adj[a1] & B_mask).count('1') != 3: continue
            if bin(adj[a2] & B_mask).count('1') != 3: continue
            if bin(adj[a3] & B_mask).count('1') != 3: continue
            
            if bin(adj[b0] & A_mask).count('1') != 3: continue
            if bin(adj[b1] & A_mask).count('1') != 3: continue
            if bin(adj[b2] & A_mask).count('1') != 3: continue
            if bin(adj[b3] & A_mask).count('1') != 3: continue
            
            return True
        return False

    def find_q3_edge_to_remove(adj):
        """Find a Q3 and return an edge in it to remove. Returns None if no Q3."""
        for A, B in pairs_AB:
            a0, a1, a2, a3 = A
            if (adj[a0] >> a1) & 1: continue
            if (adj[a0] >> a2) & 1: continue
            if (adj[a0] >> a3) & 1: continue
            if (adj[a1] >> a2) & 1: continue
            if (adj[a1] >> a3) & 1: continue
            if (adj[a2] >> a3) & 1: continue
            
            b0, b1, b2, b3 = B
            if (adj[b0] >> b1) & 1: continue
            if (adj[b0] >> b2) & 1: continue
            if (adj[b0] >> b3) & 1: continue
            if (adj[b1] >> b2) & 1: continue
            if (adj[b1] >> b3) & 1: continue
            if (adj[b2] >> b3) & 1: continue
            
            B_mask = (1 << b0) | (1 << b1) | (1 << b2) | (1 << b3)
            A_mask = (1 << a0) | (1 << a1) | (1 << a2) | (1 << a3)
            
            if bin(adj[a0] & B_mask).count('1') != 3: continue
            if bin(adj[a1] & B_mask).count('1') != 3: continue
            if bin(adj[a2] & B_mask).count('1') != 3: continue
            if bin(adj[a3] & B_mask).count('1') != 3: continue
            
            if bin(adj[b0] & A_mask).count('1') != 3: continue
            if bin(adj[b1] & A_mask).count('1') != 3: continue
            if bin(adj[b2] & A_mask).count('1') != 3: continue
            if bin(adj[b3] & A_mask).count('1') != 3: continue
            
            # Found a Q3. Return an edge between A and B.
            # We can return any edge. Let's pick (a0, b0) if it exists, else find one.
            # Since degrees are 3, there are 12 edges between A and B.
            # Just return (min(a0,b0), max(a0,b0)) if it's an edge?
            # It might not be an edge.
            # Let's find the first existing edge.
            for a in A:
                for b in B:
                    if (adj[a] >> b) & 1:
                        return (min(a, b), max(a, b))
        return None

    def try_add_edge(adj, edges, u, v):
        """Try to add edge (u,v). Returns True if added successfully."""
        e = (min(u,v), max(u,v))
        if e in edges:
            return True
        edges.add(e)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not check_q3_fast(adj):
            return True
        else:
            edges.remove(e)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    # Multiple restarts with different densities
    random.seed(42)
    
    for trial in range(8):
        if time.time() > deadline - 30:
            break
        
        density = 0.75 + 0.05 * trial
        edges = set()
        adj = [0] * n
        for e in all_edges:
            if random.random() < density:
                edges.add(e)
                u, v = e
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Phase 1: Remove edges to make Q3-free
        removals = 0
        while time.time() < deadline - 25 and removals < 50:
            e_to_remove = find_q3_edge_to_remove(adj)
            if e_to_remove is None:
                break
            edges.remove(e_to_remove)
            u, v = e_to_remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            removals += 1
        
        # Verify Q3-free
        if check_q3_fast(adj):
            # Still has Q3, try more removals
            while time.time() < deadline - 20:
                e_to_remove = find_q3_edge_to_remove(adj)
                if e_to_remove is None:
                    break
                edges.remove(e_to_remove)
                u, v = e_to_remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Phase 2: Try to add edges
        added_any = True
        while added_any and time.time() < deadline - 15:
            added_any = False
            non_edges = [e for e in all_edges if e not in edges]
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                if try_add_edge(adj, edges, u, v):
                    added_any = True
                    break
        
        # Update best
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            # Save a copy
            best_edges = set(edges)

    # Final polish on best
    if best_edges and time.time() < deadline - 10:
        adj = [0] * n
        for (u, v) in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try adding more edges
        added_any = True
        while added_any and time.time() < deadline - 5:
            added_any = False
            non_edges = [e for e in all_edges if e not in best_edges]
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                e = (min(u,v), max(u,v))
                if e not in best_edges:
                    best_edges.add(e)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not check_q3_fast(adj):
                        added_any = True
                        break
                    else:
                        best_edges.remove(e)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
