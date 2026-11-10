"""Improved K3,3-free graph construction on 20 vertices using bitmask-based local search."""

import time
import random
import numpy as np

N = 20
DEADLINE_FRACTION = 0.85  # use 85% of the 150s budget

def _triple_common_count(adj, a, b, c):
    """Count common neighbors of vertices a, b, c using bitmasks."""
    return bin(adj[a] & adj[b] & adj[c]).count('1')

def _has_violation(adj):
    """Check if there exists a triple (a,b,c) with >= 3 common neighbors."""
    # For each pair (a,b), count common neighbors
    # A K3,3 exists iff there exist 3 vertices with 3 common neighbors
    # Equivalently: there exist 6 vertices partitioned into two sets of 3 with all 9 edges present
    # Better: for each triple (a,b,c), if they share >=3 common neighbors, violation
    # But that's O(n^3 * bitop) which is 8000 * cheap = fine for 20 vertices
    
    # More efficient: for each pair (i,j), find common neighbors. 
    # If any triple of vertices has 3+ common neighbors, violation.
    # Actually the condition is: no 3 vertices have 3 common neighbors.
    # So we need: for all triples (a,b,c), common_neighbors(a,b,c) <= 2.
    
    # For n=20, triples = C(20,3) = 1140. Each check is 2 ANDs and a popcount.
    # That's very fast.
    
    for a in range(N):
        for b in range(a+1, N):
            ab = adj[a] & adj[b]
            for c in range(b+1, N):
                if (ab & adj[c]) & ((ab & adj[c]) - 1):
                    # This means (ab & adj[c]) has at least 2 bits set
                    # We need to check if it has at least 3 bits
                    common = ab & adj[c]
                    # Check if popcount >= 3
                    # Quick check: if common has bits in at least 3 positions
                    if common & (common - 1):
                        # Has at least 2 bits, check for 3rd
                        temp = common
                        if temp & (temp - 1):
                            temp &= temp - 1
                            if temp & (temp - 1):
                                return True
    return False

def _violations_count(adj):
    """Count number of violating triples (triples with >=3 common neighbors)."""
    count = 0
    for a in range(N):
        for b in range(a+1, N):
            ab = adj[a] & adj[b]
            for c in range(b+1, N):
                common = ab & adj[c]
                # Count bits
                bits = 0
                temp = common
                while temp:
                    bits += 1
                    temp &= temp - 1
                if bits >= 3:
                    count += 1
    return count

def _check_add_edge(adj, u, v):
    """Check if adding edge (u,v) would create a violation. Returns True if safe."""
    # After adding edge (u,v), we need to check all triples that include u or v
    # Actually, adding edge (u,v) can only create new violations involving u or v
    # because it increases common neighbor counts for triples containing both u and v... 
    # No wait: adding edge (u,v) changes adj[u] and adj[v].
    # A triple (a,b,c) has common neighbors = adj[a]&adj[b]&adj[c].
    # Adding edge (u,v) sets bit v in adj[u] and bit u in adj[v].
    # So for a triple (a,b,c), the new common neighbors = old_common & (mask without the new edge effects)
    # Actually the common neighbor set can only grow or stay same when adding edges.
    # So a triple that was safe might become unsafe.
    # The affected triples are those where at least one of a,b,c is u or v,
    # AND the new common neighbor includes the other endpoint.
    # Specifically:
    # - Triple (u, b, c): common neighbors gain v if v is neighbor of b and c (and was already neighbor of u... no)
    #   Wait: common neighbors of (u,b,c) = adj[u] & adj[b] & adj[c].
    #   After adding (u,v): adj[u] gains bit v.
    #   So new common = (old_adj[u] | (1<<v)) & adj[b] & adj[c] = old_common | ((1<<v) & adj[b] & adj[c])
    #   So v is added to common neighbors of (u,b,c) iff v is neighbor of both b and c.
    #   Similarly for triples containing v instead of u.
    # - Triple (u, v, c): common = adj[u]&adj[v]&adj[c]. After adding edge, adj[u] has v, adj[v] has u.
    #   So v is in adj[u] and u is in adj[v], but for common neighbors we look at neighbors of all three.
    #   Common neighbors of (u,v,c) = adj[u]&adj[v]&adj[c].
    #   The edge (u,v) doesn't directly add u or v to common neighbors of (u,v,c) because
    #   u is not a neighbor of u, v is not a neighbor of v. But u is now a neighbor of v and v is neighbor of u.
    #   So u could be a common neighbor of (v, c, x) for some x... this is getting complex.
    #   
    # Simpler approach: just check all triples after tentatively adding the edge.
    # For n=20, that's 1140 triples, each a 2-AND + popcount. Very fast.
    
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    safe = True
    for a in range(N):
        for b in range(a+1, N):
            ab = adj[a] & adj[b]
            for c in range(b+1, N):
                common = ab & adj[c]
                # Check if >= 3 bits
                if common & (common - 1):
                    temp = common
                    temp &= temp - 1
                    if temp & (temp - 1):
                        safe = False
                        break
            if not safe:
                break
        if not safe:
            break
    
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    
    return safe

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    best_adj = [0] * N
    best_edges = 0
    
    # Initial construction: greedy random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 2:
            break
        
        adj = [0] * N
        edges = 0
        
        # Start with a random sparse graph and build up greedily
        # Try adding edges in random order
        candidate_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(candidate_pairs)
        
        for u, v in candidate_pairs:
            if time.time() > deadline - 1:
                break
            if _check_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges += 1
        
        # Local search: try to add more edges
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if (adj[u] >> v) & 1:
                        continue
                    if _check_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges += 1
                        improved = True
        
        # Simulated annealing to escape local optima
        if edges > best_edges:
            best_adj = adj[:]
            best_edges = edges
        
        # SA phase
        current_adj = adj[:]
        current_edges = edges
        T = 2.0
        cooling = 0.995
        
        sa_start = time.time()
        sa_deadline = min(deadline - 0.5, sa_start + 20)
        
        while time.time() < sa_deadline:
            T = max(0.1, T * cooling)
            
            if random.random() < 0.6:
                # Try to add a random edge
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (current_adj[u] >> v) & 1:
                    continue
                if _check_add_edge(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_adj = current_adj[:]
                        best_edges = current_edges
            else:
                # Try to remove a random edge (to escape local optima)
                if current_edges == 0:
                    continue
                # Pick a random existing edge
                found = False
                for _ in range(100):
                    u = random.randint(0, N-1)
                    v = random.randint(0, N-1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (current_adj[u] >> v) & 1:
                        found = True
                        break
                if not found:
                    continue
                
                # Remove it
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges -= 1
                
                # Accept with probability exp(-1/T)
                if random.random() > np.exp(-1.0 / T):
                    # Reject, put it back
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
        
        if current_edges > best_edges:
            best_adj = current_adj[:]
            best_edges = current_edges
    
    # Construct edge list from best_adj
    final_edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                final_edges.append([u, v])
    
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
