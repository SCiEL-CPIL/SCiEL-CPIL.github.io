import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build polarity graph (93 edges)
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    polarity_edges = len(edges)
    
    if polarity_edges < m:
        # Fallback
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    # Greedily remove (polarity_edges - m) edges with highest C4 participation
    # Since it's a srg, we can just remove any 2, but let's try to be smart.
    # Actually, let's just remove the first 2 and let the local search fix it if needed?
    # No, local search preserves edge count. We MUST start with 91.
    # Let's remove 2 edges.
    
    # Calculate C4 count for a given adj mask
    def count_c4s(adj_mask):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                common = adj_mask[i] & adj_mask[j]
                c = common.bit_count()
                total += c * (c - 1) // 2
        return total // 2
    
    # Remove 2 edges
    # To minimize C4s, we should remove edges that are in many C4s.
    # In a srg, all edges are in the same number of C4s.
    # So we can just remove any 2.
    # Let's remove (0,1) and (0,2) if they exist, or just the first 2 in the list.
    # But wait, the local search will move around.
    # Let's just pick 2 random edges to remove to break symmetry?
    # Or deterministic first 2.
    
    edges_to_remove = edges[:2]
    for u, v in edges_to_remove:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.remove((u, v))
        
    # Now we have 91 edges.
    current_adj = adj[:]
    current_c4 = count_c4s(current_adj)
    best_adj = current_adj[:]
    best_c4 = current_c4
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of all pairs for random selection?
    # Or just generate random u,v and check.
    
    while time.time() < deadline - 2.0 and T > min_T:
        # Pick random edge to remove
        # We need to pick an edge that exists.
        # We can pick random u, v and check if edge exists.
        # Or maintain a list of edges.
        # Maintaining a list is faster for picking.
        # But updating the list is O(1) if we just swap and pop.
        
        # Let's use a set of edges for O(1) check and list for iteration?
        # Or just random u, v.
        # n=31, so 31*30/2 = 465 pairs.
        # Random u, v:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if not (current_adj[u] >> v) & 1:
            continue
            
        # Pick random non-edge to add
        w = random.randint(0, n-1)
        x = random.randint(0, n-1)
        if w == x:
            continue
        if (current_adj[w] >> x) & 1:
            continue
            
        # Ensure we don't add the same edge we removed
        if (w == u and x == v) or (w == v and x == u):
            continue
            
        # Calculate delta
        # Affected nodes: u, v, w, x
        A = {u, v, w, x}
        
        # List of affected pairs
        affected_pairs = []
        for a in A:
            for b in range(n):
                if a < b:
                    affected_pairs.append((a, b))
                elif b < a:
                    affected_pairs.append((b, a))
        
        # Remove duplicates and sort?
        # The loop above might add duplicates if A has duplicates? No, A is a set.
        # But (a,b) and (b,a) are handled by a<b check.
        # However, if a=1, b=2 is added when a=1, and a=2, b=1 is skipped.
        # But if a=1 is in A, we add (1,2).
        # If a=2 is in A, we add (1,2) again?
        # Yes. So we need unique pairs.
        
        affected_pairs = list(set(affected_pairs))
        
        # Save old counts
        old_counts = {}
        for a, b in affected_pairs:
            c = (current_adj[a] & current_adj[b]).bit_count()
            old_counts[(a,b)] = c
            
        # Apply move
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        current_adj[w] |= (1 << x)
        current_adj[x] |= (1 << w)
        
        # Calculate new counts and delta
        delta_sum = 0
        for a, b in affected_pairs:
            new_c = (current_adj[a] & current_adj[b]).bit_count()
            old_c = old_counts[(a,b)]
            delta_sum += (new_c * (new_c - 1) // 2) - (old_c * (old_c - 1) // 2)
            
        delta_c4 = delta_sum // 2
        
        # Acceptance criterion
        if delta_c4 < 0 or random.random() < np.exp(-delta_c4 / T):
            current_c4 += delta_c4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = current_adj[:]
        else:
            # Revert
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_adj[w] &= ~(1 << x)
            current_adj[x] &= ~(1 << w)
            
        T *= cooling
    
    # Return best found
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    # Verify
    assert len(result) == m, f"Expected {m} edges, got {len(result)}"
    for e in result:
        u, v = e
        assert 0 <= u < n and 0 <= v < n
        assert u != v
        
    return result
# EVOLVE-BLOCK-END
