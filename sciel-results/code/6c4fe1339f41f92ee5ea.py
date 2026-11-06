import time
import random
from itertools import combinations

_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _has_c4(adj_matrix, n):
    """Check if graph has a C4 using bitmasks for speed."""
    # For each pair of vertices, count common neighbors
    # If any pair has >= 2 common neighbors, C4 exists
    # Use numpy for speed
    import numpy as np
    A = np.array(adj_matrix, dtype=np.int8)
    # Count common neighbors: A @ A gives number of length-2 paths
    # But we need to exclude the edge itself
    # Common neighbors of u,v = (A[u] & A[v]).count()
    # For C4-free: for all u != v, common neighbors <= 1
    # This is O(n^3) naive, O(n^2 * n/64) with bit ops
    # For n up to 43, n^2 = 1849 pairs, each with 43 bits -> fast
    for u in range(n):
        for v in range(u+1, n):
            common = 0
            for w in range(n):
                if adj_matrix[u][w] and adj_matrix[v][w]:
                    common += 1
                    if common > 1:
                        return True
    return False

def _max_independent_set_size(adj, n, timeout=5.0):
    """Find size of maximum independent set using branch and bound."""
    start = time.time()
    best = [0]
    
    # Greedy lower bound
    def greedy_is():
        remaining = set(range(n))
        size = 0
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = min(len([v for v in remaining if not adj[u][v]]) for u in remaining)
            candidates = [u for u in remaining if len([v for v in remaining if not adj[u][v]]) == min_deg]
            u = candidates[0]
            size += 1
            remaining.discard(u)
            remaining = {v for v in remaining if not adj[u][v]}
        return size
    
    best[0] = greedy_is()
    
    # Branch and bound
    def branch(remaining, current_size):
        if time.time() - start > timeout:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(remaining) <= best[0]:
            return
        
        # Pick a vertex to branch on
        u = next(iter(remaining))
        # Option 1: include u in independent set
        new_remaining = {v for v in remaining if v != u and not adj[u][v]}
        branch(new_remaining, current_size + 1)
        # Option 2: exclude u
        remaining_copy = set(remaining)
        remaining_copy.discard(u)
        branch(remaining_copy, current_size)
    
    branch(set(range(n)), 0)
    return best[0]

def _is_c4_free(adj, n):
    """Check C4-free using bitmasks."""
    # Convert to bitmask representation
    masks = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                masks[i] |= (1 << j)
    
    for u in range(n):
        for v in range(u+1, n):
            common = masks[u] & masks[v]
            # Count bits
            if bin(common).count('1') > 1:
                return False
    return True

def _indep_number(adj, n, timeout=3.0):
    """Compute independence number with branch and bound."""
    start = time.time()
    best = [0]
    
    # Precompute neighbor masks
    masks = [0] * n
    non_masks = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if adj[i][j]:
                    masks[i] |= (1 << j)
                else:
                    non_masks[i] |= (1 << j)
    
    def popcount(x):
        return bin(x).count('1')
    
    def branch(available_mask, current_size):
        if time.time() - start > timeout:
            return
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + popcount(available_mask) <= best[0]:
            return
        
        # Pick vertex with fewest options
        # Try all vertices in available_mask
        m = available_mask
        # Find a vertex to branch on - pick one that maximizes pruning
        # Simple: pick first
        u_bit = m & (-m)
        u = u_bit.bit_length() - 1
        
        # Include u: remove u and its neighbors
        new_avail = available_mask & ~(masks[u] | (1 << u))
        branch(new_avail, current_size + 1)
        
        # Exclude u
        branch(available_mask & ~(1 << u), current_size)
    
    full_mask = (1 << n) - 1
    branch(full_mask, 0)
    return best[0]

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_n = 30
    best_edges = []
    
    # Start with 10 triangles
    adj = [[False]*30 for _ in range(30)]
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        adj[a][b] = adj[b][a] = True
        adj[a][c] = adj[c][a] = True
        adj[b][c] = adj[c][b] = True
    
    # Try to add vertices 30, 31, ..., up to 43
    for new_n in range(31, 44):
        if _time_remaining() < 15.0:
            break
        
        # Try multiple times to add a vertex
        success = False
        attempts = 0
        max_attempts = 50
        
        while not success and attempts < max_attempts and _time_remaining() > 5.0:
            attempts += 1
            
            # Create new adjacency matrix
            new_adj = [row[:] for row in adj]
            new_adj.append([False]*new_n)
            for i in range(new_n - 1):
                new_adj[i].append(False)
            
            v = new_n - 1
            
            # Decide which vertices to connect v to
            # To maintain C4-free: for any two existing vertices u1, u2 that are both neighbors of v,
            # they must have at most 1 common neighbor (excluding v)
            # Equivalently: the set of neighbors of v must be a "C4-free set" in the existing graph
            
            # Greedy: try to add edges to v
            # Start with no edges, then add edges greedily
            # Or: try random subsets
            
            # Strategy: pick a random subset of vertices to connect to v
            # The subset must satisfy: for any two vertices in the subset, 
            # they have at most 1 common neighbor in the existing graph (not counting v)
            
            # This is equivalent to: the induced subgraph on the subset has the property that
            # no two vertices share 2+ common neighbors in the FULL graph restricted to non-subset vertices
            # Actually simpler: for u1, u2 both in S (neighbors of v), 
            # |N(u1) ∩ N(u2) ∩ (V \ {v})| <= 1
            
            # Let's build S greedily
            S = set()
            vertices = list(range(new_n - 1))
            random.shuffle(vertices)
            
            for u in vertices:
                # Check if adding u to S maintains C4-free
                ok = True
                for w in S:
                    # Count common neighbors of u and w in existing graph (excluding v)
                    common = 0
                    for x in range(new_n - 1):
                        if x != u and x != w and adj[u][x] and adj[w][x]:
                            common += 1
                            if common > 1:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    S.add(u)
            
            # Now check if adding v with neighbors S keeps alpha <= 10
            # The independent set in the new graph: either includes v (then can't include S) or not
            # If includes v: independent set is {v} ∪ (independent set in G - (S ∪ {v}))
            # If not: independent set is independent set in G - {v}
            
            # For the new graph, alpha <= 10 means:
            # max(alpha(G - {v}), 1 + alpha(G - (S ∪ {v}))) <= 10
            
            # G - {v} is just the original graph, which has alpha <= 10 (we maintain this)
            # So we need: 1 + alpha(G - (S ∪ {v})) <= 10, i.e., alpha(G - (S ∪ {v})) <= 9
            
            # Check: remove S and v from graph, compute alpha
            remaining = [i for i in range(new_n - 1) if i not in S]
            if len(remaining) <= 9:
                # Trivially alpha <= 9
                alpha_rem = len(remaining)
            else:
                # Build subgraph on remaining vertices
                rem_adj = [[False]*len(remaining) for _ in range(len(remaining))]
                rem_idx = {v: i for i, v in enumerate(remaining)}
                for i, u in enumerate(remaining):
                    for j, w in enumerate(remaining):
                        if i != j and adj[u][w]:
                            rem_adj[i][j] = True
                
                # Quick check: if remaining has no edges, alpha = len(remaining)
                has_edges = any(rem_adj[i][j] for i in range(len(remaining)) for j in range(i+1, len(remaining)))
                if not has_edges:
                    alpha_rem = len(remaining)
                else:
                    # Use branch and bound with short timeout
                    alpha_rem = _indep_number(rem_adj, len(remaining), timeout=1.0)
            
            if alpha_rem <= 9:
                # Valid! Add v with neighbors S
                for u in S:
                    new_adj[v][u] = True
                    new_adj[u][v] = True
                adj = new_adj
                success = True
                break
            
            # If not valid, try again with different random S
        
        if not success:
            # Try a different strategy: try all single-vertex connections
            # Or just give up on this vertex
            # Try connecting v to as few vertices as possible
            # Minimum: connect to 0 vertices (isolated) - but that increases alpha by 1
            # So we can only add an isolated vertex if current alpha <= 9
            
            # Check current alpha
            cur_alpha = _indep_number(adj, new_n - 1, timeout=2.0)
            if cur_alpha <= 9:
                # Add isolated vertex
                new_adj = [row[:] for row in adj]
                new_adj.append([False]*new_n)
                for i in range(new_n - 1):
                    new_adj[i].append(False)
                adj = new_adj
                success = True
            else:
                # Cannot add vertex, stop
                break
        
        if success:
            # Verify C4-free
            if _is_c4_free(adj, new_n):
                # Verify alpha <= 10
                alpha = _indep_number(adj, new_n, timeout=3.0)
                if alpha <= 10:
                    best_n = new_n
                else:
                    break
            else:
                break
        else:
            break
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
