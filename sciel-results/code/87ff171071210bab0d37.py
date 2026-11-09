import time
import sys

# EVOLVE-BLOCK-START

def _generate_polarity_graph_q5():
    """
    Generate the polarity graph of PG(2,5) with polarity x1*x1' + x2*x2' + x3*x3' = 0.
    Returns adjacency list as list of sets. 31 vertices.
    """
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n = len(points)
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            dot = (pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    return adj

def _has_independent_set_k_fast(adj, n, k):
    """
    Fast check for independent set of size k using branch and bound.
    Returns True if such a set exists.
    """
    if k <= 1:
        return True
    if k > n:
        return False
    
    # Convert to bitmasks for speed
    # n <= 43, so we can use Python ints as bitmasks
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    all_mask = (1 << n) - 1
    
    def _search(remaining_mask, count):
        if count == k:
            return True
        if remaining_mask == 0:
            return False
        # Upper bound: popcount of remaining
        if count + bin(remaining_mask).count('1') < k:
            return False
        
        # Pick a vertex from remaining
        # Use lowest set bit for determinism
        v_bit = remaining_mask & (-remaining_mask)
        v = v_bit.bit_length() - 1
        remaining_mask ^= v_bit
        
        # Option 1: include v
        new_remaining = remaining_mask & ~adj_mask[v]
        if _search(new_remaining, count + 1):
            return True
        
        # Option 2: exclude v
        if _search(remaining_mask, count):
            return True
        
        return False
    
    return _search(all_mask, 0)

def _verify_c4_free(adj, n):
    """Check if graph is C4-free."""
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate PG(2,5) polarity graph
    adj_31 = _generate_polarity_graph_q5()
    n_31 = 31
    
    # Verify C4-free
    if not _verify_c4_free(adj_31, n_31):
        # Should not happen, but fallback
        return {'n': 10, 'edges': []}
    
    # Collect base edges
    base_edges = []
    for u in range(n_31):
        for v in adj_31[u]:
            if u < v:
                base_edges.append([u, v])
    
    best_n = 0
    best_edges = []
    
    # Try adding k disjoint edges to the 31-vertex graph
    # For disjoint union, alpha_total = alpha_31 + k
    # We need alpha_total < 11, so k <= 10 - alpha_31
    # But we don't know alpha_31 exactly, so we test each k
    
    for k in range(0, 12):
        if time.time() > deadline - 20:
            break
        
        total_n = n_31 + 2 * k
        if total_n > 43:
            break
        
        # Build full adjacency
        adj_total = [set() for _ in range(total_n)]
        for i in range(n_31):
            adj_total[i] = set(adj_31[i])
        
        offset = n_31
        for i in range(k):
            u = offset + 2 * i
            v = offset + 2 * i + 1
            adj_total[u].add(v)
            adj_total[v].add(u)
        
        # Verify C4-free (disjoint union of C4-free graphs is C4-free)
        # The edges we add are isolated, so no new C4 can form
        # But let's verify to be safe
        if not _verify_c4_free(adj_total, total_n):
            continue
        
        # Check no independent set of 11
        if _has_independent_set_k_fast(adj_total, total_n, 11):
            continue
        
        # Valid
        if total_n > best_n:
            best_n = total_n
            best_edges = base_edges[:]
            offset = n_31
            for i in range(k):
                u = offset + 2 * i
                v = offset + 2 * i + 1
                best_edges.append([u, v])
    
    # If no valid configuration found with edges, check base alone
    if best_n == 0:
        if not _has_independent_set_k_fast(adj_31, n_31, 11):
            best_n = n_31
            best_edges = base_edges
        else:
            # Fallback: return a small valid graph
            best_n = 10
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
