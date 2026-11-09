"""
Greedy sequential vertex addition to construct a large graph with no regular induced subgraph on 6+ vertices.
"""
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin
    
    # Target n=20
    target_n = 20
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * target_n
    
    # Greedy construction
    for k in range(1, target_n):
        if time.time() > deadline - 10:
            break
            
        # Find best connection pattern for vertex k to vertices 0..k-1
        best_pattern = None
        best_violations = float('inf')
        
        # Try 100 random patterns
        rng = random.Random(k * 123 + 456)
        for _ in range(100):
            pattern = 0
            for i in range(k):
                if rng.random() < 0.5:
                    pattern |= (1 << i)
            
            violations = _count_new_violations(adj, k, pattern)
            if violations < best_violations:
                best_violations = violations
                best_pattern = pattern
                if violations == 0:
                    break
        
        if best_pattern is None:
            best_pattern = 0
        
        adj[k] = best_pattern
        for i in range(k):
            if (best_pattern >> i) & 1:
                adj[i] |= (1 << k)
        
        # If we found a pattern with 0 violations, we are good.
        # Otherwise, we might need to repair later, but greedy is fast.
        
    # Convert to edges
    edges = []
    for i in range(target_n):
        for j in range(i + 1, target_n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
                
    # Verify validity
    if _is_valid_fast(adj, target_n):
        return {'n': target_n, 'edges': edges}
    
    # If invalid, try local search to fix
    # We can flip edges to reduce violations
    rng = random.Random(999)
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        if _is_valid_fast(adj, target_n):
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': target_n, 'edges': edges}
        
        # Find a violation
        mask, size = _find_violation_fast(adj, target_n, rng)
        if mask is None:
            # Random flip
            u, v = rng.sample(range(target_n), 2)
            _flip_edge(adj, u, v)
        else:
            # Flip edge in violation
            vertices = [i for i in range(target_n) if (mask >> i) & 1]
            u, v = rng.sample(vertices, 2)
            _flip_edge(adj, u, v)
            
    # Fallback to smaller n if 20 failed
    for n in range(target_n - 1, 5, -1):
        if time.time() > deadline - 2:
            break
        adj_small = adj[:n]
        # Need to rebuild adj for small n? No, adj is global.
        # But we need to ensure no edges to vertices >= n.
        # In our construction, adj[i] only has bits for j < i or j > i?
        # We set adj[k] for k < target_n.
        # So adj[i] might have bits for j >= n.
        # We need to mask them out.
        mask_limit = (1 << n) - 1
        adj_masked = [a & mask_limit for a in adj_small]
        
        if _is_valid_fast(adj_masked, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_masked[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return {'n': 5, 'edges': []}

def _flip_edge(adj, u, v):
    if (adj[u] >> v) & 1:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    else:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def _is_regular_mask(adj, mask, n):
    vertices = [i for i in range(n) if (mask >> i) & 1]
    size = len(vertices)
    if size < 6:
        return False
    degrees = []
    for v in vertices:
        deg = bin(adj[v] & mask).count('1')
        degrees.append(deg)
    return all(d == degrees[0] for d in degrees)

def _is_valid_fast(adj, n):
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_mask(adj, mask, n):
                return False
    return True

def _find_violation_fast(adj, n, rng):
    for _ in range(50):
        subset = rng.sample(range(n), 6)
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_regular_mask(adj, mask, n):
            return mask, 6
    return None, 0

def _count_new_violations(adj, k, pattern):
    # Count how many subsets T of {0..k-1} with size >= 5
    # would form a regular subgraph T U {k} with connection pattern 'pattern'
    count = 0
    # Only check subsets of size 5 and 6 for speed?
    # Size 5 subsets: T has 5 vertices. T U {k} has 6.
    # Size 6 subsets: T has 6 vertices. T U {k} has 7.
    # We check size 5 and 6 subsets of old vertices.
    
    # Precompute masks for subsets of size 5 and 6?
    # Just iterate combinations for speed in this limited context.
    for size in [5, 6]:
        for subset in itertools.combinations(range(k), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check if T U {k} is regular
            # m = number of neighbors of k in T
            m = bin(pattern & mask).count('1')
            
            # Check condition:
            # For all u in T, deg_T(u) + I(u~k) = m
            is_reg = True
            for u in subset:
                deg_u = bin(adj[u] & mask).count('1')
                conn = 1 if (pattern >> u) & 1 else 0
                if deg_u + conn != m:
                    is_reg = False
                    break
            
            if is_reg:
                count += 1
    return count
