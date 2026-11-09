import sys
import time
import random

P = {'n': 15}

def _construct_graph(n):
    """
    Build a valid graph on n vertices (15) maximizing edge count.
    Valid: no simple 3-edge path s-u-v-t has deg(s) == deg(t).
    Uses bitset-based validation for speed.
    """
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    n = P['n']
    
    def is_valid(adj_bits, n):
        """Fast validation using bitsets. adj_bits[i] is bitmask of neighbors of i."""
        deg = [adj_bits[i].bit_count() for i in range(n)]
        # Group vertices by degree
        from collections import defaultdict
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)
        
        # For each degree group, check if any pair has a 3-edge path between them
        for d, group in deg_groups.items():
            if len(group) < 2:
                continue
            # For each pair in group, check if there's a 3-edge path
            for idx in range(len(group)):
                s = group[idx]
                for idx2 in range(idx+1, len(group)):
                    t = group[idx2]
                    # Check if there exists u, v with: s-u-v-t
                    # s adjacent to u, u adjacent to v, v adjacent to t
                    # u != s, u != t, v != s, v != t, u != v
                    found = False
                    for u in range(n):
                        if u == s or u == t:
                            continue
                        if not (adj_bits[s] >> u) & 1:
                            continue
                        # u is neighbor of s
                        # Now need v neighbor of u and v neighbor of t, v != s, v != t, v != u
                        candidates_v = adj_bits[u] & adj_bits[t]
                        # Remove s, t, u from candidates
                        candidates_v &= ~(1 << s)
                        candidates_v &= ~(1 << t)
                        candidates_v &= ~(1 << u)
                        if candidates_v:
                            found = True
                            break
                    if found:
                        return False
        return True
    
    def count_edges(adj_bits, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    count += 1
        return count
    
    best_edges = []
    best_count = -1
    
    # Try multiple random starts with different densities
    random.seed(42)
    densities = [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    
    for density in densities:
        if time.time() > deadline:
            break
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Create random graph with given density
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < density:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Greedy edge removal to fix violations
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                if is_valid(adj_bits, n):
                    break
                
                # Find a violating pair and remove an edge
                deg = [adj_bits[i].bit_count() for i in range(n)]
                from collections import defaultdict
                deg_groups = defaultdict(list)
                for i in range(n):
                    deg_groups[deg[i]].append(i)
                
                removed = False
                for d, group in deg_groups.items():
                    if removed or len(group) < 2:
                        continue
                    for idx in range(len(group)):
                        if removed:
                            break
                        s = group[idx]
                        for idx2 in range(idx+1, len(group)):
                            if removed:
                                break
                            t = group[idx2]
                            # Find a 3-edge path s-u-v-t and remove an edge
                            for u in range(n):
                                if removed:
                                    break
                                if u == s or u == t:
                                    continue
                                if not (adj_bits[s] >> u) & 1:
                                    continue
                                candidates_v = adj_bits[u] & adj_bits[t]
                                candidates_v &= ~(1 << s)
                                candidates_v &= ~(1 << t)
                                candidates_v &= ~(1 << u)
                                if candidates_v:
                                    # Pick a v
                                    v = (candidates_v & -candidates_v).bit_length() - 1
                                    # Remove one of the edges in the path
                                    # Remove (s, u) to break the path
                                    adj_bits[s] &= ~(1 << u)
                                    adj_bits[u] &= ~(1 << s)
                                    removed = True
                                    break
                            if removed:
                                break
                        if removed:
                            break
                    if removed:
                        break
                
                if not removed:
                    break
            
            if is_valid(adj_bits, n):
                c = count_edges(adj_bits, n)
                if c > best_count:
                    best_count = c
                    best_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj_bits[i] >> j) & 1:
                                best_edges.append([i, j])
    
    # Also try a structured approach: start from K15 and remove edges
    if time.time() < deadline:
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
        
        # Greedily remove edges to fix violations
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            if is_valid(adj_bits, n):
                break
            
            deg = [adj_bits[i].bit_count() for i in range(n)]
            from collections import defaultdict
            deg_groups = defaultdict(list)
            for i in range(n):
                deg_groups[deg[i]].append(i)
            
            removed = False
            for d, group in deg_groups.items():
                if removed or len(group) < 2:
                    continue
                for idx in range(len(group)):
                    if removed:
                        break
                    s = group[idx]
                    for idx2 in range(idx+1, len(group)):
                        if removed:
                            break
                        t = group[idx2]
                        for u in range(n):
                            if removed:
                                break
                            if u == s or u == t:
                                continue
                            if not (adj_bits[s] >> u) & 1:
                                continue
                            candidates_v = adj_bits[u] & adj_bits[t]
                            candidates_v &= ~(1 << s)
                            candidates_v &= ~(1 << t)
                            candidates_v &= ~(1 << u)
                            if candidates_v:
                                v = (candidates_v & -candidates_v).bit_length() - 1
                                adj_bits[s] &= ~(1 << u)
                                adj_bits[u] &= ~(1 << s)
                                removed = True
                                break
                        if removed:
                            break
                    if removed:
                        break
                if removed:
                    break
            
            if not removed:
                break
        
        if is_valid(adj_bits, n):
            c = count_edges(adj_bits, n)
            if c > best_count:
                best_count = c
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_bits[i] >> j) & 1:
                            best_edges.append([i, j])
    
    if best_count > 0:
        return best_edges
    
    # Fallback: return a star
    return [[0, v] for v in range(1, n)]

def construct():
    edges = _construct_graph(P['n'])
    # Ensure types
    result = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    return result
