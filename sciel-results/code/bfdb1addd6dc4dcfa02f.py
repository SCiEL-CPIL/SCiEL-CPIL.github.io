import numpy as np
import time
import random
import math

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # 1. Construct the base graph: Polarity Graph of PG(2,5)
    # Points in PG(2,5)
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    # Use bitmask representation for fast C4 counting
    # adj_mask[i] is a bitmask of neighbors of vertex i
    adj_mask = [0] * n
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    
    # O(1) delta calculation using bitmasks
    # C4 count for adding edge (u,v): number of common neighbors w,x where (w,x) is an edge
    # Actually, adding edge (u,v) creates C4s where u-w-x-v-u forms a cycle
    # This equals: for each w in N(u), for each x in N(v), if (w,x) is edge, that's a C4
    # But we need to be careful: the C4 is u-w-x-v, so w != v, x != u, and (w,x) must be an edge
    
    def count_c4_add(u, v):
        """Count C4s created by adding edge (u,v) to current graph."""
        # N(u) without v, N(v) without u
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        count = 0
        # For each w in N(u)\{v}, count edges from w to N(v)\{u}
        temp = nu
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp ^= w_bit
            # Count neighbors of w that are in nv
            count += (adj_mask[w] & nv).bit_count()
        return count
    
    def count_c4_remove(u, v):
        """Count C4s destroyed by removing edge (u,v) from current graph."""
        # Same formula as add, since the C4s containing edge (u,v) are exactly
        # those formed by u-w-x-v where w in N(u)\{v}, x in N(v)\{u}, (w,x) edge
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        count = 0
        temp = nu
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp ^= w_bit
            count += (adj_mask[w] & nv).bit_count()
        return count
    
    def get_delta(u, v, x, y):
        """Delta in C4 count when removing (u,v) and adding (x,y)."""
        lost = count_c4_remove(u, v)
        
        # Temporarily remove (u,v)
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
        gained = count_c4_add(x, y)
        
        # Restore (u,v)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return gained - lost
    
    # 2. Greedy addition of remaining edges
    edges_to_add = m - current_edges
    
    # Candidate edges: all non-edges
    all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
        best_edge = None
        min_c4s = float('inf')
        
        for u, v in all_non_edges:
            c4s = count_c4_add(u, v)
            if c4s < min_c4s:
                min_c4s = c4s
                best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add(best_edge)
            all_non_edges.remove(best_edge)
    
    # Compute initial C4 count using bitmask
    def count_total_c4s():
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj_mask[i]:
                    # Count common neighbors of i and j
                    common = adj_mask[i] & adj_mask[j]
                    common &= ~(1 << i) & ~(1 << j)
                    c = common.bit_count()
                    total += c * (c - 1) // 2
        return total
    
    current_c4 = count_total_c4s()
    
    best_c4 = current_c4
    best_adj_mask = adj_mask[:]
    best_edge_set = set(edge_set)
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_edges_list = list(edge_set)
    
    # Precompute all non-edges for faster random selection
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        
        # Pick random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u, v = e_rm
        
        # Pick random non-edge to add
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                break
        
        delta = get_delta(u, v, x, y)
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            # Apply removal
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edge_set.remove(e_rm)
            current_edges_list[idx_rm] = (x, y)
            
            # Apply addition
            adj_mask[x] |= (1 << y)
            adj_mask[y] |= (1 << x)
            edge_set.add((x, y))
            
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj_mask = adj_mask[:]
                best_edge_set = set(edge_set)
                
        T *= cooling_rate
    
    # 4. Final polish: deterministic local search on best solution
    if time.time() < deadline - 3:
        adj_mask = best_adj_mask[:]
        edge_set = set(best_edge_set)
        current_c4 = best_c4
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            edges_list = list(edge_set)
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
            
            for e_rm in edges_list:
                if time.time() > deadline - 3:
                    break
                u, v = e_rm
                for x, y in non_edges:
                    if (x, y) == (u, v):
                        continue
                    delta = get_delta(u, v, x, y)
                    if delta < 0:
                        # Apply improvement
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        edge_set.remove(e_rm)
                        adj_mask[x] |= (1 << y)
                        adj_mask[y] |= (1 << x)
                        edge_set.add((x, y))
                        current_c4 += delta
                        improved = True
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_adj_mask = adj_mask[:]
                            best_edge_set = set(edge_set)
                        break
    
    # Return best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
