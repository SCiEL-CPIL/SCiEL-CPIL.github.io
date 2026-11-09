"""Rewritten program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Strategy: 
1. Construct a C4-free core with 90 edges using the polarity graph of PG(2,5).
2. Use local search (edge swaps) with bitset-accelerated C4 counting to minimize C4s.
"""
import numpy as np
import time
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    n = 31
    q = 5

    # --- Step 1: Build C4-free core with 90 edges using PG(2,5) polarity ---
    # Points of PG(2,5): normalized triples (x,y,z) with first nonzero = 1
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: B(p,q) = x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    def adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        return (x1*x2 + y1*y2 - z1*z2) % 5 == 0
    
    edges_core = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adjacent(i, j):
                edges_core.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    if len(edges_core) != 90:
        # Fallback: build a C4-free graph via greedy
        adj = [set() for _ in range(n)]
        edges_core = []
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        import random
        random.seed(42)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges_core) >= 90:
                break
            creates_c4 = False
            for a in adj[u]:
                if a in adj[v]:
                    creates_c4 = True
                    break
            if not creates_c4:
                edges_core.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
        if len(edges_core) < 90:
            for u, v in all_pairs:
                if len(edges_core) >= 90:
                    break
                if v not in adj[u]:
                    edges_core.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)
    
    # --- Step 2: Find best single edge to add to minimize C4s ---
    def count_added_c4s(u, v, adj):
        """Count C4s created by adding edge (u,v)."""
        count = 0
        for a in adj[u]:
            for b in adj[v]:
                if b in adj[a]:
                    count += 1
        return count
    
    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, n):
            if v in adj[u]:
                continue
            add = count_added_c4s(u, v, adj)
            if add < best_add:
                best_add = add
                best_edge = [u, v]
    
    if best_edge is None:
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break
    
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # --- Step 3: Local search with bitset-accelerated C4 counting ---
    # Use integer bitsets for adjacency (31 bits fit in a Python int)
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    def count_c4_bitset(adj_bits):
        """Count total C4s using bitset operations."""
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = bin(adj_bits[u] & adj_bits[v]).count('1')
                total += cn * (cn - 1) // 2
        return total // 4
    
    current_c4s = count_c4_bitset(adj_bits)
    
    # Local search: try removing edge e1 and adding edge e2, keep if improves
    import random
    random.seed(123)
    
    edges_set = set(tuple(sorted(e)) for e in edges_core)
    edges_set.add(tuple(sorted(best_edge)))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
    
    def apply_swap(e1, e2):
        """Apply swap: remove e1, add e2. Returns delta C4s."""
        u1, v1 = e1
        u2, v2 = e2
        
        # Remove edge (u1, v1)
        adj_bits[u1] &= ~(1 << v1)
        adj_bits[v1] &= ~(1 << u1)
        
        # Add edge (u2, v2)
        adj_bits[u2] |= (1 << v2)
        adj_bits[v2] |= (1 << u2)
        
        new_c4s = count_c4_bitset(adj_bits)
        delta = new_c4s - current_c4s
        return delta, new_c4s
    
    def revert_swap(e1, e2):
        """Revert swap: add back e1, remove e2."""
        u1, v1 = e1
        u2, v2 = e2
        
        # Remove edge (u2, v2)
        adj_bits[u2] &= ~(1 << v2)
        adj_bits[v2] &= ~(1 << u2)
        
        # Add back edge (u1, v1)
        adj_bits[u1] |= (1 << v1)
        adj_bits[v1] |= (1 << u1)
    
    # Phase 1: Random local search with many iterations
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        edges_list = list(edges_set)
        non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
        
        if not non_edges_list:
            break
            
        for _ in range(500):
            if time.time() > deadline - 15:
                break
            # Pick a random edge to remove
            e1 = random.choice(edges_list)
            # Pick a random non-edge to add
            e2 = random.choice(non_edges_list)
            
            delta, new_c4s = apply_swap(e1, e2)
            
            if delta < 0:
                # Apply swap
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s = new_c4s
                improved = True
                # Rebuild lists
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
            else:
                # Revert
                revert_swap(e1, e2)
    
    # Phase 2: Exhaustive best-improvement local search
    if time.time() < deadline - 5:
        for phase in range(10):  # Multiple passes of best-improvement
            if time.time() > deadline - 5:
                break
            edges_list = list(edges_set)
            non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
            
            best_improvement = 0
            best_swap = None
            
            for e1 in edges_list:
                if time.time() > deadline - 5:
                    break
                for e2 in non_edges_list:
                    if time.time() > deadline - 5:
                        break
                    delta, new_c4s = apply_swap(e1, e2)
                    if delta < best_improvement:
                        best_improvement = delta
                        best_swap = (e1, e2)
                    revert_swap(e1, e2)
            
            if best_swap is not None and best_improvement < 0:
                e1, e2 = best_swap
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s += best_improvement
            else:
                break  # No improvement possible
    
    # Phase 3: Simulated annealing for escaping local optima
    if time.time() < deadline - 3:
        import math
        T = 10.0
        T_min = 0.1
        cool_rate = 0.9995
        edges_list = list(edges_set)
        non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
        
        best_c4s = current_c4s
        best_edges = set(edges_set)
        
        for step in range(5000):
            if time.time() > deadline - 3:
                break
            if not non_edges_list:
                break
            e1 = random.choice(edges_list)
            e2 = random.choice(non_edges_list)
            
            delta, new_c4s = apply_swap(e1, e2)
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                # Accept
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s = new_c4s
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
                
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_edges = set(edges_set)
            else:
                revert_swap(e1, e2)
            
            T *= cool_rate
            if T < T_min:
                T = T_min
        
        # Restore best found
        if best_c4s < current_c4s:
            edges_set = set(best_edges)
            current_c4s = best_c4s
    
    # Build final edge list
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges
