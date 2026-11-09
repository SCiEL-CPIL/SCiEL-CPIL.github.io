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
    # For a C4-free base, adding edge (u,v) creates C4s = sum over common neighbors
    # of pairs of common neighbors that are connected.  In a C4-free graph,
    # common neighbors of any two vertices form an independent set, so the C4s
    # created by (u,v) = C(|N(u)∩N(v)|, 2) + (pairs of common neighbors that
    # are themselves adjacent -- but in C4-free these can't exist for the
    # "new" C4s; actually we need to count all C4s in the resulting graph).
    # Simpler: just count total C4s after adding.
    
    def total_c4s_after_add(u, v, adj):
        """Total C4s in graph after adding edge (u,v) to C4-free adj."""
        # In C4-free graph, C4s only involve the new edge.
        # A C4 containing (u,v): u-a-v-b-u where a in N(u), b in N(v),
        # a != b, a in N(v) is NOT required; rather:
        # u-a, a-v, v-b, b-u with a in N(u)∩N(v) and b in N(u)∩N(v), a!=b
        # Plus: u-a, a-v, v-b, b-u where a in N(u)\{v}, b in N(v)\{u},
        #   a-v means a in N(v), b-u means b in N(u).
        # So C4s through (u,v) = C(|N(u)∩N(v)|, 2) where N doesn't include
        # the new edge yet.
        common = adj[u] & adj[v]
        return len(common) * (len(common) - 1) // 2
    
    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, n):
            if v in adj[u]:
                continue
            add = total_c4s_after_add(u, v, adj)
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
    
    def swap_delta_c4(adj_bits, e1, e2):
        """Compute C4 delta for removing e1=(u1,v1) and adding e2=(u2,v2) without modifying adj_bits.
        Returns delta = new_c4s - old_c4s."""
        u1, v1 = e1
        u2, v2 = e2
        
        def common_count(a, b):
            return bin(adj_bits[a] & adj_bits[b]).count('1')
        
        affected_verts = {u1, v1, u2, v2}
        
        # Compute old contributions for affected pairs
        old_contrib = 0
        for a in range(n):
            for b in range(a+1, n):
                if a in affected_verts or b in affected_verts:
                    c = common_count(a, b)
                    old_contrib += c * (c - 1) // 2
        
        # Simulate the swap
        old_u1 = adj_bits[u1]
        old_v1 = adj_bits[v1]
        old_u2 = adj_bits[u2]
        old_v2 = adj_bits[v2]
        
        adj_bits[u1] = old_u1 & ~(1 << v1)
        adj_bits[v1] = old_v1 & ~(1 << u1)
        adj_bits[u2] = old_u2 | (1 << v2)
        adj_bits[v2] = old_v2 | (1 << u2)
        
        # Compute new contributions
        new_contrib = 0
        for a in range(n):
            for b in range(a+1, n):
                if a in affected_verts or b in affected_verts:
                    c = common_count(a, b)
                    new_contrib += c * (c - 1) // 2
        
        # Revert
        adj_bits[u1] = old_u1
        adj_bits[v1] = old_v1
        adj_bits[u2] = old_u2
        adj_bits[v2] = old_v2
        
        delta_S = new_contrib - old_contrib
        return delta_S // 4
    
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
        
        # Use incremental delta computation
        delta = swap_delta_c4(adj_bits, (u1, v1), (u2, v2))
        # But wait, swap_delta_c4 reverts internally. Let me restructure.
        # Actually swap_delta_c4 does apply and revert. So I need to
        # re-apply after calling it. This is inefficient.
        # Let me just use the full recount for now in apply_swap,
        # and use swap_delta_c4 in the search loops where we do many evaluations.
        
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
    
    def eval_swap_no_apply(e1, e2):
        """Evaluate C4 delta for swap without applying. Returns delta."""
        return swap_delta_c4(adj_bits, e1, e2)
    
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
            
            # Use fast incremental evaluation
            delta = eval_swap_no_apply(e1, e2)
            
            if delta < 0:
                # Actually apply the swap
                u1, v1 = e1
                u2, v2 = e2
                adj_bits[u1] &= ~(1 << v1)
                adj_bits[v1] &= ~(1 << u1)
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s += delta
                improved = True
                # Rebuild lists
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
    
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
                    delta = eval_swap_no_apply(e1, e2)
                    if delta < best_improvement:
                        best_improvement = delta
                        best_swap = (e1, e2)
            
            if best_swap is not None and best_improvement < 0:
                e1, e2 = best_swap
                u1, v1 = e1
                u2, v2 = e2
                adj_bits[u1] &= ~(1 << v1)
                adj_bits[v1] &= ~(1 << u1)
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
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
            
            delta = eval_swap_no_apply(e1, e2)
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                # Accept - actually apply
                u1, v1 = e1
                u2, v2 = e2
                adj_bits[u1] &= ~(1 << v1)
                adj_bits[v1] &= ~(1 << u1)
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s += delta
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
                
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_edges = set(edges_set)
            
            T *= cool_rate
            if T < T_min:
                T = T_min
        
        # Restore best found
        if best_c4s < current_c4s:
            edges_set = set(best_edges)
            current_c4s = best_c4s
            # Rebuild adj_bits from edges_set
            adj_bits = [0] * n
            for u, v in edges_set:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            current_c4s = count_c4_bitset(adj_bits)
    
    # Build final edge list
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    # Final verification: recount C4s
    adj_bits_final = [0] * n
    for u, v in final_edges:
        adj_bits_final[u] |= (1 << v)
        adj_bits_final[v] |= (1 << u)
    final_c4s = count_c4_bitset(adj_bits_final)
    
    return final_edges
