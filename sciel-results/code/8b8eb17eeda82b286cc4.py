"""Rewritten program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Strategy: 
1. Construct a C4-free core with 90 edges using the polarity graph of PG(2,5).
2. Use fast local search (edge swaps with O(deg) delta computation) to minimize C4s.
3. Polish phase: try all single swaps with exact delta.
"""
import numpy as np
import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # safety margin under 120s (evaluator kills at 150s)

    n = 31
    q = 5

    # --- Step 1: Build C4-free core with 90 edges using PG(2,5) polarity ---
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
        adj = [set() for _ in range(n)]
        edges_core = []
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
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

    # --- Helper functions for fast C4 counting ---
    def count_c4_full(adj):
        """Count total C4s: sum over all pairs (u,v) of C(|N(u)∩N(v)|, 2) / 4."""
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = len(adj[u] & adj[v])
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 4

    def delta_add_edge(u, v, adj):
        """Delta in C4 count when adding edge (u,v) to adj (u,v not yet an edge).
        New C4s = number of pairs (a,b) with a in N(u), b in N(v), a != b, a in N(b).
        Each such pair forms a C4: u-a-b-v-u (or u-v-b-a-u).
        Actually: C4s containing new edge (u,v): for each common structure...
        A C4 u-v-x-y-u requires x~v, y~u, x~y, x!=y.
        So count = number of pairs (x,y) where x in N(v), y in N(u), x~y, x!=y.
        But also need x != u, y != v (guaranteed since u,v not adjacent yet, and no self-loops).
        Also need x != y (handled by x!=y).
        """
        count = 0
        Nu = adj[u]
        Nv = adj[v]
        for x in Nv:
            if x in Nu:
                continue  # x is common neighbor, but we need x~y where y in Nu
            # x in Nv, x not in Nu
            for y in Nu:
                if y == x:
                    continue
                if y in adj[x]:
                    count += 1
        return count

    def delta_remove_edge(u, v, adj):
        """Delta in C4 count when removing edge (u,v) from adj (u,v is currently an edge).
        C4s destroyed = number of C4s that use edge (u,v).
        A C4 using (u,v): u-v-x-y-u where x~v, y~u, x~y, x!=y, x!=u, y!=v.
        """
        count = 0
        Nu = adj[u] - {v}
        Nv = adj[v] - {u}
        for x in Nv:
            if x in Nu:
                continue
            for y in Nu:
                if y == x:
                    continue
                if y in adj[x]:
                    count += 1
        return count

    def delta_swap(e1, e2, adj):
        """Compute delta C4 for removing e1=(u1,v1) and adding e2=(u2,v2).
        Assumes e1 is currently an edge and e2 is NOT currently an edge.
        Returns change in total C4 count.
        
        We need to account for interactions: removing e1 changes N(u1), N(v1),
        which affects how many C4s e2 creates.
        
        Strategy: compute delta_remove(e1) + delta_add(e2) but with correction
        for the fact that e2's C4 count is computed after e1 is removed.
        
        For correctness, simulate: remove e1, compute add delta, then revert.
        This is O(deg(u1)+deg(v1)+deg(u2)+deg(v2)) per swap.
        """
        u1, v1 = e1
        u2, v2 = e2
        
        # Save state
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        
        # Compute C4s added by e2 in the modified graph
        add_delta = delta_add_edge(u2, v2, adj)
        
        # Restore
        adj[u1].add(v1)
        adj[v1].add(u1)
        
        # Compute C4s removed by e1 in the original graph
        rem_delta = delta_remove_edge(u1, v1, adj)
        
        return add_delta - rem_delta

    # --- Step 2: Find best single edge to add ---
    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 80:
            break
        for v in range(u+1, n):
            if v in adj[u]:
                continue
            add = delta_add_edge(u, v, adj)
            if add < best_add:
                best_add = add
                best_edge = (u, v)
    
    if best_edge is None:
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    best_edge = (u, v)
                    break
            if best_edge:
                break
    
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    edges_set = set(tuple(sorted(e)) for e in edges_core)
    edges_set.add(best_edge)
    
    current_c4s = count_c4_full(adj)
    
    # --- Step 3: Local search with fast delta ---
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges_set = set(all_pairs) - edges_set
    
    random.seed(123)
    improved = True
    stall_count = 0
    
    while improved and time.time() < deadline - 20:
        improved = False
        # Try random swaps
        for _ in range(500):
            if time.time() > deadline - 20:
                break
            e1 = random.choice(list(edges_set))
            if not non_edges_set:
                break
            e2 = random.choice(list(non_edges_set))
            
            delta = delta_swap(e1, e2, adj)
            if delta < 0:
                # Apply swap
                adj[e1[0]].remove(e1[1])
                adj[e1[1]].remove(e1[0])
                edges_set.discard(e1)
                non_edges_set.add(e1)
                
                adj[e2[0]].add(e2[1])
                adj[e2[1]].add(e2[0])
                edges_set.add(e2)
                non_edges_set.discard(e2)
                
                current_c4s += delta
                improved = True
                stall_count = 0
            else:
                stall_count += 1
        
        if not improved:
            break

    # --- Step 4: Exhaustive polish - try all single swaps ---
    if time.time() < deadline - 5:
        polish_improved = True
        while polish_improved and time.time() < deadline - 3:
            polish_improved = False
            best_delta = 0
            best_swap = None
            
            edges_list = list(edges_set)
            non_edges_list = list(non_edges_set)
            
            for e1 in edges_list:
                if time.time() > deadline - 3:
                    break
                for e2 in non_edges_list:
                    if time.time() > deadline - 3:
                        break
                    delta = delta_swap(e1, e2, adj)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e1, e2)
            
            if best_swap is not None and best_delta < 0:
                e1, e2 = best_swap
                adj[e1[0]].remove(e1[1])
                adj[e1[1]].remove(e1[0])
                edges_set.discard(e1)
                non_edges_set.add(e1)
                
                adj[e2[0]].add(e2[1])
                adj[e2[1]].add(e2[0])
                edges_set.add(e2)
                non_edges_set.discard(e2)
                
                current_c4s += best_delta
                polish_improved = True

    # --- Step 5: Second polish pass (tighter) ---
    if time.time() < deadline - 2:
        polish_improved = True
        while polish_improved and time.time() < deadline - 1:
            polish_improved = False
            best_delta = 0
            best_swap = None
            
            edges_list = list(edges_set)
            non_edges_list = list(non_edges_set)
            
            for e1 in edges_list:
                if time.time() > deadline - 1:
                    break
                for e2 in non_edges_list:
                    if time.time() > deadline - 1:
                        break
                    delta = delta_swap(e1, e2, adj)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e1, e2)
            
            if best_swap is not None and best_delta < 0:
                e1, e2 = best_swap
                adj[e1[0]].remove(e1[1])
                adj[e1[1]].remove(e1[0])
                edges_set.discard(e1)
                non_edges_set.add(e1)
                
                adj[e2[0]].add(e2[1])
                adj[e2[1]].add(e2[0])
                edges_set.add(e2)
                non_edges_set.discard(e2)
                
                current_c4s += best_delta
                polish_improved = True

    # Build final edge list
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges
