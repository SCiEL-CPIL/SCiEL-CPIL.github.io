"""Minimize C4s in 31-vertex, 91-edge graph.
Strategy:
1. Build C4-free 90-edge core via PG(2,5) polarity graph.
2. Add best 91st edge (minimizing C4s created).
3. Local search with INCREMENTAL C4 delta computation (only count C4s involving swapped edges).
4. Multiple phases: greedy first-improvement, best-improvement, simulated annealing.
"""
import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

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
    
    # --- Step 2: Find best single edge to add to minimize C4s ---
    def count_added_c4s(u, v, adj_sets):
        """Count C4s created by adding edge (u,v).
        A C4 containing new edge (u,v) is u-v-b-a-u where a~u, b~v, a~b, a!=v, b!=u.
        """
        count = 0
        for a in adj_sets[u]:
            if a == v:
                continue
            for b in adj_sets[v]:
                if b == u or b == a:
                    continue
                if b in adj_sets[a]:
                    count += 1
        return count // 2  # each C4 counted twice (a,b and b,a)
    
    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 10:
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
    
    # --- Step 3: Set up bitsets and INCREMENTAL C4 delta ---
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    def count_c4_full(adj_bits_local):
        """Count total C4s using bitset operations."""
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = bin(adj_bits_local[u] & adj_bits_local[v]).count('1')
                total += cn * (cn - 1) // 2
        return total // 4
    
    current_c4s = count_c4_full(adj_bits)
    
    random.seed(123)
    
    edges_set = set(tuple(sorted(e)) for e in edges_core)
    edges_set.add(tuple(sorted(best_edge)))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edges_list = list(edges_set)
    non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
    
    def compute_swap_delta(e1, e2):
        """Compute delta in C4 count for removing e1=(u1,v1) and adding e2=(u2,v2).
        INCREMENTAL: only count C4s involving e1 or e2.
        
        A C4 is a 4-cycle: 4 vertices {a,b,c,d} with edges ab,bc,cd,da.
        A C4 contains edge (u,v) if u and v are adjacent in the cycle.
        
        C4s destroyed by removing (u1,v1): C4s where (u1,v1) is an edge.
        Such a C4 has form u1-v1-x-y-u1 where:
          - x is neighbor of u1, x != v1
          - y is neighbor of v1, y != u1  
          - x is neighbor of y
          - x != y
        Each such C4 is counted once per (x,y) pair.
        
        C4s created by adding (u2,v2): C4s where (u2,v2) is an edge, in the graph
        AFTER removing e1. Such a C4 has form u2-v2-x-y-u2 where:
          - x is neighbor of u2 (after removal), x != v2
          - y is neighbor of v2 (after removal), y != u2
          - x is neighbor of y (after removal)
          - x != y
        BUT we must exclude C4s that also contain e1 (those are net-zero).
        Actually, since we remove e1 first, the graph for counting created C4s
        has e1 removed. So C4s containing both e1 and e2 won't be counted as
        "created" because e1 is gone. And they won't be counted as "destroyed"
        either if we're careful... 
        
        Wait, let me think again. The net change is:
        delta = C4s_in_new_graph - C4s_in_old_graph
        
        C4s in old graph that are NOT in new graph: those containing e1 but not e2
        C4s in new graph that are NOT in old graph: those containing e2 but not e1
        
        A C4 containing both e1 and e2: exists in old (has e1), doesn't exist in new
        (e1 removed), but also would be created by adding e2... but it can't be
        created because e1 is removed. So it's destroyed.
        
        Actually simpler: 
        - destroyed = # C4s in old graph containing edge e1
        - created = # C4s in new graph containing edge e2
        - delta = created - destroyed
        
        Because: C4s not containing e1 or e2 are unchanged.
        C4s containing e1: all destroyed (since e1 is removed).
        C4s containing e2 in new graph: all created (since e2 wasn't there before).
        C4s containing both: they contain e1 so they're in "destroyed". In new graph,
        they can't contain e2 because e1 is removed and the C4 would need e1.
        Wait, a C4 containing both e1 and e2: in old graph it exists. In new graph,
        e1 is removed so it doesn't exist. So it's destroyed. It's NOT created because
        it's not in the new graph. So: destroyed counts it, created doesn't. Correct.
        
        A C4 containing e2 but not e1: not in old graph (no e2). In new graph, it exists.
        So created counts it, destroyed doesn't. Correct.
        
        A C4 containing e1 but not e2: in old graph. Not in new graph (e1 removed).
        Destroyed counts it. Created doesn't. Correct.
        
        So delta = created - destroyed is correct.
        """
        u1, v1 = e1
        u2, v2 = e2
        
        # Count C4s destroyed by removing (u1,v1) from current graph
        destroyed = 0
        # C4 containing (u1,v1): u1-v1-x-y-u1
        # x in N(u1) \ {v1}, y in N(v1) \ {u1}, x~y, x!=y
        # For each x in N(u1)\{v1}, count y in N(v1)\{u1,x} such that x~y
        for x in adj[u1]:
            if x == v1:
                continue
            # y must be in N(v1) \ {u1, x} and x in N(y) i.e. y in N(x)
            # Count y in (N(v1) \ {u1, x}) ∩ N(x)
            ny_mask = adj_bits[v1] & ~(1 << u1) & ~(1 << x)
            cnt = bin(ny_mask & adj_bits[x]).count('1')
            destroyed += cnt
        # Each C4 is counted once (for the specific x,y pair), so no division needed
        # Actually wait: a C4 u1-v1-x-y-u1 is identified by the pair (x,y).
        # But could the same C4 be counted for different (x,y)? No, because the
        # C4 is uniquely determined by which two non-adjacent vertices are x and y.
        # In a C4 u1-v1-x-y-u1, the vertices are u1,v1,x,y. The edge (u1,v1) is fixed.
        # x is the neighbor of u1 (other than v1) and y is the neighbor of v1 (other than u1).
        # So each C4 is counted exactly once.
        
        # Now temporarily remove e1 to compute created C4s
        adj_bits[u1] &= ~(1 << v1)
        adj_bits[v1] &= ~(1 << u1)
        
        # Count C4s created by adding (u2,v2) to the graph without e1
        created = 0
        # C4 containing (u2,v2): u2-v2-x-y-u2
        # x in N(u2)\{v2}, y in N(v2)\{u2}, x~y, x!=y
        for x in range(n):
            if x == u2 or x == v2:
                continue
            if not ((adj_bits[u2] >> x) & 1):
                continue
            # x is neighbor of u2 (in graph without e1)
            # y must be in N(v2)\{u2,x} and x in N(y)
            ny_mask = adj_bits[v2] & ~(1 << u2) & ~(1 << x)
            cnt = bin(ny_mask & adj_bits[x]).count('1')
            created += cnt
        
        # Restore e1
        adj_bits[u1] |= (1 << v1)
        adj_bits[v1] |= (1 << u1)
        
        delta = created - destroyed
        return delta
    
    def apply_swap(e1, e2):
        u1, v1 = e1
        u2, v2 = e2
        adj_bits[u1] &= ~(1 << v1)
        adj_bits[v1] &= ~(1 << u1)
        adj_bits[u2] |= (1 << v2)
        adj_bits[v2] |= (1 << u2)
    
    def revert_swap(e1, e2):
        u1, v1 = e1
        u2, v2 = e2
        adj_bits[u2] &= ~(1 << v2)
        adj_bits[v2] &= ~(1 << u2)
        adj_bits[u1] |= (1 << v1)
        adj_bits[v1] |= (1 << u1)
    
    # Phase 1: Greedy first-improvement local search
    improved = True
    stall_count = 0
    while improved and time.time() < deadline - 35:
        improved = False
        edges_list = list(edges_set)
        non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
        for _ in range(5000):
            if time.time() > deadline - 35:
                break
            if not non_edges_list:
                break
            e1 = random.choice(edges_list)
            e2 = random.choice(non_edges_list)
            
            delta = compute_swap_delta(e1, e2)
            
            if delta < 0:
                apply_swap(e1, e2)
                edges_set.discard(e1)
                new_e2 = tuple(sorted(e2))
                edges_set.add(new_e2)
                current_c4s += delta
                improved = True
                stall_count = 0
            else:
                stall_count += 1
    
    # Phase 2: Best-improvement local search
    if time.time() < deadline - 20:
        for phase in range(30):
            if time.time() > deadline - 20:
                break
            edges_list = list(edges_set)
            non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
            
            best_improvement = 0
            best_swap = None
            
            for e1 in edges_list:
                if time.time() > deadline - 20:
                    break
                for e2 in non_edges_list:
                    if time.time() > deadline - 20:
                        break
                    delta = compute_swap_delta(e1, e2)
                    if delta < best_improvement:
                        best_improvement = delta
                        best_swap = (e1, e2)
            
            if best_swap is not None and best_improvement < 0:
                e1, e2 = best_swap
                apply_swap(e1, e2)
                edges_set.discard(e1)
                new_e2 = tuple(sorted(e2))
                edges_set.add(new_e2)
                current_c4s += best_improvement
            else:
                break
    
    # Phase 3: Simulated annealing for escaping local optima
    if time.time() < deadline - 5:
        T = 5.0
        T_min = 0.05
        cool_rate = 0.9998
        edges_list = list(edges_set)
        non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
        
        best_c4s = current_c4s
        best_edges = set(edges_set)
        
        for step in range(50000):
            if time.time() > deadline - 5:
                break
            if not non_edges_list:
                break
            e1 = random.choice(edges_list)
            e2 = random.choice(non_edges_list)
            
            delta = compute_swap_delta(e1, e2)
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                apply_swap(e1, e2)
                edges_set.discard(e1)
                new_e2 = tuple(sorted(e2))
                edges_set.add(new_e2)
                current_c4s += delta
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
                
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_edges = set(edges_set)
            
            T *= cool_rate
            if T < T_min:
                T = T_min
        
        if best_c4s < current_c4s:
            edges_set = set(best_edges)
            current_c4s = best_c4s
            adj_bits = [0] * n
            for (u, v) in edges_set:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    # Final verification
    final_c4s = count_c4_full(adj_bits)
    if final_c4s != current_c4s:
        # Recompute to be safe
        current_c4s = final_c4s
    
    # Build final edge list
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges
