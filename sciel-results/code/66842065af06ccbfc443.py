import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # The task is specifically for k=3, g=16 based on the prompt context.
    # However, the interface accepts k and g.
    # If k != 3 or g != 16, we fall back to a simple valid construction if possible,
    # or just the seed algorithm if it handles it.
    # Given the "cage problem n(3,16)" context, we optimize for this case.
    
    if k != 3:
        # Fallback: use a simple cycle or other structure? 
        # For robustness, if k is not 3, we might not have a specific algebraic construction.
        # The seed program handled k=4 with XOR. 
        # For now, if k!=3, we return a trivial valid graph if possible or the seed baseline.
        # Since the prompt focuses on 3-regular, we assume k=3.
        # If k is different, we can try to construct a k-regular graph with girth g.
        # This is hard. We will return the seed baseline logic if it works, else error?
        # The prompt says "evaluator passes this task's degree k and girth g".
        # It implies k=3, g=16 is the target.
        
        # Let's implement a generic fallback using the seed's algebraic method 
        # if k is a prime or power of 2, otherwise random search.
        # But for this specific task, we focus on the 3-regular case.
        
        pass

    # Strategy for k=3:
    # We construct a 3-regular graph using a finite field F_q.
    # Vertices are elements of F_q.
    # We define 3 neighbors for each x in F_q.
    # A common method for high girth is to use a "cage" construction based on 
    # a specific recurrence or a voltage graph over a small base graph.
    
    # Alternative Strategy: 
    # Use the known explicit construction for the (3,16)-cage if we can find one,
    # or generate a large graph using a pseudo-random but structured method.
    
    # The seed program generated a graph of size ~500-1000.
    # The known best is 936.
    # Can we do better? Probably not easily without deep algebraic insights.
    # However, the current program's random search is very weak.
    # A better "different" strategy is to use **Greedy Construction with Backtracking** 
    # or **Simulated Annealing** on edge swaps to maximize girth, starting from a random 3-regular graph.
    
    # But wait, the prompt asks for a strategy *materially different* from random restarts.
    # Let's try a **Deterministic Algebraic Construction** using a Linear Feedback Shift Register (LFSR) 
    # or a similar recurrence over a finite field.
    
    # For k=3, g=16:
    # We can model the graph as a cover of a small graph.
    # Or, we can use the fact that the incidence matrix of a (3,16)-cage can be constructed
    # using specific polynomials.
    
    # Let's try a simpler, robust approach: 
    # **Iterative Shortest Cycle Elimination via Edge Swaps (2-opt)**
    # This is a local search, but *systematic* and *deterministic* in its cycle-finding, 
    # unlike the random restarts.
    # However, the current program already tries local search but poorly.
    
    # Let's try a **different design family**: 
    # **Bipartite Graph Construction using a Bipartite Cage**.
    # A (3,16)-cage is not necessarily bipartite, but many high-girth 3-regular graphs are.
    # If we construct a bipartite graph with girth 16, it is a valid (3,16)-cage candidate.
    # The Moore bound for bipartite (3,16) is 2 * (2^7 - 1) = 510.
    # We can try to construct a bipartite graph with 2 * 256 = 512 vertices?
    # No, the Moore bound for bipartite (d, g) with g even is 2 * sum_{i=0}^{g/2 - 1} (d-1)^i.
    # For d=3, g=16: 2 * (1 + 2 + 4 + ... + 128) = 2 * 255 = 510.
    # Since n must be even, 510 is possible. But a (3,16)-cage of size 510 doesn't exist (lower bound is 512 for general, but bipartite might be higher).
    
    # Let's use a **Randomized Greedy Construction** that builds the graph vertex by vertex,
    # ensuring no short cycles are formed. This is "backtracking search" but guided by girth constraints.
    
    # Given the time limit (150s), a full backtracking search for n=512 is likely too slow.
    # The seed program produces a valid graph.
    # The "different strategy" should be **using a known explicit algebraic family** that is 
    # faster to generate and potentially smaller, or at least equally good but more robust.
    
    # Let's implement the **Lubotzky-Phillips-Sarnak (LPS)** construction for k=4? No, k=3.
    # There is no simple LPS for k=3.
    
    # Let's try **Constructing via a Voltage Graph**.
    # Base graph: A 3-regular graph on 2 vertices? No, that's a multigraph.
    # Base graph: K4 (4 vertices, 3-regular). Girth 3.
    # We lift K4 to a cover of size N.
    # The girth of the cover is related to the length of the shortest non-contractible cycle in the base graph weighted by the voltage.
    
    # Let's stick to a **High-Performance Randomized Local Search** but with a **much better initialization** 
    # and **cycle-aware moves**.
    # The current program's local search is `pass`. We will implement a real one.
    
    # But the directive says "MATERIALLY DIFFERENT".
    # Randomized Local Search is the same family as the current program's intent.
    
    # **Different Family: Deterministic Algebraic Construction using Finite Fields.**
    # We will construct a graph on vertices F_q x {0,1} or similar.
    # For k=3, we can use the following:
    # Let q be a prime power.
    # Vertices: (x, i) where x in F_q, i in {0, 1, 2}? No, degree 3.
    # 
    # Actually, there is a known construction for (3,16) cages using **quasigroups** or **Latin Squares**.
    # 
    # Let's try a **Bipartite Construction** with vertices (u, i) and (v, j).
    # We want to connect (u, i) to (v, j) if v = f_i(u) + c_j.
    # 
    # Let's implement a **Greedy Edge Addition** that maximizes girth.
    # Start with n empty vertices. Add edges one by one.
    # Choose the pair (u, v) that maximizes the minimum girth of the resulting graph.
    # This is O(n^2) per edge, O(n^3) total. For n=512, 512^3 ~ 134M operations. Might be too slow in Python.
    
    # Let's try **Simulated Annealing** on the **Edge Swap** neighborhood.
    # This is distinct from "Random Restarts" in that it maintains a single trajectory and uses temperature.
    # But is it materially different? It's a local search.
    
    # Let's go with **Algebraic Construction via Recurrence**.
    # We define a graph where vertices are indices 0..n-1.
    # Neighbors of i are (i + 1) % n, (i - 1) % n, and (i * m + c) % n?
    # This creates a Cayley graph of Z_n.
    # The girth of a Cayley graph of Z_n with generators {1, -1, m} is the length of the shortest relation.
    # We can search for n and m that maximize the girth.
    # This is a 1D search over n and m. Very fast!
    
    # Strategy:
    # 1. Iterate n from Moore Bound up to some limit.
    # 2. For each n, iterate m from 2 to n-1.
    # 3. Construct the Cayley graph on Z_n with generators {1, -1, m}.
    # 4. Check if the graph is simple (no loops/multiedges). 
    #    - Loop if 1=0, -1=0, m=0. (n>2)
    #    - Multiedge if 1 = -1 (n=2), 1=m, -1=m, etc.
    #    - We need {1, -1, m} to be distinct mod n.
    # 5. Calculate the girth. If girth >= g, return this graph.
    
    # This is a completely different strategy: **Cayley Graph Search**.
    
    if k == 3:
        moore_bound = 2 * ((1 << (g // 2)) - 1) if g % 2 == 0 else 2 * (1 << (g // 2))
        if moore_bound % 2 != 0:
            moore_bound += 1
        
        # We know n must be even for 3-regular.
        # Start from moore_bound.
        
        # To speed up girth calculation, we can use BFS.
        # For n ~ 500, BFS is fast.
        # Number of m's to check: ~n.
        # Total checks: n * n * BFS(n) ~ 500 * 500 * 500 = 125M. Too slow in Python?
        # We can prune.
        
        # Optimization: 
        # The girth of a Cayley graph is at least the length of the shortest relation.
        # We can check small relations first.
        
        # Let's limit the search to a few promising n values.
        # The seed program produced a graph of size ~500-1000.
        # Let's try n from moore_bound to moore_bound + 100.
        
        best_n = None
        best_edges = None
        
        max_n_to_try = 600 # Limit search space
        
        for n in range(moore_bound, max_n_to_try + 1, 2):
            if time.time() > deadline - 10:
                break
            
            # Check if n is valid for 3-regular Cayley graph with 3 generators.
            # Generators: 1, n-1 (-1), m.
            # Conditions:
            # 1 != 0, n-1 != 0, m != 0
            # 1 != n-1 => n != 2
            # 1 != m => m != 1
            # n-1 != m => m != n-1
            # Also, the graph must be simple.
            # If m = 2, edges are (i, i+1), (i, i-1), (i, i+2).
            # If i+2 == i-1 mod n => 3=0 mod n => n=3.
            # If i+2 == i+1 mod n => 1=0 mod n => n=1.
            # So for n > 3, these are distinct.
            
            for m in range(2, n // 2 + 1):
                if time.time() > deadline - 10:
                    break
                
                # Skip if m creates multi-edges
                if m == 1 or m == n - 1:
                    continue
                if n % 2 == 0 and m == n // 2:
                    # Edge (i, i+n/2) is same as (i+n/2, i).
                    # This is fine for simple graph, but let's check if it creates a 2-cycle?
                    # No, it's a single edge.
                    # But wait, if m = n/2, then neighbor is i + n/2.
                    # Is it possible that i + n/2 = i + 1? No.
                    # Is it possible that i + n/2 = i - 1? No.
                    # So it's a valid simple graph.
                    pass
                
                # Construct edges
                # We don't need to store all edges, just check girth.
                # But we need to return edges if valid.
                
                # Check Girth
                # BFS from 0.
                # Since it's a Cayley graph, the girth is the shortest path from 0 to 0 
                # that doesn't use the trivial backtracking?
                # Actually, girth is the shortest cycle.
                # In a Cayley graph, a cycle corresponds to a relation s1...sk = e.
                
                # BFS on the Cayley graph is equivalent to BFS on the group.
                # We can just do BFS on the vertices 0..n-1.
                
                adj = [[] for _ in range(n)]
                # Build adjacency
                for i in range(n):
                    adj[i].append((i + 1) % n)
                    adj[i].append((i - 1) % n)
                    adj[i].append((i + m) % n)
                
                # Find shortest cycle
                # BFS from each node is O(n^2). 
                # For n=500, 500*500 = 250k ops. 
                # For 500 values of m, 125M ops. 
                # This is borderline in Python. 
                # We can optimize by only checking if girth >= g.
                # We can stop BFS early if we find a cycle < g.
                
                min_girth = float('inf')
                valid = True
                for start in range(n):
                    if not valid:
                        break
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    parent[start] = -2
                    queue = deque([start])
                    
                    while queue:
                        u = queue.popleft()
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u]:
                                # Found a cycle
                                # Cycle length is dist[u] + dist[v] + 1?
                                # Only if u and v are in the same BFS tree level?
                                # Actually, the shortest cycle through start is found by 
                                # looking for back edges.
                                # But this BFS finds the shortest cycle *through start*.
                                # The girth is the min over all starts.
                                
                                # Wait, standard BFS from start finds shortest path to all nodes.
                                # A cycle exists if there is an edge (u, v) where v is already visited 
                                # and v is not the parent of u.
                                # The length of the cycle formed by tree path start->u, edge u-v, 
                                # and tree path v->start is NOT simply dist[u]+dist[v]+1 
                                # unless v is an ancestor of u or vice versa.
                                # If v is not an ancestor, the path start->v and start->u 
                                # merge at LCA.
                                # The cycle length is dist[u] + dist[v] - 2*dist[lca] + 1.
                                # This is complex to compute on the fly.
                                
                                # Simpler: Just do BFS from each node and look for the shortest 
                                # path back to the start node? No, that's not right either.
                                
                                # Correct way to find girth:
                                # For each edge (u, v), remove it and find shortest path u->v.
                                # Girth = min(dist(u, v) + 1).
                                # This is O(E * (V+E)) = O(n * n) = O(n^2) per edge? No.
                                # E = 1.5n.
                                # Total O(n^3). Too slow.
                                
                                # Alternative: 
                                # The girth of a Cayley graph is the minimum length of a word 
                                # in the generators that equals the identity, 
                                # such that no proper prefix is the identity (primitive word).
                                # We can enumerate words of length 2, 3, ..., up to g-1.
                                # Number of words of length L is 3 * 2^(L-1).
                                # For L=15, 3 * 2^14 = 49152.
                                # For each word, check if it sums to 0 mod n.
                                # This is fast!
                                
                                pass
                            
                            # Since the BFS logic above is tricky for girth, 
                            # let's use the word enumeration method for Cayley graphs.
                            
                            # Reset valid check
                            pass
                        
                    # End start loop
                    pass
                
                # Let's re-implement the check using word enumeration.
                # This is much faster and exact for Cayley graphs.
                
                # Generators: g1=1, g2=-1, g3=m.
                # We want to find the shortest sequence s1...sk such that 
                # sum(s_i * g_{s_i}) = 0 mod n, and the sequence is not 
                # "backtracking" (i.e., s_i != s_{i+1}^{-1}).
                # Since the graph is simple and undirected, backtracking 
                # (e.g., 1 then -1) creates a 2-cycle, which is not a cycle in 
                # the simple graph sense (it's just going back and forth on an edge).
                # The girth is the length of the shortest *simple* cycle.
                # In a Cayley graph, a simple cycle corresponds to a 
                # "non-backtracking" word that evaluates to 0.
                
                # We can do a BFS on the "non-backtracking" state space.
                # State: (current_sum, last_generator).
                # Start: (0, None).
                # Transitions: from (sum, last), we can go to (sum + g_j, j) 
                # for all j != last_inv.
                
                # We want to find the shortest path to (0, last) where last != None 
                # and the path length >= 3?
                # Actually, we want to return to 0.
                # Start at 0.
                # Step 1: go to g_j. State (g_j, j).
                # Step 2: go to g_j + g_k. State (g_j + g_k, k).
                # ...
                # Step k: sum = 0. State (0, last).
                # If we reach (0, last) with length k, and the path was non-backtracking, 
                # we have a cycle of length k.
                
                # BFS on states (sum, last_gen).
                # Number of states: n * 3.
                # For n=500, 1500 states.
                # BFS is very fast.
                
                # Let's implement this check.
                
                def has_short_cycle_cayley(n, m, g_limit):
                    # Generators: 1, n-1, m
                    # Inverses: n-1, 1, n-m
                    # We define neighbors as adding the generator.
                    # To avoid backtracking, we cannot add the inverse of the last added generator.
                    
                    # State: (current_pos, last_gen_idx)
                    # last_gen_idx: 0 for 1, 1 for -1, 2 for m
                    # Inverse of 0 is 1.
                    # Inverse of 1 is 0.
                    # Inverse of 2 is 2 (if m = n-m, i.e., 2m=n) or... wait.
                    # The generator is m. Its inverse is -m = n-m.
                    # If m == n-m, then the generator is its own inverse.
                    # In that case, backtracking means adding m again?
                    # No, backtracking means undoing the last step.
                    # If we added m, to undo we add -m.
                    # So we must not add -m if the last was m.
                    
                    gens = [1, n - 1, m]
                    invs = [n - 1, 1, n - m]
                    
                    # dist[state] = length
                    # We start at 0 with no last gen.
                    # We can simulate the first step separately or use a dummy state.
                    
                    # Let's use a queue of (pos, last_idx, dist)
                    # Start: (0, -1, 0)
                    # But we don't want to count the start as a step.
                    # We want to find a cycle, so we must leave 0 and return to 0.
                    
                    # BFS from (0, -1)
                    # If we reach (0, j) with dist > 0, we have a cycle of length dist.
                    # Wait, if we reach (0, j), the last step was adding gens[j].
                    # The path is a sequence of generators.
                    # The sum is 0.
                    # This forms a closed walk.
                    # Is it a simple cycle?
                    # Non-backtracking ensures no immediate reversals.
                    # It does not prevent longer repetitions, but the shortest 
                    # non-backtracking closed walk in a Cayley graph 
                    # is the girth.
                    
                    dist = {}
                    queue = deque()
                    queue.append((0, -1, 0))
                    dist[(0, -1)] = 0
                    
                    while queue:
                        pos, last, d = queue.popleft()
                        
                        if pos == 0 and last != -1:
                            # Found a cycle
                            return d < g_limit # We want to know if girth < g_limit
                            # Actually, we want to check if girth >= g.
                            # So if we find a cycle of length d, and d < g, 
                            # then girth < g.
                            # We can return d.
                            return d
                        
                        for i in range(3):
                            # Check if i is the inverse of last
                            if last != -1:
                                if i == invs[last] or last == invs[i]:
                                    # Wait, invs[last] is the index of the inverse?
                                    # No, invs[last] is the *value* of the inverse.
                                    # We need to know if gens[i] == -gens[last].
                                    if gens[i] == n - gens[last]:
                                        continue
                            
                            new_pos = (pos + gens[i]) % n
                            state = (new_pos, i)
                            
                            if state not in dist:
                                dist[state] = d + 1
                                queue.append((new_pos, i, d + 1))
                                
                                # Early termination if we find a short cycle
                                if new_pos == 0 and d + 1 < g_limit:
                                    return d + 1
                                    
                    return float('inf') # No cycle found (impossible for finite group)
                
                girth = has_short_cycle_cayley(n, m, g)
                
                if girth >= g:
                    # We found a valid graph!
                    # Construct edges
                    edges = []
                    for i in range(n):
                        for j in range(3):
                            v = (i + gens[j]) % n
                            if i < v:
                                edges.append((i, v))
                            elif i == v:
                                # Loop, invalid
                                pass
                    # Ensure simple graph
                    # The construction above might add duplicate edges if gens are not distinct mod n.
                    # We checked distinctness.
                    
                    best_n = n
                    best_edges = edges
                    break # Found a valid graph for this n, try smaller n? 
                    # No, we are iterating n upwards. 
                    # We want the *smallest* n.
                    # So if we find one for n, we should break out of m loop 
                    # and check if we can find a smaller n?
                    # No, we started from moore_bound and went up.
                    # So the first n we find is the smallest.
                    # But we might find a valid m for n=512.
                    # We should return it.
                    
                    return n, edges
            
            if best_n is not None:
                return best_n, best_edges
                
    # Fallback: Use the seed algorithm's baseline if the Cayley search failed
    # (e.g. no valid m found for n in range, or k != 3)
    
    # Re-implement seed baseline quickly
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

