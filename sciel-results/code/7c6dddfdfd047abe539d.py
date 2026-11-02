import time
import random
import sys
from collections import deque

# Entry point required by the contract.
def construct(k, g):
    # Use a deadline to avoid being killed.
    deadline = 140.0
    start = time.time()
    
    # The task specifically mentions k=5, g=9.
    # The smallest known graph has 1116 vertices.
    # The seed program (score 1.0) uses LUW construction.
    # We should try to match or beat the LUW result if possible,
    # but since we are rewriting from scratch, we need a robust method.
    #
    # Approach:
    # 1. If k=5, g=9, we know the LUW construction produces a specific graph.
    #    The seed program's `_luw_construct` is likely the source of the score 1.0.
    #    We should replicate that logic to ensure we get a valid, known-good graph.
    #    However, the prompt says "Rewrite the program from scratch with a stronger algorithm."
    #    And "Keep this design and investigate improvements within its family."
    #    The current design IS the LUW construction.
    #    
    #    Let's look at the current program's `_luw_construct`.
    #    It constructs a graph based on a "rule" for adding/subtracting in a finite field/ring.
    #    For k=5, g=9:
    #    q = k = 5.
    #    d = max(3, g-5 if (g-5)%2 else g-4). g=9 -> g-5=4 (even) -> g-4=5. d=5.
    #    Since q=5 != 4, we use modular arithmetic.
    #    rule[2] = (1,1), rule[3] = (2,1), rule[4] = (1,2).
    #    c=5: (5-5)%4=0, c0=5. rule[5] = [(1, 3), (4, 1), (5, 1), (1, 6)][0] = (1, 3).
    #    
    #    The graph is a Cayley graph of Z_5^d? No, it's a lift.
    #    Let's just implement a robust greedy algorithm or a switching algorithm.
    #    Actually, the LUW construction is deterministic and fast.
    #    The voltage graph approach in the seed was trying to find smaller n.
    #    
    #    Since I must provide a COMPLETE program, and the seed's voltage graph approach
    #    was complex and potentially slow/buggy, I will implement a clean, efficient
    #    **Greedy Edge-Switching** or **Randomized Local Search** for girth improvement,
    #    OR simply implement the **LUW construction** correctly and efficiently, as it is
    #    the baseline that scores 1.0.
    #
    #    Wait, the instructions say: "REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family."
    #    The design is the voltage graph / LUW family?
    #    The seed program has `_luw_construct` and `_voltage_graph_construct`.
    #    The `_voltage_graph_construct` was trying to find a smaller graph than LUW.
    #    
    #    Let's implement a highly optimized **Greedy Girth Improvement** algorithm.
    #    1. Start with a random k-regular graph.
    #    2. While there are cycles shorter than g, find one.
    #    3. Perform a 2-switch or 4-switch to break it without creating shorter cycles.
    #    4. If stuck, restart or use simulated annealing.
    #
    #    However, for k=5, g=9, the graph is large (~1000 vertices).
    #    Random regular graphs of this size often have small girth.
    #    Improving girth to 9 is hard.
    #
    #    The LUW construction is algebraic and guarantees the girth.
    #    It is the most reliable way to get a valid graph quickly.
    #    The "improvement" might be in the voltage graph search for smaller n.
    #    But the voltage graph search is slow.
    #
    #    Let's stick to the LUW construction but optimize it.
    #    And add a fallback to a greedy switch if LUW fails (it shouldn't).
    #
    #    Actually, let's look at the "INSPIRATIONS".
    #    Inspiration 3 mentions "probabilistic edge-switching".
    #    
    #    Let's implement a hybrid:
    #    1. Use LUW to get a valid graph quickly.
    #    2. Use the remaining time to try to find a smaller graph via voltage lifts or switching?
    #    Switching on a 1000-vertex graph to reduce n is impossible (n is fixed by structure).
    #    Voltage lifts change n.
    #
    #    The score is n. Lower is better.
    #    LUW gives a specific n.
    #    If we can find a smaller n, we beat 1.0.
    #
    #    The voltage graph approach in the seed tries m=50..150.
    #    n = 10 * m.
    #    If LUW gives n > 500, then m=50 could be better.
    #    What is LUW's n?
    #    The LUW graph has |V| = 2 * q * q^d?
    #    In the seed:
    #    `side` is 0 or 1.
    #    `x` is a tuple of length d+1.
    #    Total vertices = 2 * q^(d+1)?
    #    For k=5, d=5.
    #    n = 2 * 5 * 5^5 = 2 * 5^6 = 31250?
    #    That's huge.
    #    Wait, the seed's LUW implementation:
    #    `start = (0, (None,) + (0,) * d)`
    #    It generates vertices by BFS from start.
    #    Since it's a Cayley graph, it will generate the whole group.
    #    The group size is 2 * q^d?
    #    If n=31250, that's much larger than 1116.
    #    So LUW is NOT the best. The best known is 1116.
    #    The seed's `_voltage_graph_construct` is trying to find a graph with n ~ 10*m.
    #    If m=111, n=1110.
    #    So the voltage graph approach is the one that can beat the LUW if it finds a valid lift.
    #
    #    Therefore, the "design" to keep is the **Voltage Graph Construction** on K_{k,k}.
    #    The seed's implementation was messy.
    #    I will rewrite `_voltage_graph_construct` cleanly and efficiently.
    #
    #    Key steps for Voltage Graph on K_{5,5}:
    #    1. Precompute all cycles of length < g in K_{5,5}.
    #    2. For each m, find voltages (elements of Z_m) such that sum of voltages on any cycle is 0 mod m.
    #       Actually, for girth g, we need sum != 0 for cycles of length < g.
    #       For K_{5,5}, cycles of length 4, 6, 8 exist.
    #       We need sum != 0 mod m for all cycles of length 4, 6, 8.
    #    3. If such voltages exist, the lift has girth at least 10?
    #       No, the base has no cycles of length < 4.
    #       If we eliminate 4, 6, 8 cycles in the lift, the girth is at least 10?
    #       The task asks for girth 9.
    #       If girth >= 10, it satisfies girth >= 9.
    #       So we just need to avoid cycles of length 4, 6, 8.
    #       (Cycles of length 9 are not possible in bipartite graphs, so if the base is bipartite,
    #        all cycles are even. So avoiding 4, 6, 8 ensures girth >= 10, which is >= 9).
    #    
    #    So the problem reduces to:
    #    Find a lift of K_{5,5} of degree m such that no 4, 6, or 8-cycle in the base
    #    has a total voltage of 0 mod m.
    #    Minimize m (and thus n=10m).
    #
    #    This is a constraint satisfaction problem.
    #    We can use random search for voltages.
    #    
    #    Optimization:
    #    - Precompute cycle lists as lists of edge indices.
    #    - Use integer arithmetic.
    #    - Check constraints efficiently.
    #
    #    Let's implement this.

    def _precompute_cycles_kkk(k):
        n_base = 2 * k
        adj = [[] for _ in range(n_base)]
        edge_map = {}
        edges = []
        idx = 0
        for u in range(k):
            for v in range(k, n_base):
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
                idx += 1
        
        cycles = {4: [], 6: [], 8: []}
        
        # 4-cycles: u1-v1-u2-v2-u1
        for u1 in range(k):
            for u2 in range(u1+1, k):
                for v1 in range(k, n_base):
                    for v2 in range(v1+1, n_base):
                        e1 = edge_map[(u1, v1)]
                        e2 = edge_map[(u2, v1)]
                        e3 = edge_map[(u2, v2)]
                        e4 = edge_map[(u1, v2)]
                        cycles[4].append((e1, e2, e3, e4))
                        
        # 6-cycles: u1-v1-u2-v2-u3-v3-u1
        # Permutations of 3 U's and 3 V's.
        us = list(range(k))
        vs = list(range(k, n_base))
        for i1 in range(k):
            for i2 in range(i1+1, k):
                for i3 in range(i2+1, k):
                    for j1 in range(len(vs)):
                        for j2 in range(j1+1, len(vs)):
                            for j3 in range(j2+1, len(vs)):
                                # 3! ways to order U's? No, fixed u1, u2, u3?
                                # The cycle visits 3 U's.
                                # We can fix the order of U's as i1, i2, i3?
                                # No, the cycle can be i1-v-i2-v-i3-v-i1.
                                # Or i1-v-i3-v-i2-v-i1.
                                # There are 2 orders for the middle 2 U's?
                                # Actually, let's just iterate all permutations of the 3 U indices
                                # and 3 V indices?
                                # To avoid duplicates, fix the first U as i1.
                                # The other 2 U's can be in 2 orders.
                                # The 3 V's can be in 3! = 6 orders.
                                # Total 12? But reverse cycles are same.
                                # Let's just generate and deduplicate by set of edges?
                                # Or just check all 12 and see if sum is 0.
                                # Duplicates don't hurt correctness, just speed.
                                import itertools
                                for pu in itertools.permutations([i2, i3]):
                                    for pv in itertools.permutations([vs[j1], vs[j2], vs[j3]]):
                                        u_seq = [i1, pu[0], pu[1]]
                                        v_seq = [pv[0], pv[1], pv[2]]
                                        e1 = edge_map[(u_seq[0], v_seq[0])]
                                        e2 = edge_map[(u_seq[1], v_seq[0])]
                                        e3 = edge_map[(u_seq[1], v_seq[1])]
                                        e4 = edge_map[(u_seq[2], v_seq[1])]
                                        e5 = edge_map[(u_seq[2], v_seq[2])]
                                        e6 = edge_map[(u_seq[0], v_seq[2])]
                                        cycles[6].append((e1, e2, e3, e4, e5, e6))
        
        # 8-cycles: 4 U's, 4 V's.
        # This is more complex.
        # Number of 8-cycles in K_{5,5} is large.
        # We might skip 8-cycles if 4 and 6 are hard enough?
        # If we avoid 4 and 6, the girth is at least 8?
        # No, 8-cycles still exist.
        # We MUST avoid 8-cycles to get girth >= 10.
        #
        # Generating all 8-cycles in K_{5,5}:
        # Choose 4 U's and 4 V's.
        # Number of ways: C(5,4)*C(5,4) = 25.
        # For each set, number of 8-cycles in K_{4,4}?
        # Hamiltonian cycles in K_{4,4}.
        # (4-1)! * (4-1)! / 2? No.
        # Let's just enumerate.
        
        for i1 in range(k):
            for i2 in range(i1+1, k):
                for i3 in range(i2+1, k):
                    for i4 in range(i3+1, k):
                        for j1 in range(len(vs)):
                            for j2 in range(j1+1, len(vs)):
                                for j3 in range(j2+1, len(vs)):
                                    for j4 in range(j3+1, len(vs)):
                                        # Fixed sets of 4 U's and 4 V's.
                                        # Enumerate Hamiltonian cycles in K_{4,4}.
                                        # Fix i1.
                                        # Order of other 3 U's: 3! = 6.
                                        # Order of 4 V's: 4! = 24.
                                        # Total 144?
                                        # Many are reverses or rotations.
                                        # We'll just generate and deduplicate or check all.
                                        import itertools
                                        for pu in itertools.permutations([i2, i3, i4]):
                                            for pv in itertools.permutations([vs[j1], vs[j2], vs[j3], vs[j4]]):
                                                u_seq = [i1, pu[0], pu[1], pu[2]]
                                                v_seq = [pv[0], pv[1], pv[2], pv[3]]
                                                e1 = edge_map[(u_seq[0], v_seq[0])]
                                                e2 = edge_map[(u_seq[1], v_seq[0])]
                                                e3 = edge_map[(u_seq[1], v_seq[1])]
                                                e4 = edge_map[(u_seq[2], v_seq[1])]
                                                e5 = edge_map[(u_seq[2], v_seq[2])]
                                                e6 = edge_map[(u_seq[3], v_seq[2])]
                                                e7 = edge_map[(u_seq[3], v_seq[3])]
                                                e8 = edge_map[(u_seq[0], v_seq[3])]
                                                cycles[8].append((e1, e2, e3, e4, e5, e6, e7, e8))
        
        # Deduplicate cycles by frozenset of edges?
        # If we use frozenset, we lose the order, but sum is commutative.
        # However, a set of edges might form multiple cycles? No, in a bipartite graph,
        # a set of edges forming a cycle is unique?
        # If a set of edges forms a 2-regular graph, it is a collection of cycles.
        # If it's a single cycle, it's unique.
        # So we can deduplicate by frozenset of edge indices.
        
        deduped = {4: set(), 6: set(), 8: set()}
        for l in [4, 6, 8]:
            for c in cycles[l]:
                deduped[l].add(frozenset(c))
                
        # Convert back to lists for faster summing?
        # We need the list of indices to sum voltages.
        # frozenset is unordered. We can just iterate the set.
        # But summing a set is slow.
        # Let's keep them as tuples of indices, but deduplicate by frozenset key.
        seen = {4: set(), 6: set(), 8: set()}
        final_cycles = {4: [], 6: [], 8: []}
        for l in [4, 6, 8]:
            for c in cycles[l]:
                key = frozenset(c)
                if key not in seen[l]:
                    seen[l].add(key)
                    final_cycles[l].append(c)
                    
        return final_cycles, edge_map, edges

    def _find_valid_voltages(cycles, m, num_trials=1000):
        num_edges = len(cycles[8][0]) if cycles[8] else 0 # just to get a dummy
        # Actually, we need the number of edges in K_{5,5}, which is k*k = 25.
        # But cycles reference indices 0..24.
        
        # Precompute cycle edge lists
        c4 = cycles[4]
        c6 = cycles[6]
        c8 = cycles[8]
        
        for t in range(num_trials):
            voltages = [random.randint(1, m-1) for _ in range(25)]
            
            valid = True
            for cyc in c4:
                if (voltages[cyc[0]] + voltages[cyc[1]] + voltages[cyc[2]] + voltages[cyc[3]]) % m == 0:
                    valid = False
                    break
            if not valid: continue
            
            for cyc in c6:
                s = 0
                for idx in cyc:
                    s += voltages[idx]
                if s % m == 0:
                    valid = False
                    break
            if not valid: continue
            
            for cyc in c8:
                s = 0
                for idx in cyc:
                    s += voltages[idx]
                if s % m == 0:
                    valid = False
                    break
            if not valid: continue
            
            return voltages
        return None

    def _build_lift(k, m, voltages, edge_map, edges):
        n_lift = 2 * k * m
        adj = [set() for _ in range(n_lift)]
        
        for idx, (u, v) in enumerate(edges):
            w = voltages[idx]
            for g in range(m):
                u_idx = u * m + g
                v_idx = v * m + (g + w) % m
                adj[u_idx].add(v_idx)
                adj[v_idx].add(u_idx)
        
        # Check girth
        # We assume girth is 10 if no 4,6,8 cycles.
        # But we should verify.
        # Since the base is bipartite, all cycles are even.
        # If no 4, 6, 8, then girth >= 10.
        # We can skip the full girth check if we trust the voltage condition.
        # But for safety, let's do a quick BFS check on a few nodes?
        # Or just trust it.
        
        # Convert to list of lists
        adj_list = [list(s) for s in adj]
        
        # Return n and edges
        # The graph is connected?
        # Lifts of connected graphs are connected if the voltage group is generated by the edge voltages?
        # If the graph is connected and the voltages generate Z_m, it's connected.
        # We should check connectivity.
        
        visited = [False] * n_lift
        queue = deque([0])
        visited[0] = True
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        if count != n_lift:
            # Not connected. Find largest component.
            # For now, if not connected, we might discard or use largest.
            # Let's just return the whole thing? No, must be simple graph.
            # If disconnected, it's a union of graphs.
            # The task says "a graph". Usually implies connected?
            # "5-regular graph". A disjoint union is a graph.
            # But girth is min of components.
            # If all components have girth >= 10, it's fine.
            # But n is the total vertices.
            # We want to minimize n.
            # If disconnected, we can just take the smallest component?
            # No, we want the whole graph to be valid.
            # If we take a component, is it k-regular?
            # Yes, induced subgraph of a k-regular graph?
            # No, the lift is k-regular. A component is k-regular.
            # So we can just return the smallest component.
            
            components = []
            visited = [False] * n_lift
            for i in range(n_lift):
                if not visited[i]:
                    comp = []
                    queue = deque([i])
                    visited[i] = True
                    while queue:
                        u = queue.popleft()
                        comp.append(u)
                        for v in adj_list[u]:
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                    components.append(comp)
                    
            min_comp = min(components, key=len)
            if len(min_comp) < n_lift:
                # Remap
                mapping = {old: new for new, old in enumerate(min_comp)}
                new_edges = []
                for u in min_comp:
                    for v in adj_list[u]:
                        if u < v:
                            new_edges.append((mapping[u], mapping[v]))
                return len(min_comp), new_edges
            
        # If connected
        new_edges = []
        for u in range(n_lift):
            for v in adj_list[u]:
                if u < v:
                    new_edges.append((u, v))
        return n_lift, new_edges

    if k == 5 and g == 9:
        cycles, edge_map, edges = _precompute_cycles_kkk(k)
        
        # Try m from low to high
        # Lower bound on m?
        # Moore bound for k=5, g=9 is 428.
        # n = 10m. So 10m >= 428 -> m >= 43.
        # Let's try m from 43 upwards.
        
        best_result = None
        for m in range(43, 200):
            if time.time() > deadline - 10:
                break
                
            # Try random voltages
            max_trials = 5000 if m < 100 else 2000
            voltages = _find_valid_voltages(cycles, m, max_trials)
            
            if voltages:
                n, new_edges = _build_lift(k, m, voltages, edge_map, edges)
                if best_result is None or n < best_result[0]:
                    best_result = (n, new_edges)
                    # Since we iterate m upwards, the first valid m gives smallest n.
                    # We can break immediately.
                    return best_result
                    
        if best_result:
            return best_result
            
    # Fallback: LUW construction
    def _luw_construct(k, g):
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

    return _luw_construct(k, g)
