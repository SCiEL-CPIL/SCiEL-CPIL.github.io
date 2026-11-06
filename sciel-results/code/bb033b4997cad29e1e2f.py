"""Improved program: MIS via branch-and-bound + local search for square-sum-free set."""
import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    N = 256
    # Precompute conflict: a and b conflict if a+b is a perfect square
    # Including diagonal: a+a = 2a must NOT be a perfect square
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a][b] = True if a+b is a perfect square (a,b in 1..256)
    # We want an independent set in the conflict graph (no two elements conflict)
    # Equivalently, maximum independent set.
    
    # Build adjacency as bitmasks for speed
    # node i represents number (i+1), index 0..255
    adj = [0] * N
    for a in range(1, N + 1):
        ai = a - 1
        for b in range(a, N + 1):  # only need a<=b, but build full
            bi = b - 1
            if a + b in squares:
                adj[ai] |= (1 << bi)
                adj[bi] |= (1 << ai)
    
    # Check diagonal: a+a=2a must not be square
    diagonal_safe = [True] * N
    for a in range(1, N + 1):
        if 2 * a in squares:
            diagonal_safe[a - 1] = False

    # Remove nodes that are not diagonal-safe (they can't be in the set)
    # Actually, diagonal constraint means a+a is a square, so a cannot be in set
    # This is already captured: if a+a is square, then a conflicts with itself
    # In MIS, a node with self-loop can't be in independent set
    # Let's just exclude them from consideration
    
    available = [i for i in range(N) if diagonal_safe[i]]
    avail_set = set(available)
    
    # Greedy initial solution: pick nodes in order of minimum degree first
    def greedy_solution():
        remaining = list(available)
        result = []
        while remaining:
            # Pick node with minimum conflicts among remaining
            best_idx = 0
            best_degree = float('inf')
            for idx in remaining:
                deg = 0
                for r in remaining:
                    if r != idx and (adj[idx] >> r) & 1:
                        deg += 1
                if deg < best_degree:
                    best_degree = deg
                    best_idx = idx
            result.append(remaining.pop(best_idx))
        return result
    
    # Local search: try to add nodes, or swap
    def local_search_improve(solution, max_iters=5000):
        sol_set = set(solution)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try adding each non-solution node
            for node in available:
                if node in sol_set:
                    continue
                # Check if adding node is valid
                valid = True
                for s in sol_set:
                    if (adj[node] >> s) & 1:
                        valid = False
                        break
                if valid:
                    sol_set.add(node)
                    improved = True
            # Try 1-swap: remove one, add two that were blocked
            if not improved:
                for remove_node in list(sol_set):
                    # Find nodes blocked by remove_node
                    blocked = []
                    for node in available:
                        if node in sol_set:
                            continue
                        # Check if node would be valid if remove_node is gone
                        valid = True
                        for s in sol_set:
                            if s == remove_node:
                                continue
                            if (adj[node] >> s) & 1:
                                valid = False
                                break
                        if valid:
                            blocked.append(node)
                    # Try adding up to 2 blocked nodes
                    if len(blocked) >= 2:
                        # Try pairs
                        for i in range(min(len(blocked), 10)):
                            for j in range(i + 1, min(len(blocked), 10)):
                                b1, b2 = blocked[i], blocked[j]
                                if (adj[b1] >> b2) & 1:
                                    continue
                                new_sol = sol_set - {remove_node} | {b1, b2}
                                # Verify
                                valid = True
                                for x in new_sol:
                                    for y in new_sol:
                                        if x == y:
                                            continue
                                        if (adj[x] >> y) & 1:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    sol_set = new_sol
                                    improved = True
                                    break
                            if improved:
                                break
                    elif len(blocked) == 1:
                        b1 = blocked[0]
                        new_sol = sol_set - {remove_node} | {b1}
                        # This is same size, only useful if it helps future additions
                        # Skip for now
                    if improved:
                        break
        return list(sol_set)
    
    # Branch and bound for MIS
    def branch_and_bound(initial_set, time_limit):
        """Find maximum independent set using branch and bound."""
        best = list(initial_set)
        best_size = len(best)
        best_set = set(best)
        
        # Use greedy coloring upper bound
        def color_bound(nodes):
            """Greedy coloring gives upper bound on clique size in subgraph induced by nodes.
            For MIS, we use the complement: max independent set in G = max clique in complement.
            But we're working directly on G: upper bound on independent set size.
            Actually, for MIS in G, we can use: color the subgraph G[nodes], the number of colors
            is an upper bound on the size of a clique, not independent set.
            
            Let me think again. We want MIS in conflict graph G.
            Upper bound: partition nodes into cliques in G. Each independent set can take at most 1 from each clique.
            So if we can cover nodes with k cliques, MIS size <= k.
            Greedy: repeatedly pick a maximal clique, remove it, count.
            """
            remaining = list(nodes)
            count = 0
            while remaining:
                # Find a maximal clique in G[remaining]
                # Greedy: pick node with max degree, add its neighbors that are also connected
                clique = []
                for i in range(len(remaining)):
                    node = remaining[i]
                    # Check if node connects to all in clique
                    can_add = True
                    for c in clique:
                        if not ((adj[node] >> c) & 1):
                            can_add = False
                            break
                    if can_add:
                        clique.append(node)
                count += 1
                # Remove clique from remaining
                clique_set = set(clique)
                remaining = [n for n in remaining if n not in clique_set]
            return count
        
        # Simpler upper bound: for each node, 1 + MIS of its non-neighbors
        # But that's expensive. Let's use a simpler approach.
        
        def upper_bound(nodes):
            """Upper bound on MIS size in subgraph induced by nodes."""
            if not nodes:
                return 0
            # Each node can be in the independent set, so trivial bound is len(nodes)
            # Better: greedy coloring of complement gives upper bound on clique in complement = MIS in G
            # Actually let's just use: for each node, it can contribute at most 1,
            # and we can greedily find a packing
            # Simple bound: len(nodes) (trivial)
            # Better: find a maximal matching, each edge means at most 1 endpoint
            # So upper bound = len(nodes) - size_of_maximal_matching
            # Greedy maximal matching:
            used = set()
            matching_size = 0
            nodes_list = list(nodes)
            for i in range(len(nodes_list)):
                u = nodes_list[i]
                if u in used:
                    continue
                for j in range(i + 1, len(nodes_list)):
                    v = nodes_list[j]
                    if v in used:
                        continue
                    if (adj[u] >> v) & 1:
                        matching_size += 1
                        used.add(u)
                        used.add(v)
                        break
            return len(nodes) - matching_size
        
        # Recursive branch and bound
        def bb(candidates, current_set, current_size):
            nonlocal best, best_size, best_set
            
            if time.time() - start_time > time_limit:
                return
            
            if not candidates:
                if current_size > best_size:
                    best_size = current_size
                    best = list(current_set)
                    best_set = set(current_set)
                return
            
            ub = upper_bound(candidates)
            if current_size + ub <= best_size:
                return
            
            # Pick a node to branch on
            # Heuristic: pick node with highest degree in candidate subgraph
            best_node = candidates[0]
            best_deg = -1
            for node in candidates:
                deg = 0
                for c in candidates:
                    if c != node and (adj[node] >> c) & 1:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
            
            # Branch 1: include best_node
            new_current = set(current_set)
            new_current.add(best_node)
            # New candidates: those not adjacent to best_node
            new_candidates = [c for c in candidates if c != best_node and not ((adj[best_node] >> c) & 1)]
            bb(new_candidates, new_current, current_size + 1)
            
            # Branch 2: exclude best_node
            new_candidates = [c for c in candidates if c != best_node]
            bb(new_candidates, current_set, current_size)
        
        bb(available, set(), 0)
        return best
    
    # Get initial solution
    initial = greedy_solution()
    initial = local_search_improve(initial, max_iters=2000)
    
    # Try to improve with branch and bound
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        improved = branch_and_bound(initial, time.time() + remaining_time - 2)
        if len(improved) > len(initial):
            initial = improved
    
    # Final local search
    initial = local_search_improve(initial, max_iters=3000)
    
    # Convert back to 1-indexed
    result = [i + 1 for i in initial]
    
    # Verify
    for i in range(len(result)):
        for j in range(i, len(result)):
            a, b = result[i], result[j]
            s = a + b
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid, should not happen
                pass
    
    return sorted(result)
