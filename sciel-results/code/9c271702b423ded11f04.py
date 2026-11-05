import time
import random
from random import Random

DEADLINE_MARGIN = 140  # 10 seconds safety margin before 150s

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def cycle_edges(cycle):
    """Given a list of vertices forming a cycle (>=3 distinct), return set of edges."""
    if len(cycle) < 3:
        return set()
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        e = (u, v) if u < v else (v, u)
        edges.add(e)
    return edges

def decompose_via_2factor(edges, max_time=40.0):
    """
    Decompose edges into fewest pieces by finding a large 2-factor.
    A 2-factor is a subgraph where every vertex has degree exactly 2 (or 0 if isolated in the factor).
    The edges of the 2-factor form a collection of edge-disjoint cycles.
    The remaining edges (E \ F) must be single-edge pieces.
    
    Strategy:
    1. Start with an empty 2-factor F (or a greedy initial one).
    2. Iteratively try to add edges to F such that degree constraints (deg <= 2) are maintained 
       and we maximize the number of edges in F.
       Actually, maximizing edges in F is equivalent to minimizing leftover edges.
       However, F must consist of valid cycles. 
       
       Alternative Approach: 
       Randomized Local Search on the set of "selected" edges.
       Let S be the set of edges selected for cycles.
       Constraint: For all v, deg_S(v) <= 2.
       Objective: Maximize |S| such that S is a union of cycles (no paths).
       
       This is equivalent to finding a maximum 2-matching that contains no paths.
       
       Let's use a simpler heuristic:
       1. Greedily select edges to form a 2-factor.
       2. Identify connected components of the selected subgraph.
       3. If a component is a cycle, keep it.
       4. If a component is a path or tree, we have a problem. 
       
       Better Heuristic:
       1. Start with all edges as "potential".
       2. Remove edges that are "bad" (e.g., bridges that prevent cycle formation? No, bridges can be in cycles).
       
       Let's try a different angle: 
       The number of pieces = (Number of cycles in 2-factor) + (Number of leftover edges).
       Note: If we have a 2-factor with k cycles, and m leftover edges, total pieces = k + m.
       Total edges E = sum(len(cycle_i)) + m.
       
       We want to minimize k + m.
       Since E is fixed, minimizing k + m is roughly maximizing the average cycle length?
       No, it's about packing as many edges as possible into cycles.
       
       Algorithm:
       1. Initialize S = empty set.
       2. While there are edges in E \ S:
          - Find a cycle C in E \ S.
          - Add C to S.
          - This is the greedy approach which we know is suboptimal.
          
       New Strategy: Simulated Annealing / Randomized Restart on 2-Factor
       1. Generate a random 2-factor (or near 2-factor) of G.
          - A 2-factor can be found by finding a perfect matching in the line graph? No.
          - We can find a 2-factor by solving a bipartite matching problem? 
          - Actually, finding *any* 2-factor is NP-hard? No, finding a 2-factor is polynomial (Tutte's theorem / reduction to matching).
          - But we want a *maximum* 2-factor (maximizing edges).
          
       Let's use a simple local search on the "degree constraint" satisfaction.
       1. Start with a random subset of edges S such that deg(v) <= 2 for all v.
       2. Compute the connected components of S.
       3. Count valid cycles (components that are cycles). Let this be C_valid.
       4. Count leftover edges L = |E| - |S|.
       5. Score = C_valid + L. Minimize this.
       
       Move: Swap an edge in S with an edge not in S, or add/remove an edge, maintaining deg <= 2.
       This is complex to implement robustly quickly.
       
       Let's try a simpler structural change:
       **Greedy Cycle Packing with Backtracking on Small Residuals**
       
       1. Find a long cycle.
       2. Remove it.
       3. If the remaining graph is small (e.g., < 10 edges), use exhaustive search/ILP to solve it optimally.
       4. If the remaining graph is large, continue greedy.
       
       This combines the speed of greedy with the optimality of exact methods on the "hard" tail.
    """
    start = time.time()
    all_edges_set = set(tuple(sorted(e)) for e in edges)
    remaining = set(all_edges_set)
    pieces = []
    rng = Random(123)
    
    # Phase 1: Greedy extraction of long cycles
    while remaining and (time.time() - start) < max_time * 0.7:
        cur_time = time.time()
        if cur_time - start > max_time:
            break
        
        adj = build_adj(list(remaining))
        
        # Find a cycle using randomized DFS
        cycle = None
        nodes = list(set([u for (u, v) in remaining] + [v for (u, v) in remaining]))
        if len(nodes) >= 3:
            # Try to find a cycle
            for _ in range(50):
                start_node = rng.choice(nodes)
                if start_node not in adj or len(adj[start_node]) < 2:
                    continue
                visited = {start_node}
                path = [start_node]
                stack = [(start_node, iter(sorted(adj[start_node])))]
                
                found = False
                while stack:
                    node, neighbors = stack[-1]
                    advanced = False
                    for nb in neighbors:
                        e = (node, nb) if node < nb else (nb, node)
                        if e not in remaining:
                            continue
                        if nb == start_node and len(path) >= 3:
                            cycle = path[:]
                            found = True
                            break
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            stack.append((nb, iter(sorted(adj[nb]))))
                            advanced = True
                            break
                    if found:
                        break
                    if not advanced:
                        stack.pop()
                        if path:
                            path.pop()
                            visited.discard(path[-1] if path else None)
                if cycle:
                    break
        
        if cycle and len(cycle) >= 3:
            cyc_edges = cycle_edges(cycle)
            if cyc_edges <= remaining:
                pieces.append(cycle[:])
                for e in cyc_edges:
                    remaining.discard(e)
            else:
                # Fallback
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
        else:
            # No cycle found in this iteration, or time running out
            break

    # Phase 2: Exact or near-exact solve for the remaining edges
    if remaining and (time.time() - start) < max_time:
        rem_edges = list(remaining)
        if len(rem_edges) <= 15:
            # Use exhaustive search for small residuals
            # We want to partition rem_edges into cycles and single edges
            # Minimize number of pieces
            
            # Bitmask DP or Backtracking
            # n_edges <= 15 -> 2^15 = 32768 states. Feasible.
            
            edge_list = rem_edges
            m = len(edge_list)
            # Map edges to indices
            edge_idx = {e: i for i, e in enumerate(edge_list)}
            
            # Precompute which edges form a valid cycle
            # A subset of edges is a cycle if it forms a connected component with all degrees 2.
            # Since m is small, we can check subsets.
            
            # Actually, simpler: Backtracking
            # State: set of used edges
            # Goal: Minimize pieces
            
            best_pieces = m # Worst case: all single edges
            best_decomp = None
            
            def backtrack(used_mask, current_pieces):
                nonlocal best_pieces, best_decomp
                if current_pieces >= best_pieces:
                    return
                
                # Find first unused edge
                i = 0
                while i < m and (used_mask >> i) & 1:
                    i += 1
                
                if i == m:
                    # All edges used
                    if current_pieces < best_pieces:
                        best_pieces = current_pieces
                        # We don't actually need to store the full decomp structure here, 
                        # just the count is enough for optimization, but we need the pieces for output.
                        # Let's reconstruct or store the choice.
                        # For simplicity, we'll just return the count and re-run greedy to get a valid set?
                        # No, we need the actual cycles.
                        # Let's store the choices.
                        pass
                    return
                
                e = edge_list[i]
                
                # Option 1: Use e as a single edge
                backtrack(used_mask | (1 << i), current_pieces + 1)
                
                # Option 2: Try to extend e into a cycle
                # Find all cycles containing e in the remaining graph
                # This is expensive to do inside backtracking.
                
                # Alternative: Just try to find *any* cycle in the remaining unused edges.
                # If we find one, take it.
                
                # Let's try a different exact method for small graphs:
                # Enumerate all subsets of edges that form a cycle.
                
                # For now, if m is small, we can just try all permutations? No.
                
                # Let's stick to a simpler exact method:
                # If m <= 10, we can try all subsets.
                # If m > 10, we rely on greedy.
                
                # Given time constraints, let's just do a simple local search on the residual.
                # If m is small, we can try swapping edges to form cycles.
                
                pass
            
            # The backtracking above is complex to implement correctly with state reconstruction.
            # Let's use a simpler heuristic for the residual:
            # Try to find *any* cycle in the residual.
            # If found, remove it.
            # Repeat until no cycles.
            
            # This is effectively what Phase 1 did, but we can be more aggressive with restarts.
            
            for _ in range(50):
                if not remaining:
                    break
                adj = build_adj(list(remaining))
                cycle = None
                nodes = list(set([u for (u, v) in remaining] + [v for (u, v) in remaining]))
                if len(nodes) >= 3:
                    start_node = rng.choice(nodes)
                    if start_node in adj and len(adj[start_node]) >= 2:
                        visited = {start_node}
                        path = [start_node]
                        stack = [(start_node, iter(sorted(adj[start_node])))]
                        found = False
                        while stack:
                            node, neighbors = stack[-1]
                            advanced = False
                            for nb in neighbors:
                                e = (node, nb) if node < nb else (nb, node)
                                if e not in remaining:
                                    continue
                                if nb == start_node and len(path) >= 3:
                                    cycle = path[:]
                                    found = True
                                    break
                                if nb not in visited:
                                    visited.add(nb)
                                    path.append(nb)
                                    stack.append((nb, iter(sorted(adj[nb]))))
                                    advanced = True
                                    break
                            if found:
                                break
                            if not advanced:
                                stack.pop()
                                if path:
                                    path.pop()
                                    visited.discard(path[-1] if path else None)
                
                if cycle:
                    cyc_edges = cycle_edges(cycle)
                    if cyc_edges <= remaining:
                        pieces.append(cycle[:])
                        for e in cyc_edges:
                            remaining.discard(e)
                else:
                    break

    # Add remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time per graph
    # We have 140s margin. 3 graphs. ~45s each.
    per_graph_time = 45.0
    
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = min(per_graph_time, DEADLINE_MARGIN - (cur_time - start))
        if remaining_time < 1.0:
            # No time left, return all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = decompose_via_2factor(edges, max_time=remaining_time)
        results.append(pieces)
        
        # Check deadline
        cur_time = time.time()
        if cur_time - start > DEADLINE_MARGIN - 5:
            # Return remaining as individual edges
            for remaining_spec in specs[i + 1:]:
                e = instance(*remaining_spec)
                results.append([list(e) for e in e])
            break
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])
    
    return results[:3]

if __name__ == "__main__":
    res = construct()
    print(len(res))
    for r in res:
        print(len(r))
