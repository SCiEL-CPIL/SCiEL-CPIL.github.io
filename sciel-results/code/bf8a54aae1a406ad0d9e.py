import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_conflict_set(edge, adj, edge_to_idx, idx_to_edge):
        # Returns set of edge indices that conflict with 'edge'
        # Conflict if they share a vertex or are joined by an edge
        u, v = edge
        conflicts = set()
        # Edges sharing u or v
        for w in adj[u]:
            e1 = (min(u, w), max(u, w))
            if e1 in edge_to_idx:
                conflicts.add(edge_to_idx[e1])
        for w in adj[v]:
            e1 = (min(v, w), max(v, w))
            if e1 in edge_to_idx:
                conflicts.add(edge_to_idx[e1])
        
        # Edges joined by an edge to u or v
        # If (u, w) is an edge, then any edge incident to w conflicts? No.
        # Strong edge coloring: no two edges in the same color class share a vertex OR are connected by an edge.
        # Let e = (u, v).
        # Neighbor edges are those sharing u or v.
        # "Joined by an edge" means if e1=(a,b) and e2=(c,d), and there is an edge (x,y) such that x in e1, y in e2.
        # So for e=(u,v), an edge e'=(a,b) conflicts if:
        # 1. {u,v} & {a,b} != empty (share vertex)
        # 2. There exists an edge (u,a), (u,b), (v,a), or (v,b).
        
        # Let's re-evaluate conflict logic carefully.
        # Two edges e1, e2 are in the same "forbidden" set if they are adjacent in the line graph L(G) or connected by an edge in L(G)?
        # No, "induced matching" means the subgraph induced by the edges is a matching.
        # So if e1=(u,v) and e2=(a,b), they cannot share a vertex.
        # AND, there cannot be an edge in G connecting a vertex of e1 to a vertex of e2.
        
        # So for a fixed edge e=(u,v), the set of edges that CANNOT have the same color as e is:
        # 1. All edges incident to u or v.
        # 2. All edges incident to any neighbor of u (except e itself if it's incident to u, which it is).
        #    Wait. If e' is incident to a neighbor of u, say u-u' is an edge, and e'=(u', w), then u' is in e' and u is in e, and (u,u') is an edge. So e and e' are joined by an edge.
        #    So yes, any edge incident to N(u) or N(v) conflicts.
        
        conflict_indices = set()
        # 1. Edges incident to u or v
        for w in adj[u]:
            e_idx = edge_to_idx.get((min(u, w), max(u, w)))
            if e_idx is not None:
                conflict_indices.add(e_idx)
        for w in adj[v]:
            e_idx = edge_to_idx.get((min(v, w), max(v, w)))
            if e_idx is not None:
                conflict_indices.add(e_idx)
                
        # 2. Edges incident to neighbors of u or v
        # We need to check edges (x, y) where x in N(u) or x in N(v) (or y in N(u) etc)
        # Actually, simpler: for any neighbor w of u, any edge incident to w conflicts with e.
        # We already added edges incident to u. Now add edges incident to N(u) \ {u}.
        # But careful not to double count or include e itself.
        
        # Let's just iterate through all edges and check? No, too slow if repeated.
        # Pre-compute conflict sets for each edge?
        # There are ~240 edges. 240*240 is small.
        
        pass # Placeholder, will implement pre-computation

    def precompute_conflicts(edges, adj):
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]
        
        for i, e1 in enumerate(edges):
            u1, v1 = e1
            for j, e2 in enumerate(edges):
                if i == j: continue
                u2, v2 = e2
                # Check if they share a vertex
                if u1 in e2 or v1 in e2 or u2 in e1 or v2 in e1:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
                # Check if they are joined by an edge
                # Is there an edge (a,b) such that a in e1, b in e2?
                # Pairs to check: (u1, u2), (u1, v2), (v1, u2), (v1, v2)
                joined = False
                if (min(u1, u2), max(u1, u2)) in edge_to_idx:
                    joined = True
                elif (min(u1, v2), max(u1, v2)) in edge_to_idx:
                    joined = True
                elif (min(v1, u2), max(v1, u2)) in edge_to_idx:
                    joined = True
                elif (min(v1, v2), max(v1, v2)) in edge_to_idx:
                    joined = True
                
                if joined:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    
        return conflicts, edge_to_idx

    def greedy_color(edges, adj, conflicts, edge_to_idx):
        # Order edges by some heuristic, e.g., descending sum of degrees of endpoints
        # Or just sorted order as required by output format? 
        # The output must follow sorted edge order. 
        # But we can color in any order and then map back.
        
        # Let's use a simple greedy: iterate edges in sorted order.
        # For each edge, find the smallest color not used by any conflicting edge.
        
        coloring = [-1] * len(edges)
        used_colors = defaultdict(set) # color -> set of edge indices
        
        for i, e in enumerate(edges):
            # Find forbidden colors
            forbidden = set()
            for j in conflicts[i]:
                if coloring[j] != -1:
                    forbidden.add(coloring[j])
            
            # Find smallest available color
            c = 0
            while c in forbidden:
                c += 1
            coloring[i] = c
            used_colors[c].add(i)
            
        return coloring

    def is_valid_coloring(coloring, conflicts):
        for i, c in enumerate(coloring):
            for j in conflicts[i]:
                if coloring[j] == c:
                    return False
        return True

    def try_reduce_colors(coloring, conflicts, n_edges, max_iterations=1000):
        # Try to reduce the number of colors by eliminating the highest color index
        current_coloring = coloring[:]
        current_max_color = max(current_coloring) if current_coloring else 0
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Find the highest color used
            used_colors = set(current_coloring)
            if not used_colors:
                break
            max_c = max(used_colors)
            
            # Get edges with color max_c
            edges_with_max_c = [i for i, c in enumerate(current_coloring) if c == max_c]
            
            if not edges_with_max_c:
                break
                
            # Try to move each edge in edges_with_max_c to a lower color
            moved_any = False
            for i in edges_with_max_c:
                # Find valid colors < max_c
                valid_colors = []
                for c in range(max_c):
                    # Check if color c is valid for edge i
                    # Valid if no conflicting edge j has color c
                    valid = True
                    for j in conflicts[i]:
                        if current_coloring[j] == c:
                            valid = False
                            break
                    if valid:
                        valid_colors.append(c)
                
                if valid_colors:
                    # Pick a color, maybe the one with fewest edges to balance? Or random?
                    # Let's pick the one with the fewest edges currently
                    # Or just the first one (lowest index) to pack colors
                    # Picking lowest index helps keep colors compact
                    chosen_c = min(valid_colors) 
                    current_coloring[i] = chosen_c
                    moved_any = True
                else:
                    # Stuck on this edge
                    # We might need to perturb
                    break
            
            if moved_any:
                # We might have reduced the max color or just moved some edges
                # Check if max color is still in use
                if max_c not in set(current_coloring):
                    # Color eliminated
                    continue # Try to eliminate next highest
            else:
                # Stuck, perform a perturbation
                # Pick a random edge and swap its color with another random edge?
                # Or move a random edge to a random valid color?
                # Let's try: pick a random edge from edges_with_max_c, 
                # and try to swap its color with a random edge of a lower color?
                # Or just random restart of the reduction process with a small random change.
                
                # Simple perturbation: swap colors of two random edges
                if len(edges_with_max_c) > 0:
                    # Pick a random edge from max_c
                    idx1 = random.choice(edges_with_max_c)
                    # Pick a random edge from any other color
                    other_edges = [i for i, c in enumerate(current_coloring) if c != max_c]
                    if other_edges:
                        idx2 = random.choice(other_edges)
                        # Swap
                        c1 = current_coloring[idx1]
                        c2 = current_coloring[idx2]
                        current_coloring[idx1] = c2
                        current_coloring[idx2] = c1
                        # Check validity after swap
                        if not is_valid_coloring(current_coloring, conflicts):
                            # Revert
                            current_coloring[idx1] = c1
                            current_coloring[idx2] = c2
                            # If invalid, try another perturbation or break
                            # For now, just revert and continue loop, hoping for a different random choice
                            continue
                    else:
                        break # Only one color used, can't reduce further
                else:
                    break
                    
        return current_coloring

    results = []
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Fallback to simple greedy if out of time
            edges = instance(*spec)
            adj = build_adjacency(edges, spec[0])
            conflicts, _ = precompute_conflicts(edges, adj)
            coloring = greedy_color(edges, adj, conflicts, None)
            results.append(coloring)
            continue
            
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = build_adjacency(edges, n)
        conflicts, edge_to_idx = precompute_conflicts(edges, adj)
        
        # Initial greedy coloring
        coloring = greedy_color(edges, adj, conflicts, edge_to_idx)
        
        # Optimize
        best_coloring = coloring[:]
        best_score = max(best_coloring) if best_coloring else 0
        
        # Run multiple restarts of the reduction process with perturbations
        # Or just one long run with internal perturbations
        # The try_reduce_colors function already has internal perturbations.
        # Let's run it for a fixed number of iterations or until timeout.
        
        # To improve quality, we can try multiple independent runs of the local search
        # and keep the best.
        
        runs = 10
        for _ in range(runs):
            if time.time() > deadline - 5: # Leave 5s for other graphs
                break
                
            # Start from the best known so far, or a fresh greedy?
            # Starting from best known is usually better.
            # But sometimes getting stuck in a local minimum requires a fresh start.
            # Let's alternate.
            if _ % 2 == 0:
                current = best_coloring[:]
            else:
                current = greedy_color(edges, adj, conflicts, edge_to_idx)
            
            optimized = try_reduce_colors(current, conflicts, len(edges), max_iterations=500)
            
            score = max(optimized) if optimized else 0
            if score < best_score:
                best_score = score
                best_coloring = optimized[:]
                
        results.append(best_coloring)
        
    return results

BET: {mechanism: "Local search with color reduction and perturbation escapes greedy local minima, significantly lowering the number of color classes compared to the seed's pure greedy approach.", q: 0.9, delta: [0.1, 0.3]}
