# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def get_instance(n, d, seed):
        """
        Generates a bounded degree graph.
        Strategy: Shuffle all possible edges and greedily add them if degree constraints are met.
        This is a common deterministic interpretation of 'bounded_degree_graph' with a seed.
        """
        rng = random.Random(seed)
        # Generate all possible edges
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                all_edges.append((i, j))
        
        rng.shuffle(all_edges)
        
        degree = [0] * n
        edges = []
        for u, v in all_edges:
            if degree[u] < d and degree[v] < d:
                edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                if len(edges) >= (n * d) // 2: # Max possible edges
                    break
        
        return sorted(edges), degree

    def build_conflicts(edges, adj):
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)

        for i, (u1, v1) in enumerate(edges):
            # Conflicts: share vertex
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            
            # Conflicts: distance 2 (induced matching constraint)
            # Edge (u1, v1) conflicts with (u2, v2) if dist((u1,v1), (u2,v2)) == 2
            # i.e., they don't share a vertex, but there is an edge connecting a vertex of one to a vertex of the other.
            # Actually, "induced matching" means no two edges in the same color class are connected by an edge in the graph.
            # So if (u1, v1) and (u2, v2) are in the same class, we cannot have (u1, u2), (u1, v2), (v1, u2), or (v1, v2) in the graph.
            # This is equivalent to saying that the distance between the edges in the line graph is > 1.
            # In the conflict graph for strong coloring, two edges conflict if they are at distance 1 or 2 in the line graph.
            # Distance 1 in line graph = share a vertex in original.
            # Distance 2 in line graph = connected by a path of length 2 in original (i.e., u1-u2-v2 or similar).
            
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    # Check if e2 is distinct and not sharing vertex with e1 (already handled by dist 1, but dist 2 implies no shared vertex)
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)
        return conflicts

    def dsatur_coloring(edges, conflicts, deadline):
        n_edges = len(edges)
        color = [-1] * n_edges
        neighbor_colors = [set() for _ in range(n_edges)]
        uncolored = set(range(n_edges))
        
        while uncolored:
            if time.time() > deadline:
                # Fallback: assign max possible color
                for i in uncolored:
                    color[i] = 0 # Placeholder, will be fixed by greedy fallback
                break
            
            # Find most saturated
            best_idx = -1
            best_sat = -1
            best_deg = -1
            for i in uncolored:
                sat = len(neighbor_colors[i])
                deg = len(conflicts[i])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_idx = i
            
            if best_idx == -1:
                break
                
            c = 0
            while c in neighbor_colors[best_idx]:
                c += 1
            color[best_idx] = c
            uncolored.remove(best_idx)
            
            for j in conflicts[best_idx]:
                if color[j] == -1:
                    neighbor_colors[j].add(c)
        
        # If timeout, fill remaining with a safe greedy
        if -1 in color:
            for i in range(n_edges):
                if color[i] == -1:
                    taken = set()
                    for j in conflicts[i]:
                        if color[j] != -1:
                            taken.add(color[j])
                    c = 0
                    while c in taken:
                        c += 1
                    color[i] = c
        
        return color

    def optimize_instance(n, d, seed, time_budget):
        edges, degree = get_instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflicts(edges, adj)
        
        # Time budget for this instance
        inst_deadline = time.time() + time_budget
        
        # DSATUR
        color = dsatur_coloring(edges, conflicts, inst_deadline)
        
        # Simple Local Search: Try to move each edge to a lower color
        # Only if we have time
        if time.time() < inst_deadline - 1:
            n_edges = len(edges)
            conflict_lists = [list(conflicts[i]) for i in range(n_edges)]
            color_count = max(color) + 1 if color else 0
            
            # Sort edges by color descending to try to eliminate high colors
            order = sorted(range(n_edges), key=lambda x: color[x], reverse=True)
            
            improved = True
            while improved and time.time() < inst_deadline - 1:
                improved = False
                for e_idx in order:
                    if time.time() > inst_deadline - 1:
                        break
                    c_current = color[e_idx]
                    if c_current == 0:
                        continue
                    for new_c in range(c_current):
                        can_move = True
                        for c_idx in conflict_lists[e_idx]:
                            if color[c_idx] == new_c:
                                can_move = False
                                break
                        if can_move:
                            color[e_idx] = new_c
                            improved = True
                            break
            
            # Recalculate min color count? Not strictly necessary for validity, but good for score.
            # The score is total colors. We just return the list.
        
        return color

    results = []
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time
    # Total time available ~145s
    # Let's give 45s, 45s, 50s roughly
    time_budgets = [40, 40, 50]
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges, _ = get_instance(n, d, seed)
            # Very fast greedy
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            conflicts = build_conflicts(edges, adj)
            color = [-1] * len(edges)
            for i, (u, v) in enumerate(edges):
                taken = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        taken.add(color[j])
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
            results.append(color)
            continue
        
        coloring = optimize_instance(n, d, seed, time_budgets[idx])
        results.append(coloring)
        
        # Check time
        if time.time() > deadline - 2:
            # If we are running out, fill remaining with fast greedy
            for j in range(idx + 1, len(instances)):
                n, d, seed = instances[j]
                edges, _ = get_instance(n, d, seed)
                adj = defaultdict(set)
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                conflicts = build_conflicts(edges, adj)
                color = [-1] * len(edges)
                for i, (u, v) in enumerate(edges):
                    taken = set()
                    for k in conflicts[i]:
                        if color[k] != -1:
                            taken.add(color[k])
                    c = 0
                    while c in taken:
                        c += 1
                    color[i] = c
                results.append(color)
            break

    return results
# EVOLVE-BLOCK-END
