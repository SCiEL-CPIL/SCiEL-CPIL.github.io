import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_vertex_cover(n, edges, seed, time_budget):
        """
        Solve minimum vertex cover for a graph with n vertices and given edges.
        A vertex cover is a set of vertices that includes at least one endpoint of every edge.
        """
        if not edges:
            return []
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Map edges to indices for tracking
        edge_list = list(edges)
        m = len(edge_list)
        
        # For each vertex, which edges it covers
        vertex_edges = defaultdict(set)
        for i, (u, v) in enumerate(edge_list):
            vertex_edges[u].add(i)
            vertex_edges[v].add(i)
        
        best_set = None
        best_size = float('inf')
        
        rng_solver = random.Random(seed + 999)
        
        def is_valid_cover(S):
            """Check if S is a valid vertex cover."""
            covered = set()
            for v in S:
                covered.update(vertex_edges[v])
            return len(covered) == m
        
        def greedy_cover(order=None):
            """Greedy vertex cover: pick vertex covering most uncovered edges."""
            if order is None:
                order = list(range(n))
                rng_solver.shuffle(order)
            
            S = set()
            covered_edges = set()
            
            for v in order:
                if len(covered_edges) == m:
                    break
                # Check if v covers any uncovered edge
                new_edges = vertex_edges[v] - covered_edges
                if new_edges:
                    S.add(v)
                    covered_edges.update(new_edges)
            
            return S
        
        def local_search(S, max_iterations=100):
            """
            Local search: try removals and swaps to minimize |S|.
            """
            S = set(S)
            covered_edges = set()
            for v in S:
                covered_edges.update(vertex_edges[v])
            
            improved = True
            iteration = 0
            while improved and iteration < max_iterations and time.time() < deadline:
                improved = False
                iteration += 1
                
                # Pass 1: Try to remove each vertex
                vs = list(S)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        return S
                    # Check if removing v still covers all edges
                    # v covers edges in vertex_edges[v]
                    # After removal, those edges must be covered by other vertices in S
                    can_remove = True
                    for e in vertex_edges[v]:
                        u1, u2 = edge_list[e]
                        # Check if u1 or u2 is in S (excluding v)
                        if u1 == v:
                            if u2 not in S:
                                can_remove = False
                                break
                        elif u2 == v:
                            if u1 not in S:
                                can_remove = False
                                break
                        else:
                            # Both u1 and u2 are different from v
                            # Check if at least one is in S
                            if u1 not in S and u2 not in S:
                                can_remove = False
                                break
                    if can_remove:
                        S.remove(v)
                        covered_edges -= vertex_edges[v]
                        # Recalculate covered_edges
                        covered_edges = set()
                        for w in S:
                            covered_edges.update(vertex_edges[w])
                        improved = True
                
                if not improved and time.time() > deadline:
                    return S
                
                # Pass 2: Try swaps - replace v in S with u not in S
                # This can help escape local optima
                if improved:
                    continue
                    
                vs = list(S)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        return S
                    # Try replacing v with each u not in S
                    candidates = [u for u in range(n) if u not in S]
                    rng_solver.shuffle(candidates)
                    for u in candidates:
                        if time.time() > deadline:
                            return S
                        # Try S' = S - {v} + {u}
                        S_trial = set(S)
                        S_trial.remove(v)
                        S_trial.add(u)
                        
                        # Check if S_trial is a valid cover
                        valid = True
                        for e in range(m):
                            u1, u2 = edge_list[e]
                            if u1 not in S_trial and u2 not in S_trial:
                                valid = False
                                break
                        
                        if valid:
                            # Check if we can remove more vertices from S_trial
                            S_improved = set(S_trial)
                            # Try to remove redundant vertices
                            removed = True
                            while removed and time.time() < deadline:
                                removed = False
                                vs2 = list(S_improved)
                                rng_solver.shuffle(vs2)
                                for w in vs2:
                                    if time.time() > deadline:
                                        break
                                    can_remove = True
                                    for e in vertex_edges[w]:
                                        u1, u2 = edge_list[e]
                                        if u1 == w:
                                            if u2 not in S_improved:
                                                can_remove = False
                                                break
                                        elif u2 == w:
                                            if u1 not in S_improved:
                                                can_remove = False
                                                break
                                        else:
                                            if u1 not in S_improved and u2 not in S_improved:
                                                can_remove = False
                                                break
                                    if can_remove:
                                        S_improved.remove(w)
                                        removed = True
                            
                            if len(S_improved) < len(S):
                                S = S_improved
                                improved = True
                                break
                    if improved:
                        break
            
            return S
        
        # Multiple strategies
        strategies = 0
        
        # Strategy 1: Greedy with different orderings
        for _ in range(50):
            if time.time() > deadline:
                break
            S = greedy_cover()
            S = local_search(S)
            if len(S) < best_size:
                best_size = len(S)
                best_set = set(S)
            strategies += 1
        
        # Strategy 2: Start from a maximal matching, then extend
        # A maximal matching gives a 2-approximation for vertex cover
        for _ in range(30):
            if time.time() > deadline:
                break
            # Find a maximal matching using greedy
            matched = set()
            matching = []
            vertices = list(range(n))
            rng_solver.shuffle(vertices)
            for v in vertices:
                if v in matched:
                    continue
                for u in adj[v]:
                    if u not in matched:
                        matching.append((v, u))
                        matched.add(v)
                        matched.add(u)
                        break
            
            # Vertex cover from matching: take one endpoint from each edge
            # Try both choices
            for choice in range(2):
                S = set()
                for i, (u, v) in enumerate(matching):
                    if choice == 0:
                        S.add(u)
                    else:
                        S.add(v)
                
                # Extend to cover all edges
                covered = set()
                for w in S:
                    covered.update(vertex_edges[w])
                
                # Add vertices to cover remaining edges
                uncovered = [e for e in range(m) if e not in covered]
                for e in uncovered:
                    u1, u2 = edge_list[e]
                    # Add the vertex that covers more uncovered edges
                    if len(vertex_edges[u1] - covered) >= len(vertex_edges[u2] - covered):
                        S.add(u1)
                        covered.update(vertex_edges[u1])
                    else:
                        S.add(u2)
                        covered.update(vertex_edges[u2])
                
                S = local_search(S)
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = set(S)
                strategies += 1
        
        # Strategy 3: Random initialization + local search
        for _ in range(30):
            if time.time() > deadline:
                break
            # Random subset
            S = set(rng_solver.sample(range(n), min(n, rng_solver.randint(n // 4, n // 2))))
            
            # Extend to valid cover
            covered = set()
            for w in S:
                covered.update(vertex_edges[w])
            
            uncovered = [e for e in range(m) if e not in covered]
            for e in uncovered:
                u1, u2 = edge_list[e]
                if len(vertex_edges[u1] - covered) >= len(vertex_edges[u2] - covered):
                    S.add(u1)
                    covered.update(vertex_edges[u1])
                else:
                    S.add(u2)
                    covered.update(vertex_edges[u2])
            
            S = local_search(S)
            if len(S) < best_size:
                best_size = len(S)
                best_set = set(S)
            strategies += 1
        
        # Strategy 4: Branch and bound for small instances or polishing
        # Use a simple branch and bound with lower bound from greedy matching
        if time.time() < deadline - 5:
            def branch_and_bound(S_partial, edges_to_cover, depth=0):
                """
                Branch and bound for minimum vertex cover.
                S_partial: current set of selected vertices
                edges_to_cover: set of edge indices that still need to be covered
                """
                nonlocal best_size, best_set
                
                if time.time() > deadline:
                    return
                
                if not edges_to_cover:
                    if len(S_partial) < best_size:
                        best_size = len(S_partial)
                        best_set = set(S_partial)
                    return
                
                # Lower bound: greedy matching on remaining edges
                # Find a maximal matching in the subgraph induced by edges_to_cover
                remaining_edges = [edge_list[e] for e in edges_to_cover]
                if not remaining_edges:
                    if len(S_partial) < best_size:
                        best_size = len(S_partial)
                        best_set = set(S_partial)
                    return
                
                # Simple lower bound: number of edges / 2 (since each vertex covers at most... actually this is weak)
                # Better: find a maximal matching, its size is a lower bound
                matched_verts = set()
                matching_size = 0
                for u, v in remaining_edges:
                    if u not in matched_verts and v not in matched_verts:
                        matching_size += 1
                        matched_verts.add(u)
                        matched_verts.add(v)
                
                lb = len(S_partial) + matching_size
                if lb >= best_size:
                    return
                
                # Branch on an edge
                u, v = remaining_edges[0]
                
                # Option 1: add u
                S_new = set(S_partial)
                S_new.add(u)
                edges_new = set(edges_to_cover)
                for e in vertex_edges[u]:
                    edges_new.discard(e)
                branch_and_bound(S_new, edges_new, depth + 1)
                
                # Option 2: add v
                S_new = set(S_partial)
                S_new.add(v)
                edges_new = set(edges_to_cover)
                for e in vertex_edges[v]:
                    edges_new.discard(e)
                branch_and_bound(S_new, edges_new, depth + 1)
            
            # Only do B&B if we have a good upper bound and the instance is small enough
            if best_size <= 20:
                all_edges = set(range(m))
                branch_and_bound(set(), all_edges)
        
        return sorted(best_set) if best_set else []

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        
        if not edges:
            return []
        
        # Calculate time budget for this instance
        # We have 3 instances, so roughly 45 seconds each
        time_budget = 45
        
        result = solve_vertex_cover(n, edges, seed, time_budget)
        
        return result

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices involved in edges
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
