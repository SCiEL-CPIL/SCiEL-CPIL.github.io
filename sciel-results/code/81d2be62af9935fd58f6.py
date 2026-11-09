<<<<<<< SEARCH
import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def solve_vertex_cover(edges, n, time_limit):
        if not edges:
            return []
        
        # Try ILP formulation of Minimum Vertex Cover
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            
            m = len(edges)
            # Constraint matrix A: m x n
            # A[i, u] = 1 if u is in edge i
            # A[i, v] = 1 if v is in edge i
            A = np.zeros((m, n), dtype=np.int8)
            for i, (u, v) in enumerate(edges):
                A[i, u] = 1
                A[i, v] = 1
                
            c = np.ones(n, dtype=np.float64)
            integrality = np.ones(n, dtype=np.int8)
            bounds = Bounds(0, 1)
            
            # x_u + x_v >= 1  =>  -x_u - x_v <= -1
            # LinearConstraint(A, 1, np.inf) is sum(A_i) >= 1
            # This is exactly what we want.
            constraints = LinearConstraint(A, 1, np.inf)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, 
                       bounds=bounds, 
                       options={"time_limit": time_limit, "presolve": True})
            
            if res.status in [0, 1, 2, 3]: # 0: success, 1: optimal, 2: best, 3: iter
                if res.x is not None:
                    sol = [i for i in range(n) if res.x[i] > 0.5]
                    # Verify
                    if is_valid_vertex_cover(sol, edges):
                        return sol
            # If result exists but status is weird, try to salvage
            if res.x is not None:
                sol = [i for i in range(n) if res.x[i] > 0.5]
                if is_valid_vertex_cover(sol, edges):
                    return sol
        except Exception:
            pass
            
        # Fallback: Greedy Vertex Cover + Local Search (Minimize)
        return greedy_vertex_cover(edges, n)

    def is_valid_vertex_cover(sol_set, edges):
        sol_set = set(sol_set)
        for u, v in edges:
            if u not in sol_set and v not in sol_set:
                return False
        return True

    def greedy_vertex_cover(edges, n):
        # Greedy: Pick vertex with max degree
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cover = []
        # We need to track remaining edges or just simulate removal
        # Simulate: keep track of vertices still "active" (not in cover) and edges between them
        # Actually, standard greedy:
        # 1. While edges exist:
        #    a. Find vertex v with max degree in current graph
        #    b. Add v to cover
        #    c. Remove v and all incident edges
        
        # Efficient simulation
        remaining_edges = set(edges)
        # To get degree quickly, we might need to update adj sets
        
        # Simpler approach for small n:
        # Copy adj
        curr_adj = [set(s) for s in adj]
        
        while remaining_edges:
            # Find vertex with max degree in remaining_edges
            max_deg = -1
            best_v = -1
            # We can just iterate over all vertices in remaining_edges
            # Or maintain degrees
            # For n=40, iterating 40 times is fast
            for v in range(n):
                # Degree is len(curr_adj[v])
                # But curr_adj must be updated when we remove vertices
                if len(curr_adj[v]) > max_deg:
                    max_deg = len(curr_adj[v])
                    best_v = v
                    
            if best_v == -1:
                break
                
            cover.append(best_v)
            
            # Remove best_v from curr_adj of all neighbors
            neighbors = list(curr_adj[best_v])
            for u in neighbors:
                curr_adj[u].remove(best_v)
            curr_adj[best_v].clear()
            
            # Update remaining_edges
            # Remove any edge incident to best_v
            # This is O(E) per step. Total O(V*E) = 40 * 1000 = 40k ops. Very fast.
            remaining_edges = {(u, v) for u, v in remaining_edges if u != best_v and v != best_v}

        # Local Search: Try to remove vertices
        cover_set = set(cover)
        changed = True
        while changed:
            changed = False
            for v in list(cover_set):
                # If we remove v, are all its neighbors in cover_set?
                # If yes, v is redundant.
                if all(u in cover_set for u in adj[v]):
                    cover_set.remove(v)
                    changed = True
        
        return sorted(cover_set)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        edges, adj = random_graph(n, p, seed)
        
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 5:
            # Not enough time, just use greedy
            sol = greedy_vertex_cover(edges, n)
        else:
            # Allocate time to ILP
            # Leave 5s buffer for other instances/overhead
            ilp_time = max(1, (time_left - 5) / 3) 
            sol = solve_vertex_cover(edges, n, ilp_time)
        
        results.append(sol)

    return results
=======
import time
import random
import numpy as np
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch_cliques(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        # Use bitmask representation for efficiency
        adj_mask = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_mask[u] = mask
        
        def bk(R, P, X):
            if P == 0 and X == 0:
                # R is a maximal clique
                if len(R) >= 2:
                    cliques.append(R)
                return
            
            # Choose pivot u from P | X
            PX = P | X
            if PX == 0:
                return
            
            # Pick pivot with max degree in P
            best_u = -1
            best_deg = -1
            temp = PX
            while temp:
                u = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = bin(P & adj_mask[u]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
            
            # Branch on P \\ N(u)
            candidates = P & ~adj_mask[best_u]
            while candidates:
                v = (candidates & -candidates).bit_length() - 1
                candidates &= candidates - 1
                new_R = R | (1 << v)
                new_P = P & adj_mask[v]
                new_X = X & adj_mask[v]
                bk(new_R, new_P, new_X)
                P &= ~(1 << v)
                X |= (1 << v)
        
        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_hitting_set(cliques, n, time_limit):
        """Branch and bound for minimum hitting set."""
        if not cliques:
            return []
        
        # Convert cliques to frozensets for fast lookup
        cliques_fs = [frozenset(c) for c in cliques]
        
        # Count frequency of each vertex
        freq = [0] * n
        for c in cliques_fs:
            for v in c:
                freq[v] += 1
        
        # Sort vertices by frequency (descending)
        sorted_vertices = sorted(range(n), key=lambda v: -freq[v])
        
        # Filter out vertices that appear in no cliques
        active_vertices = [v for v in sorted_vertices if freq[v] > 0]
        
        if not active_vertices:
            return []
        
        # Build mapping: for each clique, which active vertices it contains
        # Represent cliques as sorted tuples of active vertex indices
        cliques_active = []
        for c in cliques_fs:
            active_in_c = tuple(v for v in sorted(c) if freq[v] > 0)
            if active_in_c:
                cliques_active.append(active_in_c)
        
        # Remove duplicate cliques
        cliques_active = list(set(cliques_active))
        cliques_active.sort(key=lambda c: len(c))
        
        # If any clique is empty, no solution exists (shouldn't happen)
        for c in cliques_active:
            if not c:
                return list(active_vertices)  # Fallback
        
        # Lower bound: max clique size (each clique needs at least one vertex)
        # Better lower bound: for each vertex, count how many cliques it covers
        # Use simple LB: 1 if any clique exists
        
        best_solution = None
        best_size = len(active_vertices)  # Upper bound: take all active vertices
        
        # Initial greedy solution
        def greedy_solution():
            remaining = set(range(len(cliques_active)))
            selected = []
            while remaining:
                # Pick vertex covering most remaining cliques
                best_v = -1
                best_count = -1
                for v in active_vertices:
                    count = 0
                    for i in remaining:
                        if v in cliques_active[i]:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_v == -1 or best_count == 0:
                    break
                selected.append(best_v)
                remaining = {i for i in remaining if best_v not in cliques_active[i]}
            return selected
        
        greedy_sol = greedy_solution()
        if greedy_sol:
            # Verify greedy solution
            covered = set()
            for i, c in enumerate(cliques_active):
                if any(v in c for v in greedy_sol):
                    covered.add(i)
            if len(covered) == len(cliques_active):
                best_solution = greedy_sol
                best_size = len(greedy_sol)
        
        # Branch and bound
        nodes_explored = 0
        start_bnb = time.time()
        
        def bnb(selected, remaining_cliques, current_size):
            nonlocal best_solution, best_size, nodes_explored
            
            # Time check
            if time.time() - start_bnb > time_limit:
                return
            
            nodes_explored += 1
            
            if not remaining_cliques:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(selected)
                return
            
            # Lower bound: current_size + ceil(remaining_cliques / max_coverage)
            # Simple LB: current_size + 1 if any remaining
            if current_size + 1 >= best_size:
                return
            
            # Pick the most constrained remaining clique (smallest)
            # Find clique with fewest candidate vertices
            min_clique_idx = -1
            min_clique_size = float('inf')
            for i in remaining_cliques:
                c = cliques_active[i]
                if len(c) < min_clique_size:
                    min_clique_size = len(c)
                    min_clique_idx = i
            
            if min_clique_idx == -1:
                return
            
            target_clique = cliques_active[min_clique_idx]
            
            # Branch on each vertex in the target clique
            for v in sorted(target_clique, key=lambda x: -freq[x]):
                new_selected = selected + [v]
                new_remaining = set()
                for i in remaining_cliques:
                    if v not in cliques_active[i]:
                        new_remaining.add(i)
                bnb(new_selected, new_remaining, current_size + 1)
        
        # Start B&B with empty selection
        all_cliques = set(range(len(cliques_active)))
        bnb([], all_cliques, 0)
        
        if best_solution is None:
            # Fallback to greedy
            best_solution = greedy_sol if greedy_sol else list(active_vertices)
        
        # Local search: try to remove redundant vertices
        sol_set = set(best_solution)
        changed = True
        while changed:
            changed = False
            for v in list(sol_set):
                # Check if removing v still covers all cliques
                can_remove = True
                for c in cliques_active:
                    if v in c and not any(u in c for u in sol_set if u != v):
                        can_remove = False
                        break
                if can_remove:
                    sol_set.remove(v)
                    changed = True
        
        # Local search: try swaps (replace one vertex with another)
        improved = True
        while improved:
            improved = False
            sol_list = list(sol_set)
            for i, v in enumerate(sol_list):
                # Try replacing v with each other active vertex
                for w in active_vertices:
                    if w in sol_set:
                        continue
                    new_set = set(sol_set)
                    new_set.remove(v)
                    new_set.add(w)
                    # Check if new_set covers all cliques
                    valid = True
                    for c in cliques_active:
                        if not any(u in c for u in new_set):
                            valid = False
                            break
                    if valid and len(new_set) < len(sol_set):
                        sol_set = new_set
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(sol_set)

    def greedy_hitting_set(cliques, n):
        """Fast greedy for hitting set."""
        if not cliques:
            return []
        
        cliques_fs = [frozenset(c) for c in cliques]
        remaining = set(range(len(cliques_fs)))
        selected = []
        
        while remaining:
            best_v = -1
            best_count = -1
            for v in range(n):
                count = sum(1 for i in remaining if v in cliques_fs[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                # No vertex covers any remaining clique, add all vertices from first remaining
                c = cliques_fs[next(iter(remaining))]
                for v in c:
                    if v not in selected:
                        selected.append(v)
                break
            selected.append(best_v)
            remaining = {i for i in remaining if best_v not in cliques_fs[i]}
        
        # Remove redundant
        sol_set = set(selected)
        changed = True
        while changed:
            changed = False
            for v in list(sol_set):
                can_remove = True
                for c in cliques_fs:
                    if v in c and not any(u in c for u in sol_set if u != v):
                        can_remove = False
                        break
                if can_remove:
                    sol_set.remove(v)
                    changed = True
        
        return sorted(sol_set)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        edges, adj = random_graph(n, p, seed)
        
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        
        if time_left < 5:
            # Not enough time, just use greedy
            cliques = bron_kerbosch_cliques(adj, n)
            sol = greedy_hitting_set(cliques, n)
        else:
            # Enumerate cliques
            cliques = bron_kerbosch_cliques(adj, n)
            
            # Allocate remaining time among remaining instances
            remaining_instances = len(instances) - idx
            bnb_time = max(1, (time_left - 2) / remaining_instances)
            
            sol = solve_hitting_set(cliques, n, bnb_time)
        
        results.append(sol)

    return results
>>>>>>> REPLACE
