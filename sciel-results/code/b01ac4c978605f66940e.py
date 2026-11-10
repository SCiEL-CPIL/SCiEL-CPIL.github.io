"""Rewritten program: decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        if not edge_set:
            return set()
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        q = deque([v for v in deg if deg[v] < 2])
        in_core = {v: True for v in deg}
        
        while q:
            v = q.popleft()
            if not in_core[v]:
                continue
            in_core[v] = False
            for nb in adj[v]:
                if in_core[nb]:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        q.append(nb)
                        
        core_edges = set()
        for u, v in edge_set:
            if in_core[u] and in_core[v]:
                core_edges.add((u, v))
        return core_edges

    def find_all_simple_cycles(edge_set, max_len=12):
        """
        Find all simple cycles up to max_len in the graph defined by edge_set.
        Returns a list of cycles, where each cycle is a list of vertices.
        """
        if not edge_set:
            return []
        
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        vertices = sorted(adj.keys())
        cycles = []
        seen_cycles = set()
        
        # To avoid duplicates, we only start DFS from the smallest vertex in the cycle
        for start in vertices:
            # DFS from start, only visiting vertices >= start
            stack = [(start, [start])]
            visited = {start}
            
            while stack:
                u, path = stack.pop()
                if len(path) > max_len:
                    continue
                
                for nb in adj[u]:
                    if nb < start:
                        continue
                    if nb == start:
                        if len(path) >= 3:
                            # Found a cycle
                            # Normalize cycle to avoid duplicates (rotate so min vertex is first)
                            cycle = path[:]
                            min_idx = cycle.index(min(cycle))
                            norm_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                            if norm_cycle not in seen_cycles:
                                seen_cycles.add(norm_cycle)
                                cycles.append(list(norm_cycle))
                    elif nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
                        
        return cycles

    def score_decomposition(cycles):
        """
        Score is sum(len(cycle) - 1). Higher is better.
        """
        return sum(len(c) - 1 for c in cycles)

    def solve_max_weight_independent_set(cycles, edge_set, time_limit):
        """
        Solve Maximum Weight Independent Set on the conflict graph of cycles.
        Nodes: cycles. Edges: cycles that share an edge.
        Weight: len(cycle) - 1.
        Uses Branch and Bound.
        """
        if not cycles:
            return []
        
        # Pre-compute edge sets for each cycle
        cycle_edges = []
        for c in cycles:
            edges = set()
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                edges.add(tuple(sorted((u, v))))
            cycle_edges.append(edges)
            
        # Build conflict graph: conflict[i][j] is True if cycle i and j share an edge
        n_cycles = len(cycles)
        conflict = [[False] * n_cycles for _ in range(n_cycles)]
        for i in range(n_cycles):
            for j in range(i + 1, n_cycles):
                if cycle_edges[i] & cycle_edges[j]:
                    conflict[i][j] = True
                    conflict[j][i] = True
                    
        # Weights
        weights = [len(c) - 1 for c in cycles]
        
        # Sort cycles by weight descending to help B&B
        indices = sorted(range(n_cycles), key=lambda i: weights[i], reverse=True)
        sorted_cycles = [cycles[i] for i in indices]
        sorted_weights = [weights[i] for i in indices]
        # Remap conflict graph
        # new_conflict[i][j] corresponds to old indices indices[i], indices[j]
        new_conflict = [[False] * n_cycles for _ in range(n_cycles)]
        for i in range(n_cycles):
            for j in range(n_cycles):
                if i != j:
                    new_conflict[i][j] = conflict[indices[i]][indices[j]]
                    
        best_score = 0
        best_set = []
        
        # B&B State: (current_set, available_indices, current_score)
        # available_indices is a list of indices in sorted_cycles that can still be added
        # We use a recursive function with memoization or just simple B&B
        
        def bound(available, current_score):
            # Optimistic bound: sum of weights of all available cycles
            return current_score + sum(sorted_weights[i] for i in available)
            
        def branch_and_bound(current_set, available, current_score, depth=0):
            nonlocal best_score, best_set
            
            if time.time() > time_limit:
                return
                
            if not available:
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                return
                
            # Prune
            if bound(available, current_score) <= best_score:
                return
                
            # Pick the first available cycle (heuristic: highest weight)
            idx = available[0]
            rest_available = available[1:]
            
            # Option 1: Include cycle idx
            new_set = current_set + [idx]
            new_score = current_score + sorted_weights[idx]
            
            # Remove conflicting cycles from available
            new_available = [j for j in rest_available if not new_conflict[idx][j]]
            
            branch_and_bound(new_set, new_available, new_score, depth + 1)
            
            # Option 2: Exclude cycle idx
            branch_and_bound(current_set, rest_available, current_score, depth + 1)
            
        # Initial available: all indices
        initial_available = list(range(n_cycles))
        branch_and_bound([], initial_available, 0)
        
        # Map back to original cycles
        return [sorted_cycles[i] for i in best_set]

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # 1. Get 2-core
        core_edges = get_2_core(edge_set)
        
        # 2. Find all simple cycles in 2-core up to length 12
        # If core is large, this might be slow, but n<=30 is small.
        cycles = find_all_simple_cycles(core_edges, max_len=12)
        
        # 3. Solve Max Weight Independent Set
        # Use remaining time for B&B
        ls_time = min(deadline - 5, time.time() + 30)
        selected_cycles = solve_max_weight_independent_set(cycles, core_edges, ls_time)
        
        # 4. If B&B didn't find a good solution or timed out, fallback to greedy
        # Actually, B&B with time limit should return best found.
        # But if cycles list is empty, selected_cycles is empty.
        
        # 5. Check if we can add more cycles from the remaining edges?
        # The B&B only considered cycles in the initial 2-core.
        # Removing cycles might reveal new cycles in the remaining edges?
        # No, cycles are subsets of edges. If an edge is removed, it can't be part of a new cycle 
        # unless the new cycle uses only remaining edges.
        # So we should run a second pass on the remaining edges.
        
        used_edges = set()
        for c in selected_cycles:
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                used_edges.add(tuple(sorted((u, v))))
                
        remaining_edges = edge_set - used_edges
        
        # Run greedy on remaining edges to pick up any small cycles missed
        # This is a safety net
        rem_cycles = []
        rem_edges = set(remaining_edges)
        while rem_edges:
            core = get_2_core(rem_edges)
            if not core:
                break
            # Find any cycle
            c = find_all_simple_cycles(core, max_len=5)
            if not c:
                break
            # Pick the longest one
            c = max(c, key=len)
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                rem_edges.discard(tuple(sorted((u, v))))
            rem_cycles.append(c)
            
        all_cycles = selected_cycles + rem_cycles
        
        # Construct pieces
        pieces = []
        for c in all_cycles:
            pieces.append(c[:])
            
        # Add remaining edges as single edge pieces
        # We need to know which edges are still unused
        used_edges_final = set()
        for c in all_cycles:
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                used_edges_final.add(tuple(sorted((u, v))))
                
        final_remaining = edge_set - used_edges_final
        for e in final_remaining:
            pieces.append(list(e))
            
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback: all edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
