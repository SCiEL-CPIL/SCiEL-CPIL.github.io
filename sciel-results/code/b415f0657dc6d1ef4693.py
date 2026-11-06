# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_greedy(adj, edges_set):
        """Greedily find edge-disjoint cycles, longest first."""
        cycles = []
        used_edges = set()
        remaining = set(edges_set)
        n = max(adj.keys()) + 1 if adj else 0
        
        # Build adjacency from remaining edges for quick lookup
        rem_adj = {i: [] for i in range(n)}
        for (u, v) in remaining:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
        
        # Try to find cycles starting from each vertex, preferring longer ones
        # Use DFS to find a cycle
        def find_cycle_from(start, used_e, adj_dict, n):
            # DFS to find a cycle back to start
            stack = [(start, [start], set())]
            best_cycle = None
            while stack:
                node, path, visited = stack.pop()
                if len(path) >= 3:
                    # Check if node is in path and we can close
                    if node in path:
                        idx = path.index(node)
                        cycle = path[idx:]
                        if len(cycle) >= 3:
                            # Check all edges in cycle are available
                            valid = True
                            for i in range(len(cycle)):
                                a, b = cycle[i], cycle[(i+1) % len(cycle)]
                                e = tuple(sorted((a, b)))
                                if e not in used_e:
                                    valid = False
                                    break
                            if valid:
                                if best_cycle is None or len(cycle) > len(best_cycle):
                                    best_cycle = cycle
                if len(path) > 15:  # limit depth
                    continue
                for nb in adj_dict.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb], visited))
                    elif nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = path + [start]
                        # Remove duplicate start at end for cycle representation
                        cycle = cycle[:-1]
                        valid = True
                        for i in range(len(cycle)):
                            a, b = cycle[i], cycle[(i+1) % len(cycle)]
                            e = tuple(sorted((a, b)))
                            if e not in used_e:
                                valid = False
                                break
                        if valid:
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
            return best_cycle
        
        # Repeatedly find a cycle, prefer longer ones
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            best_c = None
            best_len = 0
            # Try from each vertex
            for v in range(n):
                if not rem_adj[v]:
                    continue
                c = find_cycle_from(v, used_edges, rem_adj, n)
                if c and len(c) > best_len:
                    best_len = len(c)
                    best_c = c
            if best_c:
                # Mark edges as used
                for i in range(len(best_c)):
                    a, b = best_c[i], best_c[(i+1) % len(best_c)]
                    e = tuple(sorted((a, b)))
                    if e in remaining:
                        remaining.discard(e)
                        used_edges.add(e)
                        rem_adj[a].remove(b)
                        rem_adj[b].remove(a)
                cycles.append(best_c)
                changed = True
        
        return cycles, remaining

    def local_improve(cycles, singles, adj, n, max_iters=50):
        """Try to improve by removing a cycle and repacking."""
        best_score = len(cycles) + len(singles)
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try removing each cycle and see if we can do better
            for ci, cyc in enumerate(cycles):
                if time.time() > DEADLINE:
                    break
                # Remove this cycle
                saved_cycle = cyc
                new_cycles = cycles[:ci] + cycles[ci+1:]
                # Add cycle edges back to singles
                new_singles = list(singles)
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i+1) % len(cyc)]
                    new_singles.append(tuple(sorted((a, b))))
                
                # Try to repack: find cycles in the new_singles
                edge_set = set(new_singles)
                adj_local = {i: [] for i in range(n)}
                for (u, v) in edge_set:
                    adj_local[u].append(v)
                    adj_local[v].append(u)
                
                sub_cycles, sub_remaining = find_cycles_greedy(adj_local, edge_set)
                new_total = len(sub_cycles) + len(sub_remaining)
                old_total = len(new_cycles) + len(new_singles)  # before repack
                
                if new_total < old_total:
                    cycles = new_cycles + sub_cycles
                    singles = list(sub_remaining)
                    improved = True
                    break
            if not improved:
                break
        return cycles, singles

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE:
            # Fallback: all edges as singles
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        edge_set = set(edges)
        
        # Build adjacency
        adj = {i: [] for i in range(n)}
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Greedy cycle packing
        cycles, remaining_edges = find_cycles_greedy(adj, edge_set)
        
        # Convert remaining to singles
        singles = [list(e) for e in remaining_edges]
        
        # Local improvement
        cycles, singles = local_improve(cycles, singles, adj, n)
        
        # Build result: cycles as vertex lists, singles as [u,v]
        pieces = []
        for cyc in cycles:
            pieces.append(list(cyc))
        for s in singles:
            pieces.append(list(s))
        
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
