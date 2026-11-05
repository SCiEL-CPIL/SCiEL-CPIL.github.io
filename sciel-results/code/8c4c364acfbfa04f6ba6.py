import time
import random
from collections import defaultdict, deque


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0

        edge_set = set(edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)

        return used_edges == edge_set

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def eulerian_decomposition(adj, edges):
        """
        Decompose an Eulerian graph (all degrees even) into a single cycle (or multiple cycles
        if not connected, but we handle connected components).
        Returns list of paths (cycles represented as paths where first==last is not required,
        but we return them as simple paths. Wait, a cycle is not a simple path in the strict sense
        if we require distinct vertices. But the problem says "simple paths".
        A cycle v0-v1-...-vk-v0 is not a simple path because v0 is repeated.
        However, we can break the cycle at any edge to make it a simple path.
        So an Eulerian cycle can be converted to 1 simple path by removing one edge.
        Actually, if the graph is Eulerian, we can decompose it into edge-disjoint cycles.
        Each cycle can be broken into a path.
        But we want to minimize the number of paths.
        If the graph is Eulerian and connected, it has an Eulerian circuit.
        We can traverse the Eulerian circuit and stop before returning to the start.
        This gives one simple path that covers all edges?
        No. An Eulerian circuit visits every edge exactly once.
        If we take the sequence of vertices v0, v1, ..., vk, v0, and remove the last v0,
        we get v0, v1, ..., vk. This is a simple path if all vertices in v0...vk are distinct.
        But in an Eulerian circuit, vertices can repeat.
        So we cannot just take one path.
        
        Actually, the standard result is:
        The minimum number of edge-disjoint paths covering all edges of a graph G is:
        max(1, (sum over components of max(1, odd_degrees/2)))
        Wait, for a connected graph:
        If all degrees are even, it has an Eulerian circuit. This circuit can be decomposed into
        one path? No, the circuit itself is a closed walk.
        We can decompose the Eulerian circuit into edge-disjoint paths.
        Actually, if we have an Eulerian circuit, we can just output it as one path?
        No, because a path must be simple (no repeated vertices).
        An Eulerian circuit generally has repeated vertices.
        
        However, there is a theorem:
        A connected graph can be decomposed into k edge-disjoint paths iff
        k >= max(1, odd/2) and ...
        
        Let's use a different approach for Eulerian graphs.
        We can find an Eulerian circuit. Then we can "break" the circuit at vertices where we
        don't want to repeat. But we want to minimize the number of paths.
        
        Actually, for an Eulerian graph, the minimum number of paths is 1?
        No. Consider a figure-8 graph (two cycles sharing a vertex).
        Degrees: shared vertex has degree 4, others degree 2. All even.
        Can we cover it with 1 simple path?
        No, because a simple path cannot visit the shared vertex twice.
        So we need at least 2 paths.
        In general, the minimum number of paths to cover an Eulerian graph is related to
        the number of "blocks" or biconnected components?
        
        Actually, a simpler bound:
        For any graph, the minimum number of edge-disjoint paths covering all edges is
        at least the number of odd-degree vertices / 2.
        For Eulerian graphs (0 odd vertices), the lower bound is 1?
        But the figure-8 example shows it's 2.
        
        Wait, the lower bound is max(1, odd/2).
        For figure-8, odd=0, so lower bound is 1. But it takes 2.
        So the lower bound is not tight for Eulerian graphs.
        
        However, we can use a greedy approach to decompose Eulerian graphs into few paths.
        Or we can use the fact that we can merge paths that share endpoints.
        
        Let's stick to the general strategy:
        1. Identify odd-degree vertices.
        2. Pair them up.
        3. Add virtual edges to make the graph Eulerian.
        4. Find Eulerian circuit.
        5. The virtual edges correspond to "breaks" in the path decomposition.
        6. Removing the virtual edges from the Eulerian circuit gives the path decomposition.
        
        This guarantees a decomposition with exactly (odd/2) paths.
        This is optimal for graphs with odd-degree vertices.
        For Eulerian graphs, this gives 1 path?
        If we add 0 virtual edges, we get 1 Eulerian circuit.
        But as seen before, an Eulerian circuit is not a simple path.
        
        So we need to handle Eulerian graphs differently.
        For Eulerian graphs, we can decompose into edge-disjoint cycles.
        Then break each cycle into a path.
        The number of paths will be the number of cycles.
        We want to minimize the number of cycles.
        This is the "Cycle Double Cover" problem? No, just cycle decomposition.
        The minimum number of cycles to cover edges is the "cycle rank" + something?
        
        Actually, we can just use the greedy path extension method which works well in practice.
        The Eulerian trick is good for ensuring we hit the lower bound when odd > 0.
        
        Let's implement the Eulerian trick for graphs with odd vertices.
        """
        # This function is a placeholder for the specific Eulerian decomposition logic
        # which will be integrated into solve_instance
        pass

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        adj = build_adjacency(edges)
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_vertices = [v for v, d in deg.items() if d % 2 == 1]
        odd_count = len(odd_vertices)
        
        # Lower bound: max(1, odd_count // 2)
        lower_bound = max(1, odd_count // 2)
        
        time_per_instance = 40.0
        inst_start = time.time()
        inst_deadline = inst_start + time_per_instance
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Eulerian augmentation for graphs with odd vertices
        if odd_count > 0:
            # We need to pair up odd vertices.
            # A good pairing minimizes the "cost" of the virtual edges, but here we just need
            # any pairing to get a valid decomposition of size odd_count/2.
            # Actually, ANY pairing works to make the graph Eulerian.
            # The number of paths will be exactly odd_count/2.
            # This is optimal.
            
            # Pair up odd vertices arbitrarily
            rng_pair = random.Random(seed + 12345)
            odd_copy = odd_vertices[:]
            rng_pair.shuffle(odd_copy)
            pairs = []
            for i in range(0, odd_count, 2):
                pairs.append((odd_copy[i], odd_copy[i+1]))
            
            # Create augmented graph
            aug_edges = list(edges)
            aug_adj = defaultdict(set)
            for u, v in aug_edges:
                aug_adj[u].add(v)
                aug_adj[v].add(u)
            
            for u, v in pairs:
                if u != v: # Self-loops not allowed in simple graphs, but virtual edges can be anything
                    aug_edges.append((u, v))
                    aug_adj[u].add(v)
                    aug_adj[v].add(u)
            
            # Find Eulerian circuit in augmented graph
            # Hierholzer's algorithm
            def find_eulerian_circuit(adj_dict, start):
                circuit = []
                stack = [start]
                while stack:
                    v = stack[-1]
                    if adj_dict[v]:
                        w = next(iter(adj_dict[v]))
                        adj_dict[v].remove(w)
                        adj_dict[w].remove(v)
                        stack.append(w)
                    else:
                        circuit.append(stack.pop())
                return circuit[::-1]
            
            # The augmented graph should be connected if the original was connected
            # (which it is, by construction of instance_graph)
            start_node = edges[0][0]
            
            # We need a fresh copy of adj for Hierholzer as it modifies it
            hier_adj = defaultdict(set)
            for u, v in aug_edges:
                hier_adj[u].add(v)
                hier_adj[v].add(u)
            
            try:
                circuit = find_eulerian_circuit(hier_adj, start_node)
                
                # The circuit is a list of vertices.
                # The virtual edges are the ones we added.
                # We need to split the circuit at the virtual edges.
                
                # Identify virtual edges in the circuit
                virtual_set = set()
                for u, v in pairs:
                    virtual_set.add(tuple(sorted((u, v))))
                
                paths = []
                current_path = []
                
                for i in range(len(circuit)):
                    v = circuit[i]
                    current_path.append(v)
                    if i < len(circuit) - 1:
                        u = circuit[i+1]
                        e = tuple(sorted((u, v)))
                        if e in virtual_set:
                            # End current path
                            if len(current_path) > 1:
                                paths.append(current_path)
                            current_path = []
                
                if len(current_path) > 1:
                    paths.append(current_path)
                
                # Validate
                if validate_paths(paths, edges):
                    best_paths = paths
                    best_count = len(paths)
                    
                    # This should be optimal (odd_count/2)
                    if best_count == lower_bound:
                        return best_paths
                        
            except Exception:
                pass
        
        # Strategy 2: Greedy targeting for remaining cases or to improve
        # If we didn't find an optimal solution via Eulerian, or if odd_count was 0
        if best_count > lower_bound:
            # Try greedy targeting
            max_targets = lower_bound + 10
            for target_k in range(lower_bound, min(len(edges), max_targets) + 1):
                if time.time() > inst_deadline - 5:
                    break
                if best_count <= target_k:
                    continue
                
                max_restarts = 500
                for restart in range(max_restarts):
                    if time.time() > inst_deadline - 5:
                        break
                    
                    seed_val = seed * 100000 + restart * 7 + target_k
                    paths = greedy_target(edges, target_k, seed_val, adj)
                    
                    if paths is not None:
                        if validate_paths(paths, edges):
                            merged = merge_paths(paths, max_iters=100)
                            if validate_paths(merged, edges):
                                if len(merged) < best_count:
                                    best_count = len(merged)
                                    best_paths = merged
                                    if best_count == lower_bound:
                                        return best_paths
                            else:
                                if target_k < best_count:
                                    best_count = target_k
                                    best_paths = paths
                                    if best_count == lower_bound:
                                        return best_paths

        if best_paths is None:
            # Fallback
            best_paths = fallback_decomposition(edges, adj, seed, inst_deadline)
        
        return best_paths

    def greedy_target(edges, k, seed_val, adj):
        """
        Attempts to cover all edges with exactly k simple paths.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None

        rng = random.Random(seed_val)
        remaining = set(edges)
        
        paths = [[] for _ in range(k)]

        edge_list = list(remaining)
        rng.shuffle(edge_list)

        start_edges = edge_list[:k]

        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]

        changed = True
        while changed and remaining:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)

            for i in idxs:
                if not paths[i]:
                    continue

                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue

                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue

        if remaining:
            return None
        return paths

    def merge_paths(paths, max_iters=1000):
        """
        Iteratively merge paths that share endpoints.
        """
        current_paths = [list(p) for p in paths]
        
        endpoint_map = defaultdict(list)
        
        def rebuild_endpoint_map():
            endpoint_map.clear()
            for i, p in enumerate(current_paths):
                if not p:
                    continue
                endpoint_map[p[0]].append((i, False))
                endpoint_map[p[-1]].append((i, True))
        
        rebuild_endpoint_map()
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            merge_candidates = []
            for v, entries in endpoint_map.items():
                if len(entries) >= 2:
                    merge_candidates.append(v)
            
            if not merge_candidates:
                break
            
            rng = random.Random(iter_count)
            rng.shuffle(merge_candidates)
            
            for v in merge_candidates:
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue
                
                path_indices = list(set([idx for idx, _ in entries]))
                
                if len(path_indices) < 2:
                    continue
                
                idx1, idx2 = path_indices[0], path_indices[1]
                
                p1 = current_paths[idx1]
                p2 = current_paths[idx2]
                
                if not p1 or not p2:
                    continue
                
                p1_end_v = (p1[-1] == v)
                p2_end_v = (p2[-1] == v)
                
                if p1_end_v and not p2_end_v:
                    new_path = p1 + p2
                elif p1_end_v and p2_end_v:
                    new_path = p1 + list(reversed(p2))
                elif not p1_end_v and not p2_end_v:
                    new_path = list(reversed(p1)) + p2
                else:
                    new_path = p2 + p1
                
                if len(set(new_path)) == len(new_path):
                    current_paths[idx1] = new_path
                    current_paths[idx2] = []
                    improved = True
                    break
            
            if improved:
                current_paths = [p for p in current_paths if p]
                rebuild_endpoint_map()
        
        return current_paths

    def fallback_decomposition(edges, adj, seed, deadline):
        rng = random.Random(seed + 9999)
        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() > deadline:
                break
            degs = defaultdict(int)
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            if not degs:
                break
            max_d = max(degs.values())
            cands = [v for v, d in degs.items() if d == max_d]
            start_u = rng.choice(cands)
            
            neighbors = [nb for nb in adj[start_u]
                         if (start_u, nb) in remaining or (nb, start_u) in remaining]
            if not neighbors:
                e = next(iter(remaining))
                start_u, start_v = e
            else:
                start_v = rng.choice(neighbors)
            
            path = [start_u, start_v]
            if (start_u, start_v) in remaining:
                remaining.remove((start_u, start_v))
            else:
                remaining.remove((start_v, start_u))
            
            changed = True
            while changed:
                changed = False
                back = path[-1]
                cands = [nb for nb in adj[back]
                         if nb not in path and
                         ((back, nb) in remaining or (nb, back) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    path.append(nb)
                    changed = True
                    continue
                front = path[0]
                cands = [nb for nb in adj[front]
                         if nb not in path and
                         ((front, nb) in remaining or (nb, front) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    path.insert(0, nb)
                    changed = True
                    continue
            
            paths.append(path)
        
        if remaining:
            for e in remaining:
                paths.append([e[0], e[1]])
        
        merged = merge_paths(paths, max_iters=100)
        if validate_paths(merged, edges):
            return merged
        else:
            return paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)

    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results


if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
