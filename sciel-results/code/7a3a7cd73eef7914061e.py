import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_paths(edges, n):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        used_edges = set()
        paths = []
        
        # We need to pick edges in a specific order. 
        # If we just iterate `for u,v in edges`, the order is fixed by the sorted edges.
        # This is deterministic.
        
        for u, v in edges:
            if (u, v) in used_edges:
                continue
            
            path = [u, v]
            used_edges.add((u, v))
            
            def available_neighbors(node, path):
                candidates = []
                for w in adj[node]:
                    # Check if edge (node, w) is used
                    if (node, w) in used_edges or (w, node) in used_edges:
                        continue
                    # Check if w is in path (simple path constraint)
                    if w in path:
                        continue
                    candidates.append(w)
                return candidates

            # Greedy extension
            # We can alternate ends or just extend one end until stuck, then the other.
            # Let's try to extend the end that has more options first? Or fewer?
            # Let's stick to a simple heuristic: extend right, then left.
            
            current_end = v
            other_end = u
            
            # Extend from current_end
            while True:
                cands = available_neighbors(current_end, path)
                if not cands:
                    break
                # Heuristic: pick the candidate with the smallest degree in remaining graph
                # This is a proxy for "most constrained".
                def rem_degree(node):
                    count = 0
                    for nb in adj[node]:
                        if (node, nb) not in used_edges and (nb, node) not in used_edges:
                            count += 1
                    return count
                
                # To keep it fast, we might just pick the first one. 
                # But let's try the min-degree heuristic for better path lengths.
                # Since n is small (<=30), this is cheap.
                best_w = min(cands, key=lambda w: rem_degree(w))
                
                path.append(best_w)
                used_edges.add((current_end, best_w))
                current_end = best_w
                
            # Extend from other_end
            while True:
                cands = available_neighbors(other_end, path)
                if not cands:
                    break
                def rem_degree(node):
                    count = 0
                    for nb in adj[node]:
                        if (node, nb) not in used_edges and (nb, node) not in used_edges:
                            count += 1
                    return count
                
                best_w = min(cands, key=lambda w: rem_degree(w))
                
                path.insert(0, best_w)
                used_edges.add((other_end, best_w))
                other_end = best_w
                
            paths.append(path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_paths(edges, n)
        results.append(paths)
        
    return results
