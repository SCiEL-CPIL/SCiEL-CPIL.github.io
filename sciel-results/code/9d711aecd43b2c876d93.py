import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _decompose(edges):
        if not edges:
            return []
        
        adj = _build_adj(edges)
        remaining_edges = set(edges)
        paths = []
        
        # Precompute initial degrees for heuristics
        def get_deg(v, rem_edges):
            count = 0
            for nb in adj[v]:
                e = tuple(sorted((v, nb)))
                if e in rem_edges:
                    count += 1
            return count

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                return paths

            # Find start vertex: min degree in remaining graph
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            if not active_vertices:
                break

            # Calculate degrees for all active vertices
            degs = {v: 0 for v in active_vertices}
            for u, v in remaining_edges:
                degs[u] += 1
                degs[v] += 1
                
            # Choose start: min degree, tie-break by vertex id
            start = min(active_vertices, key=lambda x: (degs[x], x))
            
            # Greedy Simple Path Extension
            path = [start]
            current = start
            visited = {start}
            
            while True:
                # Find available neighbors
                avail = [nb for nb in adj[current] 
                         if tuple(sorted((current, nb))) in remaining_edges 
                         and nb not in visited]
                
                if not avail:
                    break
                
                # Heuristic: Choose neighbor with lowest remaining degree (leaf-preference)
                # This tends to extend paths into dead-ends, keeping the core for later
                avail.sort(key=lambda x: (degs[x], x))
                next_v = avail[0]
                
                # Update state
                edge = tuple(sorted((current, next_v)))
                remaining_edges.remove(edge)
                
                # Update degrees for subsequent steps
                degs[current] -= 1
                degs[next_v] -= 1
                
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths

    def _try_merge(paths, edges_set):
        """Try to merge two paths if they are edge-disjoint and can be connected."""
        # This is a placeholder for local search improvement.
        # For now, we rely on the greedy construction which is already quite good.
        # A proper merge would check if end of path A connects to start of path B
        # or vice versa, and if the resulting path is simple.
        pass

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        edges = instance(*spec)
        paths = _decompose(edges)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
    print(f"Total: {sum(len(r) for r in res)} paths")
