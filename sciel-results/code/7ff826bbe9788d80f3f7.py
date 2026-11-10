import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 140.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _bfs_path(adj, s, t, forbidden):
        """Shortest simple path from s to t, internal vertices not in forbidden."""
        if s == t:
            return [s]
        if s in forbidden or t in forbidden:
            return None
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if t not in prev:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        # Check if internal vertices are forbidden
        if len(path) >= 3 and set(path[1:-1]) & forbidden:
            return None
        return path

    def _pair_odd_vertices(adj, odd_vertices, max_restarts=50):
        """
        Find vertex-disjoint paths pairing up all odd vertices.
        Returns a list of paths (vertex lists) or None if failed.
        """
        if not odd_vertices:
            return []
        
        # Try multiple random orderings of odd vertices to find a valid disjoint pairing
        for _ in range(max_restarts):
            remaining = odd_vertices[:]
            random.shuffle(remaining)
            used_internal = set()
            pair_paths = []
            success = True
            
            while len(remaining) >= 2:
                # Try to find a pair
                found_pair = False
                # Sort remaining to prioritize shorter paths? Or just iterate.
                # We need to pick 2 vertices from remaining.
                # To be efficient, we can try all pairs in the current 'remaining' list?
                # Or just pick the first two that work?
                # Greedy: try to find the shortest path between any pair in remaining.
                best_path = None
                best_len = float('inf')
                best_i, best_j = -1, -1
                
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        s, t = remaining[i], remaining[j]
                        p = _bfs_path(adj, s, t, used_internal)
                        if p is not None and len(p) < best_len:
                            best_len = len(p)
                            best_path = p
                            best_i, best_j = i, j
                
                if best_path is None:
                    success = False
                    break
                
                # Commit this path
                used_internal |= set(best_path[1:-1])
                pair_paths.append(best_path)
                
                # Remove the endpoints from remaining
                # Note: best_i and best_j are indices in the 'remaining' list
                # We must remove them carefully
                # It's safer to reconstruct the list
                new_remaining = []
                for idx, v in enumerate(remaining):
                    if idx == best_i or idx == best_j:
                        continue
                    new_remaining.append(v)
                remaining = new_remaining
            
            if success:
                return pair_paths
        
        return None

    def _eulerian_circuit(adj):
        """
        Hierholzer's algorithm for multigraph.
        adj: dict of vertex -> list of (neighbor, edge_id)
        Returns the circuit as a list of vertices.
        """
        if not adj:
            return []
        
        # Ensure all vertices in adj have a list
        # Find a start vertex with edges
        start = None
        for v in adj:
            if adj[v]:
                start = v
                break
        if start is None:
            return []
            
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            if adj[v]:
                # Take the first available edge
                u, _ = adj[v].pop()
                adj[u].remove((v, _)) # Need to remove the reverse edge too? 
                # Wait, adj is built with edge indices.
                # Let's use a different structure for Hierholzer to be safe.
                pass
        
        # Let's implement a standard Hierholzer using edge indices to avoid mutation issues.
        return [] # Placeholder, will rewrite below

    def _solve_one_gallai(edges, n):
        if not edges:
            return []
            
        # Build adjacency for original graph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find odd vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Eulerian graph: 1 path (circuit)
            # Use Hierholzer
            hadj = {i: list(adj[i]) for i in range(n)}
            # Flatten to list of neighbors for Hierholzer
            # Standard Hierholzer
            circuit = []
            stack = [0]
            while stack:
                v = stack[-1]
                if hadj[v]:
                    u = hadj[v].pop()
                    hadj[u].remove(v)
                    stack.append(u)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            # Ensure it's a simple path? An Eulerian circuit is a trail, not necessarily a simple path.
            # However, for the purpose of the task, if the graph is Eulerian, we can return the circuit.
            # But the task requires SIMPLE paths.
            # An Eulerian circuit in a graph with repeated vertices is NOT a simple path.
            # So we cannot just return the circuit if it repeats vertices.
            # However, if |O|=0, the min number of paths is 1.
            # But that 1 path must be simple.
            # If the graph is not Hamiltonian, we can't have 1 simple path covering all edges.
            # Wait. Gallai's theorem is about EDGE-DISJOINT PATH COVERS.
            # The paths must be simple.
            # If |O|=0, we can decompose into 1 trail, but not necessarily 1 simple path.
            # Actually, the theorem says min number of paths is max(1, |O|/2).
            # This implies that if |O|=0, it IS possible to cover with 1 simple path?
            # No, it's possible to cover with 1 TRAIL.
            # If the graph is not Hamiltonian, you can't cover it with 1 simple path.
            # So the "1 path" in Gallai's theorem refers to a path in the sense of a walk?
            # "Paths must be simple".
            # If the theorem guarantees 1 path, it must be a simple path.
            # This is only true if the graph has a Hamiltonian path.
            # Random graphs are likely to have Hamiltonian paths.
            # If not, we might need more paths.
            # But for now, let's assume the Eulerian circuit can be treated as a path if we don't repeat vertices.
            # If we do repeat vertices, we are stuck.
            # For the odd case, the cuts produce simple paths because the virtual edges are disjoint.
            
            return [circuit]
            
        # Pair odd vertices
        pair_paths = _pair_odd_vertices(adj, odd_vertices)
        
        if pair_paths is None:
            # Fallback to greedy if pairing fails
            return _greedy_decompose(edges, n)
            
        # Build multigraph H = G + pair_paths
        # We need to distinguish original edges from added edges.
        # We'll use a list of edges with flags.
        # Edge format: (u, v, is_added)
        
        edge_list = []
        for u, v in edges:
            edge_list.append((u, v, False))
            
        for pp in pair_paths:
            for i in range(len(pp) - 1):
                u, v = pp[i], pp[i+1]
                edge_list.append((u, v, True))
                
        # Build adjacency for Hierholzer
        # adj_h[v] = list of (neighbor, edge_index)
        adj_h = {i: [] for i in range(n)}
        for idx, (u, v, is_added) in enumerate(edge_list):
            adj_h[u].append((v, idx))
            adj_h[v].append((u, idx))
            
        # Hierholzer
        used_edge = [False] * len(edge_list)
        stack = [0] # Start at 0
        circuit_vertices = []
        edge_seq = [] # List of edge indices in order
        
        while stack:
            v = stack[-1]
            advanced = False
            # Find an unused edge
            while adj_h[v]:
                u, idx = adj_h[v].pop()
                if not used_edge[idx]:
                    used_edge[idx] = True
                    # Remove reverse edge reference? 
                    # We don't strictly need to remove from adj_h[u] if we check used_edge, 
                    # but it's cleaner to pop it.
                    # However, finding the reverse index is O(deg). 
                    # For small graphs, it's fine.
                    # Let's just check used_edge.
                    stack.append(u)
                    edge_seq.append(idx)
                    advanced = True
                    break
            if not advanced:
                circuit_vertices.append(stack.pop())
                
        circuit_vertices.reverse()
        edge_seq.reverse()
        
        # Now, walk the circuit and cut at added edges.
        # edge_seq[i] is the index of the edge used at step i.
        # The vertices are circuit_vertices.
        # circuit_vertices[0] is the start.
        # The path is circuit_vertices[0] -> circuit_vertices[1] -> ...
        # Edge i connects circuit_vertices[i] and circuit_vertices[i+1].
        
        paths = []
        current_path = [circuit_vertices[0]]
        
        for i in range(len(edge_seq)):
            idx = edge_seq[i]
            is_added = edge_list[idx][2]
            u = circuit_vertices[i]
            v = circuit_vertices[i+1]
            
            current_path.append(v)
            
            if is_added:
                # Cut here
                if len(current_path) >= 2:
                    # Verify simple path
                    if len(set(current_path)) == len(current_path):
                        paths.append(current_path)
                    else:
                        # If not simple, we have a problem. 
                        # This shouldn't happen with vertex-disjoint pairing.
                        # Fallback: add as is or split?
                        paths.append(current_path)
                current_path = []
                
        # If there's a remaining path (cyclic case where we didn't end on an added edge?)
        # Actually, we cut at every added edge.
        # The last segment after the last cut goes until the end of the circuit.
        # Since the circuit is closed, the last segment connects back to the start?
        # No, the "cuts" break the cycle into paths.
        # If we have k added edges, we get k paths.
        # The path starts AFTER an added edge and ends BEFORE the next added edge.
        # My loop above appends v to current_path.
        # When is_added is True, we finalize the path.
        # This means the path includes the added edge?
        # No, the added edge is the CUT.
        # The path in the ORIGINAL graph does NOT include the added edge.
        # So, if we traverse an added edge, we should NOT include it in the path.
        
        # Let's re-evaluate the extraction.
        # The circuit is a sequence of edges.
        # We want the segments of ORIGINAL edges between ADDED edges.
        
        # Reset
        paths = []
        current_path = []
        
        for i in range(len(edge_seq)):
            idx = edge_seq[i]
            is_added = edge_list[idx][2]
            u = circuit_vertices[i]
            v = circuit_vertices[i+1]
            
            if is_added:
                # We just crossed an added edge.
                # The segment before this edge is a path.
                # The segment starts at 'u' (which is the end of the previous original segment)
                # Wait, the added edge connects u and v.
                # The path we just finished ends at u.
                # So we should finalize current_path here.
                if current_path:
                    if len(set(current_path)) == len(current_path):
                        paths.append(current_path)
                    else:
                        paths.append(current_path)
                    current_path = []
                # The next path will start at v.
                current_path = [v]
            else:
                # Original edge.
                if not current_path:
                    current_path = [u]
                current_path.append(v)
                
        # After the loop, if current_path is not empty, it's the last segment.
        if current_path:
            if len(set(current_path)) == len(current_path):
                paths.append(current_path)
            else:
                paths.append(current_path)
                
        # Filter out empty paths
        paths = [p for p in paths if len(p) >= 2]
        
        if not paths:
            return _greedy_decompose(edges, n)
            
        return paths

    def _greedy_decompose(edges, n, max_time=5.0):
        # Fallback greedy
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        edge_set = set(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        def has_edge(u, v, eset):
            return (u, v) in eset or (v, u) in eset

        def find_longest_path(start, eset, adj_local, visited_set):
            path = [start]
            visited = set(visited_set) | {start}
            current = start
            
            while True:
                available = []
                for nb in adj_local[current]:
                    if nb not in visited and has_edge(current, nb, eset):
                        available.append(nb)
                
                if not available:
                    break
                
                min_deg = float('inf')
                best_nb = None
                for nb in available:
                    deg = 0
                    for x in adj_local[nb]:
                        if x not in visited and has_edge(nb, x, eset):
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_nb = nb
                
                if random.random() < 0.2 and len(available) > 1:
                    best_nb = random.choice(available)
                
                u, v = current, best_nb
                if (u, v) in eset:
                    eset.discard((u, v))
                else:
                    eset.discard((v, u))
                
                path.append(best_nb)
                visited.add(best_nb)
                current = best_nb
                
            return path

        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < max_time and _remaining() > 2.0:
            remaining = set(edge_set)
            paths = []
            
            deg = {i: 0 for i in range(n)}
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
            
            while remaining:
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
                
                if rem_odd and random.random() < 0.8:
                    start_v = random.choice(rem_odd)
                else:
                    start_v = random.choice([v for v in rem_deg if rem_deg[v] > 0])
                
                path = find_longest_path(start_v, remaining, adj, set())
                
                if not path or len(path) < 2:
                    found_edge = False
                    for u, v in remaining:
                        if u == start_v:
                            path = [start_v, v]
                            remaining.discard((u, v))
                            found_edge = True
                            break
                        elif v == start_v:
                            path = [start_v, u]
                            remaining.discard((u, v))
                            found_edge = True
                            break
                    if not found_edge:
                        break
                else:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if (u, v) in remaining:
                            remaining.discard((u, v))
                        else:
                            remaining.discard((v, u))
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        max_t = 10.0
        if _remaining() < max_t:
            max_t = max(1.0, _remaining() - 1.0)
            
        # Try Gallai first
        paths = _solve_one_gallai(edges, n)
        
        # If Gallai fails or returns too many paths, try Greedy?
        # Actually, Gallai should be optimal.
        # But if the "simple path" constraint is violated by the Eulerian circuit logic (e.g. in the |O|=0 case),
        # we might need to verify.
        
        # Verify paths
        valid = True
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            if len(set(path)) != len(path):
                valid = False
                break
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if not valid:
                break
                
        if not valid or len(used_edges) != len(edges):
            # Fallback to Greedy
            paths = _greedy_decompose(edges, n, max_time=max_t)
            
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
