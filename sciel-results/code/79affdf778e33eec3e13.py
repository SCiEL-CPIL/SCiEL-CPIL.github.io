import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Build a sparse pancyclic graph: Hamiltonian ring + chords.
    # We start with the 19-chord fan from vertex 0 (edges 0-k for k=2..20)
    # and then attempt to remove chords one by one if all cycle lengths
    # can still be covered by the remaining structure.

    edges = set()

    # Ring: 0-1-2-...-39-0
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))

    # Chords from vertex 0 to vertices 2..20
    chords = list(range(2, 21))
    for k in chords:
        edges.add((0, k))

    def find_cycle_of_length(L, edge_set):
        """Find a cycle of exactly L vertices in the graph defined by edge_set.
        Returns a list of L distinct vertices forming a cycle, or None."""
        if L < 3 or L > N:
            return None
        if L == N:
            # Check if the full ring is intact
            for i in range(N):
                a, b = i, (i + 1) % N
                if (min(a, b), max(a, b)) not in edge_set:
                    return None
            return list(range(N))
        
        # Build adjacency list
        adj = {i: [] for i in range(N)}
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)
            
        # Sort neighbors to ensure deterministic behavior and potentially better pruning
        for i in range(N):
            adj[i].sort()

        # Try to find a cycle of length L
        # We look for a path of length L-1 from start to end, where (start, end) is an edge
        # We iterate over all edges as potential closing edges
        # To speed up, we prioritize edges incident to 0 (chords) or ring edges
        
        # Heuristic: Try chords first
        candidate_edges = []
        for (u, v) in edge_set:
            if u == 0 or v == 0:
                candidate_edges.append((u, v))
            else:
                candidate_edges.append((u, v))
        
        # Shuffle candidate edges to vary search
        random.shuffle(candidate_edges)

        for (u, v) in candidate_edges:
            # We need a path of length L-1 from u to v
            # Path: u - x1 - x2 - ... - v
            # Length of path is L-1 edges, so L-2 intermediate vertices
            # Total vertices in cycle: L
            
            # BFS/DFS from u to find path of length L-1 to v
            # Since L can be up to 40, DFS is risky.
            # Use DFS with pruning.
            
            visited = [False] * N
            visited[u] = True
            path = [u]
            
            def dfs(current, remaining):
                if remaining == 0:
                    if current == v:
                        return path[:]
                    return None
                for nb in adj[current]:
                    if nb == v:
                        if remaining == 1:
                            path.append(v)
                            res = path[:]
                            path.pop()
                            return res
                        else:
                            continue # Cannot go to v yet
                    if not visited[nb]:
                        visited[nb] = True
                        path.append(nb)
                        res = dfs(nb, remaining - 1)
                        if res is not None:
                            return res
                        path.pop()
                        visited[nb] = False
                return None

            # We want a path of L-1 edges from u to v
            # So we start at u, need to take L-1 steps to reach v
            # dfs(u, L-1)
            
            # Optimization: If L-1 is large, we can try from both ends
            # But let's stick to one for now with a time limit
            
            # To avoid infinite loops or long runs, we limit the number of nodes visited
            # This is a simple heuristic. 
            # A better approach: Use BFS to check connectivity first?
            
            # Let's just run DFS. If it's too slow, we might fail.
            # To mitigate, we try fewer candidate edges or break early.
            
            result = dfs(u, L - 1)
            if result is not None:
                # result is [u, ..., v]
                # Cycle is u, ..., v, u
                # But we return vertex list of cycle, usually distinct vertices
                # The cycle vertices are result
                # Check distinctness
                if len(result) == L:
                    return result
                else:
                    return None

        return None

    def all_lengths_covered(edge_set):
        """Check if all lengths 3..40 have a valid cycle in the graph."""
        for L in range(3, 41):
            if find_cycle_of_length(L, edge_set) is None:
                return False, L
        return True, None

    # Greedy chord removal
    # Sort chords by how critical they are
    # We'll try removing in a random order
    
    chord_list = list(chords)
    random.seed(42)
    random.shuffle(chord_list)
    
    for k in chord_list:
        if time.time() > deadline - 5.0:
            break
        
        # Try removing chord (0, k)
        edges.discard((0, k))
        
        # Check if all lengths are still covered
        covered, missing_L = all_lengths_covered(edges)
        
        if not covered:
            # Restore the chord
            edges.add((0, k))
    
    # Build cycle certificates for each length 3..40
    cycles = {}
    for L in range(3, 41):
        cycle = find_cycle_of_length(L, edges)
        if cycle is None:
            # Fallback: Restore original construction if failed
            # This should not happen if all_lengths_covered passed, 
            # but DFS might have failed due to time limit or search space
            pass
        else:
            cycles[L] = cycle
    
    # If any cycle is missing, fall back to the original 19-chord construction
    if any(c is None for c in cycles.values()) or len(cycles) < 38:
        edges = set()
        for i in range(N):
            a, b = i, (i + 1) % N
            edges.add((min(a, b), max(a, b)))
        for k in range(2, 21):
            edges.add((0, k))
        
        cycles = {}
        cycles[40] = list(range(N))
        for k in range(2, 21):
            l1 = k + 1
            l2 = 41 - k
            if l1 not in cycles:
                cycles[l1] = list(range(k + 1))
            if l2 not in cycles:
                cycles[l2] = [0] + list(range(N - 1, k - 1, -1))
    
    # Assemble final list in order: length 3, 4, ..., 40
    final_cycles = []
    for L in range(3, 41):
        final_cycles.append(cycles[L])
    
    # Convert edges to sorted list of tuples
    edge_list = sorted(edges)
    
    return {'edges': edge_list, 'cycles': final_cycles}

if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}")
    print(f"Cycles: {len(res['cycles'])}")
    # Quick validation
    edge_set = {frozenset(e) for e in res['edges']}
    for c in res['cycles']:
        assert len(c) == len(set(c)), f"Duplicate vertex in cycle of length {len(c)}"
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            assert frozenset((u, v)) in edge_set, f"Missing edge ({u},{v}) in cycle of length {len(c)}"
    print("All cycles valid.")
