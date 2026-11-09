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
    #
    # Key insight: A chord (0, k) provides cycles of length k+1 (forward)
    # and 41-k (backward) via the ring. But other chords may also help form
    # cycles of those lengths through more complex paths.
    #
    # We'll do a greedy removal: for each chord, try removing it and check
    # if every length 3..40 still has a valid cycle. If yes, keep it removed.

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
        
        # Try to find a cycle of length L starting from vertex 0
        # Use DFS with path tracking
        # Limit search to avoid timeout
        max_depth = L
        visited_in_path = [False] * N
        path = []
        
        def dfs(current, remaining, visited):
            if remaining == 0:
                # Check if current connects back to path[0]
                if path[0] in adj[current]:
                    return path[:] + [current]
                return None
            for nb in adj[current]:
                if not visited[nb] and nb != path[0]:
                    visited[nb] = True
                    path.append(nb)
                    result = dfs(nb, remaining - 1, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited[nb] = False
            return None
        
        # Try starting from vertex 0
        visited_in_path[0] = True
        path.append(0)
        result = dfs(0, L - 1, visited_in_path)
        if result is not None:
            # Remove the last element which is the duplicate of start
            if len(result) == L + 1 and result[-1] == result[0]:
                return result[:-1]
            elif len(result) == L:
                return result
            else:
                return None
        
        # Try other starting vertices
        for start in range(1, N):
            visited = [False] * N
            visited[start] = True
            path = [start]
            result = dfs(start, L - 1, visited)
            if result is not None:
                if len(result) == L + 1 and result[-1] == result[0]:
                    return result[:-1]
                elif len(result) == L:
                    return result
                else:
                    continue
        
        return None

    def all_lengths_covered(edge_set):
        """Check if all lengths 3..40 have a valid cycle in the graph."""
        for L in range(3, 41):
            if find_cycle_of_length(L, edge_set) is None:
                return False, L
        return True, None

    # Greedy chord removal
    # Sort chords by how critical they are (chords that provide unique length coverage first)
    # For each chord k, it provides lengths k+1 and 41-k
    # We'll try removing in order of least criticality
    
    # For each chord, determine which lengths it uniquely provides
    # A length L is uniquely provided by chord k if no other chord provides L
    # Chord k provides L if L == k+1 or L == 41-k
    
    # Let's just try removing each chord and check
    # To be efficient, we'll try removals in a random order and keep successful ones
    
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
            # This shouldn't happen if all_lengths_covered passed
            # But let's handle it by restoring the original construction
            cycles[L] = None
        else:
            cycles[L] = cycle
    
    # If any cycle is missing, fall back to the original 19-chord construction
    if any(c is None for c in cycles.values()):
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
