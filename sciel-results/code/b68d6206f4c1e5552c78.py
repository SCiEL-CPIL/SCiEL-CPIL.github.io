"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time
import itertools

def _valid_solution(edges, cycles):
    """Validate that the solution meets all requirements."""
    # Check edges are simple, distinct, and within range
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= N or v < 0 or v >= N:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Check cycles
    if len(cycles) != N - 2:  # lengths 3 to 40
        return False
    
    for i, cycle in enumerate(cycles):
        expected_len = 3 + i
        if len(cycle) != expected_len:
            return False
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            return False
        # Check all vertices in range
        for v in cycle:
            if v < 0 or v >= N:
                return False
        # Check each consecutive pair and wrap-around is an edge
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                return False
    
    return True

def _find_cycle_in_graph(start, end, length, edge_set, visited=None, path=None):
    """Find a simple cycle of given length from start to end in graph."""
    if visited is None:
        visited = set()
    if path is None:
        path = []
    
    path.append(start)
    visited.add(start)
    
    if start == end and len(path) == length:
        return path
    
    if len(path) >= length:
        path.pop()
        visited.discard(start)
        return None
    
    # Determine which neighbors to explore
    neighbors = []
    if start == end:
        # We're at the endpoint, need to close the cycle
        # But we only reach here if len(path) < length, which shouldn't happen
        # if we're at end. Actually, we should check neighbors normally
        pass
    
    for v in range(N):
        if v in visited:
            continue
        key = (min(start, v), max(start, v))
        if key in edge_set:
            neighbors.append(v)
    
    for v in neighbors:
        # If v is end and len(path) == length - 1, we can close
        if v == end and len(path) == length - 1:
            path.append(v)
            result = path[:]
            path.pop()
            visited.discard(start)
            return result
        
        result = _find_cycle_in_graph(v, end, length, edge_set, visited, path)
        if result is not None:
            visited.discard(start)
            return result
    
    path.pop()
    visited.discard(start)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Start with a reasonable base: path graph plus some chords
    # Path: 0-1-2-...-39
    edges = [(i, i+1) for i in range(N-1)]
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # Add chord (0, 39) to complete the 40-cycle
    edges.append((0, 39))
    edge_set.add((0, 39))
    
    # We need to add chords to cover cycles of lengths 3-39
    # Strategy: add chords from vertex 0 to various vertices
    
    # First, try to find a minimal set of chords from 0
    # A chord (0, k) creates a cycle of length k (0-1-...-k-0)
    # So adding chords (0, k) for various k gives us cycles of length k
    
    # But we can also form cycles using combinations of chords
    
    # Let's start by adding chords for all even lengths 4, 6, ..., 38
    # Cycle of length L using chord (0, L-1): 0-1-...-(L-1)-0
    # Wait, that's a path of length L-1, so cycle length is L
    
    # Actually, chord (0, k) gives cycle 0-1-...-k-0 of length k+1
    # So for cycle length L, we need chord (0, L-1)
    
    # Let's add chords (0, L-1) for L = 3, 4, 5, ..., 39
    # That's chords (0,2), (0,3), (0,4), ..., (0,38)
    # That's 37 chords, total edges = 39 + 1 + 37 = 77
    
    # This is the same as the seed. Let's try to do better.
    
    # Better approach: use a sparse set of chords and verify we can find all cycles
    
    # Let's try chords from 0 to a carefully chosen set of vertices
    
    # Known result: for a path 0-1-...-(N-1) with chord (0, N-1),
    # we need additional chords to cover all cycle lengths.
    
    # Let's use a greedy approach:
    # 1. Start with path + (0, 39)
    # 2. For each length L from 3 to 39, try to find a cycle of length L
    # 3. If no cycle exists, add a chord to enable one
    
    current_edges = edges[:]
    current_edge_set = edge_set.copy()
    
    # Check which cycle lengths we already have
    available_lengths = set()
    
    # Length 40: 0-1-2-...-39-0
    available_lengths.add(40)
    
    # Try to find cycles of other lengths
    for L in range(3, 40):
        # Try to find a cycle of length L
        # Simple approach: try all pairs (start, end) where end-start+1 <= L
        # and the path start->start+1->...->end exists in the graph
        # and (start, end) is an edge or can be connected
        
        # For now, just check if chord (0, L-1) exists
        if L <= N:
            key = (0, L-1)
            if key in current_edge_set:
                available_lengths.add(L)
    
    # Add chords to cover missing lengths
    missing_lengths = set(range(3, 40)) - available_lengths
    
    # Greedy: for each missing length, add the most useful chord
    for L in sorted(missing_lengths, reverse=True):  # Start with larger lengths
        if time.time() > deadline:
            break
        
        # Try to find a chord that helps
        # A chord (u, v) can help create cycles of various lengths
        # Simple heuristic: add chord (0, L-1) for the missing length L
        
        if L <= N:
            u, v = 0, L-1
            key = (min(u,v), max(u,v))
            if key not in current_edge_set:
                current_edges.append((u, v))
                current_edge_set.add(key)
                available_lengths.add(L)
    
    # Now verify we have all cycles and construct them
    # This is where it gets complex - we need to actually find the cycles
    
    # Let's use a simpler approach: build the cycle certificates
    cycles = []
    
    for L in range(3, N+1):
        if time.time() > deadline:
            # Return what we have with a fallback
            break
        
        if L == 40:
            cycles.append(list(range(N)))
            continue
        
        # For cycle of length L, try to find it
        # First try: 0-1-2-...-(L-1)-0 using chord (0, L-1)
        if (0, L-1) in current_edge_set or (L-1, 0) in current_edge_set:
            cycle = list(range(L))
            cycles.append(cycle)
            continue
        
        # Try to find a cycle using DFS
        found = False
        for start in range(N):
            if time.time() > deadline:
                break
            for end in range(start+1, N):
                if end - start + 1 > L:
                    break
                # Check if path start->start+1->...->end exists
                path_exists = True
                for i in range(start, end):
                    key = (i, i+1)
                    if key not in current_edge_set:
                        path_exists = False
                        break
                
                if not path_exists:
                    continue
                
                # Check if (start, end) is an edge
                key = (min(start, end), max(start, end))
                if key in current_edge_set:
                    # Cycle: start->start+1->...->end->start
                    cycle_len = end - start + 1
                    if cycle_len == L:
                        cycles.append(list(range(start, end+1)))
                        found = True
                        break
                
                # Try to extend with chords
                # This is getting complex, let's use a simpler fallback
        
        if not found:
            # Fallback: use chord (0, L-1) if available, or find another way
            # For now, use a default cycle
            cycles.append(list(range(L)))
    
    # Validate the solution
    if not _valid_solution(current_edges, cycles):
        # Fall back to a known valid solution
        # Use the dense approach: path + all chords from 0
        fallback_edges = [(i, i+1) for i in range(N-1)]
        for L in range(3, N):
            fallback_edges.append((0, L-1))
        fallback_edges.append((0, N-1))
        
        fallback_cycles = []
        for L in range(3, N+1):
            if L == N:
                fallback_cycles.append(list(range(N)))
            else:
                fallback_cycles.append(list(range(L)))
        
        return {'edges': fallback_edges, 'cycles': fallback_cycles}
    
    return {'edges': current_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
