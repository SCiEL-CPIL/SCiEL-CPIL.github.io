"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices using a circular difference set approach.
    
    Strategy:
    1. Start with a Hamiltonian cycle (N edges).
    2. Select chords such that their spans (distances along the cycle) cover all required values.
    3. Use a combinatorial approach to minimize the number of chords needed.
    
    For a cycle of length L, we need a chord connecting vertices at distance L-1 or N-(L-1) apart.
    A chord with span s covers cycle lengths s+1 and N-s+1.
    
    We need to cover all cycle lengths 3..N-1, which corresponds to spans 2..N-2.
    Since span s and span N-s are equivalent (same chord), we only need to cover spans 2..floor(N/2).
    For N=40, that's spans 2..20.
    
    We use a greedy set cover to select minimum chords covering all required spans.
    """
    n = N
    
    # Spans needed: for cycle length L, we need span L-1 or N-(L-1)
    # Cycle lengths 3..n-1 correspond to spans 2..n-2
    # But span s and span n-s are the same chord, so we need spans 2..floor(n/2)
    # For n=40: spans 2..20
    
    max_span = n // 2
    needed_spans = set(range(2, max_span + 1))
    
    # For each possible chord, determine which spans it covers
    # Chord (i, j) with i < j has span min(j-i, n-(j-i))
    # It covers cycle lengths span+1 and n-span+1
    # In terms of the needed spans (2..max_span), it covers span itself if span <= max_span
    
    # Greedy set cover: pick chords that cover the most uncovered spans
    uncovered = set(needed_spans)
    chosen_chords = []
    
    # Precompute all possible chords and their covered spans
    chord_options = []
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            if d == 1 or d == n-1:
                continue  # already in Hamiltonian cycle
            span = min(d, n-d)
            if span in needed_spans:
                chord_options.append(((i, j), {span}))
    
    # Sort by number of spans covered (all cover 1 span, so just pick greedily)
    # Actually, each chord covers exactly one span from needed_spans
    # So we just need one chord for each span
    
    # For each span, pick a chord
    span_to_chord = {}
    for span in range(2, max_span + 1):
        # Find a chord with this span
        # Chord (0, span) has span 'span'
        span_to_chord[span] = (0, span)
    
    # But we can do better: some chords might cover multiple spans if we consider
    # that a chord with span s also gives cycle length n-s+1
    # However, since we're only tracking spans 2..max_span, each chord covers exactly one
    
    # Actually, let me reconsider. The key insight is:
    # A chord with span s creates cycles of length s+1 AND n-s+1.
    # So if we pick a chord with span s, we cover both cycle lengths s+1 and n-s+1.
    # In terms of needed spans, we need to cover spans 2..n-2.
    # But span s and span n-s represent the same chord.
    # So we need to pick chords such that for every L in 3..n-1, there's a chord with span L-1 or n-(L-1).
    
    # Let me redo this properly:
    # For each cycle length L (3..n-1), we need a chord with span = L-1 or span = n-(L-1).
    # Let required_spans = set of all (L-1) for L in 3..n-1, plus all (n-(L-1)) for L in 3..n-1.
    # But span s and span n-s are the same, so we just need spans 2..n-2, but we can use
    # either span s or span n-s to cover a requirement.
    
    # Simpler: we need to cover cycle lengths 3..n-1.
    # A chord with span s covers cycle lengths s+1 and n-s+1.
    # We want to minimize chords such that all cycle lengths 3..n-1 are covered.
    
    cycle_lengths_needed = set(range(3, n))  # 3 to n-1
    
    # For each chord, compute which cycle lengths it covers
    chord_coverage = {}
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            if d == 1 or d == n-1:
                continue
            span = min(d, n-d)
            cl1 = span + 1
            cl2 = n - span + 1
            covered = set()
            if cl1 in cycle_lengths_needed:
                covered.add(cl1)
            if cl2 in cycle_lengths_needed:
                covered.add(cl2)
            chord_coverage[(i,j)] = covered
    
    # Greedy set cover
    uncovered = set(cycle_lengths_needed)
    chosen_chords = []
    
    while uncovered:
        best_chord = None
        best_cover = set()
        for chord, covers in chord_coverage.items():
            if chord in chosen_chords:
                continue
            new_cover = covers & uncovered
            if len(new_cover) > len(best_cover):
                best_cover = new_cover
                best_chord = chord
        if best_chord is None:
            break
        chosen_chords.append(best_chord)
        uncovered -= best_cover
    
    # Build edge list
    edges = []
    # Hamiltonian cycle
    for i in range(n):
        j = (i + 1) % n
        if i < j:
            edges.append((i, j))
        else:
            edges.append((j, i))
    
    # Add chords
    for (i, j) in chosen_chords:
        edges.append((i, j))
    
    # Remove duplicates and ensure canonical form
    edge_set = set()
    for e in edges:
        if e[0] > e[1]:
            e = (e[1], e[0])
        edge_set.add(e)
    edges = sorted(edge_set)
    
    # Now construct cycles for each length L = 3..n
    cycles = []
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    def has_edge(u, v):
        return (u, v) in edge_set
    
    for L in range(3, n+1):
        if L == n:
            # Hamiltonian cycle
            cycles.append(list(range(n)))
        else:
            # Find a cycle of length L
            # Try: consecutive vertices on the Hamiltonian cycle with a closing chord
            found = False
            for start in range(n):
                # Take vertices start, start+1, ..., start+L-1 (mod n)
                verts = [(start + k) % n for k in range(L)]
                # Check if all consecutive edges exist
                ok = True
                for k in range(L-1):
                    if not has_edge(verts[k], verts[k+1]):
                        ok = False
                        break
                if ok and has_edge(verts[L-1], verts[0]):
                    cycles.append(verts)
                    found = True
                    break
            if not found:
                # Try BFS/DFS to find any cycle of length L
                found = _find_cycle_of_length(n, edge_set, L)
                if found:
                    cycles.append(found)
                else:
                    # Fallback
                    cycles.append(list(range(L)))
    
    return {'edges': edges, 'cycles': cycles}

def _find_cycle_of_length(n, edge_set, L):
    """Find a simple cycle of length L using DFS with pruning."""
    if L > n:
        return None
    if L == 3:
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in edge_set:
                    continue
                for w in range(v+1, n):
                    if (v, w) in edge_set and (w, u) in edge_set:
                        return [u, v, w]
        return None
    
    def dfs(current, path, visited):
        if len(path) == L:
            if (current, path[0]) in edge_set:
                return list(path)
            return None
        for neighbor in range(n):
            if neighbor in visited:
                continue
            if (current, neighbor) not in edge_set:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            result = dfs(neighbor, path, visited)
            if result:
                return result
            path.pop()
            visited.remove(neighbor)
        return None
    
    for start in range(n):
        visited = {start}
        path = [start]
        result = dfs(start, path, visited)
        if result:
            return result
    return None

# EVOLVE-BLOCK-END
