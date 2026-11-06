import time
import sys
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # Base cycle edges: 0-1-2-...-39-0
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set((min(e), max(e)) for e in base_edges)
    
    # For each potential chord (u, v) where u < v and (u,v) not in base_edge_set,
    # determine which cycle lengths it covers.
    # A chord (u, v) creates two paths around the cycle:
    #   Path 1: u -> u+1 -> ... -> v, length = v - u + 1
    #   Path 2: u -> u-1 -> ... -> v (going the other way), length = N - (v - u) + 1
    # So chord (u,v) covers lengths: (v-u+1) and (N - v + u + 1)
    
    # We need to cover cycle lengths 3 through 39 (length 40 is the base cycle)
    target_lengths = set(range(3, N))  # 3 to 39
    
    # Generate all possible chords and their coverage
    chord_coverages = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in base_edge_set:
                continue
            # Compute the two cycle lengths this chord creates
            len1 = v - u + 1
            len2 = N - (v - u) + 1
            # Both lengths must be >= 3 to be useful
            covered = set()
            if 3 <= len1 <= 39:
                covered.add(len1)
            if 3 <= len2 <= 39:
                covered.add(len2)
            if covered:
                chord_coverages.append(((u, v), covered))
    
    # Greedy set cover with lookahead
    uncovered = set(target_lengths)
    selected_chords = []
    
    while uncovered and (time.time() - start_time) < deadline:
        best_chord = None
        best_coverage = set()
        best_lookahead_score = -1
        
        for chord, covered in chord_coverages:
            if chord in [c for c, _ in selected_chords]:
                continue
            new_coverage = covered & uncovered
            if not new_coverage:
                continue
            
            # Lookahead: for each length in new_coverage, count how many 
            # other chords could also cover it (lower = more critical)
            # We want chords that cover critical lengths (fewer alternatives)
            lookahead = 0
            for length in new_coverage:
                # Count how many chords cover this length
                alt_count = 0
                for c2, cov2 in chord_coverages:
                    if c2 != chord and length in cov2:
                        alt_count += 1
                # Lower alt_count means this length is more critical
                lookahead += (1.0 / (1.0 + alt_count))
            
            score = len(new_coverage) + 0.1 * lookahead
            
            if score > best_lookahead_score:
                best_lookahead_score = score
                best_chord = chord
                best_coverage = new_coverage
        
        if best_chord is None:
            break
            
        selected_chords.append((best_chord, best_coverage))
        uncovered -= best_coverage
    
    # Build final edge set
    all_edges = list(base_edges)
    for chord, _ in selected_chords:
        u, v = chord
        edge = (min(u, v), max(u, v))
        if edge not in base_edge_set:
            all_edges.append(edge)
    
    # Build edge set for verification
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Generate cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Base cycle
            cycles.append(list(range(N)))
            continue
        
        # Find a chord that creates a cycle of length L
        # A chord (u, v) with v - u + 1 == L or N - (v - u) + 1 == L
        found = False
        for u in range(N):
            for v in range(u + 1, N):
                len1 = v - u + 1
                len2 = N - (v - u) + 1
                if len1 == L or len2 == L:
                    edge = (min(u, v), max(u, v))
                    if edge in edge_set:
                        if len1 == L:
                            # Cycle: u, u+1, ..., v, u
                            cycle = list(range(u, v + 1))
                        else:
                            # Cycle: u, u-1, ..., v, u (going the long way)
                            # This means the short path has length len2 = L
                            # So v - u + 1 = N - L + 1, meaning the short arc is N-L+1
                            # The cycle goes u -> u-1 -> ... -> v -> u
                            # Which is vertices: u, (u-1)%N, (u-2)%N, ..., v
                            # Number of vertices: N - (v - u) = N - v + u
                            # Wait, let me reconsider.
                            # If len2 = N - (v-u) + 1 = L, then the cycle uses the long path
                            # from u to v going counterclockwise: u, u-1, u-2, ..., v
                            # The vertices are: u, (u-1)%N, (u-2)%N, ..., v
                            # That's L vertices
                            cycle = [u]
                            cur = (u - 1) % N
                            while cur != v:
                                cycle.append(cur)
                                cur = (cur - 1) % N
                            cycle.append(v)
                        # Verify the cycle
                        valid = True
                        for i in range(len(cycle)):
                            a = cycle[i]
                            b = cycle[(i + 1) % len(cycle)]
                            if (min(a, b), max(a, b)) not in edge_set:
                                valid = False
                                break
                        if valid and len(set(cycle)) == L:
                            cycles.append(cycle)
                            found = True
                            break
            if found:
                break
        
        if not found:
            # Fallback: try all possible cycles of length L
            # This shouldn't happen with a good chord selection
            for u in range(N):
                # Try consecutive vertices
                cycle = [(u + i) % N for i in range(L)]
                valid = True
                for i in range(L):
                    a = cycle[i]
                    b = cycle[(i + 1) % L]
                    if (min(a, b), max(a, b)) not in edge_set:
                        valid = False
                        break
                if valid and len(set(cycle)) == L:
                    cycles.append(cycle)
                    found = True
                    break
            if not found:
                # Last resort: use a simple construction
                cycles.append(list(range(L)))
    
    # Final verification
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            assert (min(u, v), max(u, v)) in edge_set, f"Edge ({u},{v}) not in graph"
        assert len(set(cycle)) == len(cycle), f"Cycle has duplicate vertices: {cycle}"
    
    return {'edges': all_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
