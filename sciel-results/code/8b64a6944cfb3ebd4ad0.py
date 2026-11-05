"""Construct sparse pancyclic graph on 40 vertices via chord-coverage search."""
import time
import random
from itertools import combinations

N = 40
DEADLINE = 55.0  # safety margin before 60s limit

def _base_edges():
    """The 40 cycle edges (i, i+1 mod N)."""
    return [(i, (i + 1) % N) for i in range(N)]

def _edge_set(edges):
    return frozenset(frozenset(e) for e in edges)

def _validate(edges, cycles):
    """Check validity: edges simple/distinct, cycles valid, all cycle edges present."""
    es = set()
    for e in edges:
        u, v = e
        if not (0 <= u < N and 0 <= v < N) or u == v:
            return False
        key = frozenset((u, v))
        if key in es:
            return False
        es.add(key)
    es_frozensets = set(es)
    for cyc in cycles:
        k = len(cyc)
        if k < 3 or k > N:
            return False
        if len(set(cyc)) != k:
            return False
        for i in range(k):
            u, v = cyc[i], cyc[(i + 1) % k]
            if frozenset((u, v)) not in es_frozensets:
                return False
    return True

def _score(edges, cycles):
    """Objective: minimize edge_count - 40."""
    return len(edges) - 40

def _try_construct(chords, deadline):
    """Given a set of chord pairs (in addition to base cycle), find cycles or return None."""
    all_edges = _base_edges() + chords
    es = _edge_set(all_edges)
    
    # For each k in 3..N, find a cycle of length k
    cycles = []
    for k in range(3, N + 1):
        if time.time() > deadline:
            return None
        found = False
        # Strategy 1: try consecutive windows on the cycle
        # A window of k consecutive vertices: need the chord (start, end) if k < N
        for start in range(N):
            if time.time() > deadline:
                return None
            verts = [(start + i) % N for i in range(k)]
            # Check if all edges exist: consecutive are in base cycle,
            # need chord (verts[0], verts[-1]) if k < N
            if k < N:
                if frozenset((verts[0], verts[-1])) not in es:
                    continue
            # For k == N, all edges are in base cycle
            # Verify all edges (should be fine)
            cycles.append(verts)
            found = True
            break
        if not found:
            # Strategy 2: try to find any cycle of length k using chords
            # This is harder; use a simple approach:
            # Try all subsets of chords to see if they can form a cycle
            # For now, try a few random starts
            if k < N:
                # Need a chord that connects two vertices at distance d apart
                # where d < k and the path of length k-1 exists
                # Actually, for a cycle of length k using the base cycle,
                # we need exactly one chord (u,v) where the distance along
                # the cycle between u and v is k-1 (or N-(k-1))
                # So the chord must connect vertices at cyclic distance k-1
                # or equivalently, the chord "skips" k-1 edges
                # Wait, let me reconsider.
                # A cycle of length k in a graph that contains the N-cycle:
                # It uses k edges. If it uses the N-cycle edges, it uses k-1 
                # consecutive N-cycle edges and 1 chord.
                # The chord must connect the endpoints of the k-1 consecutive edges.
                # So if the path is v0, v1, ..., v_{k-1} where v_{i+1} = (v_i+1)%N,
                # then the chord is (v0, v_{k-1}).
                # The cyclic distance from v0 to v_{k-1} is k-1.
                # So we need a chord between vertices at cyclic distance k-1.
                # Let's check if any such chord exists.
                for u in range(N):
                    v = (u + k - 1) % N
                    if frozenset((u, v)) in es:
                        # Build the cycle
                        cyc = [(u + i) % N for i in range(k)]
                        # Verify all edges
                        valid = True
                        for i in range(k):
                            a, b = cyc[i], cyc[(i+1) % k]
                            if frozenset((a, b)) not in es:
                                valid = False
                                break
                        if valid:
                            cycles.append(cyc)
                            found = True
                            break
                    if time.time() > deadline:
                        return None
            if not found:
                return None
    if not _validate(all_edges, cycles):
        return None
    return {'edges': all_edges, 'cycles': cycles}

def _greedy_chord_placement(deadline):
    """
    Greedy: start with base cycle, add chords to cover all cycle lengths.
    Each chord (u,v) with cyclic distance d covers cycle lengths d+1 and N-d+1.
    """
    # For each k in 3..N, we need a chord at cyclic distance k-1 (for k<N)
    # or the full cycle (k=N, no chord needed)
    # Chord at distance d covers: cycle of length d+1 (using d+1 consecutive vertices)
    # and cycle of length N-d+1 (using N-d+1 consecutive vertices the other way)
    
    # What distances do we need?
    # For k=3: need distance 2
    # For k=4: need distance 3
    # ...
    # For k=N-1: need distance N-2
    # For k=N: no chord needed
    
    # A chord at distance d covers cycles of length d+1 and N-d+1
    # So we need to cover all k in 3..N-1 using chords
    # k=3 -> d=2, covers k=3 and k=N-1
    # k=4 -> d=3, covers k=4 and k=N-2
    # etc.
    
    # So we need chords at distances 2, 3, 4, ..., N-2
    # But a chord at distance d is the same as distance N-d
    # So distance d and N-d are the same chord type
    
    # Let's think: we need to cover all k in 3..N-1
    # Chord at distance d covers k=d+1 and k=N-d+1
    # We need: for each k in 3..N-1, there exists a chord at distance k-1 or N-k
    
    # Actually, let's just collect which distances are needed:
    # For k=3: need d=2 (or d=N-2=38, but that's the same chord)
    # For k=4: need d=3
    # ...
    # For k=20: need d=19
    # For k=21: need d=20 (or d=N-21=19, same)
    # For k=39: need d=38 (or d=2, same)
    
    # So we need chords at distances 2, 3, ..., 19 (since 20 covers 21, etc.)
    # Wait: distance d covers k=d+1 and k=N-d+1
    # d=2: covers k=3 and k=39
    # d=3: covers k=4 and k=38
    # ...
    # d=19: covers k=20 and k=22
    # d=20: covers k=21 and k=21 (same)
    
    # So we need distances 2 through 19 (18 chords minimum)
    # Plus we can pick specific vertex pairs
    
    # Greedy: for each needed distance, add a chord
    needed_distances = set()
    for k in range(3, N):
        d = k - 1
        # d and N-d are equivalent
        needed_distances.add(min(d, N - d))
    
    # needed_distances = {2, 3, 4, ..., 19, 20}
    # min(d, N-d) for d=2..38 gives {2,3,...,19,20}
    # So we need 19 distinct chord distances
    
    chords = []
    es = _edge_set(_base_edges())
    
    for d in sorted(needed_distances):
        if time.time() > deadline:
            break
        # Add a chord at cyclic distance d
        # Pick u=0, v=d
        u, v = 0, d
        if frozenset((u, v)) not in es:
            chords.append((u, v))
            es.add(frozenset((u, v)))
    
    result = _try_construct(chords, deadline)
    if result:
        return result
    
    # If greedy with u=0 for all doesn't work, try random placements
    for attempt in range(200):
        if time.time() > deadline:
            break
        chords = []
        es = _edge_set(_base_edges())
        for d in sorted(needed_distances):
            u = random.randint(0, N - 1)
            v = (u + d) % N
            if frozenset((u, v)) not in es:
                chords.append((u, v))
                es.add(frozenset((u, v)))
        result = _try_construct(chords, deadline)
        if result:
            return result
    
    return None

def _local_search(deadline):
    """
    Start from a valid construction, try to remove edges (chords) while maintaining validity.
    """
    base = _greedy_chord_placement(deadline)
    if not base:
        return None
    
    edges = base['edges']
    cycles = base['cycles']
    
    # Try to remove each chord
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Get current chords (non-base edges)
        base_es = _edge_set(_base_edges())
        current_es = _edge_set(edges)
        chords = [e for e in edges if frozenset(e) not in base_es]
        
        for chord in chords:
            if time.time() > deadline:
                break
            # Try removing this chord
            new_edges = [e for e in edges if e != chord and (e[0], e[1]) != (chord[1], chord[0])]
            # Rebuild edge set
            new_es = _edge_set(new_edges)
            
            # Check if all cycles still valid
            valid = True
            for cyc in cycles:
                k = len(cyc)
                for i in range(k):
                    u, v = cyc[i], cyc[(i + 1) % k]
                    if frozenset((u, v)) not in new_es:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                edges = new_edges
                improved = True
                break
    
    if not _validate(edges, cycles):
        return None
    return {'edges': edges, 'cycles': cycles}

def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    # Try local search approach
    result = _local_search(deadline)
    if result:
        return result
    
    # Fallback: return the seed construction
    return {'edges': [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)],
            'cycles': [list(range(k)) for k in range(3, N + 1)]}

