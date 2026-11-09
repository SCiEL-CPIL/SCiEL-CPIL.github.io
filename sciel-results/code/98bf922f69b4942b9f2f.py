"""Sparse pancyclic graphs on 40 vertices: dual-hub decomposition."""
import time

N = 40


def _verify_cyc(cyc, edge_set):
    """Check a cycle: distinct vertices, all edges present."""
    k = len(cyc)
    if len(set(cyc)) != k:
        return False
    for i in range(k):
        u, v = cyc[i], cyc[(i + 1) % k]
        if (min(u, v), max(u, v)) not in edge_set:
            return False
    return True


def _build_dual_hub_cycles(edge_set):
    """Build cycles for dual-hub (0 and 20) construction."""
    cycs = []
    cset = set(edge_set)
    for k in range(3, N + 1):
        if k == N:
            c = list(range(N))
        elif k <= 19:
            # Use hub 0: cycle 0, 1, 2, ..., k-1, 0
            c = list(range(k))
        elif k <= 38:
            # Use hub 20: cycle 20, 21, ..., 19+k, 20
            # Vertices: 20, 21, ..., 19+k (that's k vertices)
            # Edges: (20,21), (21,22), ..., (18+k,19+k), (19+k,20)
            # Rim edges cover consecutive pairs
            # Need chord (19+k, 20)
            c = list(range(20, 20 + k))
            if c[-1] >= 40:
                # Wrap around: use modular arithmetic
                c = [20] + list(range(21, 20 + k))
        else:
            # k == 39 or 40 handled above
            c = list(range(N))
        
        if not _verify_cyc(c, cset):
            return None
        cycs.append(c)
    return cycs


def construct():
    t0 = time.time()
    DL = t0 + 70.0

    # ---- Dual-hub construction ----
    # Rim: 40 edges (cycle 0-1-2-...-39-0)
    rim = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        rim.add((min(a, b), max(a, b)))

    # Hub 0 chords: connect 0 to vertices 2, 3, ..., 18 (17 chords)
    # This gives cycles of length 3 through 19 using hub 0
    hub0_chords = set()
    for d in range(2, 19):
        hub0_chords.add((0, d))

    # Hub 20 chords: connect 20 to vertices 22, 23, ..., 38 (17 chords)
    # This gives cycles of length 3 through 19 using hub 20
    hub20_chords = set()
    for d in range(2, 19):
        hub20_chords.add((20, 20 + d))

    # Bridge between hubs: connect 0 and 20
    bridge = (0, 20)

    # Total: 40 rim + 17 + 17 + 1 = 75 edges (worse than 58, but let's verify)
    # Actually, we can be smarter about which chords we need

    # Let's try a more efficient dual-hub approach:
    # Hub 0 handles cycles 3-20 (needs chords to 2..19, that's 18 chords)
    # Hub 20 handles cycles 21-40 (needs different structure)
    
    # Alternative: Use the rim more cleverly
    # For cycle of length k using hub h:
    # If k <= 20: h, h+1, h+2, ..., h+k-1, h (needs chord (h+k-1, h))
    # If k > 20: h, h-1, h-2, ..., h-k+1, h (needs chord (h-k+1, h))
    
    # Let's try: hub 0 for small cycles, hub 20 for large cycles
    # Hub 0: cycles 3-20 need chords (0,2), (0,3), ..., (0,19) = 18 chords
    # Hub 20: cycles 21-40 need chords (20,19), (20,18), ..., (20,2) = 18 chords
    # But (20,19) and (20,2) are rim edges! So we only need chords for non-rim connections
    
    # For hub 20, cycle of length k: 20, 19, 18, ..., 20-k+1, 20
    # Edges: (20,19) rim, (19,18) rim, ..., (21-k+1, 20-k+1) rim, (20-k+1, 20) chord
    # So we need chord (20-k+1, 20) for each k
    # For k=21: chord (0, 20) - this is the bridge!
    # For k=22: chord (39, 20) - not rim
    # For k=23: chord (38, 20) - not rim
    # ...
    # For k=40: chord (20, 20) - invalid, use Hamiltonian cycle instead
    
    # Let's construct and verify
    edges = set(rim)
    
    # Hub 0 chords for cycles 3-19 (k=3: chord (0,2), k=4: chord (0,3), ..., k=19: chord (0,18))
    for k in range(3, 20):
        edges.add((0, k))
    
    # Hub 20 chords for cycles 20-39
    # k=20: cycle 20,19,...,1,20 needs chord (1,20)
    # k=21: cycle 20,19,...,0,20 needs chord (0,20)
    # k=22: cycle 20,19,...,39,20 needs chord (39,20)
    # ...
    # k=39: cycle 20,19,...,21,20 needs chord (21,20) - rim edge!
    # k=40: Hamiltonian cycle
    
    for k in range(20, 40):
        # Cycle: 20, 19, 18, ..., (20-k+1 mod 40), 20
        # Last vertex before returning to 20 is (20-k+1) mod 40
        last = (20 - k + 1) % 40
        if last != 19 and last != 21:  # Not rim edges
            edges.add((min(last, 20), max(last, 20)))
    
    # Add bridge if not already there
    edges.add((0, 20))
    
    # Build and verify cycles
    cset = set(edges)
    cycs = []
    valid = True
    for k in range(3, N + 1):
        if k == N:
            c = list(range(N))
        elif k <= 19:
            # Hub 0: 0, 1, 2, ..., k-1, 0
            c = list(range(k))
        elif k == 20:
            # Hub 20: 20, 19, 18, ..., 1, 20
            c = [20] + list(range(19, 0, -1))
        else:
            # Hub 20: 20, 19, ..., (20-k+1 mod 40), 20
            c = [20]
            v = 19
            for _ in range(k - 2):
                c.append(v)
                v = (v - 1) % 40
            # Last vertex should connect back to 20
            if c[-1] != (20 - k + 1) % 40:
                valid = False
                break
        
        if not _verify_cyc(c, cset):
            valid = False
            break
        cycs.append(c)
    
    if valid:
        best_edges = sorted(edges)
        best_cycs = cycs
    else:
        # Fallback to original 58-edge construction
        rim2 = set()
        for i in range(N):
            a, b = i, (i + 1) % N
            rim2.add((min(a, b), max(a, b)))
        
        fan_chords = set()
        for d in range(2, 19):
            fan_chords.add((0, d))
        
        bridge2 = (2, 21)
        e58 = frozenset(rim2 | fan_chords | {bridge2})
        
        best_edges = sorted(e58)
        best_cycs = []
        for k in range(3, N + 1):
            if k == N:
                c = list(range(N))
            elif k <= 19:
                c = list(range(k))
            elif k == 20:
                c = list(range(2, 22))
            elif k == 21:
                c = [0, 2] + list(range(21, 40))
            elif k == 22:
                c = [2, 1, 0] + list(range(39, 20, -1))
            else:
                dp = 41 - k
                c = [0] + list(range(39, dp - 1, -1))
            best_cycs.append(c)

    # Final validation
    final_es = set(best_edges)
    assert len(final_es) == len(best_edges), "Duplicate edges"
    for c in best_cycs:
        assert _verify_cyc(c, final_es), f"Invalid cycle: {c}"

    return {'edges': best_edges, 'cycles': best_cycs}
