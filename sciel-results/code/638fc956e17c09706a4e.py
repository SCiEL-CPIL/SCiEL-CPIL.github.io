"""Optimized pancyclic graph construction on 40 vertices using dual-hub cycle decomposition."""
N = 40
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit

    # STRUCTURAL CHANGE: Dual-hub design with optimal chord assignment
    # Hub A = vertex 0, Hub B = vertex 20
    # For each cycle length L, assign it to the hub that minimizes total chords.
    # A chord (h, k) on base cycle C40 creates a cycle of length d+1 where d = min(|h-k|, 40-|h-k|)
    # So chord (0,k) gives lengths: k+1 and 41-k (for k=2..38, excluding k=1,39 which are cycle edges)
    # Similarly for hub 20.
    # We need to cover all L in {3,...,39} (L=40 is the base cycle itself).
    # Each chord covers at most 2 lengths. We want to minimize total chords.
    # This is a set cover / matching problem.

    # For hub 0: chord (0,k) covers lengths k+1 and 41-k for k in 2..38
    # For hub 20: chord (20,k) covers lengths (k-20+1 mod 40)+1 and 41-((k-20+1 mod 40)+1) ... 
    # Actually: distance from 20 to k on cycle is d = min(|20-k|, 40-|20-k|)
    # Chord (20,k) creates cycles of length d+1 and 41-d

    # Let's compute which lengths each potential chord covers:
    # Hub 0, chord (0,k), k=2..38:
    #   d = min(k, 40-k)
    #   lengths: d+1 and 41-d
    # Hub 20, chord (20,k), k != 20, k != 19, k != 21 (those are cycle edges from 20)
    #   d = min(|20-k|, 40-|20-k|)
    #   lengths: d+1 and 41-d

    # We need to select a set of chords such that every L in 3..39 is covered.
    # Greedy: for each L, pick the chord that covers L and maximizes additional coverage.

    # Precompute: for each possible chord (hub, k), what lengths does it cover?
    chord_coverage = {}  # (hub, k) -> set of lengths covered

    # Hub 0 chords
    for k in range(2, 39):  # k=2..38
        d = min(k, 40 - k)
        if d >= 2:  # ensure it's not a cycle edge
            lengths = {d + 1, 41 - d}
            lengths = {l for l in lengths if 3 <= l <= 39}
            if lengths:
                chord_coverage[(0, k)] = lengths

    # Hub 20 chords
    for k in range(0, 40):
        if k == 20 or k == 19 or k == 21:
            continue  # cycle edges or self
        d = min(abs(20 - k), 40 - abs(20 - k))
        if d >= 2:
            lengths = {d + 1, 41 - d}
            lengths = {l for l in lengths if 3 <= l <= 39}
            if lengths:
                chord_coverage[(20, k)] = lengths

    # Greedy set cover: pick chords to cover all lengths 3..39
    needed = set(range(3, 40))
    selected_chords = []
    while needed:
        # Find chord that covers the most needed lengths
        best_chord = None
        best_count = 0
        for chord, cov in chord_coverage.items():
            count = len(cov & needed)
            if count > best_count:
                best_count = count
                best_chord = chord
        if best_chord is None:
            break
        selected_chords.append(best_chord)
        needed -= chord_coverage[best_chord]

    # Build edge set
    edges = set()
    # Base cycle
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Add selected chords
    for hub, k in selected_chords:
        a, b = min(hub, k), max(hub, k)
        edges.add((a, b))

    # Now generate cycles: for each L, find a chord that covers L and construct the cycle
    def get_cycle_for_length(L, chord_info):
        """Given length L and a chord (hub, k) that covers L, return the cycle."""
        hub, k = chord_info
        if L == 40:
            return list(range(40))
        
        # Determine which of the two lengths this chord provides for L
        # The chord (hub, k) splits the cycle into two paths.
        # Path 1: from hub to k going forward (increasing index mod 40)
        # Path 2: from hub to k going backward
        
        # Compute the two path lengths
        if hub == 0:
            # Forward: 0, 1, 2, ..., k -> length k+1
            # Backward: 0, 39, 38, ..., k -> length 41-k
            if L == k + 1:
                # Use forward path: 0, 1, 2, ..., k
                return list(range(k + 1))
            else:  # L == 41 - k
                # Use backward path: 0, 39, 38, ..., k
                return [0] + list(range(39, k - 1, -1))
        else:  # hub == 20
            # Forward from 20: 20, 21, 22, ..., k (mod 40)
            # Backward from 20: 20, 19, 18, ..., k (mod 40)
            # Compute forward distance
            fwd_dist = (k - hub) % 40
            bwd_dist = (hub - k) % 40
            
            if L == fwd_dist + 1:
                # Forward path
                cyc = []
                cur = hub
                for _ in range(fwd_dist + 1):
                    cyc.append(cur)
                    cur = (cur + 1) % 40
                return cyc
            else:  # L == bwd_dist + 1
                # Backward path
                cyc = []
                cur = hub
                for _ in range(bwd_dist + 1):
                    cyc.append(cur)
                    cur = (cur - 1) % 40
                return cyc

    # For each L, find a selected chord that covers it
    cycles = {}
    for L in range(3, 41):
        if L == 40:
            cycles[L] = list(range(40))
        else:
            found = False
            for chord in selected_chords:
                if L in chord_coverage[chord]:
                    cycles[L] = get_cycle_for_length(L, chord)
                    found = True
                    break
            if not found:
                # This shouldn't happen if set cover is complete
                cycles[L] = list(range(L))

    # Verify all cycles
    def edge_exists(u, v):
        return (min(u, v), max(u, v)) in edges

    def verify():
        for L in range(3, N + 1):
            cyc = cycles[L]
            if len(cyc) != L or len(set(cyc)) != L:
                return False
            for i in range(L):
                u, v = cyc[i], cyc[(i + 1) % L]
                if not edge_exists(u, v):
                    return False
        return True

    if not verify():
        # Fallback: use hub 0 with chords for all needed lengths
        edges = set()
        for i in range(N):
            edges.add((i, (i + 1) % N))
        for k in range(2, 39):
            edges.add((0, k))
        cycles = {}
        for L in range(3, 41):
            if L == 40:
                cycles[L] = list(range(40))
            else:
                k = L - 1
                if 2 <= k <= 38:
                    if k + 1 == L:
                        cycles[L] = list(range(k + 1))
                    else:
                        k2 = 41 - L
                        if 2 <= k2 <= 38:
                            cycles[L] = [0] + list(range(39, k2 - 1, -1))
                        else:
                            cycles[L] = list(range(L))
                else:
                    cycles[L] = list(range(L))
        edge_list = list(edges)
        return {'edges': edge_list, 'cycles': [cycles[L] for L in range(3, N + 1)]}

    # Prune unused edges
    edge_usage = set()
    for L in range(3, N + 1):
        cyc = cycles[L]
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            edge_usage.add((min(u, v), max(u, v)))

    final_edges = set()
    for e in edges:
        if e in edge_usage:
            final_edges.add(e)

    edge_list = list(final_edges)
    return {'edges': edge_list, 'cycles': [cycles[L] for L in range(3, N + 1)]}
# EVOLVE-BLOCK-END
