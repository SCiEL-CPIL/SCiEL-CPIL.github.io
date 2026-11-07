import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        num_lift_vertices = 2 * n
        parent = list(range(num_lift_vertices))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                union(u * 2, v * 2)
                union(u * 2 + 1, v * 2 + 1)
            else:
                union(u * 2, v * 2 + 1)
                union(u * 2 + 1, v * 2)
        root = find(0)
        for i in range(1, num_lift_vertices):
            if find(i) != root:
                return False
        return True
    
    # The prism graph has m top cycle edges (indices 0..m-1),
    # m bottom cycle edges (indices m..2m-1), and m matching edges (indices 2m..3m-1).
    # Total edges = 3m. Number of independent cycles = 3m - 2m + 1 = m + 1.
    # 
    # Key insight: the spectrum of the signed adjacency matrix is invariant under
    # switching (conjugation by a diagonal ±1 matrix). So we only need to check
    # 2^(m+1) switching classes.
    #
    # We can parameterize: fix the matching edges to +1 (one representative per
    # switching class that has matching edges fixed), and vary the signs on a
    # spanning tree complement. Actually, let's think more carefully.
    #
    # A switching is determined by a vertex labeling s: V -> {±1}. The switched
    # sign on edge (u,v) is s[u]*s[v]*original_sign[u,v]. Two signings are in
    # the same switching class iff their ratio is a switching.
    #
    # The number of distinct switching classes = 2^(E - V + 1) = 2^(3m - 2m + 1) = 2^(m+1).
    #
    # To enumerate: fix a spanning tree, and vary the signs on the non-tree edges.
    # The tree edges are then determined by the requirement that the product around
    # each fundamental cycle gives the class.
    #
    # Simpler approach: just enumerate all 2^(m+1) possible patterns on a set of
    # m+1 "free" edges, and for each, compute the full signing that represents
    # that switching class (by fixing a reference signing and applying switchings).
    #
    # Even simpler: since we just need to find the min spectral radius over all
    # signings in each switching class, and the spectrum is the same for all
    # signings in a class, we just need to pick ONE signing per class and check
    # if it gives a connected lift.
    #
    # Let's use: fix all matching edges to +1. Then the remaining freedom is
    # in the top and bottom cycles. The top cycle has m edges, bottom has m edges.
    # But they're connected through the matching edges. 
    #
    # Actually, let me just do brute force over all 2^(m+1) switching classes
    # by fixing a canonical form. The simplest: enumerate all sign patterns on
    # the m top cycle edges and m bottom cycle edges, but mod out by switchings.
    #
    # Cleanest approach: enumerate all 2^(3m) signings, but group by switching class.
    # For m=9, that's 2^27 = 134M which is too slow.
    #
    # Better: the switching class is determined by the products of signs around
    # each independent cycle. There are m+1 independent cycles. Let's pick:
    # - m cycles: each top cycle edge + corresponding matching + bottom matching + bottom cycle edge back
    #   Actually, let's use: for each i, the cycle: top_i -> top_{i+1} -> bottom_{i+1} -> bottom_i -> top_i
    #   That's m cycles. Plus one more: the full top cycle or full bottom cycle.
    #
    # Let me just enumerate all 2^(m+1) patterns on m+1 specific edges and compute
    # the unique signing in that class that has all matching edges = +1.
    
    # Identify edge indices:
    # Edges are: top cycle (0..m-1), bottom cycle (m..2m-1), matching (2m..3m-1)
    # Let me verify by checking the edge list structure
    
    top_edges = list(range(m))       # indices 0 to m-1
    bottom_edges = list(range(m, 2*m))  # indices m to 2m-1
    match_edges = list(range(2*m, 3*m)) # indices 2m to 3m-1
    
    # For each switching class (specified by m+1 cycle products), find the signing
    # where all matching edges are +1.
    #
    # The m+1 independent cycles I'll use:
    # For i in 0..m-1: cycle_i goes through top edge i, matching at (i+1)%m, 
    #   bottom edge (i+1)%m (reversed), matching at i. 
    #   Hmm, this is getting complicated. Let me use a different approach.
    
    # Alternative: just enumerate all 2^(m+1) switching classes by varying
    # vertex potentials. A switching is determined by s: V -> {±1}. 
    # s has 2m vertices, but s and -s give the same switching. So 2^(2m-1) switchings.
    # But the number of switching classes is 2^(m+1), so each class has 
    # 2^(2m-1)/2^(m+1) = 2^(m-2) signings... wait that doesn't work for small m.
    
    # Let me just be direct: enumerate all 2^(3m) possible sign assignments? No, too slow for m=9.
    
    # OK, simplest correct approach: 
    # The spectrum depends only on the switching class. The switching class is 
    # determined by the product of signs around each cycle in a cycle basis.
    # For the prism graph, a cycle basis consists of:
    # - The m "ladder" cycles: for each i, the 4-cycle top_i - top_{i+1} - bot_{i+1} - bot_i
    # - The top m-cycle
    # - The bottom m-cycle
    # But these aren't independent. The m ladder cycles + top cycle determine the bottom cycle.
    # So we have m+1 independent cycle products.
    
    # Let me define: for signing s, the "invariant" is the vector of products around
    # m+1 specific cycles. Two signings are in the same class iff they have the same invariants.
    
    # Ladder cycle i (for i=0..m-1): edges are top_i, match_{i+1}, bottom_{i+1} (reversed), match_i
    # In terms of edge indices:
    #   top edge i: index i
    #   matching from (i+1)%m to (i+1)%m + m: index 2*m + (i+1)%m  
    #   bottom edge from (i+1)%m+m to i+m: this is bottom edge (i+1)%m but reversed, index m + (i+1)%m
    #   matching from i to i+m: index 2*m + i
    # Product: s[i] * s[2m + (i+1)%m] * s[m + (i+1)%m] * s[2m + i]
    
    # Top cycle: product of s[0]*s[1]*...*s[m-1] (with appropriate orientation)
    # Bottom cycle: product of s[m]*s[m+1]*...*s[2m-1]
    
    # Let me just compute the invariants for a given signing and use them to identify classes.
    
    def cycle_invariants(signs):
        """Return the m+1 cycle product invariants."""
        invariants = []
        # m ladder cycles
        for i in range(m):
            ip = (i + 1) % m
            # Cycle: top_i -> top_{ip} -> bot_{ip} -> bot_i -> top_i
            # Edges: top_i (forward), match_ip (top_ip to bot_ip), bottom_ip (bot_ip to bot_i, reversed), match_i (bot_i to top_i, reversed)
            # Sign contribution: s[top_i] * s[match_ip] * s[bottom_ip] * s[match_i]
            # (Reversing an edge doesn't change the sign in the product since we're just multiplying)
            prod = signs[i] * signs[2*m + ip] * signs[m + ip] * signs[2*m + i]
            invariants.append(prod)
        # Top cycle: product of all top edges in order
        top_prod = 1
        for i in range(m):
            top_prod *= signs[i]
        invariants.append(top_prod)
        # Bottom cycle
        bot_prod = 1
        for i in range(m):
            bot_prod *= signs[m + i]
        invariants.append(bot_prod)
        # Note: bottom_prod = top_prod * (product of all ladder invariants) in some sense,
        # but let's keep all m+2 and note that one is redundant. Actually for m+1 independent,
        # let's use the m ladder invariants + top cycle invariant.
        return invariants[:m+1]  # m ladder + top cycle
    
    # Now, for each of the 2^(m+1) possible invariant patterns, find one signing 
    # that realizes it AND has all matching edges = +1, then check connectivity.
    
    # Given a target invariant pattern, can we find a signing with matching edges = +1?
    # If matching edges are all +1, then:
    #   Ladder invariant i: signs[i] * 1 * signs[m+ip] * 1 = signs[i] * signs[m+ip]
    #   Top invariant: product of signs[i] for i=0..m-1
    # So we need: signs[i] * signs[m+(i+1)%m] = target_ladder_i for all i
    #   and product(signs[i]) = target_top
    
    # From the ladder constraints: signs[m+ip] = target_ladder_i * signs[i]
    # So signs[m+0] = target_ladder_{m-1} * signs[m-1]
    #    signs[m+1] = target_ladder_0 * signs[0]
    #    signs[m+2] = target_ladder_1 * signs[1]
    #    ...
    #    signs[m+i] = target_ladder_{i-1} * signs[i-1] (with i-1 mod m)
    
    # So once we pick signs[0], all other signs are determined:
    # signs[i] for i=1..m-1 are free? No, they're constrained by the ladder equations.
    # Actually: signs[m+ip] = target_ladder_i * signs[i]
    # This determines the bottom signs from the top signs. So the top signs (signs[0..m-1])
    # are free, but constrained by: product(signs[i]) = target_top
    # And we also need consistency: the bottom cycle product must match.
    # product(signs[m+i]) = product(target_ladder_{i-1} * signs[i-1]) = (product of target_ladders) * (product of signs)
    # The bottom cycle invariant should equal product(signs[m+i]).
    # But we only specified m+1 invariants (m ladders + top), so the bottom is determined.
    
    # So: for a given (target_ladders, target_top), the top signs must satisfy:
    #   product(signs[i]) = target_top
    # And signs[i] ∈ {±1}. The number of free variables is m-1 (fix m-1, the last is determined).
    # So for each invariant pattern, there are 2^(m-1) signings with matching=+1.
    # But they're all in the same switching class! So we just need to pick ONE.
    
    # Pick: signs[0] = +1, signs[1] = +1, ..., signs[m-2] = +1, signs[m-1] = target_top
    # Then compute bottom signs from ladder constraints.
    
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^(m+1) invariant patterns
    for pattern_idx in range(2 ** (m + 1)):
        if time.time() > deadline - 5:
            break
        
        # Decode pattern
        bits = [(pattern_idx >> k) & 1 for k in range(m + 1)]
        target_ladders = [1 if b == 0 else -1 for b in bits[:m]]
        target_top = 1 if bits[m] == 0 else -1
        
        # Construct a signing with matching edges = +1
        signs = np.zeros(num_edges, dtype=int)
        # Top edges
        for i in range(m - 1):
            signs[i] = 1
        # Last top edge determined by top cycle product
        signs[m - 1] = target_top
        
        # Bottom edges from ladder constraints: signs[m+ip] = target_ladder_i * signs[i]
        for i in range(m):
            ip = (i + 1) % m
            signs[m + ip] = target_ladders[i] * signs[i]
        
        # Matching edges = +1
        for i in range(m):
            signs[2*m + i] = 1
        
        # Check connectivity
        if not is_connected_lift(signs):
            continue
        
        # Compute spectral radius
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
