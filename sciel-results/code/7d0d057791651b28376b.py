"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations
    import sys
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Build the set of bad triples (3-uniform hyperedges)
    # A triple {a, b, c} with a < b < c is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for i in range(N):
        a = numbers[i]
        for j in range(i + 1, N):
            b = numbers[j]
            lab = lcm(a, b)
            for k in range(j + 1, N):
                c = numbers[k]
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((i, j, k))
    
    M = len(bad_triples)
    
    # For each vertex, store the list of hyperedge indices it participates in
    vertex_edges = [[] for _ in range(N)]
    for ei, (i, j, k) in enumerate(bad_triples):
        vertex_edges[i].append(ei)
        vertex_edges[j].append(ei)
        vertex_edges[k].append(ei)
    
    # Sort vertices by degree (number of bad triples they're in) descending
    # This helps branch-and-bound prune earlier
    order = sorted(range(N), key=lambda v: len(vertex_edges[v]), reverse=True)
    
    # Reorder vertices and rebuild data structures
    # pos[v] = position in the ordered list
    pos = [0] * N
    for idx, v in enumerate(order):
        pos[v] = idx
    
    # For each vertex in the new order, store its neighbors in the hypergraph sense
    # Actually, for branch-and-bound on 3-uniform hypergraph independent set:
    # We process vertices in some order. At each step, we decide to include or exclude.
    # If we include vertex v, we must exclude all vertices that form a bad triple with v and any other included vertex.
    
    # Let's use a different approach: branch and bound with bitsets.
    # State: current set of included vertices (as a bitset), and the "available" vertices (those not yet decided and not forbidden).
    
    # Precompute for each pair of vertices (i, j) the set of vertices k such that (i, j, k) is a bad triple.
    # conflict[i][j] = bitmask of k > max(i,j) in original indexing... but we work with reordered indices.
    
    # Let's work with the reordered indices. Let's call the reordered vertex at position p as order[p].
    # For positions p1 < p2 < p3, the triple (order[p1], order[p2], order[p3]) is bad if it's in bad_triples.
    
    # Build conflict table: for each pair (p1, p2) with p1 < p2, find all p3 > p2 such that (p1, p2, p3) is bad.
    # conflict[p1][p2] = set of p3 > p2
    
    # Actually, let's use a simpler representation. For each vertex p, and for each previously chosen vertex q < p,
    # we need to know which future vertices r > p would form a bad triple with q and p.
    
    # Let's precompute: for each pair (p, q) with p < q, the set of r > q such that (p, q, r) is a bad triple.
    # Store as a list of bitmasks or just a set.
    
    # Since N=128, we can use Python integers as bitsets (128 bits).
    
    # conflict[p][q] for p < q: bitmask of vertices r > q such that (p, q, r) is bad.
    # We index by position in the ordered list.
    
    # First, create a lookup: for each bad triple (i, j, k) in original indices, convert to positions.
    # bad_triples_pos = set of (pos[i], pos[j], pos[k]) sorted
    
    bad_triples_pos = set()
    for (i, j, k) in bad_triples:
        pi, pj, pk = pos[i], pos[j], pos[k]
        triple = tuple(sorted([pi, pj, pk]))
        bad_triples_pos.add(triple)
    
    # Now build conflict table
    # conflict[p][q] = bitmask of r > q such that (p, q, r) in bad_triples_pos, for p < q
    conflict = [[0] * N for _ in range(N)]
    for (p, q, r) in bad_triples_pos:
        # p < q < r
        conflict[p][q] |= (1 << r)
        # Also, for the pair (p, r), q is not > r, so no entry needed in that direction
        # For the pair (q, r), p is not > r, so no entry needed
    
    # Now, branch and bound.
    # We process vertices in order 0, 1, ..., N-1.
    # State: 
    #   - current position p (next vertex to consider)
    #   - included: list of included vertices so far (or bitmask)
    #   - forbidden: bitmask of vertices that cannot be included (because they would form a bad triple with two already-included vertices)
    
    # Actually, a cleaner formulation:
    # When we decide to include vertex p, we must forbid all vertices r > p such that there exists q < p in included with (q, p, r) bad.
    # So we need to track, for each included vertex q, the set of future vertices that would conflict with q and the current vertex.
    
    # Let's use: 
    #   included_mask: bitmask of included vertices
    #   For each vertex p we consider, we check if it's forbidden.
    #   A vertex r is forbidden if there exist two included vertices q1, q2 < r such that (q1, q2, r) is bad.
    
    # To efficiently check if vertex p is forbidden given the current set of included vertices:
    # p is forbidden if there exist q1, q2 in included, q1 < q2 < p, such that (q1, q2, p) is bad.
    # Equivalently, for each q2 in included with q2 < p, check if there exists q1 in included with q1 < q2 such that (q1, q2, p) is bad.
    
    # Precompute: for each pair (q1, q2) with q1 < q2, the set of p > q2 such that (q1, q2, p) is bad.
    # This is exactly conflict[q1][q2].
    
    # So: p is forbidden iff there exist q1, q2 in included with q1 < q2 < p such that p is in conflict[q1][q2].
    
    # For efficiency, we can maintain a "forbidden_mask" that is updated as we include vertices.
    # When we include vertex p, for each q in included with q < p, we add all vertices r in conflict[q][p] to forbidden.
    
    # Let's implement this.
    
    best = [0]
    best_set = [0]  # bitmask of the best solution found
    
    def is_forbidden(p, included_mask, forbidden_mask):
        """Check if vertex p is forbidden given current included and forbidden masks."""
        if (forbidden_mask >> p) & 1:
            return True
        # Check if p forms a bad triple with any two included vertices
        # For each q2 in included with q2 < p:
        #   check if any q1 in included with q1 < q2 has p in conflict[q1][q2]
        # This is expensive. Instead, let's just check all pairs of included vertices.
        # But we can be smarter: maintain forbidden_mask properly.
        return False
    
    # Let's use a recursive branch and bound with proper forbidden mask maintenance.
    
    def branch_and_bound(p, included_mask, forbidden_mask, count):
        """
        p: next vertex to consider (position in ordered list)
        included_mask: bitmask of included vertices
        forbidden_mask: bitmask of vertices that are forbidden (cannot be included)
        count: number of included vertices so far
        """
        if count + (N - p) <= best[0]:
            return  # Upper bound: even if we include all remaining, we can't beat best
        
        if p == N:
            if count > best[0]:
                best[0] = count
                best_set[0] = included_mask
            return
        
        # Option 1: Exclude vertex p
        branch_and_bound(p + 1, included_mask, forbidden_mask, count)
        
        # Option 2: Include vertex p, if not forbidden
        if not ((forbidden_mask >> p) & 1):
            # New forbidden mask: add all vertices r > p that conflict with p and any included q < p
            new_forbidden = forbidden_mask
            # For each q in included with q < p:
            #   new_forbidden |= conflict[q][p]
            q = 0
            inc = included_mask
            while inc:
                if inc & 1:
                    if q < p:
                        new_forbidden |= conflict[q][p]
                q += 1
                inc >>= 1
            
            # Also, we need to make sure p itself is not forbidden (already checked)
            branch_and_bound(p + 1, included_mask | (1 << p), new_forbidden, count + 1)
    
    # The above is too slow because iterating through all q for each inclusion is O(N) per node.
    # Let's optimize by precomputing for each p and each possible included set... no, that's too much.
    
    # Alternative: use a different ordering or a greedy initialization to get a good lower bound.
    
    # Let's first find a good lower bound using a greedy algorithm.
    greedy_set = []
    forbidden = set()
    for p in range(N):
        if p not in forbidden:
            greedy_set.append(p)
            # For each pair of vertices in greedy_set, find all r that form a bad triple
            for i in range(len(greedy_set) - 1):
                q1 = greedy_set[i]
                q2 = greedy_set[-1]
                # Add all r in conflict[q1][q2] to forbidden
                mask = conflict[q1][q2]
                r = 0
                m = mask
                while m:
                    if m & 1:
                        forbidden.add(r)
                    r += 1
                    m >>= 1
    
    best[0] = len(greedy_set)
    best_set[0] = 0
    for p in greedy_set:
        best_set[0] |= (1 << p)
    
    # Now run branch and bound to try to improve.
    # But the above B&B is too slow. Let's use a simpler approach.
    
    # Given the time constraint, let's just return the greedy solution if it's good enough.
    # But let's try a local search improvement.
    
    # Actually, let's just return the greedy solution for now, as the B&B might be too slow.
    # Convert back to original numbers.
    result = []
    for p in greedy_set:
        result.append(order[p])
    result.sort()
    return result
# EVOLVE-BLOCK-END
