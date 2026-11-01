"""Construct a small 3-regular graph with large girth via random init + girth-preserving 2-switches."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(12345)

    # For k=3, g=16: target n. Moore bound is 512, but we can try a few sizes.
    # The seed program produces ~q*(2^d) vertices which for q=3, d=11 is 3*2048=6144.
    # We want to do better. Let's aim for n around 512-1000.
    # Actually, for 3-regular, the Moore bound for girth 16 is 2*sum_{i=0}^{g/2-1} (k-1)^i = 2*sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510.
    # So n must be even and >= 510. Let's try n=512, 514, ..., up to maybe 2000.
    
    # Strategy:
    # 1. Generate a random 3-regular graph on n vertices.
    # 2. Check girth. If girth < g, try to fix via edge swaps.
    # 3. If we can't fix it quickly, try larger n or restart.
    
    # We'll try a few candidate sizes.
    candidate_sizes = []
    # Start from Moore bound, go up in steps of 2 (must be even for 3-regular).
    moore = 2 * ((1 << (g // 2)) - 1) if g % 2 == 0 else 2 * ((1 << ((g - 1) // 2)) - 1) + (1 << ((g - 1) // 2))
    # For g=16: 2 * (2^8 - 1) = 510.
    # Let's try a range of sizes.
    for n in range(moore, moore + 2000, 2):
        candidate_sizes.append(n)
    
    best_n = None
    best_edges = None
    
    # We will attempt to construct a valid graph for each candidate size, starting from the smallest.
    # If we find a valid graph, we can stop. But we might want to try a few to see if a smaller one is possible?
    # No, we want the *smallest* n. So we should try from the smallest up.
    # However, generating a random 3-regular graph and fixing girth is hard.
    # Maybe we should just try to make a valid graph with a reasonably small n and return it.
    
    # Let's refine: try to find *any* valid graph with n as small as possible.
    # Since 510 is the theoretical minimum, and 936 is the known best, there is a gap.
    # Random 3-regular graphs on ~512 vertices will have many short cycles (girth ~ O(log n)).
    # Girth 16 is very large for n=512.
    # So random init + local search might not work well for small n.
    
    # Alternative: Use the Lazebnik-Ustimenko-Woldar (LUW) construction but try to optimize parameters or find a better family.
    # The seed program uses LUW. Let's stick with a constructive approach but maybe try different q or d?
    # The seed program is fixed to q=k=3.
    # Is there a better known construction?
    # The task says "smallest known such graph has 936 vertices".
    # The seed program produces n = 3 * (2^d - 1) or similar?
    # In the seed: q=3, d=11.
    # Number of points = q^d = 3^11 = 177147? No, the code uses tuples of length d+1 with first element None.
    # Wait, the seed code:
    # start = (0, (None,) + (0,) * d)
    # neighbours generates 1 - side, tuple(y) where y has length d+1.
    # The number of vertices in the connected component of the zero point.
    # For LUW graphs, the size is often q^d or similar.
    # Let's check the size of the seed program's output for k=3, g=16.
    # d = max(3, 16-5=11 if odd else 12). 11 is odd, so d=11.
    # q=3.
    # The graph is bipartite.
    # The number of vertices in the component of 0.
    # This is a known family. The size is typically 2 * q * (q^d - 1) / (q - 1) ? No.
    # Actually, for LUW graphs, the order is often 2 * q^d or similar.
    # Let's assume the seed produces a valid graph with some n.
    # The score is 1.0, meaning it's the baseline.
    # We need to do better (smaller n).
    
    # If we can't beat the seed, we should at least match it.
    # But the directive says "Make a STRUCTURAL change".
    # Let's try a different construction.
    # What about the Biggs-Sims graph? No, that's girth 12.
    # What about the Hoffman-Singleton graph? Girth 5.
    
    # Let's try to implement a "voltage graph" lifting or a different algebraic construction.
    # Or, let's try to optimize the LUW parameters if possible? The seed fixes q=k.
    # Can we use q != k? The interface says "evaluator passes this task's degree k".
    # The graph must be k-regular.
    # LUW graphs are q-regular. So q must be k.
    # So we are stuck with q=3 for k=3.
    
    # Is there another family?
    # Maybe a product graph?
    # Or a random graph with a larger n, but optimized?
    # The known best is 936.
    # If the seed produces, say, 3000 vertices, we can easily beat it by producing a 936-vertex graph.
    # But we don't know the seed's output size.
    # Let's estimate.
    # In the seed, d=11, q=3.
    # The number of vertices is likely around 2 * 3 * (3^11 - 1) / 2 ?
    # Actually, LUW graphs of type CD(d, q) have 2 * q * (q^d - 1) / (q - 1) vertices?
    # No, the standard LUW graph has 2 * q^d vertices?
    # Let's check a reference. The LUW graph for girth 16 and degree 3 is known to have 3 * 2^11 = 6144?
    # If the seed produces 6144 vertices, and we can produce 936, we win big.
    # But we don't have a construction for 936.
    # The 936-vertex graph is from a specific paper (Exoo et al.).
    # We can't just guess its edges.
    
    # So, what can we do?
    # 1. Try to find a better algebraic construction.
    # 2. Try to use a smaller d in LUW?
    #    If we reduce d, the girth reduces.
    #    Girth is roughly d + 5.
    #    For g=16, d=11.
    #    If we use d=10, girth is ~15. Not enough.
    #    So we are stuck with d=11 for LUW.
    
    # 3. Try a different family: The "Kronecker cover" or "double cover"?
    # 4. Try a random search for a 936-vertex graph?
    #    That's infeasible.
    
    # Let's look at the "girth-preserving edge-switching" strategy mentioned in the prompt.
    # "starts from a random 3-regular graph on 512 vertices and iteratively performs 2-switches... to break the shortest existing cycles".
    # This is a local search.
    # For n=512, the expected girth of a random 3-regular graph is small (~6-8).
    # Breaking cycles of length 8 to get to 16 is a huge hill to climb.
    # It's unlikely to work in 120s.
    
    # However, maybe we can start from a graph with a larger girth?
    # The LUW graph has girth 16.
    # Can we "shrink" it?
    # No, shrinking usually reduces girth.
    
    # Let's reconsider the LUW construction.
    # Is there a parameter choice that yields a smaller graph?
    # The seed uses q=k.
    # What if we use a different algebraic structure?
    # For example, using a finite field of size q=3 is fixed.
    
    # Let's try to implement a different known family.
    # The "Tutte 8-cage" is 3-regular, girth 8, 30 vertices.
    # The "Biggs-Sims" is 3-regular, girth 12, 100 vertices.
    # The "Hoffman-Singleton" is 7-regular, girth 5, 50 vertices.
    
    # For girth 16, the next known cages are large.
    # Maybe we can use a "voltage graph" on a smaller base graph?
    # A voltage graph on a 2-vertex base (an edge) with voltage group G of order N yields a 3-regular graph on 2N vertices?
    # If the base is a single edge with 3 parallel edges (a multigraph), the lift is 3-regular.
    # The girth of the lift is related to the girth of the base and the voltages.
    # If the base is a loop or a small graph, the lift can have large girth.
    # This is the "Cayley graph" or "voltage lift" approach.
    
    # Let's try to construct a 3-regular graph as a Cayley graph of a group G with generators {a, a^-1, b} or similar.
    # If G is abelian, G = Z_m x Z_n.
    # Generators: (1, 0), (0, 1), (1, 1)?
    # The girth is the length of the shortest non-trivial relation.
    # We want no relation of length < 16.
    # This means we need a group and generators such that the word metric has no short loops.
    # This is equivalent to finding a "cage" in the Cayley graph.
    
    # Let's try to search for a Cayley graph of an abelian group Z_m x Z_n.
    # n = m * n.
    # We want n to be small.
    # Let's try n = 512, 514, ...
    # For n = 512 = 2^9, Z_512.
    # Generators: 1, -1, and some other element?
    # If we use {1, -1, x}, the graph is 3-regular if x is not +/-1.
    # The girth is the shortest cycle.
    # A cycle of length L corresponds to a word of length L that sums to 0.
    # We want the shortest such word to be >= 16.
    # This is a "girth of a Cayley graph" problem.
    
    # Let's try to find such a generator set for small n.
    # We can brute-force search for generator sets for small n.
    # For each n, try pairs of generators (a, b) (since -a and -b are implied by undirectedness, and we need a 3rd generator c? No, 3-regular means 3 neighbors. In a Cayley graph, the degree is the size of the generating set S. If S = {a, b, c}, degree is 3. But if a=b^-1, it's 2-regular. So we need 3 distinct generators, none of which is the inverse of another, and none is identity.
    # Actually, for an undirected Cayley graph, S must be symmetric: s in S => s^-1 in S.
    # So if we pick a, we must pick a^-1. That's 2 generators.
    # To get degree 3, we need one more generator b such that b = b^-1 (an involution).
    # So S = {a, a^-1, b} with b^2 = e.
    # So we are looking for a group G, an element a, and an involution b, such that the Cayley graph has girth >= 16.
    
    # Let's try G = Z_m.
    # b must be an involution: 2b = 0 mod m. So b = m/2 (if m even).
    # So S = {a, -a, m/2}.
    # We need to find a and m such that the shortest cycle is >= 16.
    # A cycle is a sequence of steps s1, s2, ..., sk such that sum si = 0.
    # The steps are in {a, -a, m/2}.
    # Let the number of +a steps be p, -a steps be n, and m/2 steps be z.
    # Total length L = p + n + z.
    # Sum = (p - n) * a + z * (m/2) = 0 mod m.
    # We want the minimum L >= 16.
    
    # Let's search for m and a.
    # m should be around 500-1000.
    # a should be coprime to m? Not necessarily, but if gcd(a, m) > 1, the graph is disconnected.
    # We want a connected graph, so gcd(a, m) = 1.
    
    # Let's write a function to check the girth of such a Cayley graph.
    # We can use BFS from the identity to find the shortest cycle.
    # Actually, we can just check all possible combinations of p, n, z with p+n+z < 16 and see if any sum to 0.
    # If none do, the girth is at least 16.
    
    # Let's implement this search.
    
    def check_cayley_girth(m, a, limit=16):
        b = m // 2
        # Check all p, n, z with p+n+z < limit
        # (p-n)*a + z*b = 0 mod m
        # z*b = z*m/2.
        # If z is even, z*m/2 is a multiple of m, so (p-n)*a = 0 mod m.
        # Since gcd(a, m)=1, p=n.
        # So if z is even, we need p=n.
        # Length L = 2p + z.
        # Min L for p=n, z even: p=1, z=0 -> L=2 (edge a, -a). Wait, that's a 2-cycle?
        # In a simple graph, we don't have 2-cycles.
        # The edge (0, a) and (a, 0) are the same edge.
        # A cycle of length 2 would mean going 0->a->0. This uses edge (0,a) twice? No, it's a backtrack.
        # Cycles are non-backtracking.
        # So we must exclude sequences where we go back and forth immediately.
        # In terms of p, n, z, a backtrack is a followed by -a.
        # This makes the "word" approach tricky because it counts walks, not cycles.
        # A better way: BFS on the graph, tracking the previous vertex to avoid backtracking.
        
        # Let's do BFS from 0, avoiding the previous vertex.
        # State: (current, previous)
        # Start: (a, 0), (-a, 0), (b, 0)
        # Target: 0 (with previous != start)
        
        from collections import deque
        visited = set()
        queue = deque()
        # Neighbors of 0: a, -a, b
        for start in [a, -a, b]:
            if start % m == 0: continue
            queue.append((start, 0, 1)) # (pos, prev, dist)
            visited.add((start, 0))
        
        while queue:
            pos, prev, dist = queue.popleft()
            if dist >= limit:
                continue
            for step in [a, -a, b]:
                nxt = (pos + step) % m
                if nxt == prev:
                    continue
                if nxt == 0:
                    return dist + 1 # Found a cycle
                if (nxt, pos) not in visited:
                    visited.add((nxt, pos))
                    queue.append((nxt, pos, dist + 1))
        return limit # No cycle found < limit
    
    # Now search for m and a.
    # We want the smallest m such that there exists an a with girth >= 16.
    # m must be even (for b=m/2).
    # m >= 512.
    
    found = False
    for m in range(512, 1000, 2):
        if time.time() > deadline:
            break
        for a in range(1, m, 2): # a must be coprime to m, so if m is even, a must be odd.
            if time.time() > deadline:
                break
            # Check gcd
            if __import__('math').gcd(a, m) != 1:
                continue
            girth = check_cayley_girth(m, a, 16)
            if girth >= 16:
                # Found a valid graph!
                # Construct the edges.
                edges = []
                for v in range(m):
                    for step in [a, -a, b]:
                        u = (v + step) % m
                        if v < u:
                            edges.append((v, u))
                return m, edges
    # If we didn't find one, fall back to the seed program's logic (LUW)
    # But we need to return *something*.
    # Let's just return the seed program's result by reimplementing it here?
    # Or we can just return a random 3-regular graph if we run out of time?
    # No, it must be valid.
    
    # Fallback: LUW construction from the seed.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
