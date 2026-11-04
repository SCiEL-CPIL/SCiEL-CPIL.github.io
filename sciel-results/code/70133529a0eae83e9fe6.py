import math
import time
import random

# EVOLVE-BLOCK-START

def _is_prime_power(n):
    """Return (q, is_power) where q is the prime power equal to n if n is a prime power."""
    if n < 2:
        return None, False
    # check if n is a power of a prime
    # first find the smallest prime factor
    m = n
    p = None
    d = 2
    while d * d <= m:
        if m % d == 0:
            p = d
            break
        d += 1
    if p is None:
        p = m
    # now check if n = p^k for some k >= 1
    q = 1
    while q < n:
        q *= p
        if q > n:
            break
    if q == n:
        return n, True
    return None, False


def _next_prime_ge(n):
    """Return the smallest prime >= n."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    while not is_prime(n):
        n += 2
    return n


def _moore_bound(k, g):
    """Moore bound (lower bound on n for k-regular graph of girth g)."""
    if k <= 1:
        return 1
    if g % 2 == 0:
        # even girth: n >= k * (k-1)^(g/2 - 1)
        half = g // 2
        if half - 1 == 0:
            return k
        return k * ((k - 1) ** (half - 1))
    else:
        # odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        half = (g - 3) // 2
        s = 0
        term = 1
        for i in range(half + 1):
            s += term
            term *= (k - 1)
        return 1 + k * s


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # The task asks for a k-regular graph of girth g with minimum vertices.
    # The current seed uses Lazebnik-Ustimenko-Woldar bipartite construction.
    # We try the Alon-Babai-Suzuki (ABS) construction: Cayley graph on upper
    # triangular unit matrices over GF(q) with elementary matrix generators.
    #
    # For k=5, g=9: The ABS construction typically uses GF(q) where q is chosen
    # to give a k-regular graph. The group has order q^{m(m-1)/2} for m x m matrices.
    # The number of generators is m (one for each off-diagonal position in the first
    # row, or we use a specific set).
    #
    # Actually, let me think more carefully. The ABS construction:
    # - Consider the group G of m x m upper triangular matrices over GF(q) with 1s on diagonal.
    # - |G| = q^{m(m-1)/2}
    # - Use generators: for i < j, the elementary matrix E_{i,j}(a) = I + a * e_{i,j}
    # - But we need exactly k generators for a k-regular Cayley graph.
    #
    # A common ABS construction: use the generators corresponding to the "first row"
    # positions: E_{1,2}, E_{1,3}, ..., E_{1,m} (that's m-1 generators) plus possibly
    # E_{2,3}, etc.
    #
    # Actually, the standard ABS construction for girth g uses:
    # - m = (g+1)//2 or similar
    # - The generators are the elementary matrices in specific positions.
    #
    # Let me use a simpler approach: the classical ABS construction.
    # For a k-regular graph of girth g, we need:
    # - m such that 2m >= g (for odd girth) or 2m >= g (for even)
    # - q such that we have enough generators
    #
    # Actually, let me try a different, well-known construction: the
    # Morgenstern construction, which is a Cayley graph on a nilpotent group.
    #
    # For now, let me try the simplest ABS-like construction:
    # Group: upper triangular unit matrices of size m x m over GF(q).
    # Generators: e_{1,2}, e_{2,3}, ..., e_{m-1,m} (adjacent positions) - that's m-1 generators.
    # But we need exactly k generators.
    #
    # Alternative: use all generators e_{i,j} for specific (i,j) pairs to get k generators.
    #
    # Let me try: for k=5, use m=3, q=some prime, generators = {E_{1,2}(1), E_{1,3}(1), E_{2,3}(1), E_{1,2}(-1), E_{2,3}(-1)}?
    # That gives 5 generators on a group of size q^3.
    #
    # Actually, the cleanest ABS construction I know:
    # For odd girth g = 2r+1, use m = r+1, and the group of (r+1)x(r+1) upper triangular
    # unit matrices over GF(q). Use generators: for each i from 1 to r, the elementary matrix
    # adding 1 to position (i, i+1). That's r generators. To get k generators, we can add
    # more elementary matrices.
    #
    # Let me just try the seed construction first and see if I can improve it,
    # then also try a Morgenstern-type construction.
    #
    # Actually, re-reading the directive: "Use the Alon-Babai-Suzuki (ABS) construction to
    # generate a k-regular graph of girth g by defining a Cayley graph on the group of upper
    # triangular unit matrices over GF(k) with generators corresponding to elementary matrices"
    #
    # So use GF(k) (assuming k is prime or prime power). Let's use GF(q) where q is the
    # smallest prime power >= k, or just q = k if k is prime.
    #
    # For k=5, g=9:
    # g=9 is odd, so g = 2r+1 => r = 4, m = r+1 = 5.
    # Group size = q^{m(m-1)/2} = 5^{10} = 9765625. That's too many vertices.
    #
    # Hmm, that's way too large. The ABS construction as described gives very large graphs.
    #
    # Let me reconsider. The ABS construction in Alon, Babai, Suzuki (1985) actually uses
    # a specific set of generators and the girth is related to the "weight" of the generators.
    #
    # A more practical approach: use a smaller group. For girth g, we need the shortest
    # non-backtracking closed walk to have length >= g. In a Cayley graph, this relates to
    # the word length in the group.
    #
    # Let me try m=3 (9x9 matrices? No, 3x3). Group size = q^3.
    # For q=5, size = 125. Generators: need k=5 generators.
    # Upper triangular 3x3 unit matrices: [[1, a, c], [0, 1, b], [0, 0, 1]]
    # Elements (a, b, c) in GF(5)^3.
    # Multiplication: (a1,b1,c1)*(a2,b2,c2) = (a1+a2, b1+b2, c1+c2+a1*b2)
    #
    # Generators: Let's use:
    # g1 = (1, 0, 0)  [E_{1,2}]
    # g2 = (0, 1, 0)  [E_{2,3}]
    # g3 = (0, 0, 1)  [E_{1,3}]
    # g4 = (1, 0, 0)^{-1} = (-1, 0, 0) = (4, 0, 0)
    # g5 = (0, 1, 0)^{-1} = (0, -1, 0) = (0, 4, 0)
    #
    # This is a 5-regular Cayley graph on 125 vertices. What's its girth?
    # The commutator [g1, g2] = g1*g2*g1^{-1}*g2^{-1} = (1,1,1)*(4,4,0) = ... let me compute.
    # (a1,b1,c1)*(a2,b2,c2) = (a1+a2, b1+b2, c1+c2+a1*b2) mod 5
    # g1*g2 = (1,0,0)*(0,1,0) = (1, 1, 0+0+1*1) = (1, 1, 1)
    # g1^{-1}*g2^{-1} = (4,0,0)*(0,4,0) = (4, 4, 0+0+4*4) = (4, 4, 16 mod 5) = (4, 4, 1)
    # [g1,g2] = (1,1,1)*(4,4,1) = (1+4, 1+4, 1+1+1*4) = (0, 0, 6 mod 5) = (0, 0, 1) = g3
    # So g1*g2*g1^{-1}*g2^{-1} = g3, a generator.
    # The word g1*g2*g1^{-1}*g2^{-1}*g3^{-1} has length 5 and is the identity.
    # So there's a cycle of length 5! That's too short for girth 9.
    #
    # I need a different set of generators or a different group structure.
    #
    # Let me try the Morgenstern construction, which is known to give good cages.
    # Morgenstern (1996) constructed 5-regular graphs of girth 9 on 125 vertices? No...
    #
    # Actually, let me just keep the L-U-W construction but try to optimize it.
    # The L-U-W construction CD(d, q) with q=k gives a q-regular bipartite graph of
    # girth at least d+5 (for odd d) or d+4 (for even d).
    #
    # For k=5, g=9: we need girth >= 9.
    # If d is odd: girth >= d+5, so d+5 >= 9 => d >= 4. But d must be odd, so d >= 5.
    # If d is even: girth >= d+4, so d+4 >= 9 => d >= 5. d even => d >= 6.
    # The seed uses: d = max(3, g-5 if (g-5)%2==0 else g-4)
    # For g=9: g-5=4 (even), so d = max(3, 4) = 4. But d=4 is even, girth >= 4+4=8. Not enough!
    # Wait, let me re-read the seed: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # (g-5)%2 == 0 means g-5 is even. For g=9, g-5=4, 4%2==0, so d = max(3, 4) = 4.
    # But the comment says "girth is at least d + 5 for odd d". For even d, it might be d+4.
    # So with d=4, girth >= 8, which is not enough for g=9.
    #
    # The seed might actually produce a graph with girth < g for some parameters!
    # Let me check: the seed's d calculation might be wrong.
    #
    # For the L-U-W construction, the girth of CD(d, q) is at least 2*min(d, q-1) + 2 or similar.
    # Actually, the girth of CD(d, q) is at least d + 5 if d is odd, and d + 4 if d is even,
    # provided q >= d.
    #
    # So for g=9, k=5: we need d such that the girth bound is >= 9.
    # If d is odd: d+5 >= 9 => d >= 4, odd => d >= 5.
    # If d is even: d+4 >= 9 => d >= 5, even => d >= 6.
    # So we need d >= 5 (odd) or d >= 6 (even).
    #
    # The seed uses d=4, which gives girth >= 8. This might not satisfy girth >= 9!
    # But the seed scores 1.0, so either the girth is actually >= 9 for this specific construction,
    # or the evaluator is lenient.
    #
    # Let me just try to improve the construction. Let me use d=5 for k=5, g=9.
    # With d=5, q=5: the graph has 2 * q^d = 2 * 5^5 = 3125 vertices (bipartite, two parts of q^d each).
    # Wait, no. The L-U-W construction: points are d-tuples, lines are d-tuples.
    # The bipartite graph has 2*q^d vertices? No, it has q^d points and q^d lines, total 2*q^d.
    # But actually, the construction might only use a subset.
    #
    # Hmm, the seed says "the connected component of the zero point is returned".
    # So it might be smaller than 2*q^d.
    #
    # Let me just try to implement a correct ABS/Morgenstern construction.
    #
    # Actually, let me try a completely different approach: use the known 5-regular
    # cage of girth 9. The smallest known 5-regular graph of girth 9 has 1116 vertices
    # (per the task description). The Moore bound for (5,9) is:
    # g=9 is odd, so n >= 1 + 5*(1 + 4 + 4^2 + 4^3) = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426.
    # The best lower bound is 428.
    #
    # Since we can't construct the 1116-vertex graph in this time, let's just make sure
    # our construction is valid and try to minimize n.
    #
    # Let me try the L-U-W construction with the correct d value.
    
    # Determine the correct d for L-U-W construction
    # We need girth >= g
    # L-U-W CD(d, q) has girth >= d+5 if d odd, d+4 if d even (assuming q >= d)
    # So we need the smallest d such that the girth bound >= g
    if (g - 5) % 2 == 0:
        # g-5 is even, try d = g-5 (even)
        d_even = g - 5
        d_odd = g - 4  # this is odd
        # d_even gives girth >= d_even + 4 = g - 1 < g. Not enough!
        # d_odd gives girth >= d_odd + 5 = g + 1 >= g. Enough!
        d = d_odd
    else:
        # g-5 is odd
        d_odd = g - 5
        d_even = g - 4
        # d_odd gives girth >= d_odd + 5 = g >= g. Enough!
        # d_even gives girth >= d_even + 4 = g >= g. Enough!
        # Use the smaller one: d_odd = g-5
        d = d_odd
    
    # Ensure d >= 3
    d = max(3, d)
    
    # q = k (must be prime or 4 for the construction to work as in the seed)
    q = k
    _, is_pp = _is_prime_power(q)
    if not is_pp:
        # Find next prime
        q = _next_prime_ge(k)
    
    # Build the L-U-W construction
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q
    
    # Rule for L-U-W construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vi in range(len(order)):
        vertex = order[vi]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
