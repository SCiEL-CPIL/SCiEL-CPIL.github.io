import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe

    # Fallback: LUW construction (guaranteed valid, but large)
    def luw_construct(k, g):
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

    # Cayley graph construction using involutions in Z_q x Z_m
    # A generator (a, b) is an involution if 2a ≡ 0 (mod q) and 2b ≡ 0 (mod m)
    # For k=3, we need 3 involutions in Z_q x Z_m
    def cayley_construct(k, g):
        """
        Try to find a Cayley graph on Z_q x Z_m with 3 involutory generators
        that has girth >= g.
        
        Key insight: In a Cayley graph, short cycles correspond to short relations
        among the generators. By choosing generators with large "commutator distances"
        in the group, we can enforce high girth.
        
        For Z_q x Z_m, an element (a,b) is an involution iff 2a ≡ 0 (mod q) and 2b ≡ 0 (mod m).
        So a = q//2 (if q even) and b = m//2 (if m even), or a=0, b=m//2, etc.
        """
        if k != 3:
            return None
        
        # For k=3, g=16: try various (q, m) pairs
        # The group order is q*m, and we need at least the Moore bound
        # Moore bound for (3,16) is 766, so q*m >= 766
        
        # Strategy: Try q and m such that both are even (so involutions exist)
        # and the generators have "independent" directions to avoid short cycles
        
        # In Z_q x Z_m, involutions are elements of order 2.
        # If q is even: (q//2, 0) is an involution
        # If m is even: (0, m//2) is an involution
        # Also (q//2, m//2) if both even
        # More generally: (a, b) with 2a ≡ 0 mod q, 2b ≡ 0 mod m
        
        # For Z_q x Z_m with q, m even:
        # Involutions: (a, b) where a ∈ {0, q//2} and b ∈ {0, m//2}
        # That gives 4 involutions: (0,0), (q//2,0), (0,m//2), (q//2,m//2)
        # But (0,0) is identity, so we have 3 non-identity involutions!
        
        # So for any even q, m, the Cayley graph with generators
        # {(q//2, 0), (0, m//2), (q//2, m//2)} is 3-regular.
        
        # The question is: what is the girth?
        # Short cycles in Cayley graphs correspond to short words in generators
        # that equal the identity.
        
        # Let s1 = (q//2, 0), s2 = (0, m//2), s3 = (q//2, m//2)
        # Note: s1 + s2 = s3 (in the group)
        # So s1 * s2 * s3^{-1} = identity, giving a 3-cycle!
        # That's bad. We need girth >= 16.
        
        # So this simple construction doesn't work. We need generators that
        # don't have short additive relations.
        
        # Alternative: Use Z_p for prime p, but then involutions don't exist
        # unless p=2. So we need composite groups.
        
        # Better approach: Use Z_q where q is even, and pick involutions
        # that are "far apart" in some sense.
        
        # Actually, in Z_q (q even), the only involution is q//2.
        # So we can only have one involution in Z_q.
        
        # We need a group with at least 3 distinct involutions.
        # Z_q x Z_m with q, m even has exactly 3 non-identity involutions
        # as noted above, but they have the relation s1+s2=s3.
        
        # What about Z_q x Z_r where q, r are even but not both powers of 2?
        # The involutions are still (a,b) with 2a≡0 mod q, 2b≡0 mod r.
        # If q=2^a * q' (q' odd), then 2a≡0 mod q means a is a multiple of q/2.
        # So a ∈ {0, q/2} if q is even. Same for b.
        # So still only 4 involutions total (including identity).
        
        # The problem is that in abelian groups, involutions form a subgroup
        # (the 2-torsion), and any three involutions a, b, c satisfy
        # a+b+c = (a+b)+c. If a+b is also an involution, then...
        
        # Actually, in any abelian group, if a, b, c are involutions,
        # then (a+b+c)^2 = a^2 + b^2 + c^2 + 2ab + 2ac + 2bc = 2ab + 2ac + 2bc.
        # This is 0 iff ab+ac+bc = 0 in the 2-torsion quotient.
        
        # For Z_q x Z_m, the 2-torsion is (Z/2) x (Z/2) if q,m even.
        # The three non-identity elements of (Z/2)^2 sum to 0.
        # So s1+s2+s3 = (q//2, 0) + (0, m//2) + (q//2, m//2) = (q, m) = (0,0) mod (q,m).
        # Hence s1*s2*s3 = identity, giving a 3-cycle.
        
        # Conclusion: Any Cayley graph on an abelian group with 3 involutory
        # generators has a 3-cycle (since the 3 involutions sum to 0 in
        # the 2-torsion subgroup when there are exactly 3).
        
        # So we need a NON-abelian group, or a different approach.
        
        # Alternative: Use a non-abelian group like S_n or a semidirect product.
        # Or use a voltage graph / lifting approach.
        
        # For now, return None to fall back to LUW.
        return None

    # Try Cayley construction first
    result = cayley_construct(k, g)
    if result is not None:
        return result

    # Fallback to LUW
    return luw_construct(k, g)
