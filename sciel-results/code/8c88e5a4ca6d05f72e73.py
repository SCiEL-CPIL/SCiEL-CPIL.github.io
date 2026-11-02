import random
import time
from itertools import combinations_with_replacement

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: LUW Construction
    def luw_construct():
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

    # Primary: Circulant Graph Search
    # We look for a circulant graph Circ(N, S) with |S|=k and girth >= g
    # N must be even for bipartite-like symmetry? No, circulant works for any N.
    # However, for odd girth, the graph cannot be bipartite.
    # Circulant graphs can be non-bipartite.
    
    # Moore bound for k=5, g=9 is 426.
    # Known best is 1116.
    # We search N from 1116 upwards.
    
    def has_short_cycle(N, S, max_len):
        # S is a list of k distinct steps from 1 to N//2
        # Check for cycles of length 2 to max_len
        # A cycle is a sequence of steps s_1, ..., s_L such that sum +/- s_i == 0 mod N
        # This is equivalent to finding a multiset of steps with signs that sum to 0.
        
        # L=2: s_i == s_j (impossible if distinct) or s_i + s_j == 0 (impossible if positive) or s_i - s_j == 0 (impossible)
        # Wait, in undirected graph, step s is same as -s.
        # Cycle of length 2: s + (-s) = 0. This is always true?
        # No, a cycle of length 2 in a simple graph implies a double edge.
        # In a circulant graph, if s is in S, edge (u, u+s) exists.
        # Edge (u+s, u) is the same edge.
        # So length 2 cycles don't exist in simple graphs.
        # Length 3: s1 + s2 + s3 = 0 or s1 + s2 - s3 = 0.
        # Since s_i > 0, s1+s2+s3=0 is impossible mod N unless sum >= N.
        # We check if any combination of steps with signs sum to 0 mod N.
        
        # Efficient check:
        # For L up to 8, we can use a set of reachable sums.
        # But we need to track the number of steps.
        
        # Let's use a BFS on the "difference" space?
        # Actually, just check all combinations of steps with signs.
        # Number of steps k=5.
        # Max length 8.
        # Combinations with replacement of size L from k items, with signs.
        # This is manageable for small k.
        
        for L in range(2, max_len + 1):
            # Generate all sequences of length L from S
            # For each sequence, check if any assignment of signs sums to 0
            # This is 2^L * k^L which is too big for L=8, k=5 (256 * 390625 = 100M)
            # We need a smarter check.
            
            # Alternative: Check if 0 is in the set of sums of L steps with signs.
            # Let D_L be the set of all possible sums of L steps with signs.
            # We can build D_L iteratively.
            # D_1 = {s, -s for s in S}
            # D_L = {d + s, d - s for d in D_{L-1}, s in S}
            # We stop if 0 is in D_L.
            
            sums = set()
            for s in S:
                sums.add(s)
                sums.add(-s)
            
            for _ in range(1, L):
                new_sums = set()
                for d in sums:
                    for s in S:
                        new_sums.add((d + s) % N)
                        new_sums.add((d - s) % N)
                sums = new_sums
                if 0 in sums:
                    return True
        return False

    best_n, best_edges = luw_construct()
    
    # Search for Circulant
    # Start from known best 1116
    for N in range(1116, 1250):
        if time.time() > deadline:
            break
            
        # Randomly sample sets S of size k from 1..N//2
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
            S = random.sample(range(1, N//2 + 1), k)
            # Avoid trivial small cycles
            # Check girth
            if not has_short_cycle(N, S, g - 1):
                # Construct edges
                edges = []
                for u in range(N):
                    for s in S:
                        v = (u + s) % N
                        if u < v:
                            edges.append((u, v))
                return N, edges
                found = True
                break
        if found:
            break
            
    return best_n, best_edges
