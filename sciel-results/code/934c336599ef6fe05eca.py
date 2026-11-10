import time
import random
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For k=5, g=9, we use a Cayley graph on Z_n with n even.
    # S = {n/2, s1, -s1, s2, -s2}
    # We search for n and s1, s2 such that girth >= g.
    
    if k != 5:
        # Fallback to LUW for other k
        return _luw_construct(k, g)
        
    # Range for n
    # Moore bound for k=5, g=9 is approx 1000?
    # Known best 1116.
    # Try n from 428 to 1500.
    
    best_n = float('inf')
    best_edges = []
    
    # Heuristic: n should be even.
    # We want to minimize n.
    
    for n in range(430, 1600, 2):
        if time.time() > deadline:
            break
            
        # If we found a valid graph with smaller n, we can stop?
        # No, we iterate n increasing, so first valid is smallest?
        # Yes, if we iterate n from lower bound upwards.
        
        # Randomize s1, s2
        # To be fast, we can try a few random pairs for each n.
        # Or use a deterministic heuristic?
        # Random is safer.
        
        # Limit trials per n to keep time bounded
        max_trials_per_n = 20
        
        for _ in range(max_trials_per_n):
            if time.time() > deadline:
                break
                
            # Pick s1, s2 from 1 to n-1, excluding 0, n/2
            # And ensuring s1 != s2, s1 != -s2
            valid_s = [x for x in range(1, n) if x != n//2]
            if len(valid_s) < 4:
                continue
                
            # Random selection
            s1, s2 = random.sample(valid_s, 2)
            if s2 == -s1 % n:
                continue
                
            S = [n//2, s1, -s1 % n, s2, -s2 % n]
            # Ensure unique
            if len(set(S)) != 5:
                continue
                
            # Check girth
            if _check_girth_cyclic(n, S, g):
                # Construct edges
                edges = []
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # If no cyclic graph found, fallback to LUW
    return _luw_construct(k, g)

def _check_girth_cyclic(n, S, g):
    # S is a set of 5 elements in Z_n.
    # Check for cycles of length < g.
    # Since g is 9, we check 4, 6, 8.
    
    # 4-cycle: a+b=c+d with distinct a,b,c,d in S?
    # Actually, in a Cayley graph, a 4-cycle corresponds to a relation a+b-c-d=0.
    # This exists if there are distinct a,b,c,d in S such that a+b=c+d.
    # Note: a,b,c,d are elements of S.
    # If a=b, then 2a=c+d. This is a "2-edge" cycle? No, simple graph.
    # The condition for girth >= 5 is that all sums of two elements of S are distinct (except for a+b=b+a).
    # i.e., S is a Sidon set.
    
    sums = defaultdict(list)
    for i in range(len(S)):
        for j in range(i, len(S)):
            s_val = (S[i] + S[j]) % n
            sums[s_val].append((i, j))
            
    for s_val, pairs in sums.items():
        # If there are more than 1 pair for a sum, we might have a 4-cycle.
        # Pairs (i,j) and (k,l).
        # If i,j,k,l are distinct, then S[i]+S[j]=S[k]+S[l] implies a 4-cycle.
        # If they share an index, e.g., i=k, then S[i]+S[j]=S[i]+S[l] => S[j]=S[l].
        # Since S has unique elements, this implies j=l, so it's the same pair.
        # So we only care if there are >= 2 distinct pairs with distinct indices.
        
        if len(pairs) > 1:
            # Check if any two pairs are disjoint
            for p1_idx in range(len(pairs)):
                for p2_idx in range(p1_idx + 1, len(pairs)):
                    p1 = pairs[p1_idx]
                    p2 = pairs[p2_idx]
                    if set(p1).isdisjoint(set(p2)):
                        return False # Girth 4
                        
    # 6-cycle: a+b+c=0 with distinct a,b,c in S?
    # Or a+b+c=d+e+f?
    # A 6-cycle in Cayley graph is a relation a+b+c-d-e-f=0.
    # Simplest: a+b+c=0.
    # Check if there are distinct a,b,c in S such that a+b+c=0.
    
    # Precompute all sums of 2
    # We already have sums.
    # For each c in S, check if -c is in sums of 2 distinct elements not including c.
    # This is slightly complex.
    # Brute force for 6-cycle:
    # Iterate all triples (i,j,k) from S.
    # If i,j,k distinct and S[i]+S[j]+S[k] == 0, then girth 6.
    # Also check a+b+c = d+e+f?
    # Actually, if a+b+c=0, that's a cycle of length 3 in the directed sense?
    # In undirected, a+b+c=0 means a+b=-c.
    # Path: 0 -> a -> a+b -> a+b+c=0. Length 3.
    # But graph is simple, so no triangles?
    # If a+b+c=0, and a,b,c distinct, we have a triangle.
    # Girth 3.
    # So we must avoid triangles.
    
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                if (S[i] + S[j] + S[k]) % n == 0:
                    return False # Girth 3 (or 6 if we consider the other direction? No, triangle is girth 3)
                    
    # Check 6-cycles that are not triangles.
    # Relation a+b+c = d+e+f with distinct a..f?
    # Or a+b+c = 0 is the only "short" one?
    # Actually, a 6-cycle is a sequence x0, x1, x2, x3, x4, x5, x6=x0.
    # Steps: s1, s2, s3, s4, s5, s6.
    # Sum s1..s6 = 0.
    # Non-backtracking: s1 != -s2, etc.
    # If we pick s1, s2, s3, -s3, -s2, -s1, sum is 0. This is a "backtracking" cycle of length 6?
    # No, that's just going out and back.
    # A "true" 6-cycle requires distinct steps or non-trivial relation.
    # If S is a Sidon set (no 4-cycles) and no triangles, the girth is at least 5.
    # We need girth >= 9.
    # So we must avoid 6-cycles and 8-cycles.
    
    # Check 6-cycles:
    # Any relation a+b+c = d+e+f with distinct indices?
    # Or a+b+c=0 (triangle, already checked).
    # What about a+b = c+d+e+f?
    # This is getting expensive.
    # Given the small size of S (5), we can enumerate all sequences of length 6 from S.
    # 5^6 = 15625.
    # Check if sum is 0 and non-backtracking.
    
    for s1 in S:
        for s2 in S:
            if s2 == -s1 % n: continue
            for s3 in S:
                if s3 == -s2 % n: continue
                for s4 in S:
                    if s4 == -s3 % n: continue
                    for s5 in S:
                        if s5 == -s4 % n: continue
                        for s6 in S:
                            if s6 == -s5 % n: continue
                            if s1 == s6: continue # Close the loop? No, s6 is the step back to 0?
                            # Sum s1..s6 must be 0.
                            if (s1+s2+s3+s4+s5+s6) % n == 0:
                                # Check if it's a trivial cycle (backtracking)
                                # If s1=-s6, s2=-s5, s3=-s4, it's a backtrack.
                                # But we already checked non-backtracking for adjacent.
                                # Is it possible to have a non-backtracking 6-cycle?
                                # Yes.
                                return False # Girth 6
    
    # Check 8-cycles:
    # 5^8 = 390625. Too many?
    # 390k * 1000 n's? No, we do this per n.
    # 390k operations is fast in Python? ~0.1s.
    # If we do this for 100 n's, it's 10s. Acceptable.
    
    for s1 in S:
        for s2 in S:
            if s2 == -s1 % n: continue
            for s3 in S:
                if s3 == -s2 % n: continue
                for s4 in S:
                    if s4 == -s3 % n: continue
                    for s5 in S:
                        if s5 == -s4 % n: continue
                        for s6 in S:
                            if s6 == -s5 % n: continue
                            for s7 in S:
                                if s7 == -s6 % n: continue
                                for s8 in S:
                                    if s8 == -s7 % n: continue
                                    if s1 == s8: continue
                                    if (s1+s2+s3+s4+s5+s6+s7+s8) % n == 0:
                                        return False # Girth 8
                                        
    return True

def _luw_construct(k, g):
    # Original LUW implementation
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
