import numpy as np
import random
import time

def construct():
    n = 20
    deadline = time.time() + 110
    
    # Build the polarity graph of PG(2,4)
    # PG(2,4) has 21 points. The collinearity graph (polarity graph) 
    # connects two points if they are incident to a common line under a fixed polarity.
    # For q=4, each point has degree (q+1) = 5 in the polarity graph? No.
    # Actually, the "polarity graph" often refers to the graph where edges connect
    # points P and Q such that P is on the polar line of Q (or vice versa, or both).
    # In PG(2,q), a non-singular polarity has no absolute points if q is even?
    # For q=4 (even), a symplectic polarity has no absolute points.
    # The graph where P~Q if P is on polar line of Q is regular of degree q+1 = 5?
    # Wait, the standard "polarity graph" construction for K_{2,3}-free graphs usually
    # refers to the graph where P~Q if P is on the polar line of Q.
    # Degree is q+1. For q=4, degree 5. 21*5/2 = 52.5 edges? No, must be integer.
    # If no absolute points, it's a 2-regular? No.
    
    # Let's look at the specific construction mentioned: "collinearity graph of PG(2,4)"
    # The collinearity graph has vertices as points, edges if collinear.
    # In PG(2,4), any two points are collinear (define a line). So it's K_21.
    # That's not right.
    
    # The "polarity graph" $G(q)$ for a polarity $\pi$ of PG(2,q) has vertex set PG(2,q)
    # and edges $PQ$ if $P$ lies on the polar line of $Q$.
    # If the polarity is non-singular (no absolute points), the graph is $q+1$-regular.
    # For q=4, degree 5. Total edges = 21*5/2 = 52.5. This is impossible.
    # Ah, for even q, a symplectic polarity has NO absolute points.
    # Wait, $21 \times 5$ is odd. So it can't be regular?
    # Let's check: In PG(2,q), a point P is on the polar line of Q iff Q is on the polar line of P.
    # So the graph is undirected.
    # Number of edges must be integer.
    # For q=4, is there a polarity with no absolute points?
    # Yes, symplectic polarity.
    # Then degree is constant?
    # Number of points on a line in PG(2,q) is q+1 = 5.
    # So each polar line contains 5 points.
    # So each vertex has degree 5.
    # Sum of degrees = 21 * 5 = 105.
    # 105 is odd. Contradiction.
    # Therefore, there MUST be absolute points or the degree is not constant?
    # In PG(2,q) for q even, every polarity has absolute points?
    # No, symplectic polarity has NO absolute points.
    # Let me re-read properties.
    # "The graph is (q+1)-regular if the polarity has no absolute points."
    # If so, 21*5=105 is odd, so it can't be a simple undirected graph?
    # Unless... I am miscounting vertices. PG(2,4) has (4^3-1)/(4-1) = 63/3 = 21 vertices. Correct.
    # Maybe the graph is directed? No, incidence is symmetric for polarities?
    # $P \in \pi(Q) \iff Q \in \pi(P)$. Yes.
    # So sum of degrees must be even.
    # Therefore, for q=4, a polarity MUST have absolute points?
    # Or maybe my degree calculation is wrong.
    # Points on polar line of Q: There are q+1 points. One of them is Q itself if Q is absolute.
    # If no absolute points, Q is NOT on its own polar line.
    # So the neighbors of Q are the q+1 points on its polar line, excluding Q?
    # No, the definition is usually: $P \sim Q$ if $P \in \pi(Q)$.
    # If $Q \notin \pi(Q)$, then the set of neighbors is exactly the points on $\pi(Q)$.
    # Size is q+1.
    # So degree is q+1.
    # This leads to the odd sum contradiction.
    
    # Resolution: For q even, a polarity of PG(2,q) ALWAYS has absolute points?
    # Actually, a symplectic polarity has no absolute points.
    # Let's check q=2. PG(2,2) has 7 vertices.
    # Degree would be 3. Sum = 21. Odd.
    # The Fano plane polarity graph?
    # In PG(2,2), a polarity has 3 absolute points.
    # So for q=2, there are absolute points.
    # Is there a polarity of PG(2,4) with no absolute points?
    # Yes, symplectic.
    # So why the contradiction?
    # Maybe the graph definition includes loops? No, simple graph.
    # Maybe the degree is q?
    # If P is on polar line of Q, and Q is not absolute, then P != Q.
    # The line has q+1 points.
    # So degree is q+1.
    
    # Let's try a different known construction.
    # The task suggests: "polarity graph of the projective plane of order 4... 60 edges per vertex".
    # 60 edges per vertex? Degree 60?
    # Max degree in a 21-vertex graph is 20.
    # So "60 edges per vertex" is likely a typo in my interpretation or the prompt.
    # "115 edges" total for 20 vertices?
    # If degree ~11.5, that's plausible.
    # 20 * 11.5 / 2 = 115.
    # So the graph on 20 vertices has 115 edges.
    # This implies the 21-vertex graph had higher degree?
    # If we delete one vertex, we lose its incident edges.
    # If the 21-vertex graph was regular of degree d, edges = 21d/2.
    # If d=11, edges = 115.5 (impossible).
    # If d=10, edges = 105.
    # If we delete a vertex of degree 10, we get 105-10 = 95 edges?
    # Or maybe the 21-vertex graph is not regular?
    
    # Let's look for "K2,3 free graph 20 vertices 115 edges".
    # ex(20, K2,3) is known to be 115?
    # According to literature, ex(n, K_{2,3}) for n=20 is 115.
    # The construction is often based on the polarity graph of PG(2,4).
    
    # Let's construct PG(2,4) explicitly.
    # Field GF(4) = {0, 1, a, a+1} where a^2 = a + 1 (primitive element).
    # Or a^2 + a + 1 = 0.
    # Elements: 0, 1, w, w+1 where w^2 = w+1.
    # Let's use integers 0,1,2,3 mapping to 0,1,w,w+1.
    # Addition table:
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1
    # 3+3=0
    # Multiplication:
    # 0*x=0
    # 1*x=x
    # 2*2 = (w)^2 = w+1 = 3
    # 2*3 = w(w+1) = w^2+w = (w+1)+w = 1
    # 3*3 = (w+1)^2 = w^2+2w+1 = (w+1)+0+1 = w = 2 (since 2w=0 in char 2)
    
    # Points of PG(2,4) are 1-dimensional subspaces of GF(4)^3.
    # Represent point by a non-zero vector (x,y,z), normalized so first non-zero is 1.
    # Forms: (1,y,z), (0,1,z), (0,0,1).
    # Total: 4*4 + 4 + 1 = 21 points.
    
    # We need a polarity. A polarity is a map P -> Line(P).
    # Symplectic polarity can be defined by a skew-symmetric matrix M.
    # P is on polar line of Q iff P^T M Q = 0.
    # For PG(2,q), a standard symplectic polarity matrix is:
    # M = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
    # In char 2, -1 = 1.
    # M = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    # Check skew-symmetry: M^T = M. In char 2, skew-symmetric means M^T = -M = M.
    # So any symmetric matrix with 0 diagonal is "skew-symmetric" in char 2?
    # Yes, because a_ii = -a_ii => 2a_ii = 0 => a_ii = 0.
    # And a_ij = -a_ji => a_ij = a_ji.
    # So we need a symmetric matrix with 0 diagonal.
    # The matrix above works.
    
    # Let's generate the 21 points.
    gf4_add_tbl = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    gf4_mul_tbl = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf4_add(a, b):
        return gf4_add_tbl[a][b]
        
    def gf4_mul(a, b):
        return gf4_mul_tbl[a][b]
        
    def gf4_neg(a):
        return a # char 2
        
    # Generate points
    points = []
    # (1, y, z)
    for y in range(4):
        for z in range(4):
            points.append((1, y, z))
    # (0, 1, z)
    for z in range(4):
        points.append((0, 1, z))
    # (0, 0, 1)
    points.append((0, 0, 1))
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Polarity matrix M
    # M = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    M = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
         
    def dot(p, q):
        # p^T M q
        # p = (x1, x2, x3), q = (y1, y2, y3)
        # M q = (y2, y1+y3, y2)
        # dot = x1*y2 + x2*(y1+y3) + x3*y2
        # = x1*y2 + x2*y1 + x2*y3 + x3*y2
        # = y2*(x1+x3) + x2*y1 + x2*y3
        res = 0
        # Calculate M q first
        mq1 = gf4_mul(M[0][0], q[0]) ^ gf4_mul(M[0][1], q[1]) ^ gf4_mul(M[0][2], q[2]) # Wait, add is XOR? No, gf4_add.
        # In GF(4), addition is not XOR unless we map to GF(2)^2.
        # My gf4_add table is correct for GF(4).
        # Let's just use the table.
        
        # M q
        mq0 = gf4_add(gf4_mul(M[0][0], q[0]), gf4_add(gf4_mul(M[0][1], q[1]), gf4_mul(M[0][2], q[2])))
        mq1 = gf4_add(gf4_mul(M[1][0], q[0]), gf4_add(gf4_mul(M[1][1], q[1]), gf4_mul(M[1][2], q[2])))
        mq2 = gf4_add(gf4_mul(M[2][0], q[0]), gf4_add(gf4_mul(M[2][1], q[1]), gf4_mul(M[2][2], q[2])))
        
        # p dot (M q)
        res = gf4_add(gf4_mul(p[0], mq0), gf4_add(gf4_mul(p[1], mq1), gf4_mul(p[2], mq2)))
        return res
        
    # Build adjacency for 21 vertices
    # P ~ Q if dot(P, Q) == 0 and P != Q
    A21 = np.zeros((21, 21), dtype=int)
    for i in range(21):
        for j in range(21):
            if i != j:
                if dot(points[i], points[j]) == 0:
                    A21[i, j] = 1
                    A21[j, i] = 1
                    
    # Check degrees
    degrees = np.sum(A21, axis=1)
    # Expected degree 5?
    # If degrees are 5, total edges 52.5 -> Impossible.
    # Let's check if there are absolute points (P on its own polar line).
    # dot(P, P) == 0?
    abs_points = []
    for i in range(21):
        if dot(points[i], points[i]) == 0:
            abs_points.append(i)
            
    # If there are absolute points, the degree might vary or be different.
    # Actually, if P is absolute, P is on its own polar line.
    # In the graph definition "P ~ Q if P on polar line of Q", does P~P count?
    # Simple graph, no loops. So P~P is ignored.
    # Neighbors of P are {Q != P | Q on polar line of P}.
    # Polar line of P contains 5 points.
    # If P is absolute, P is one of them. So 4 neighbors.
    # If P is not absolute, P is not on its own polar line. So 5 neighbors.
    # Sum of degrees = 4*(#abs) + 5*(21-#abs).
    # If #abs = 3 (common for q=2, maybe q=4?), sum = 12 + 5*18 = 102. Edges = 51.
    # If #abs = 0, sum = 105. Impossible.
    # So there must be absolute points.
    
    # For PG(2,4), a symplectic polarity has NO absolute points.
    # Wait, I might be confusing types of polarities.
    # Orthogonal polarities have absolute points.
    # Symplectic have none.
    # If symplectic has none, then degree is 5 for all.
    # Sum = 105.
    # This implies the graph is NOT simple undirected? Or my understanding of "P on polar line of Q" is slightly off for the graph definition used in extremal graph theory.
    
    # Alternative: The graph connects P and Q if P is on the polar line of Q OR Q is on the polar line of P?
    # For a polarity, these are equivalent.
    
    # Let's check the literature value for ex(20, K2,3).
    # It is 115.
    # 115 edges on 20 vertices. Average degree 11.5.
    # The polarity graph of PG(2,4) has 21 vertices.
    # If we delete one vertex, we get 20 vertices.
    # To get 115 edges, the 21-vertex graph must have had ~125+ edges?
    # Or the graph is the COMPLEMENT of the polarity graph?
    # Complement of a 5-regular graph on 21 vertices is 15-regular.
    # 21*15/2 = 157.5 edges.
    # Delete one vertex (degree 15).
    # Remaining edges = 157.5 - 15 = 142.5? No.
    # If the 21-vertex graph has 157 edges (if regular degree 15? No, 21*15 is odd).
    
    # Let's try the "collinearity graph" interpretation again.
    # Maybe it's the graph where P~Q if they are NOT collinear? No, all are collinear.
    
    # Let's look at the specific number 115.
    # 115 = 20 * 11.5 / 2.
    # This suggests a near-regular graph of degree 11 or 12.
    
    # Let's assume the construction is:
    # 1. Take PG(2,4).
    # 2. Define graph G where P~Q if P is NOT on the polar line of Q.
    #    (Complement of the polarity graph).
    #    If polarity graph is 5-regular (ignoring the odd sum issue for a moment, maybe it's 4 or 5 mixed),
    #    complement degree is 20 - deg_polarity.
    #    If deg_polarity is 5, deg_comp is 15.
    #    If deg_polarity is 4 (absolute points), deg_comp is 16.
    
    # Let's calculate the actual edges of the polarity graph I built.
    edges_21 = int(np.sum(A21)) // 2
    # If this is 51 or 52, the complement has 210 - 51 = 159 edges.
    # Delete one vertex.
    # If we delete a vertex of degree 15 in the complement, we lose 15 edges.
    # 159 - 15 = 144.
    # This is > 115.
    # Is the complement K2,3 free?
    # The polarity graph is K2,3 free (known result).
    # Is the complement K2,3 free?
    # Not necessarily.
    
    # However, the prompt says: "polarity graph ... is known to be K2,3-free ... delete one vertex ... 115 edges".
    # This implies the POLARITY GRAPH ITSELF has 115+ edges?
    # But I calculated ~50 edges.
    # Contradiction.
    
    # Re-read: "polarity graph of the projective plane of order 4 ... 60 edges per vertex".
    # "60 edges per vertex" is definitely wrong for 21 vertices.
    # Maybe it means "60 edges" total?
    # If 60 edges total, 21*60/2 = 630? No.
    # Maybe "degree 6"?
    # If degree 6, edges = 21*6/2 = 63.
    # Delete one vertex (deg 6). 63-6 = 57.
    # Still not 115.
    
    # What if the graph is on the LINES?
    # Same structure.
    
    # What if q is different?
    # PG(2,9) has 91 vertices.
    
    # Let's ignore the specific number 115 for a moment and trust the "polarity graph" structure.
    # If the polarity graph is K2,3 free, and we delete a vertex, it remains K2,3 free.
    # We want to MAXIMIZE edges.
    # If the polarity graph is sparse (~50 edges), it's a bad start for maximizing edges (target ~115).
    # We should start with the DENSEST known K2,3 free graph.
    
    # Known fact: ex(n, K2,3) is approx n^2/4 for large n? No, that's K2,2.
    # For K2,3, ex(n, K2,3) = O(n^{3/2})?
    # No, K2,3 is a bipartite graph.
    # Zarankiewicz problem.
    # ex(n, K2,3) ~ c n^{3/2}.
    # For n=20, 20^{1.5} = 89.
    # 115 is close to this bound.
    
    # So the polarity graph must be dense.
    # How can a polarity graph be dense?
    # Only if the definition is different.
    # Maybe "P~Q if P is on the polar line of Q" is the SPARSE one.
    # And the DENSE one is the complement?
    # But is the complement K2,3 free?
    
    # Let's test the complement of my A21.
    # If A21 has ~50 edges, complement has ~160 edges.
    # Delete one vertex -> ~145 edges.
    # If it is K2,3 free, 145 > 115.
    # So if the complement is K2,3 free, we can achieve 145.
    
    # Let's check if the complement is K2,3 free.
    # This is a property of the specific polarity.
    
    # Plan:
    # 1. Construct the polarity graph A21 (sparse).
    # 2. Construct the complement B21.
    # 3. Check if B21 is K2,3 free.
    # 4. If yes, delete a vertex to get 20 vertices.
    # 5. Use local search to add more edges.
    
    # If B21 is NOT K2,3 free, we might need to use A21 and add edges?
    # But A21 is sparse.
    
    # Let's implement the construction and check validity.
    
    # First, verify the GF(4) arithmetic and polarity.
    
    # Let's run a quick check in the code.
    
    # For now, let's assume the complement is the intended dense graph.
    # If it fails validity, we fall back to a random dense graph + repair.
    
    best_edges = -1
    best_A = None
    
    def is_valid_mask(adj_masks):
        # adj_masks[i] is a bitmask of neighbors of i
        for u in range(20):
            for v in range(u + 1, 20):
                # common neighbors = popcount(adj[u] & adj[v])
                common = adj_masks[u] & adj_masks[v]
                if bin(common).count('1') >= 3:
                    return False
        return True
        
    def count_edges_mask(adj_masks):
        total = 0
        for i in range(20):
            total += bin(adj_masks[i]).count('1')
        return total // 2
        
    # Construct PG(2,4) polarity graph
    # Points
    points = []
    for y in range(4):
        for z in range(4):
            points.append((1, y, z))
    for z in range(4):
        points.append((0, 1, z))
    points.append((0, 0, 1))
        
    M = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
         
    def dot(p, q):
        mq0 = gf4_add(gf4_mul(M[0][0], q[0]), gf4_add(gf4_mul(M[0][1], q[1]), gf4_mul(M[0][2], q[2])))
        mq1 = gf4_add(gf4_mul(M[1][0], q[0]), gf4_add(gf4_mul(M[1][1], q[1]), gf4_mul(M[1][2], q[2])))
        mq2 = gf4_add(gf4_mul(M[2][0], q[0]), gf4_add(gf4_mul(M[2][1], q[1]), gf4_mul(M[2][2], q[2])))
        res = gf4_add(gf4_mul(p[0], mq0), gf4_add(gf4_mul(p[1], mq1), gf4_mul(p[2], mq2)))
        return res
        
    A21 = np.zeros((21, 21), dtype=int)
    for i in range(21):
        for j in range(21):
            if i != j:
                if dot(points[i], points[j]) == 0:
                    A21[i, j] = 1
                    A21[j, i] = 1
                    
    # Complement
    B21 = 1 - A21
    np.fill_diagonal(B21, 0)
    
    # Try deleting each vertex from B21 and check validity
    best_candidate = None
    best_candidate_edges = -1
    
    for del_idx in range(21):
        # Create 20x20 matrix
        B20 = np.zeros((20, 20), dtype=int)
        idx_map = []
        k = 0
        for i in range(21):
            if i != del_idx:
                idx_map.append(i)
                k += 1
                
        for i in range(20):
            for j in range(20):
                B20[i, j] = B21[idx_map[i], idx_map[j]]
                
        # Check validity
        K = B20 @ B20
        valid = True
        for u in range(20):
            for v in range(u + 1, 20):
                if K[u, v] >= 3:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            edges = int(np.sum(B20)) // 2
            if edges > best_candidate_edges:
                best_candidate_edges = edges
                best_candidate = B20.copy()
                
    # If we found a valid candidate, use it as seed
    if best_candidate is not None:
        A = best_candidate.copy()
        best_edges = best_candidate_edges
        best_A = A
    else:
        # Fallback: Random dense graph + repair
        A = np.zeros((20, 20), dtype=int)
        for u in range(20):
            for v in range(u + 1, 20):
                if random.random() < 0.6:
                    A[u, v] = 1
                    A[v, u] = 1
        # Repair
        K = A @ A
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(20):
                if time.time() >= deadline:
                    break
                for v in range(u + 1, 20):
                    if K[u, v] >= 3:
                        # Remove an edge
                        # Find common neighbor z
                        zs = [z for z in range(20) if A[u, z] == 1 and A[v, z] == 1]
                        if zs:
                            z = random.choice(zs)
                            A[u, z] = 0
                            A[z, u] = 0
                            # Update K
                            for w in range(20):
                                if A[u, w] == 1 and w != z:
                                    K[w, z] -= 1
                                    K[z, w] -= 1
                                if A[z, w] == 1 and w != u:
                                    K[u, w] -= 1
                                    K[w, u] -= 1
                            changed = True
                            break
                if changed:
                    break
        best_edges = int(np.sum(A)) // 2
        best_A = A.copy()
        
    # Local search to add edges
    # Use bitmask for speed
    adj_masks = [0] * 20
    for u in range(20):
        for v in range(20):
            if best_A[u, v] == 1:
                adj_masks[u] |= (1 << v)
                
    def try_add_local_search(adj_masks, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(20):
                if time.time() >= deadline:
                    return adj_masks
                for v in range(u + 1, 20):
                    if not (adj_masks[u] & (1 << v)):
                        # Check if we can add u-v
                        # u-v can be added if for all x in N(u), K[x, v] < 2
                        # and for all y in N(v), K[u, y] < 2
                        # K[x, v] = popcount(adj[x] & adj[v])
                        
                        ok = True
                        # Check neighbors of u
                        mask_u = adj_masks[u]
                        # Iterate bits in mask_u
                        m = mask_u
                        while m:
                            lsb = m & -m
                            x = lsb.bit_length() - 1
                            m -= lsb
                            if x == v:
                                continue
                            common = adj_masks[x] & adj_masks[v]
                            if bin(common).count('1') >= 2:
                                ok = False
                                break
                        if ok:
                            # Check neighbors of v
                            mask_v = adj_masks[v]
                            m = mask_v
                            while m:
                                lsb = m & -m
                                y = lsb.bit_length() - 1
                                m -= lsb
                                if y == u:
                                    continue
                                common = adj_masks[u] & adj_masks[y]
                                if bin(common).count('1') >= 2:
                                    ok = False
                                    break
                                    
                        if ok:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)
                            changed = True
                            
        return adj_masks
        
    adj_masks = try_add_local_search(adj_masks, deadline)
    
    # Convert back to matrix
    final_A = np.zeros((20, 20), dtype=int)
    for u in range(20):
        for v in range(20):
            if adj_masks[u] & (1 << v):
                final_A[u, v] = 1
                
    final_edges = int(np.sum(final_A)) // 2
    
    if final_edges > best_edges:
        best_edges = final_edges
        best_A = final_A
        
    # Random restarts with local search if time permits
    # We can perturb the best solution and run local search again
    
    for restart in range(100):
        if time.time() >= deadline - 10:
            break
            
        # Perturb: remove 5 random edges
        A_pert = best_A.copy()
        edges_list = []
        for u in range(20):
            for v in range(u + 1, 20):
                if A_pert[u, v] == 1:
                    edges_list.append((u, v))
                    
        if len(edges_list) > 5:
            random.shuffle(edges_list)
            for u, v in edges_list[:5]:
                A_pert[u, v] = 0
                A_pert[v, u] = 0
                
        # Convert to masks
        adj_pert = [0] * 20
        for u in range(20):
            for v in range(20):
                if A_pert[u, v] == 1:
                    adj_pert[u] |= (1 << v)
                    
        adj_pert = try_add_local_search(adj_pert, deadline)
        
        # Count edges
        pert_edges = sum(bin(m).count('1') for m in adj_pert) // 2
        
        if pert_edges > best_edges:
            best_edges = pert_edges
            # Convert back to matrix for next perturbations
            best_A = np.zeros((20, 20), dtype=int)
            for u in range(20):
                for v in range(20):
                    if adj_pert[u] & (1 << v):
                        best_A[u, v] = 1

    edges = []
    for u in range(20):
        for v in range(u + 1, 20):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges
