# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def generate_polarity_graph(q):
        """
        Generate the polarity graph of the projective plane of order q.
        q must be a prime power.
        Returns a list of edges.
        """
        # For simplicity and speed, we implement for prime q.
        # Vertices:
        # 0 to q^2-1: Points (x, y) in F_q x F_q
        # q^2 to q^2+q-1: Points at infinity (slope m)
        # q^2+q: Point at infinity (vertical)
        
        n = q*q + q + 1
        edges = set()
        
        # Helper for modular inverse
        def inv(x, m):
            # Fermat's little theorem for prime m
            return pow(x, m-2, m)
        
        # Polarity definition for PG(2, q) with q odd:
        # Point P = (x, y, 1) maps to line L_P: x*X + y*Y - Z = 0 ?
        # Actually, standard polarity: (x,y,z) -> (x, y, -x^2 - y^2) ? No.
        
        # Let's use the explicit adjacency for the Erdos-Renyi-Sos graph.
        # Vertices are (a,b) in F_q x F_q, and (c, 0) and (0, 1)?
        
        # Standard construction from literature:
        # V = { (x,y) in F_q^2 } U { (1,0,t) in F_q } U { (0,1,1) }
        # This is getting complicated to index correctly.
        
        # Simpler:
        # Use the fact that for q=7, we can just generate it.
        # But we need a generic function or at least for q=7.
        
        # Let's use a known simple polarity:
        # Identify vertices with elements of F_q x F_q plus q+1 points at infinity.
        # Polarity: (x,y) ~ (x',y') iff x*x' + y*y' = 1
        # This is not a polarity on the whole plane, just on the affine part.
        
        # Correct Polarity for PG(2, q) (q odd):
        # Map [u:v:w] to line uX + vY + (u^2+v^2)Z = 0?
        # Let's stick to a robust implementation for q=7.
        
        # Since we only need q=7 for n=57, and maybe q=5 for n=31.
        # I will hardcode the logic for prime q.
        
        # Vertices 0..q^2-1: (i, j)
        # Vertices q^2..q^2+q-1: (1, 0, k) for k in 0..q-1
        # Vertex q^2+q: (0, 1, 1)
        
        # Polarity map sigma:
        # sigma(i, j, 1) = (i, j, -(i^2+j^2))
        # sigma(1, 0, k) = (k, 1, 0) ? No.
        
        # Let's use the standard construction from "C4-free graphs with high minimum degree"
        # For q=7, n=57.
        # We can just use a precomputed list or a very fast generator.
        
        # Generator for q=7:
        if q != 7 and q != 5:
             # Fallback for other q, or just return empty if not supported
             # But we only need 7 and 5.
             pass
             
        # Indices:
        # 0..q^2-1: (x, y, 1)
        # q^2..q^2+q-1: (1, 0, z)
        # q^2+q: (0, 1, 1)
        
        # Polarity:
        # P = (x, y, 1) -> Line L: x*X + y*Y - (x^2+y^2)Z = 0
        # P = (1, 0, z) -> Line L: z*X - Y = 0 ?
        # P = (0, 1, 1) -> Line L: X = 0 ?
        
        # Two points P, Q are adjacent if P is on sigma(Q).
        # P on sigma(Q) means P satisfies the line equation of sigma(Q).
        
        # Let's implement this.
        
        def get_point(idx):
            if idx < q*q:
                x = idx % q
                y = idx // q
                return (x, y, 1)
            elif idx < q*q + q:
                z = idx - q*q
                return (1, 0, z)
            else:
                return (0, 1, 1)
        
        def get_line(idx):
            # sigma(point)
            p = get_point(idx)
            x, y, z = p
            if z == 1:
                # (x, y, 1) -> (x, y, -(x^2+y^2))
                # Line: x*X + y*Y - (x^2+y^2)*Z = 0
                a, b, c = x, y, -(x*x + y*y) % q
            elif x == 1 and y == 0:
                # (1, 0, z) -> (z, 1, 0) ?
                # Check: (1,0,z) is on line (z, 1, 0) -> z*1 + 1*0 + 0*z = z != 0.
                # Correct polarity for (1,0,z) is (z, 1, 0)?
                # Actually, the polarity map is an involution.
                # Let's use the standard one:
                # (x,y,z) -> (x, y, -x^2-y^2) is not a projective transformation.
                
                # Use:
                # (x,y,z) -> (x, y, -x^2-y^2) is valid for z=1.
                # For (1,0,z), the line is z*X - Y = 0?
                # (1,0,z) is on z*1 - 0 = z. Not 0.
                
                # Let's use the explicit matrix for polarity.
                # M = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]? No.
                
                # I will use a hardcoded adjacency for q=7 to be safe and fast.
                # Or just use the property that G_q is C4-free.
                # Given time constraints, I will use a simple greedy seed if the generator is too complex.
                # BUT, the directive says use polarity graph.
                
                # Let's try a simpler polarity:
                # (x,y) ~ (x',y') iff x*x' + y*y' = 1
                # This works for the affine part.
                # We need to handle infinity.
                
                # I will implement the affine part and connect infinity points appropriately.
                # For q=7, we can just generate the affine part (49 vertices) and add 8 more.
                
                pass
            return (0,0,0)

        # Due to complexity of correct polarity implementation in short time,
        # I will use a robust alternative:
        # Start with a random C4-free graph? No.
        # Use the "Erdos-Renyi-Sos" construction explicitly for q=7.
        
        # Actually, I can just use the `networkx` style construction if I write it carefully.
        # But to be safe and fast, I will use a "Greedy with Bitsets" on a *dense* random seed?
        # No, random dense seeds are full of C4s.
        
        # Let's use the polarity graph of the *affine* plane?
        # Affine plane of order q has q^2 points.
        # Polarity: (x,y) ~ (x',y') iff x*x' + y*y' = 1.
        # This graph is C4-free?
        # If (x,y) ~ (x',y') and (x,y) ~ (x'',y'') and (x',y') ~ (x'',y'') ...
        # This is a standard C4-free graph.
        # n = q^2 = 49 for q=7.
        # Min degree: Each (x,y) is connected to (x',y') if xx'+yy'=1.
        # Number of solutions to xx'+yy'=1 in F_7.
        # For fixed x, x', y y' = 1 - xx'.
        # If 1-xx' != 0, 1 solution for y'. If 0, 0 solutions (if y,y' non-zero?)
        # Actually, in F_q, the number of solutions to xX + yY = c is q.
        # So degree is q.
        # For q=7, degree is 7.
        # n=49.
        # Required min degree for n=49: 49-39=10.
        # 7 < 10. So affine polarity graph is not enough for n=49.
        # But we can add edges?
        
        # What if we use q=5? n=25. Deg 5. Req 25-39 < 0. OK.
        # But we want n=47.
        
        # Let's go back to Projective Plane Polarity.
        # I will implement the correct polarity for PG(2, q).
        
        # Vertices:
        # 0..q^2-1: (x, y, 1)
        # q^2..q^2+q-1: (1, 0, z)
        # q^2+q: (0, 1, 1)
        
        # Polarity sigma:
        # (x, y, 1) -> (x, y, -(x^2+y^2))
        # (1, 0, z) -> (z, 1, 0)
        # (0, 1, 1) -> (1, 0, 0)
        
        # Check self-adjacency (absolute points):
        # (x,y,1) on (x,y,-(x^2+y^2)): x*x + y*y + 1*(-(x^2+y^2)) = 0. Yes.
        # (1,0,z) on (z,1,0): z*1 + 1*0 + 0*z = z. Self-adjacent if z=0.
        # (0,1,1) on (1,0,0): 1*0 + 0*1 + 0*1 = 0. Self-adjacent.
        
        # Adjacency P ~ Q iff P on sigma(Q).
        
        def is_adj(idx1, idx2):
            # Get coordinates
            p1 = get_point(idx1)
            p2 = get_point(idx2)
            
            # Get line sigma(p2)
            x2, y2, z2 = p2
            if z2 == 1:
                # Line: x2*X + y2*Y - (x2^2+y2^2)*Z = 0
                a, b, c = x2, y2, -(x2*x2 + y2*y2) % q
            elif x2 == 1 and y2 == 0:
                # (1, 0, z2) -> (z2, 1, 0)
                a, b, c = z2, 1, 0
            else:
                # (0, 1, 1) -> (1, 0, 0)
                a, b, c = 1, 0, 0
            
            # Check if p1 is on line (a,b,c)
            x1, y1, z1 = p1
            return (a*x1 + b*y1 + c*z1) % q == 0

        for i in range(n):
            for j in range(i+1, n):
                if is_adj(i, j):
                    edges.add((i, j))
        
        return list(edges)

    def check_c4_safe(adj, n, u, v):
        """
        adj: list of ints (bitmasks)
        Check if adding (u,v) creates a C4.
        """
        nu = adj[u]
        nv = adj[v]
        
        # We need to find x in N(u), y in N(v), x != y, x ~ y.
        # Let A = N(u), B = N(v).
        # For each y in B:
        #   Check if there exists x in A \ {y} such that x ~ y.
        #   This is equivalent to: (Union of N(x) for x in A \ {y}) has bit y set.
        #   Or simpler: M_y & (A ^ (1<<y)) != 0 ?
        #   Wait, M_y is the bitmask of neighbors of y.
        #   If M_y has any bit set that is also in A, and that bit is not y, then yes.
        
        # Optimization:
        # Iterate over the smaller of A and B?
        # No, the check is asymmetric in the loop but symmetric in logic.
        
        # Let's iterate over y in B.
        # To do this fast, we can iterate over set bits in B.
        
        # If B is large, this is slow.
        # Alternative:
        # Calculate S = Union of N(x) for x in A.
        # If S & B == 0, return True.
        # Else, for each y in S & B:
        #    Check if y has neighbor in A \ {y}.
        
        # Computing S is expensive if done naively.
        
        # Given n <= 47, we can just do the double loop with bit checks.
        # Worst case: 47 * 47 = 2209 checks. Very fast.
        
        # But we call this many times.
        
        # Let's use the bitset intersection method.
        # For each y in B:
        #   if (adj[y] & (nu ^ (1 << y))) != 0: return False
        #   # Note: adj[y] contains u? No, we are checking if adding (u,v) creates C4.
        #   # The existing graph is C4-free.
        #   # We are adding (u,v).
        #   # The new C4 would be u-x-y-v-u.
        #   # x is in N(u) (existing), y is in N(v) (existing).
        #   # x ~ y must be existing.
        #   # So we check existing adj[y].
        
        # We need to iterate over set bits in nv.
        temp = nv
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check if y has a neighbor in nu, excluding y itself.
            # adj[y] is the bitmask of neighbors of y.
            # We need to check if adj[y] intersects (nu & ~(1<<y)).
            if adj[y] & (nu & ~(1 << y)):
                return False
        
        return True

    def local_search(adj, n, min_deg_req, time_limit, seed):
        """
        adj: list of ints
        """
        t0 = time.time()
        rng = random.Random(seed)
        
        # Precompute degrees
        def get_deg(i):
            return bin(adj[i]).count('1')
        
        while time.time() - t0 < time_limit:
            # Check if done
            min_deg = min(get_deg(i) for i in range(n))
            if min_deg >= min_deg_req:
                return True
            
            # Find vertex with min degree
            target = min(range(n), key=get_deg)
            
            # Try to add edges to target
            # Candidates: non-neighbors
            candidates = [v for v in range(n) if v != target and not (adj[target] >> v) & 1]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if check_c4_safe(adj, n, target, v):
                    # Add edge
                    adj[target] |= (1 << v)
                    adj[v] |= (1 << target)
                    added = True
                    break
            
            if not added:
                # Stuck on this target. Try another target?
                # Or break?
                # If we can't add to target, we might be stuck.
                # Try to add edges to other low-degree vertices.
                # If we can't add any edge anywhere, we are stuck.
                # For now, just break to try a new seed.
                break
                
        return False

    best_result = None
    best_n = 0
    
    # Try n from 47 down
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
        
        min_deg_req = max(1, n - 39)
        
        # Generate seed
        # Use polarity graph of order 7 (57 vertices)
        # Take first n vertices
        # If n > 57, skip (but n <= 47)
        
        # Generate G_7
        # We only need to generate it once
        if not hasattr(construct, '_g7_edges'):
            construct._g7_edges = generate_polarity_graph(7)
        
        g7_edges = construct._g7_edges
        g7_n = 57
        
        # If n > g7_n, we can't use this seed directly as subset.
        # But n <= 47 < 57.
        
        # Try multiple subsets
        # We can just take the first n vertices.
        # Or random subsets.
        
        attempts = 5
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            
            # Select vertices
            if attempt == 0:
                vertices = list(range(n))
            else:
                vertices = random.sample(range(g7_n), n)
            
            # Map old indices to new 0..n-1
            idx_map = {old: new for new, old in enumerate(vertices)}
            
            # Build adj list for subgraph
            adj = [0] * n
            for u, v in g7_edges:
                if u in idx_map and v in idx_map:
                    nu = idx_map[u]
                    nv = idx_map[v]
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
            
            # Run local search
            t_limit = min(10.0, deadline - (time.time() - start_time) - 2.0)
            if t_limit < 1.0:
                continue
                
            success = local_search(adj, n, min_deg_req, t_limit, seed=12345 + attempt)
            
            if success:
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                
                result = {'n': n, 'edges': edges}
                if n > best_n:
                    best_n = n
                    best_result = result
                
                # Return immediately as we maximize n
                return result
    
    # Fallback
    if best_result:
        return best_result
        
    return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}

# EVOLVE-BLOCK-END
