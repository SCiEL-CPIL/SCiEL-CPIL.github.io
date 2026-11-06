import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit (or 40s for 150s limit? Prompt says 120s evaluator, 150s kill. Use 110s safe margin)
    
    # 1. Construct Polarity Graph of PG(2, 9)
    # GF(9) representation
    # We need arithmetic in GF(9)
    # Let's use a simple table or polynomial representation.
    # Irreducible polynomial x^2 + 1 over GF(3) since -1 is not a square in GF(3)? 
    # Squares in GF(3): 0^2=0, 1^2=1, 2^2=4=1. So -1=2 is not a square.
    # So x^2+1 is irreducible.
    # Elements are (a, b) representing a + b*x where x^2 = -1 = 2.
    # Addition is component-wise mod 3.
    # Multiplication: (a,b)(c,d) = (ac - bd, ad + bc) mod 3?
    # (a+bx)(c+dx) = ac + adx + bcx + bd x^2 = ac + 2bd + (ad+bc)x.
    # So (ac+2bd mod 3, (ad+bc) mod 3).
    
    def gf9_add(a, b):
        return ((a[0]+b[0])%3, (a[1]+b[1])%3)
        
    def gf9_mul(a, b):
        return ((a[0]*b[0] + 2*a[1]*b[1])%3, (a[0]*b[1] + a[1]*b[0])%3)
        
    def gf9_sub(a, b):
        return ((a[0]-b[0])%3, (a[1]-b[1])%3)
        
    def gf9_neg(a):
        return ((-a[0])%3, (-a[1])%3)
        
    def gf9_inv(a):
        # a^2 + b^2? No, norm is a^2 - 2b^2 = a^2 + b^2 mod 3?
        # (a+bx)(a-bx) = a^2 - b^2 x^2 = a^2 - 2b^2 = a^2 + b^2.
        # If a=0,b=0 no inv.
        norm = (a[0]*a[0] + a[1]*a[1]) % 3
        if norm == 0:
            return (0,0)
        # inv of norm in GF(3)
        inv_norm = 1 if norm == 1 else 2
        return gf9_mul(a, (inv_norm, 0))
        
    # Generate all points in PG(2, 9)
    # Points are 1D subspaces of GF(9)^3.
    # Canonical form: (x,y,1) or (x,0,1) or (0,0,1) ?
    # Standard enumeration:
    # (x, y, 1) for x,y in GF(9) -> 81 points
    # (x, 1, 0) for x in GF(9) -> 9 points
    # (1, 0, 0) -> 1 point
    # Total 91 points.
    
    points = []
    idx_to_point = {}
    
    # Helper to get canonical representative
    def canon(p):
        x, y, z = p
        if z != (0,0):
            inv_z = gf9_inv(z)
            return (gf9_mul(x, inv_z), gf9_mul(y, inv_z), (1,0))
        elif y != (0,0):
            inv_y = gf9_inv(y)
            return (gf9_mul(x, inv_y), (1,0), (0,0))
        else:
            return ((1,0), (0,0), (0,0))
            
    # We need to be careful with GF(9) zero
    ZERO = (0,0)
    
    for x in [(i, j) for i in range(3) for j in range(3)]:
        for y in [(i, j) for i in range(3) for j in range(3)]:
            p = canon((x, y, (1,0)))
            if p not in idx_to_point:
                idx_to_point[p] = len(points)
                points.append(p)
                
    for x in [(i, j) for i in range(3) for j in range(3)]:
        p = canon((x, (1,0), (0,0)))
        if p not in idx_to_point:
            idx_to_point[p] = len(points)
            points.append(p)
            
    p = canon(((1,0), (0,0), (0,0)))
    if p not in idx_to_point:
        idx_to_point[p] = len(points)
        points.append(p)
        
    N = len(points)
    assert N == 91, f"PG(2,9) should have 91 points, got {N}"
    
    # Build adjacency matrix for Polarity Graph
    # P is adjacent to Q if P lies on polar line of Q.
    # Polar line of Q=(x,y,z) is X*x + Y*y + Z*z = 0.
    # P=(u,v,w) is on it if u*x + v*y + w*z = 0.
    
    adj = [[False]*N for _ in range(N)]
    
    for i in range(N):
        pi = points[i]
        for j in range(N):
            pj = points[j]
            # Check dot product
            # pi[0]*pj[0] + pi[1]*pj[1] + pi[2]*pj[2] == 0
            dot = gf9_add(gf9_add(gf9_mul(pi[0], pj[0]), gf9_mul(pi[1], pj[1])), gf9_mul(pi[2], pj[2]))
            if dot == ZERO:
                adj[i][j] = True
                
    # Verify C4-free and degree
    # Degree should be q+1 = 10.
    for i in range(N):
        deg = sum(adj[i])
        if deg != 10:
            # Polarity of PG(2,q) is regular of degree q+1
            # Exception: if q is even, there are absolute points (self-adjacent)?
            # For q=9 (odd), no absolute points?
            # Actually, for odd q, the polarity graph is (q+1)-regular.
            # For even q, it is (q+1)-regular with loops? No, simple graph.
            # Let's check.
            pass

    # We want a subset of size n with high min degree.
    # Target n=47, min_deg >= 8.
    # Target n=46, min_deg >= 7.
    
    best_result = None
    best_score = 0
    
    def evaluate_subset(S):
        # S is a set of indices
        n = len(S)
        S_set = set(S)
        min_deg = n # Init high
        for v in S:
            deg = 0
            for u in S:
                if u != v and adj[v][u]:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
        # Check C4 free?
        # The induced subgraph of a C4-free graph is C4-free.
        # So we just need to check min_deg condition.
        # Condition: min_deg >= n - 39
        if min_deg >= n - 39:
            # Valid
            return n, S, min_deg
        else:
            return 0, S, min_deg
            
    # Greedy Randomized Search
    # We will try multiple random initial subsets and improve them.
    
    for n_target in [47, 46, 45]:
        target_min_deg = n_target - 39
        found = False
        
        for _ in range(200): # 200 restarts
            if time.time() > deadline:
                break
                
            # Random initial subset
            S = set(random.sample(range(N), n_target))
            
            # Local Search: Swap vertices to improve min degree
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Calculate degrees in S
                degs = {}
                for v in S:
                    d = 0
                    for u in S:
                        if u != v and adj[v][u]:
                            d += 1
                    degs[v] = d
                    
                min_deg = min(degs.values())
                if min_deg >= target_min_deg:
                    # Success!
                    found = True
                    # Store result
                    edges = []
                    S_list = list(S)
                    for i in range(len(S_list)):
                        for j in range(i+1, len(S_list)):
                            if adj[S_list[i]][S_list[j]]:
                                edges.append([S_list[i], S_list[j]])
                    # Map vertices to 0..n-1
                    map_idx = {v: i for i, v in enumerate(S_list)}
                    new_edges = []
                    for e in edges:
                        new_edges.append([map_idx[e[0]], map_idx[e[1]]])
                    
                    result = {'n': n_target, 'edges': new_edges}
                    if best_result is None or n_target > best_result['n']:
                        best_result = result
                        best_score = n_target
                    break
                else:
                    # Try to swap a low-degree vertex in S with a vertex not in S
                    # Sort S by degree
                    sorted_S = sorted(S, key=lambda v: degs[v])
                    # Try swapping the lowest degree vertex
                    v_low = sorted_S[0]
                    
                    # Candidates: vertices not in S
                    # We want a vertex u such that replacing v_low with u increases min degree
                    # Ideally u has high degree within S \ {v_low}
                    
                    best_swap = None
                    best_swap_score = -1
                    
                    # To speed up, only check a random sample of outside vertices or all?
                    # N=91, n=47. Outside = 44. Checking all is fast.
                    outside = [v for v in range(N) if v not in S]
                    
                    for u in outside:
                        # Calculate degree of u in S \ {v_low}
                        deg_u_new = 0
                        for w in S:
                            if w != v_low and adj[u][w]:
                                deg_u_new += 1
                        # New min degree would be min(min_deg without v_low, deg_u_new)
                        # Actually, removing v_low might increase min degree if v_low was the unique min.
                        # Let's just calculate new min degree roughly.
                        # New S' = S - {v_low} + {u}
                        # We need min(deg(w) for w in S', where deg is in S')
                        # This is O(n^2) per swap. 44 * 47^2 ~ 97k ops. Fast.
                        
                        # Optimization: Only consider u if deg_u_new > min_deg
                        if deg_u_new <= min_deg:
                            continue
                            
                        # Calculate new min degree
                        new_min = min(deg_u_new, min(degs[w] for w in S if w != v_low))
                        if new_min > min_deg:
                            if new_min > best_swap_score:
                                best_swap_score = new_min
                                best_swap = u
                                
                    if best_swap is not None:
                        S.remove(v_low)
                        S.add(best_swap)
                        improved = True
                    else:
                        # Try swapping a different low vertex?
                        # Or break
                        pass
                        
            if found:
                break
                
        if found:
            break

    if best_result is None:
        # Fallback: Empty graph on 39 vertices
        return {'n': 39, 'edges': []}
        
    return best_result
