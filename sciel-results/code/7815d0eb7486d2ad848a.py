import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # 1. Try Voltage Lift on K6
    # K6 is 5-regular.
    # Vertices 0..5
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    
    # Find all closed walks of length 3 to 8 in K6
    # We represent walks as sequences of vertices.
    # Since K6 is complete, any sequence of distinct vertices (except start=end) is a walk?
    # No, backtracking is allowed in walks, but for girth we care about simple cycles?
    # No, girth is shortest *cycle*. A cycle is a simple closed walk.
    # So we only need to check simple cycles.
    # Simple cycles in K6:
    # Length 3: Triangles.
    # Length 4: Squares.
    # Length 5: Pentagons.
    # Length 6: Hexagons.
    
    # We only need to ensure that the voltage sum around any simple cycle of length < g is not 0.
    # If a non-simple walk sums to 0, it contains a simple cycle that sums to 0?
    # Not necessarily, but if all simple cycles have non-zero sum, does it imply girth >= g?
    # Yes, because any closed walk can be decomposed into simple cycles.
    # If the sum of the walk is 0, and it's a concatenation of cycles $C_1, C_2, \dots$,
    # then $\sum v(C_i) = 0$.
    # This doesn't imply any $v(C_i) = 0$.
    # However, for girth, we care about the *shortest* closed walk with sum 0.
    # If we ensure all simple cycles of length < g have non-zero sum,
    # is it possible a non-simple walk of length < g has sum 0?
    # A non-simple walk of length $L < g$ must repeat a vertex.
    # It contains a simple cycle of length $L' < L < g$.
    # If the sum of the simple cycle is non-zero, can the whole walk sum to 0?
    # Yes, if the "tail" cancels it.
    # But the tail is a path.
    # Actually, the standard condition for lift girth is that no *closed walk* of length < g has sum 0.
    # But checking all closed walks is exponential.
    # However, for small base graphs, we can just check all simple cycles.
    # It is known that if all simple cycles of length < g have non-zero voltage, the lift has girth >= g?
    # No, counterexamples exist.
    # But for K6, the number of simple cycles is small.
    # Let's check all simple cycles of length 3 to 8.
    
    simple_cycles = []
    # Generate all simple cycles in K6
    # Since K6 is small, we can just iterate over permutations?
    # Or use a DFS.
    def find_cycles(graph, n):
        cycles = []
        for start in range(n):
            # DFS to find cycles starting and ending at start
            # Only allow vertices > start to avoid duplicates?
            # No, we just need unique sets of edges.
            # Let's just generate all sequences of length L from start, with distinct vertices.
            for L in range(3, g):
                for perm in __import__('itertools').permutations(range(n), L-1):
                    if start in perm: continue
                    # Check if start is adjacent to last (always true in K6)
                    # Check if all consecutive are adjacent (always true in K6)
                    cycle = [start] + list(perm)
                    # Add to list
                    simple_cycles.append(cycle)
        return simple_cycles

    # This is inefficient. Let's just hardcode or generate K6 cycles properly.
    # K6 has 20 triangles, 45 squares, 12 pentagons, etc.
    
    # Let's try a specific m
    # If we find a valid voltage assignment, we build the lift.
    
    def try_lift(m):
        # Initialize random voltages
        voltages = {e: random.randint(0, m-1) for e in edges_k6}
        
        # Get simple cycles
        # We can generate them on the fly or precompute.
        # For K6, we can just check all triangles, squares, pentagons.
        
        # Precompute simple cycles of K6
        cycles = []
        # Triangles
        for i in range(6):
            for j in range(i+1, 6):
                for l in range(j+1, 6):
                    cycles.append([(i, j), (j, l), (l, i)])
        # Squares
        # A square is 4 distinct vertices a,b,c,d. Edges (a,b), (b,c), (c,d), (d,a).
        # In K6, any 4 vertices form 3 squares.
        for i in range(6):
            for j in range(i+1, 6):
                for k in range(j+1, 6):
                    for l in range(k+1, 6):
                        # Vertices i, j, k, l
                        # Squares: (i,j,k,l), (i,j,l,k), (i,k,j,l)
                        # Edges for (i,j,k,l): (i,j), (j,k), (k,l), (l,i)
                        cycles.append([(i, j), (j, k), (k, l), (l, i)])
                        cycles.append([(i, j), (j, l), (l, k), (k, i)])
                        cycles.append([(i, k), (k, j), (j, l), (l, i)])
        # Pentagons
        # 5 distinct vertices. (5-1)!/2 = 12 per set of 5.
        # Total sets of 5: 6.
        # Total pentagons: 6 * 12 = 72?
        # Let's just generate all permutations of 5 vertices from 6.
        # This is getting complex.
        # Let's just use the non-backtracking check on the *lift* directly?
        # No, that's what we want to avoid.
        
        # Let's assume checking triangles and squares is enough for a first pass?
        # No, we need girth 9.
        
        # Let's try a different approach:
        # Just generate the lift for a random voltage assignment and check girth.
        # If girth < 9, adjust voltages.
        
        def build_lift(voltages, m):
            n_lift = 6 * m
            edges_lift = []
            # Vertex (v, i) where v in 0..5, i in 0..m-1
            # Edge between (u, i) and (v, i + voltage(u,v))
            for u, v in edges_k6:
                vol = voltages[(u, v)]
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + ((i + vol) % m)
                    edges_lift.append((u_lift, v_lift))
            return n_lift, edges_lift

        def check_girth_lift(n, edges, g):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check for cycles < g
            # Use non-backtracking BFS from each vertex
            for v in range(n):
                # Start from v
                # Queue: (curr, prev, depth)
                q = deque()
                for u in adj[v]:
                    q.append((u, v, 1))
                visited = set()
                for u in adj[v]:
                    visited.add((u, v))
                    
                while q:
                    curr, prev, depth = q.popleft()
                    if curr == v and depth < g:
                        return False
                    for next_v in adj[curr]:
                        if next_v == prev:
                            continue
                        if (next_v, curr) not in visited:
                            visited.add((next_v, curr))
                            q.append((next_v, curr, depth + 1))
            return True

        # Local search on voltages
        for iter in range(100):
            n_lift, edges_lift = build_lift(voltages, m)
            if check_girth_lift(n_lift, edges_lift, g):
                return n_lift, edges_lift
            
            # Find a short cycle and modify voltage
            # This is hard to implement quickly.
            # Just re-randomize?
            voltages = {e: random.randint(0, m-1) for e in edges_k6}
            
        return None

    # Try small m
    for m in [10, 20, 50, 100, 200, 500]:
        if time.time() > deadline - 10: break
        res = try_lift(m)
        if res:
            return res

    # Fallback to LUW
    # ... (existing LUW code) ...
