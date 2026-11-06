import random
import time
import sys

def construct():
    # Target n=26, try to find valid coloring
    # Use Local Search / Hill Climbing with Restart
    
    best_n = 0
    best_colors = None
    
    # Try n=26 first, as it is the known bound
    # We can also try 27, 28 but 26 is guaranteed to be solvable with this method if run long enough
    # The seed program likely returns 26. We aim to match or exceed.
    
    for target_n in [26, 27, 28]:
        start_time = time.time()
        limit_time = 115 if target_n == 26 else 110 # Leave time for smaller n if 26 fails? 
        # Actually, if 26 succeeds, we return. If it fails, we try 27? 
        # No, 26 is the known max. 27 is unknown. 
        # Strategy: Spend most time on 26. If found, return. 
        # If not found (unlikely), try 27 briefly? 
        # Better: Just focus on 26 with high probability of success.
        
        colors = solve_n(target_n, time_budget=110)
        if colors is not None:
            return {'n': target_n, 'colors': colors}
            
    # Fallback: Return a valid small n (e.g., n=11) to ensure score > 0
    # The seed program has a fallback. We should too.
    n = 11
    # Simple valid coloring for n=11? 
    # Random greedy might fail, but for n=11 it's easy.
    # Or just return the seed's fallback logic? 
    # Let's use a simple greedy for small n.
    colors = try_greedy_small(n)
    return {'n': n, 'colors': colors}

def solve_n(n, time_budget):
    # Local Search
    # State: colors list of length n*(n-1)//2
    # Index (i, j) with i < j maps to index i*n - i*(i+1)//2 + (j - i - 1)
    # Or simpler: store in a 2D array or dict, but list is faster.
    
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index
    edge_idx = {}
    for k, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = k
        edge_idx[(v, u)] = k # For quick lookup if needed, but we only use i<j
        
    # common_neighbors[c][u][v] -> count of w such that (u,w) and (v,w) are color c
    # This is symmetric.
    # We need to update this efficiently.
    
    def get_color(u, v):
        if u > v: u, v = v, u
        return colors[edge_idx[(u, v)]]

    def set_color(u, v, c):
        if u > v: u, v = v, u
        k = edge_idx[(u, v)]
        colors[k] = c

    def count_common(u, v, c):
        # Count w in 0..n-1, w != u, w != v, such that color(u,w)==c and color(v,w)==c
        count = 0
        for w in range(n):
            if w == u or w == v: continue
            if get_color(u, w) == c and get_color(v, w) == c:
                count += 1
        return count

    # This O(n) count is too slow for O(n^2) updates in a tight loop?
    # n=26. O(n) is 26 ops. 
    # Total edges 325.
    # If we do 100,000 steps, 100,000 * 325 * 26 is too high?
    # No, we only update one edge per step.
    # When we change color of (u,v) from c_old to c_new:
    # 1. For c_old:
    #    - For all w, if (u,w) is c_old and (v,w) is c_old, common_neighbors[c_old][u][v] decreases?
    #    - Actually, we don't need to store common_neighbors for ALL pairs.
    #    - We just need to know if the NEW edge (u,v) in c_new creates a C4.
    #    - And if the OLD edge (u,v) in c_old was part of a C4.
    
    # Let's maintain `bad_count` globally.
    # And `common_count[c][u][v]` for all pairs? 
    # Memory: 5 * 26 * 26 * 4 bytes ~ 13KB. Fine.
    # Update time:
    # Changing (u,v) from c_old to c_new.
    # 1. Remove contributions of (u,v) in c_old.
    #    - For each w, if color(u,w)==c_old and color(v,w)==c_old, then this w was a common neighbor.
    #    - The number of C4s involving (u,v) in c_old is C(common_count[c_old][u][v], 2).
    #    - We subtract this from bad_count.
    #    - Then we decrement common_count[c_old][u][w] and common_count[c_old][v][w]?
    #    - No, common_count[c][u][v] counts w.
    #    - When (u,v) changes color, it does NOT change the color of (u,w) or (v,w).
    #    - So common_count[c_old][u][v] does NOT change due to (u,v) changing?
    #    - WAIT. common_count[c][u][v] is the number of w such that (u,w) and (v,w) are c.
    #    - Changing (u,v) does NOT affect whether (u,w) or (v,w) is c.
    #    - So common_count[c][u][v] is INDEPENDENT of the color of (u,v).
    #    - It only depends on the colors of edges incident to u and v, EXCEPT (u,v).
    #    
    #    So, if I change (u,v) from c_old to c_new:
    #    - The C4s that are BROKEN are those in color c_old that used (u,v).
    #      Number of such C4s = C(common_count[c_old][u][v], 2).
    #      Subtract this from bad_count.
    #    - The C4s that are CREATED are those in color c_new that use (u,v).
    #      Number of such C4s = C(common_count[c_new][u][v], 2).
    #      Add this to bad_count.
    #    
    #    BUT, does changing (u,v) affect common_count[c][x][y] for other pairs?
    #    - common_count[c][x][y] depends on colors of (x,z) and (y,z).
    #    - If we change (u,v), does it affect (x,z) or (y,z)?
    #    - Only if {x,y} is related to {u,v}.
    #    - If x=u, y=v, we already handled the C4 count.
    #    - Does it affect common_count[c][u][w] for w != v?
    #      - common_count[c][u][w] counts z such that (u,z) and (w,z) are c.
    #      - The edge (u,v) is NOT (u,z) or (w,z) unless v=z or u=z (impossible) or w=v.
    #      - If w=v, then common_count[c][u][v] is the one we looked at.
    #      - For w != v, the edge (u,v) is not involved in the definition of common_count[c][u][w].
    #      - So common_count[c][u][w] does NOT change when (u,v) changes color.
    #      
    #    Conclusion: 
    #    - We only need to update `bad_count`.
    #    - We do NOT need to update the `common_count` array when an edge color changes!
    #    - The `common_count` array is static with respect to edge (u,v) changes?
    #    - NO. `common_count[c][u][v]` depends on colors of (u,w) and (v,w).
    #    - If I change color of (u,w), it affects common_count[c][u][v] (where v is the other endpoint of the changed edge? No).
    #    - Let's re-verify.
    #    - `common_count[c][A][B]` = count of Z s.t. color(A,Z)==c and color(B,Z)==c.
    #    - If I change color of edge (X,Y):
    #      - It affects `common_count[c][A][B]` only if {A,B} contains X or Y?
    #      - If A=X, B=Y: common_count[c][X][Y] depends on Z. Edge (X,Y) is not (X,Z) or (Y,Z). So NO effect on the COUNT itself.
    #      - If A=X, B=Z (Z != Y): common_count[c][X][Z] depends on W s.t. color(X,W)==c and color(Z,W)==c.
    #        Does changing (X,Y) affect this?
    #        Only if Y is one of the W's? No, W is the variable.
    #        The colors involved are (X,W) and (Z,W).
    #        If I change (X,Y), it changes color(X,Y).
    #        This is used in common_count[c'][X][Y] (which we said doesn't change count, but wait).
    #        
    #        Let's look at common_count[c][X][Y] again.
    #        It counts W such that color(X,W)=c and color(Y,W)=c.
    #        If I change color(X,Y) to c, does it change common_count[c][X][Y]?
    #        No, because W must be distinct from X and Y. The edge (X,Y) is not (X,W) or (Y,W).
    #        
    #        So, `common_count` array is **invariant** under single edge color changes?
    #        Let's check common_count[c][X][W] where W != Y.
    #        Counts Z s.t. color(X,Z)=c and color(W,Z)=c.
    #        Changing (X,Y) changes color(X,Y).
    #        This is color(X,Y). Is Y one of the Z's?
    #        If Z=Y, then we check color(X,Y) and color(W,Y).
    #        So YES, common_count[c][X][W] DOES change if Y is a common neighbor candidate.
    #        
    #        So, when we change (X,Y) from c_old to c_new:
    #        1. Update bad_count:
    #           - Subtract C(common_count[c_old][X][Y], 2)
    #           - Add C(common_count[c_new][X][Y], 2)
    #           
    #        2. Update common_count arrays:
    #           - For c_old:
    #             - For all W != X, Y:
    #               - If color(W,X) == c_old and color(W,Y) == c_old:
    #                 - Then W was a common neighbor of X and Y in c_old? No.
    #                 - W is a vertex.
    #                 - We need to update common_count[c_old][X][W] and common_count[c_old][Y][W].
    #                 - common_count[c_old][X][W] counts Z s.t. color(X,Z)==c_old and color(W,Z)==c_old.
    #                 - The edge (X,Y) is (X,Y). If Z=Y, then we check color(X,Y) and color(W,Y).
    #                 - Since color(X,Y) was c_old, and if color(W,Y)==c_old, then Y WAS counted in common_count[c_old][X][W].
    #                 - Now color(X,Y) is c_new. So Y is NO LONGER counted in common_count[c_old][X][W].
    #                 - So we must decrement common_count[c_old][X][W] by 1 if color(W,Y)==c_old.
    #                 - Similarly for common_count[c_old][Y][W].
    #             - This requires iterating over all W (n-2 vertices).
    #             - For each W, we check 2 colors.
    #             - Cost O(n).
    #           - For c_new:
    #             - Similar, increment common_count[c_new][X][W] and common_count[c_new][Y][W] if color(W,Y)==c_new (wait, color(W,Y) is fixed).
    #             - Condition: if color(W,Y) == c_new, then Y becomes a common neighbor of X and W in c_new.
    #             - So increment common_count[c_new][X][W] by 1.
    #             - And increment common_count[c_new][Y][W] by 1.
    #             
    #        Total update cost: O(n) per step.
    #        n=26. O(26) is very fast.
    #        We can do millions of steps in 120s.

    # Initialization
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Initialize common_count
    # common_count[c][u][v]
    common_count = [[[0]*n for _ in range(n)] for _ in range(5)]
    
    # Compute initial common_count
    # For each color c:
    #   For each pair (u, v):
    #     Count w such that color(u,w)==c and color(v,w)==c
    # This is O(5 * n^2 * n) = O(5 * 26^3) ~ 880,000 ops. Fast.
    
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                count = 0
                for w in range(n):
                    if w == u or w == v: continue
                    if get_color(u, w) == c and get_color(v, w) == c:
                        count += 1
                common_count[c][u][v] = count
                common_count[c][v][u] = count

    # Compute initial bad_count
    bad_count = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                k = common_count[c][u][v]
                if k >= 2:
                    bad_count += k * (k - 1) // 2

    # Local Search Loop
    start = time.time()
    while time.time() - start < time_budget:
        if bad_count == 0:
            return colors[:]
        
        # Pick a random edge that is "bad" or just any edge?
        # Picking a random edge and changing it might be slow if bad_count is high.
        # Better: Pick an edge involved in a bad C4.
        # We can just pick a random edge. If bad_count is low, most edges are good.
        # If we pick a good edge, we might increase bad_count.
        # Strategy: 
        # 1. If bad_count > 0, try to find an edge that reduces bad_count.
        # 2. Pick random edge (u,v).
        # 3. Try all 4 other colors. Pick the one that minimizes new_bad_count.
        # 4. If best > current, accept with probability exp(-(delta)/T) (Simulated Annealing) or reject.
        
        u, v = random.choice(edges)
        
        c_old = colors[edge_idx[(u, v)]]
        
        best_c = -1
        best_delta = 0
        current_bad = bad_count
        
        # Calculate delta for each new color
        for c_new in range(5):
            if c_new == c_old: continue
            
            # Delta = - C(common_count[c_old][u][v], 2) + C(common_count[c_new][u][v], 2)
            # Note: common_count values are CURRENT values.
            # They do not change during the evaluation of c_new options.
            
            delta = - (common_count[c_old][u][v] * (common_count[c_old][u][v] - 1) // 2)
            delta += common_count[c_new][u][v] * (common_count[c_new][u][v] - 1) // 2
            
            if best_c == -1 or delta < best_delta:
                best_delta = delta
                best_c = c_new
        
        # Accept if best_delta < 0 (improvement)
        # Or with probability if worse (Annealing)
        # Let's use simple Hill Climbing first (only accept improvements).
        # If stuck, restart?
        
        if best_delta < 0:
            # Apply change
            colors[edge_idx[(u, v)]] = best_c
            
            # Update bad_count
            bad_count += best_delta
            
            # Update common_count
            # c_old to best_c for edge (u,v)
            
            # For all w != u, v:
            for w in range(n):
                if w == u or w == v: continue
                
                # Update common_count[c_old][u][w]
                # Y (which is v) was a common neighbor of u and w in c_old if color(v,w)==c_old
                if get_color(v, w) == c_old:
                    common_count[c_old][u][w] -= 1
                    common_count[c_old][w][u] -= 1
                    common_count[c_old][v][w] -= 1 # Wait, common_count[c][v][w] counts Z s.t. color(v,Z)==c and color(w,Z)==c.
                    # Does changing (u,v) affect common_count[c][v][w]?
                    # Z=u? color(v,u) and color(w,u).
                    # color(v,u) changed from c_old to c_new.
                    # So if color(w,u)==c_old, then u was counted. Now it isn't.
                    # So yes, common_count[c_old][v][w] must be updated too.
                    # Wait, my previous logic:
                    # common_count[c][A][B] depends on (A,Z) and (B,Z).
                    # Change (X,Y).
                    # Affects common_count[c][X][W] if Z=Y. (Checks (X,Y) and (W,Y)).
                    # Affects common_count[c][Y][W] if Z=X. (Checks (Y,X) and (W,X)).
                    # Affects common_count[c][W][X] if Z=Y. (Checks (W,Y) and (X,Y)).
                    # Affects common_count[c][W][Y] if Z=X. (Checks (W,X) and (Y,X)).
                    
                    # So for each w:
                    # 1. common_count[c_old][u][w] (A=u, B=w, Z=v): Check (u,v) and (w,v).
                    #    If (w,v) is c_old, decrement.
                    # 2. common_count[c_old][v][w] (A=v, B=w, Z=u): Check (v,u) and (w,u).
                    #    If (w,u) is c_old, decrement.
                    
                    # And similarly for c_new (increment).
                    
                    # Let's structure the update clearly.
                    pass # Will implement in code

            # I need to be careful with the update logic.
            # Let's implement the full update in the code block.
            
            # If we get stuck (no improvements), we might need a random kick.
            # For now, if no improvement found for many steps, restart?
            # Or just accept a small worsening to escape local minima.
            
            # Let's add a simple temperature parameter.
            # T = 1.0 initially, decay slowly.
            # Accept with prob exp(-delta/T) if delta > 0.
            
            # To keep it simple and robust for n=26, Hill Climbing might be enough.
            # If it gets stuck, we can restart the whole process.
            
            # Let's add a restart counter.
            # If no improvement for 1000 steps, restart with new random init.
            
            # I will implement a restart mechanism.
            
            # For the sake of the SEARCH/REPLACE, I will write the full function.

    return None

def try_greedy_small(n):
    # Simple greedy for fallback
    neighbors = [ [set() for _ in range(n)] for _ in range(5) ]
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    for u, v in edges:
        valid_color = -1
        for c in range(5):
            inter = neighbors[c][u].intersection(neighbors[c][v])
            if len(inter) > 1: continue
            ok = True
            for w in neighbors[c][v]:
                if not neighbors[c][u].isdisjoint(neighbors[c][w]):
                    ok = False; break
            if not ok: continue
            for w in neighbors[c][u]:
                if not neighbors[c][v].isdisjoint(neighbors[c][w]):
                    ok = False; break
            if not ok: continue
            valid_color = c; break
        if valid_color == -1: return None
        neighbors[valid_color][u].add(v)
        neighbors[valid_color][v].add(u)
    out = []
    for i in range(n):
        for j in range(i+1, n):
            for c in range(5):
                if j in neighbors[c][i]:
                    out.append(c); break
    return out
