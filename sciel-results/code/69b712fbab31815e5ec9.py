import random
import time

def construct():
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16) (the Clebsch graph three times),
    # doubled: two copies with colour 3 on all edges between them (a bipartite colour class).
    # This is the seed solution for n=32. We will try to exceed this.
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # We will try to construct up to n=50.
    # If we fail, we return the best n found.
    
    best_n = 32
    best_colors = None
    
    # Initialize seed solution for n=32 to have a fallback
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    best_colors = seed_colors

    def try_construct(target_n):
        nonlocal best_n, best_colors
        color = [[0] * target_n for _ in range(target_n)]
        # color[i][j] is the color of edge (i, j)
        
        # Start with n=1
        n = 1
        # No edges for n=1
        
        while n < target_n:
            if time.time() - start_time > time_limit:
                return False
            
            # We want to add vertex n (index n)
            # Edges to add: (n, u) for u in 0..n-1
            # Let new_colors[u] be the color of (n, u)
            
            # We use a local search to find new_colors
            # Objective: Minimize triangles, then maximize min-slack
            
            def evaluate(new_colors):
                # Returns (num_triangles, min_slack)
                # num_triangles: count of monochromatic triangles involving vertex n
                # min_slack: min over all edges (i,j) of the graph (including new ones) of slack(i,j)
                
                tri_count = 0
                # Check triangles involving n
                # Triangle (n, u, w) with u < w < n
                # Color of (n,u) is new_colors[u]
                # Color of (n,w) is new_colors[w]
                # Color of (u,w) is color[u][w]
                for u in range(n):
                    cu = new_colors[u]
                    for w in range(u + 1, n):
                        cw = new_colors[w]
                        if cu == cw and cu == color[u][w]:
                            tri_count += 1
                
                # If triangles exist, we don't care about slack much
                if tri_count > 0:
                    return (tri_count, 0)
                
                # Calculate min slack for the whole graph
                # Slack(i, j) for edge (i,j) of color c:
                # Number of k != i,j such that NOT (color(i,k)==c AND color(j,k)==c)
                # This is equivalent to: (n_total - 2) - (number of k such that color(i,k)==c AND color(j,k)==c)
                # Let common_c(i,j) be the count of such k.
                # Slack = (n_total - 2) - common_c(i,j)
                
                min_slack = float('inf')
                n_total = n + 1 # Current total vertices after adding n
                
                # We only need to check edges that might have changed or are new?
                # Actually, adding a vertex changes the slack of ALL existing edges because n_total increases.
                # And the common_c counts for existing edges might change?
                # No, common_c(i,j) for existing edges (i,j) depends on vertices k.
                # The new vertex n is a new k.
                # So common_c(i,j) increases by 1 if color(i,n)==c and color(j,n)==c.
                # So we need to recalculate or update.
                
                # For efficiency, we can just calculate for all edges.
                # O(n^3) might be too slow for n=50 in Python inside a local search loop.
                # n=50, edges ~1225. For each edge, check 50 vertices. 1225 * 50 = 60k ops.
                # Local search steps: 1000. 60M ops. Might be slow in Python.
                
                # Optimization:
                # We can precompute common counts for existing graph.
                # But for now, let's try a simpler slack definition or approximation.
                # Or, just use tri_count as the primary metric and a simpler slack.
                
                # Let's use a faster slack:
                # For the NEW edges (n, u), the slack is (n-1) - common_new(n, u).
                # common_new(n, u) is the number of w such that new_colors[w]==new_colors[u] and color[u][w]==new_colors[u].
                # This is exactly the number of triangles involving edge (n,u).
                # Since we require 0 triangles, this is 0.
                # So new edges have max slack.
                
                # The risk is in EXISTING edges.
                # Adding n might create "near-triangles" for existing edges.
                # For an existing edge (u,w) of color c, the new vertex n is a threat if
                # color(u,n)==c and color(w,n)==c.
                # This increases common_c(u,w) by 1.
                # So slack(u,w) decreases by 1.
                
                # So we want to minimize the number of existing edges (u,w) that are "threatened" by n.
                # Threatened(u,w) = 1 if color(u,n)==color(w,n)==color(u,w).
                
                threats = 0
                for u in range(n):
                    cu = new_colors[u]
                    for w in range(u + 1, n):
                        if cu == new_colors[w] and cu == color[u][w]:
                            # This is a triangle, already counted in tri_count
                            # But if tri_count==0, this never happens.
                            pass
                        # Wait, if tri_count==0, then for any u,w, it is NOT the case that
                        # new_colors[u]==new_colors[w]==color[u][w].
                        # So no existing edge is part of a monochromatic triangle with n.
                        # However, the "slack" is reduced if new_colors[u]==c and new_colors[w]==c
                        # where c is NOT color[u][w]? No.
                        # Slack reduction happens if new_colors[u]==color[u][w] AND new_colors[w]==color[u][w].
                        # i.e., new_colors[u] == new_colors[w] == color[u][w].
                        # This is exactly the triangle condition.
                        # So if tri_count==0, no existing edge's slack is reduced?
                        # Let's re-read slack definition.
                        # Slack(i,j) = #k such that NOT(color(i,k)==c AND color(j,k)==c).
                        # If we add n, and color(n,i)==c and color(n,j)==c, then for edge (i,j) of color c,
                        # the term for k=n becomes "color(n,i)==c AND color(n,j)==c" which is True.
                        # So the "NOT" becomes False.
                        # So the count of "NOT" decreases by 1.
                        # This happens IF color(n,i)==c AND color(n,j)==c.
                        # It does NOT require color(i,j)==c.
                        # Ah! My previous triangle logic was for triangles involving n.
                        # But slack reduction for existing edge (i,j) happens if n connects to both i and j with color c,
                        # REGARDLESS of the color of (i,j).
                        # Wait, if color(i,j) is d != c, then the edge (i,j) is not of color c.
                        # The slack is defined for the edge's color.
                        # So for edge (i,j) of color d, we look for k such that color(i,k)!=d OR color(j,k)!=d.
                        # If color(n,i)==d and color(n,j)==d, then for k=n, the condition is False.
                        # So slack decreases.
                        # So, threats to existing edge (u,w) of color c are:
                        # new_colors[u] == c AND new_colors[w] == c.
                        
                for u in range(n):
                    cu = new_colors[u]
                    for w in range(u + 1, n):
                        if cu == new_colors[w] and cu == color[u][w]:
                             # This is a triangle, already handled
                             pass
                        # Actually, the threat condition is:
                        # new_colors[u] == color[u][w] AND new_colors[w] == color[u][w]
                        # This is exactly cu == color[u][w] and cw == color[u][w].
                        # Which implies cu == cw.
                        # So yes, it's the same condition as forming a triangle with n.
                        # So if tri_count == 0, then for all u,w, it is NOT the case that
                        # new_colors[u] == color[u][w] AND new_colors[w] == color[u][w].
                        # So no existing edge's slack is reduced?
                        # Let's check.
                        # Edge (u,w) color c.
                        # Threat if new_colors[u]==c and new_colors[w]==c.
                        # If new_colors[u]==c and new_colors[w]==c, then since color[u][w]==c,
                        # we have a triangle (n,u,w).
                        # So if tri_count==0, there are NO threats to existing edges' slack?
                        # This implies that if we maintain 0 triangles, the slack of existing edges doesn't drop?
                        # That seems too good to be true.
                        # Let's re-evaluate.
                        # Slack(i,j) for color c: #k where NOT(color(i,k)==c AND color(j,k)==c).
                        # If we add n, and color(n,i)==c and color(n,j)==c, then for k=n, the condition is False.
                        # So slack decreases by 1.
                        # This requires color(n,i)==c and color(n,j)==c.
                        # It does NOT require color(i,j)==c.
                        # My previous deduction "implying cu==cw" is correct.
                        # But does it imply a triangle?
                        # Triangle requires color(i,j)==c as well.
                        # So if color(i,j) != c, then color(n,i)==c and color(n,j)==c does NOT form a triangle.
                        # But it DOES reduce the slack of edge (i,j) if edge (i,j) is of color c?
                        # No, if edge (i,j) is of color d != c, we don't care about c.
                        # We care about d.
                        # So for edge (i,j) of color d, threat is color(n,i)==d and color(n,j)==d.
                        # So, yes, the threat condition for edge (i,j) of color d is
                        # new_colors[i]==d and new_colors[j]==d.
                        # This is independent of whether new_colors[i]==new_colors[j].
                        # Wait, if new_colors[i]==d and new_colors[j]==d, then they are equal.
                        # So the condition is: new_colors[i] == new_colors[j] == color[i][j].
                        # This IS the triangle condition!
                        # So if tri_count == 0, then for all i,j, it is NOT the case that
                        # new_colors[i] == new_colors[j] == color[i][j].
                        # So no existing edge's slack is reduced?
                        # Yes, it seems so.
                        # So the "slack" of the existing graph is preserved if we add a vertex with 0 new triangles.
                        # The only "slack" change is for the NEW edges.
                        # And new edges have max slack if 0 triangles.
                        # So if we always add with 0 triangles, the graph remains "maximally slack"?
                        # That suggests the greedy approach is very powerful if it can always find a 0-triangle assignment.
                        
                return (0, n) # Placeholder slack

            # Local Search
            new_colors = [random.randint(0, 3) for _ in range(n)]
            best_local = None
            best_score = None
            
            for _ in range(500):
                tri, _ = evaluate(new_colors)
                score = tri * 10000
                if best_score is None or score < best_score:
                    best_score = score
                    best_local = new_colors[:]
                
                if tri == 0:
                    break
                    
                # Pick random u
                u = random.randint(0, n - 1)
                old_c = new_colors[u]
                new_c = random.choice([c for c in range(4) if c != old_c])
                new_colors[u] = new_c
                
            if best_local is not None:
                tri, _ = evaluate(best_local)
                if tri == 0:
                    # Commit
                    for u in range(n):
                        color[n][u] = best_local[u]
                        color[u][n] = best_local[u]
                    n += 1
                    continue
            # If stuck, restart
            return False

        # If we reached target_n
        res_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                res_colors.append(color[i][j])
        
        if n > best_n:
            best_n = n
            best_colors = res_colors
        return True

    # Try to reach 50
    # We can try multiple times with different seeds
    attempts = 0
    while time.time() - start_time < time_limit and best_n < 50:
        attempts += 1
        random.seed(attempts)
        try_construct(50)
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
