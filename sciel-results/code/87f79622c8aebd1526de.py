"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    # Target n
    target_n = 50
    max_time = 115
    start_time = time.time()
    
    # Initialize random coloring
    n = target_n
    num_edges = n * (n - 1) // 2
    edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = len(edges)
            edge_to_idx[(j, i)] = len(edges)
            edges.append((i, j))
            
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # violations[e] = number of monochromatic triangles containing edge e
    violations = [0] * num_edges
    
    def update_violation_for_edge(e_idx):
        """Recalculate violations for a specific edge."""
        u, v = edges[e_idx]
        c_uv = colors[e_idx]
        count = 0
        for w in range(n):
            if w == u or w == w == v: # Typo in thought, fix in code
                continue
            # Check if (u,v,w) is monochromatic
            # Edges: (u,v), (u,w), (v,w)
            e_uw = edge_to_idx[(u, w)]
            e_vw = edge_to_idx[(v, w)]
            if colors[e_uw] == c_uv and colors[e_vw] == c_uv:
                count += 1
        violations[e_idx] = count

    def calculate_delta(e_idx, new_color):
        """Calculate change in total cost if edge e_idx is set to new_color."""
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        if old_color == new_color:
            return 0
            
        # Current contribution of this edge to total cost is violations[e_idx]
        # But wait, violations[e_idx] counts triangles where THIS edge is part of a mono triangle.
        # If we change the color, the set of mono triangles changes.
        
        # Let's just compute the new violation count for this edge under new_color
        # and compare with current violations[e_idx].
        # However, changing (u,v) also affects violations of (u,w) and (v,w).
        # The total cost is sum(violations).
        # Delta = (New Sum) - (Old Sum)
        
        # It is safer to compute the delta for the specific edge (u,v) first,
        # then add deltas for neighbors.
        
        # 1. Delta for edge (u,v) itself
        # Old count: violations[e_idx]
        # New count: number of w such that c(u,w)==new_color and c(v,w)==new_color
        new_count_uv = 0
        for w in range(n):
            if w == u or w == v:
                continue
            e_uw = edge_to_idx[(u, w)]
            e_vw = edge_to_idx[(v, w)]
            if colors[e_uw] == new_color and colors[e_vw] == new_color:
                new_count_uv += 1
        
        delta = new_count_uv - violations[e_idx]
        
        # 2. Delta for edges (u,w) and (v,w)
        # For a neighbor edge (u,w), it is part of triangle (u,v,w).
        # The triangle (u,v,w) is mono if c(u,v)==c(u,w)==c(v,w).
        # Before: c(u,v)=old_color.
        # After: c(u,v)=new_color.
        
        # If old_color == c(u,w) == c(v,w), then (u,v,w) was mono.
        #   It contributed 1 to violations[(u,w)] and 1 to violations[(v,w)] and 1 to violations[(u,v)].
        #   If we change to new_color:
        #   If new_color == c(u,w) == c(v,w), it is still mono. Delta 0 for these.
        #   If new_color != c(u,w), it is no longer mono.
        #      So violations[(u,w)] decreases by 1.
        #      violations[(v,w)] decreases by 1.
        #      violations[(u,v)] decreases by 1 (handled above).
        
        # If old_color != c(u,w) or c(u,w) != c(v,w), it was NOT mono.
        #   It contributed 0.
        #   If we change to new_color:
        #   If new_color == c(u,w) == c(v,w), it becomes mono.
        #      violations[(u,w)] increases by 1.
        #      violations[(v,w)] increases by 1.
        #      violations[(u,v)] increases by 1 (handled above).

        for w in range(n):
            if w == u or w == v:
                continue
            e_uw = edge_to_idx[(u, w)]
            e_vw = edge_to_idx[(v, w)]
            c_uw = colors[e_uw]
            c_vw = colors[e_vw]
            
            # Check if triangle (u,v,w) was mono with old_color
            was_mono = (old_color == c_uw and old_color == c_vw)
            # Check if triangle (u,v,w) will be mono with new_color
            is_mono = (new_color == c_uw and new_color == c_vw)
            
            if was_mono and not is_mono:
                # Triangle stops being mono.
                # (u,w) loses 1 violation
                # (v,w) loses 1 violation
                delta -= 1
                delta -= 1
            elif not was_mono and is_mono:
                # Triangle starts being mono.
                # (u,w) gains 1 violation
                # (v,w) gains 1 violation
                delta += 1
                delta += 1
                
        return delta

    def update_violations_after_change(e_idx, old_color, new_color):
        """Update the violations array after changing edge e_idx."""
        u, v = edges[e_idx]
        
        # Update (u,v)
        new_count_uv = 0
        for w in range(n):
            if w == u or w == v:
                continue
            e_uw = edge_to_idx[(u, w)]
            e_vw = edge_to_idx[(v, w)]
            if colors[e_uw] == new_color and colors[e_vw] == new_color:
                new_count_uv += 1
        violations[e_idx] = new_count_uv
        
        # Update neighbors
        for w in range(n):
            if w == u or w == v:
                continue
            e_uw = edge_to_idx[(u, w)]
            e_vw = edge_to_idx[(v, w)]
            c_uw = colors[e_uw]
            c_vw = colors[e_vw]
            
            # Recalculate violations for (u,w) and (v,w)
            # Actually, we can just recalculate from scratch for these O(n) edges.
            # It's O(n) per edge, so O(n^2) total. Acceptable.
            
            # Recalc (u,w)
            count_uw = 0
            for x in range(n):
                if x == u or x == w:
                    continue
                e_ux = edge_to_idx[(u, x)]
                e_wx = edge_to_idx[(w, x)]
                if colors[e_ux] == c_uw and colors[e_wx] == c_uw:
                    count_uw += 1
            violations[e_uw] = count_uw
            
            # Recalc (v,w)
            count_vw = 0
            for x in range(n):
                if x == v or x == w:
                    continue
                e_vx = edge_to_idx[(v, x)]
                e_wx = edge_to_idx[(w, x)]
                if colors[e_vx] == c_vw and colors[e_wx] == c_vw:
                    count_vw += 1
            violations[e_vw] = count_vw

    # Initial calculation of violations
    for i in range(num_edges):
        update_violation_for_edge(i)
        
    total_violations = sum(violations)
    
    # Local Search
    temperature = 10.0
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        if total_violations == 0:
            break
            
        # Find edges with violations
        violating_edges = [i for i in range(num_edges) if violations[i] > 0]
        
        if not violating_edges:
            break
            
        # Pick random violating edge
        e_idx = random.choice(violating_edges)
        
        # Try all 4 colors
        best_delta = float('inf')
        best_color = -1
        
        for c in range(4):
            if c == colors[e_idx]:
                continue
            d = calculate_delta(e_idx, c)
            if d < best_delta:
                best_delta = d
                best_color = c
                
        # Acceptance
        if best_delta < 0:
            accept = True
        elif best_delta == 0:
            accept = random.random() < 0.5
        else:
            # Simulated annealing
            accept = random.random() < pow(2.71828, -best_delta / temperature)
            
        if accept:
            old_color = colors[e_idx]
            colors[e_idx] = best_color
            total_violations += best_delta
            update_violations_after_change(e_idx, old_color, best_color)
            
        temperature *= cooling
        
        # If stuck, random restart? Or just keep going.
        # With n=50, it should converge.
        
    if total_violations == 0:
        return {'n': n, 'colors': colors}
    else:
        # Fallback to seed's n=32 if we failed
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
        colors_32 = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors_32}

# EVOLVE-BLOCK-END
