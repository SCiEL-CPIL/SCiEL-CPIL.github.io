import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate the known valid coloring for n=32 (Paley over GF(16))
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_seed_coloring(n=32):
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                # The seed logic:
                # i, j in 0..31
                # Split into high/low 4 bits?
                # The original code: log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                # This implies a block structure.
                # Let's reuse the exact logic from the seed program to ensure validity.
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                colors[i][j] = c
                colors[j][i] = c
        return colors

    seed_colors = get_seed_coloring(32)
    best_result_n = 32
    best_colors = seed_colors

    # 2. MinConflicts Solver for Extension
    def solve_minconflicts(target_n, time_limit):
        if target_n <= best_result_n:
            return None

        # Initialize colors
        colors = [[0] * target_n for _ in range(target_n)]
        
        # Copy seed
        for i in range(32):
            for j in range(i + 1, 32):
                colors[i][j] = seed_colors[i][j]
                colors[j][i] = seed_colors[i][j]
                
        # Randomize new edges
        for i in range(32, target_n):
            for j in range(i + 1, target_n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
        # Edges between 0..31 and 32..target_n-1
        for i in range(32):
            for j in range(32, target_n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]

        # Precompute neighbors for speed?
        # We need to update conflict counts.
        # conflict_count[i][j] = number of k such that (i,j,k) is mono.
        conflict_count = [[0] * target_n for _ in range(target_n)]
        total_conflicts = 0

        # Initial count
        for i in range(target_n):
            for j in range(i + 1, target_n):
                c = colors[i][j]
                cnt = 0
                for k in range(target_n):
                    if k == i or k == j: continue
                    if colors[i][k] == c and colors[j][k] == c:
                        cnt += 1
                conflict_count[i][j] = cnt
                conflict_count[j][i] = cnt
                total_conflicts += cnt
        total_conflicts //= 3 # Each triangle counted 3 times

        if total_conflicts == 0:
            return target_n, colors

        # MinConflicts Loop
        t_end = time.time() + time_limit
        # List of edges for random access
        edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                edges.append((i, j))
        
        # To pick an edge in conflict, we can just pick a random edge and check if it has conflicts?
        # Or maintain a list of conflicting edges?
        # Simpler: Pick random edge. If it has 0 conflicts, pick another?
        # Better: Pick random edge. If conflict_count > 0, process it.
        # If many edges are 0, we might pick 0s often.
        # Heuristic: Pick a random edge. If it's clean, skip or pick another.
        # Given n is small, picking a random edge and checking is O(1).
        
        while time.time() < t_end:
            # Pick a random edge
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            if conflict_count[u][v] == 0:
                continue # This edge is not in any conflict, don't change it (standard MinConflicts)
                
            c_old = colors[u][v]
            
            # Evaluate all 4 colors
            best_delta = float('inf')
            best_c = c_old
            best_conf = total_conflicts
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                    
                # Calculate delta
                # Triangles affected are (u, v, k) for k != u, v
                # For each k:
                # Old: contributes 1 if colors[u][k]==c_old and colors[v][k]==c_old
                # New: contributes 1 if colors[u][k]==c_new and colors[v][k]==c_new
                
                delta = 0
                for k in range(target_n):
                    if k == u or k == v: continue
                    cu = colors[u][k]
                    cv = colors[v][k]
                    
                    old_contrib = 1 if (cu == c_old and cv == c_old) else 0
                    new_contrib = 1 if (cu == c_new and cv == c_new) else 0
                    
                    # Note: The triangle (u,v,k) is counted in conflict_count[u][v], conflict_count[u][k], conflict_count[v][k]
                    # We are tracking total_conflicts which is sum of conflict_count / 3.
                    # So delta in total_conflicts is (delta_sum_of_counts) / 3.
                    # delta_sum_of_counts = (new_contrib - old_contrib) * 3?
                    # No.
                    # If a triangle becomes mono:
                    #   conflict_count[u][v] increases by 1
                    #   conflict_count[u][k] increases by 1
                    #   conflict_count[v][k] increases by 1
                    #   Sum increases by 3.
                    #   total_conflicts increases by 1.
                    # So delta_total = sum(new_contrib - old_contrib) for all k.
                    
                    delta += (new_contrib - old_contrib)
                
                # We want to minimize total_conflicts
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
                    best_conf = total_conflicts + delta

            # If best_c is still c_old, it means no change improves it. 
            # In MinConflicts, we usually pick the best available, even if it's worse?
            # No, we only pick if it's better or equal? 
            # Standard: Pick the color that minimizes conflicts. If all are worse, we might still pick the least bad one to escape local minima?
            # Actually, if best_delta > 0, we are stuck. 
            # To handle plateaus, we can allow random moves if stuck for a while, or just accept the best (least bad) move.
            # Let's accept the best move even if it increases conflicts, to allow exploration?
            # No, that's SA. MinConflicts is greedy.
            # If best_delta >= 0, we might be in a local minimum.
            # We can try to pick a random color among those with min delta?
            # Or just break/restart?
            # Let's accept the best move regardless of sign, but if delta > 0, we are going up.
            # To be safe, let's only accept if delta <= 0? 
            # If we only accept <= 0, we get stuck easily.
            # Let's accept the best move always.
            
            if best_c == c_old:
                continue

            # Apply change
            colors[u][v] = best_c
            colors[v][u] = best_c
            
            # Update conflict counts
            # For each k:
            for k in range(target_n):
                if k == u or k == v: continue
                cu = colors[u][k]
                cv = colors[v][k]
                
                # Old state
                old_u_v = c_old
                # The triangle (u,v,k) was mono if cu==cv==old_u_v
                # New state
                new_u_v = best_c
                # The triangle (u,v,k) is mono if cu==cv==new_u_v
                
                # Update conflict_count[u][v]
                if cu == cv:
                    if cu == c_old:
                        conflict_count[u][v] -= 1
                        conflict_count[v][u] -= 1
                        # Also need to update conflict_count[u][k] and conflict_count[v][k]
                        # The triangle (u,v,k) contributes to all 3 pairs.
                        # If it was mono, it contributed 1 to each.
                        # Now it's not mono (since cu==cv != best_c? No, cu==cv==c_old != best_c)
                        # So it loses 1 from all 3.
                        conflict_count[u][k] -= 1
                        conflict_count[k][u] -= 1
                        conflict_count[v][k] -= 1
                        conflict_count[k][v] -= 1
                        total_conflicts -= 1
                    elif cu == best_c:
                        conflict_count[u][v] += 1
                        conflict_count[v][u] += 1
                        conflict_count[u][k] += 1
                        conflict_count[k][u] += 1
                        conflict_count[v][k] += 1
                        conflict_count[k][v] += 1
                        total_conflicts += 1
                else:
                    # cu != cv, so it was never mono, and still not mono.
                    # But wait, what if cu != cv but one of them matches?
                    # Mono requires ALL 3 equal.
                    # So if cu != cv, it's not mono.
                    # So no change.
                    pass

            if total_conflicts == 0:
                return target_n, colors

        return None

    # Try to extend
    # We try increasing n.
    # Start from 33 up to 50.
    # Time budget: 120s total.
    # We have ~115s.
    # Let's give 5s to each n=33..45, and 2s to 46..50?
    # Or just loop until time is up.
    
    # To maximize chance, we should try smaller n first as they are easier.
    # If we find n=33, we update best.
    
    current_n = 32
    while current_n < 50:
        if time.time() > deadline - 5:
            break
            
        current_n += 1
        
        # Allocate time
        # 33-40: 10s each
        # 41-50: 5s each
        if current_n <= 40:
            t_lim = 10.0
        else:
            t_lim = 5.0
            
        # Ensure we don't exceed deadline
        t_lim = min(t_lim, deadline - time.time() - 2.0)
        if t_lim <= 0:
            break
            
        result = solve_minconflicts(current_n, t_lim)
        if result:
            n, colors = result
            out_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    out_colors.append(colors[i][j])
            return {'n': n, 'colors': out_colors}
            
    # Fallback to seed
    out_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            out_colors.append(seed_colors[i][j])
    return {'n': 32, 'colors': out_colors}
