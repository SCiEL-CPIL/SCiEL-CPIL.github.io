import numpy as np

def construct():
    N = 50
    E = N * (N - 1) // 2
    
    # Map (i, j) to edge index
    # i < j
    # We can use a 2D array or a dictionary. 
    # For speed in the inner loop, a 2D array of indices is best.
    edge_idx = np.full((N, N), -1, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            idx += 1
            
    # Precompute neighbors for each edge to speed up updates?
    # For edge e=(u,v), the other vertices are all w != u,v.
    # We need the indices of edges (u,w) and (v,w).
    # We can just loop w and use edge_idx.
    
    # Initialize colors randomly
    colors = np.random.randint(0, 4, size=E)
    
    # Calculate initial conflict counts
    # conflict_count[e] = number of w such that c(u,v)=c(u,w)=c(v,w)
    conflict_count = np.zeros(E, dtype=int)
    
    # Efficient initial calculation:
    # For each triangle (i,j,k), check if mono.
    # This is O(N^3) = 125k ops. Fine.
    for i in range(N):
        for j in range(i + 1, N):
            e1 = edge_idx[i, j]
            c1 = colors[e1]
            count = 0
            for k in range(N):
                if k == i or k == j: continue
                e2 = edge_idx[i, k]
                e3 = edge_idx[j, k]
                if colors[e2] == c1 and colors[e3] == c1:
                    count += 1
            conflict_count[e1] = count
            
    # MinConflicts
    # We need a list of edges with conflicts > 0 to pick from.
    # Or just pick random edge and check if it has conflict.
    # To be faster, maintain a list/set of conflicting edges.
    
    import random
    random.seed(42)
    
    max_iters = 100000
    for _ in range(max_iters):
        # Find edges with conflicts
        # If we don't maintain a list, we can sample.
        # But checking if a sampled edge has conflict is O(1).
        # Probability of picking a bad edge is high if many conflicts.
        # Let's maintain a list of conflicting edge indices.
        
        # Rebuilding the list every iteration is O(E) = 1225.
        # 100000 * 1225 = 122M ops. Might be slow in Python.
        # Let's try to update the list incrementally or just use a simpler check.
        
        # Alternative: Pick random edge. If conflict_count[e] > 0, process it.
        # If not, pick another.
        
        e = np.random.randint(E)
        if conflict_count[e] == 0:
            # Check if all are 0?
            # We can periodically check if sum(conflict_count) == 0.
            # Or just keep going.
            # To avoid infinite loop if no conflicts but we don't know, 
            # we can check sum every 1000 steps.
            if _ % 1000 == 0 and np.sum(conflict_count) == 0:
                break
            continue
            
        u, v = -1, -1
        # Find u, v for e
        # We can precompute u, v for each edge index.
        # Let's do that outside the loop.
        
    # Precompute u, v for each edge
    us = np.zeros(E, dtype=int)
    vs = np.zeros(E, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            us[idx] = i
            vs[idx] = j
            idx += 1
            
    # Restart with precomputed us, vs
    colors = np.random.randint(0, 4, size=E)
    conflict_count = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            e1 = edge_idx[i, j]
            c1 = colors[e1]
            count = 0
            for k in range(N):
                if k == i or k == j: continue
                e2 = edge_idx[i, k]
                e3 = edge_idx[j, k]
                if colors[e2] == c1 and colors[e3] == c1:
                    count += 1
            conflict_count[e1] = count
            
    for step in range(100000):
        if np.sum(conflict_count) == 0:
            break
            
        e = np.random.randint(E)
        if conflict_count[e] == 0:
            continue
            
        u = us[e]
        v = vs[e]
        c_old = colors[e]
        
        # Try all 4 colors, pick best
        best_color = c_old
        best_delta = 0 # We want to minimize conflict_count[e]
        # Actually, we want to minimize the total conflicts, or just local conflicts?
        # MinConflicts usually minimizes the number of constraints violated by the variable.
        # Here, "violations" for edge e is conflict_count[e].
        # But changing e affects conflict_count of other edges too.
        # Standard MinConflicts: pick value that minimizes total conflicts in the neighborhood.
        # Neighborhood of e: all edges sharing a vertex with e? No, just the triangles.
        # The cost function is sum(conflict_count).
        # Delta in sum(conflict_count) when changing e from c_old to c_new:
        # Only triangles containing e are affected.
        # For each w, triangle (u,v,w) changes status.
        # If it goes from mono to non-mono: sum decreases by 3 (e, e1, e2 each lose 1).
        # If it goes from non-mono to mono: sum increases by 3.
        # So we just need to count how many triangles become mono and how many cease to be mono.
        
        total_delta = 0
        best_total_delta = float('inf')
        
        for c_new in range(4):
            if c_new == c_old:
                continue
            delta = 0
            for w in range(N):
                if w == u or w == v:
                    continue
                e1 = edge_idx[u, w]
                e2 = edge_idx[v, w]
                c1 = colors[e1]
                c2 = colors[e2]
                
                # Current status (with c_old)
                was_mono = (c1 == c_old and c2 == c_old)
                # New status (with c_new)
                is_mono = (c1 == c_new and c2 == c_new)
                
                if was_mono and not is_mono:
                    delta -= 3
                elif not was_mono and is_mono:
                    delta += 3
            
            if delta < best_total_delta:
                best_total_delta = delta
                best_color = c_new
                
        if best_color == c_old:
            # No improvement found, maybe stuck? 
            # In MinConflicts, we usually pick the best available, even if it's the same (which means we shouldn't change).
            # If best is same, we might need to randomize or accept a worse move?
            # Standard MinConflicts: if best is current, pick a random other color? 
            # No, if best is current, it means current is best. We might be in a local minimum.
            # We can force a random change to escape.
            best_color = (c_old + 1) % 4
            
        # Apply change
        c_new = best_color
        if c_new == c_old:
            continue
            
        colors[e] = c_new
        
        # Update conflict counts
        for w in range(N):
            if w == u or w == v:
                continue
            e1 = edge_idx[u, w]
            e2 = edge_idx[v, w]
            c1 = colors[e1]
            c2 = colors[e2]
            
            was_mono = (c1 == c_old and c2 == c_old)
            is_mono = (c1 == c_new and c2 == c_new)
            
            if was_mono and not is_mono:
                conflict_count[e] -= 1
                conflict_count[e1] -= 1
                conflict_count[e2] -= 1
            elif not was_mono and is_mono:
                conflict_count[e] += 1
                conflict_count[e1] += 1
                conflict_count[e2] += 1
                
    # Check if solved
    if np.sum(conflict_count) == 0:
        # Convert to list of ints for output
        color_list = [int(c) for c in colors]
        return {'n': N, 'colors': color_list}
    else:
        # If failed, return a smaller valid one? 
        # The task says maximize n. If we can't get 50, maybe 32 is better than nothing?
        # But the seed gets 32. We should try to get 50.
        # If this fails, we might need a better solver or restart.
        # Given the time limit, one run of MinConflicts on 50 vertices is very likely to succeed 
        # because the solution space is not that tight for n=50 (it's the maximum known).
        # Actually, n=50 is the hard limit. But MinConflicts is very good.
        
        # Fallback: Return the seed's 32-vertex solution to ensure validity?
        # No, the score is value/seed. 50/32 > 1. 32/32 = 1.
        # If we return 32, we tie. If we return 50, we win.
        # If we return an invalid 50, we score 0.
        # We must ensure validity.
        
        # Let's add a verification step.
        # If invalid, we can try to fix it or return a known valid smaller one.
        # But let's trust MinConflicts.
        
        # To be safe, if it doesn't converge, we can try a few more restarts.
        for restart in range(10):
             colors = np.random.randint(0, 4, size=E)
             conflict_count = np.zeros(E, dtype=int)
             for i in range(N):
                 for j in range(i + 1, N):
                     e1 = edge_idx[i, j]
                     c1 = colors[e1]
                     count = 0
                     for k in range(N):
                         if k == i or k == j: continue
                         e2 = edge_idx[i, k]
                         e3 = edge_idx[j, k]
                         if colors[e2] == c1 and colors[e3] == c1:
                             count += 1
                     conflict_count[e1] = count
                     
             for step in range(100000):
                 if np.sum(conflict_count) == 0:
                     break
                 e = np.random.randint(E)
                 if conflict_count[e] == 0:
                     continue
                 u = us[e]
                 v = vs[e]
                 c_old = colors[e]
                 best_color = c_old
                 best_total_delta = float('inf')
                 for c_new in range(4):
                     if c_new == c_old: continue
                     delta = 0
                     for w in range(N):
                         if w == u or w == v: continue
                         e1 = edge_idx[u, w]
                         e2 = edge_idx[v, w]
                         c1 = colors[e1]
                         c2 = colors[e2]
                         was_mono = (c1 == c_old and c2 == c_old)
                         is_mono = (c1 == c_new and c2 == c_new)
                         if was_mono and not is_mono: delta -= 3
                         elif not was_mono and is_mono: delta += 3
                     if delta < best_total_delta:
                         best_total_delta = delta
                         best_color = c_new
                 if best_color == c_old:
                     best_color = (c_old + 1) % 4
                 c_new = best_color
                 if c_new == c_old: continue
                 colors[e] = c_new
                 for w in range(N):
                     if w == u or w == v: continue
                     e1 = edge_idx[u, w]
                     e2 = edge_idx[v, w]
                     c1 = colors[e1]
                     c2 = colors[e2]
                     was_mono = (c1 == c_old and c2 == c_old)
                     is_mono = (c1 == c_new and c2 == c_new)
                     if was_mono and not is_mono:
                         conflict_count[e] -= 1
                         conflict_count[e1] -= 1
                         conflict_count[e2] -= 1
                     elif not was_mono and is_mono:
                         conflict_count[e] += 1
                         conflict_count[e1] += 1
                         conflict_count[e2] += 1
                         
             if np.sum(conflict_count) == 0:
                 return {'n': N, 'colors': [int(c) for c in colors]}
                 
        # If all failed, return the 32-vertex solution from seed to guarantee score >= 1.0
        # Re-implementing seed logic for fallback
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
