import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses incremental energy update for fast local search.
    """
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def compute_energy(adj, n):
        """Count C4s: for each color, sum over pairs (i,j) of C(common_neighbors, 2)"""
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    best_n = 1
    best_colors = [0]
    
    for target_n in range(8, 29):
        if time_left() < 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            # Phase 1: Simulated annealing with faster cooling
            temp = 3.0
            cooling = 0.9998
            min_temp = 0.1
            max_steps = 300000
            stuck_count = 0
            max_stuck = 15000
            
            for step in range(max_steps):
                if time_left() < 3:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                
                ei = random.randint(0, num_edges - 1)
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                # Calculate delta using only relevant neighbors
                delta = 0
                
                # Remove from c_old: check common neighbors of u and v in c_old
                # For each w that is adjacent to both u and v in c_old, removing edge (u,v) 
                # reduces common neighbors of pairs (u,w) and (v,w) by 1 each
                # Also for each w adjacent to u (not v) in c_old, and w adjacent to v (not u),
                # the common neighbor count for (u,v) changes
                # 
                # More precisely: energy contribution from color c for pair (a,b) is C(|N_c(a) ∩ N_c(b)|, 2)
                # When we remove edge (u,v) from color c_old:
                #   - For pair (u,v): |N(u) ∩ N(v)| decreases by 0 (u,v not in each other's N in simple graph sense for C4 counting)
                #   Actually wait - in C4 counting, we count pairs (i,j) and their common neighbors.
                #   Edge (u,v) in color c means u ∈ N_c(v) and v ∈ N_c(u).
                #   Removing (u,v) from color c:
                #     For pair (u,v): common neighbors lose... no, u and v are still "connected" in the sense
                #     but actually the edge removal means v is no longer in N_c(u) and u is no longer in N_c(v).
                #     So for any pair (a,b) where a=u or b=u or a=v or b=v, the common neighbor set changes.
                #
                # Let me think more carefully. The C4 count for color c is:
                #   sum over all pairs (i,j), i<j, of C(|N_c(i) ∩ N_c(j)|, 2)
                # where N_c(i) = {j : edge (i,j) has color c}
                #
                # When we change edge (u,v) from color c_old to c_new:
                #   In c_old: v removed from N(u), u removed from N(v)
                #   In c_new: v added to N(u), u added to N(v)
                #
                # For c_old, the affected pairs are those involving u or v:
                #   Pair (u,v): |N(u) ∩ N(v)| loses... well u∉N(u), v∉N(v) always.
                #   Actually N(u) ∩ N(v) for pair (u,v) = common neighbors. Removing edge (u,v) doesn't change
                #   common neighbors of u and v directly (those are other vertices w). But it does change
                #   N(u) and N(v) which affects pairs (u,w) and (v,w) for all w.
                #
                # For pair (u,w) where w≠v: N(u) loses v. So |N(u) ∩ N(w)| may decrease by 1 if v ∈ N(w).
                # For pair (v,w) where w≠u: N(v) loses u. So |N(v) ∩ N(w)| may decrease by 1 if u ∈ N(w).
                # For pair (u,v): N(u) loses v, N(v) loses u. But v∉N(v) and u∉N(u) so no change to N(u)∩N(v).
                #   Wait, that's not right either. N(u) ∩ N(v) = {w : w adjacent to both u and v in color c}.
                #   Removing edge (u,v) doesn't affect whether w is adjacent to u or v (for w≠u,v).
                #   So N(u) ∩ N(v) is unchanged!
                #
                # So the only changes are:
                #   For each w ≠ u,v: if v ∈ N_c_old(w), then |N(u) ∩ N(w)| decreases by 1 (since v left N(u))
                #   For each w ≠ u,v: if u ∈ N_c_old(w), then |N(v) ∩ N(w)| decreases by 1 (since u left N(v))
                #
                # Energy change for c_old:
                #   For each w with v ∈ N_c_old(w): change in C(|N(u)∩N(w)|, 2) when |N(u)∩N(w)| → |N(u)∩N(w)|-1
                #     = C(k-1,2) - C(k,2) = -k+1 where k = |N(u)∩N(w)| BEFORE removal
                #     But wait, after removal k' = k-1, so C(k-1,2) - C(k,2) = (k-1)(k-2)/2 - k(k-1)/2 = (k-1)(k-2-k)/2 = -(k-1)
                #   Similarly for u ∈ N_c_old(w): change = -(k-1) where k = |N(v)∩N(w)| BEFORE removal
                #
                # For c_new, similar but adding:
                #   For each w with v ∈ N_c_new(w): |N(u)∩N(w)| increases by 1, change = C(k+1,2) - C(k,2) = k
                #   For each w with u ∈ N_c_new(w): |N(v)∩N(w)| increases by 1, change = k
                #
                # This matches the original code! The original code is correct.
                # The issue is just speed. Let's optimize by only iterating over relevant vertices.
                
                # For c_old removal:
                # Vertices w where v ∈ N_c_old(w) = neighbors of v in c_old (excluding u)
                # Vertices w where u ∈ N_c_old(w) = neighbors of u in c_old (excluding v)
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    mask_w = adj[c_old][w]
                    
                    if mask_w & (1 << v):
                        k = (adj[c_old][u] & mask_w).bit_count()
                        delta -= (k - 1)
                    
                    if mask_w & (1 << u):
                        k = (adj[c_old][v] & mask_w).bit_count()
                        delta -= (k - 1)
                
                # For c_new addition:
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    mask_w = adj[c_new][w]
                    
                    if mask_w & (1 << v):
                        k = (adj[c_new][u] & mask_w).bit_count()
                        delta += k
                    
                    if mask_w & (1 << u):
                        k = (adj[c_new][v] & mask_w).bit_count()
                        delta += k
                
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    colors[e_idx] = c_new
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    stuck_count += 1
                
                temp *= cooling
            
            if found:
                break
            
            # Phase 2: Polish with targeted moves
            if time_left() > 5 and best_energy > 0:
                polish_steps = 50000
                polish_temp = 0.5
                polish_cooling = 0.9999
                polish_stuck = 0
                polish_max_stuck = 5000
                
                for pstep in range(polish_steps):
                    if time_left() < 3:
                        break
                    if polish_temp < 0.05:
                        break
                    if polish_stuck > polish_max_stuck:
                        break
                    
                    ei = random.randint(0, num_edges - 1)
                    u, v, e_idx = edge_list[ei]
                    c_old = colors[e_idx]
                    c_new = random.randint(0, 4)
                    if c_new == c_old:
                        continue
                    
                    delta = 0
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        
                        mask_w = adj[c_old][w]
                        
                        if mask_w & (1 << v):
                            k = (adj[c_old][u] & mask_w).bit_count()
                            delta -= (k - 1)
                        
                        if mask_w & (1 << u):
                            k = (adj[c_old][v] & mask_w).bit_count()
                            delta -= (k - 1)
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        
                        mask_w = adj[c_new][w]
                        
                        if mask_w & (1 << v):
                            k = (adj[c_new][u] & mask_w).bit_count()
                            delta += k
                        
                        if mask_w & (1 << u):
                            k = (adj[c_new][v] & mask_w).bit_count()
                            delta += k
                    
                    new_energy = energy + delta
                    
                    if delta <= 0 or random.random() < math.exp(-delta / polish_temp):
                        energy = new_energy
                        
                        adj[c_old][u] &= ~(1 << v)
                        adj[c_old][v] &= ~(1 << u)
                        adj[c_new][u] |= (1 << v)
                        adj[c_new][v] |= (1 << u)
                        colors[e_idx] = c_new
                        
                        if energy < best_energy:
                            best_energy = energy
                            best_colors_local = colors[:]
                            polish_stuck = 0
                            if energy == 0:
                                best_n = n
                                best_colors = colors[:]
                                found = True
                                break
                    else:
                        polish_stuck += 1
                    
                    polish_temp *= polish_cooling
                
                if found:
                    break
                
                energy = compute_energy(adj, n)
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                
                if best_energy == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break

        if found:
            continue

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
