import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We want to maximize n.
    # Theoretical max is around 30.
    # Seed gives 11.
    # Let's try to find a valid coloring for n=12, 13, ...
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Helper to check validity
    def is_valid(n, colors):
        # colors is a list of length n*(n-1)/2
        # Map (i,j) to index
        # idx = i*n - i*(i+1)//2 + (j-i-1)
        # Or simpler: build adjacency matrices
        
        # Count C4s
        # k[u][v][c] = number of common neighbors in color c
        # This is expensive to build from scratch every time.
        # But for final check, it's fine.
        
        # Efficient check:
        # For each color c, build adjacency list.
        # For each pair (u,v), count common neighbors.
        # If count >= 2, invalid.
        
        adj = [[] for _ in range(5)]
        # adj[c] is a list of sets? Or just check pairs.
        
        # Let's use a 3D array or list of dicts for common neighbors?
        # Simpler: For each color c, create a set of edges.
        # Then for each pair (u,v), count common neighbors.
        
        edges_by_color = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                edges_by_color[c].add((i, j))
                idx += 1
        
        for c in range(5):
            # For each pair (u,v), count common neighbors
            # Common neighbors of u,v are w such that (u,w) in E and (v,w) in E
            # To speed up, we can precompute neighbors for each vertex in this color
            neighbors = [set() for _ in range(n)]
            for u, v in edges_by_color[c]:
                neighbors[u].add(v)
                neighbors[v].add(u)
                
            for u in range(n):
                for v in range(u+1, n):
                    # Count common neighbors
                    # Intersection of neighbors[u] and neighbors[v]
                    if len(neighbors[u].intersection(neighbors[v])) >= 2:
                        return False
        return True

    def local_search(n, colors, time_limit):
        """
        colors: list of length n*(n-1)/2
        Returns (success, colors)
        """
        m = n * (n - 1) // 2
        
        # Precompute indices for fast access
        # idx(i, j)
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        # State:
        # k[c][u][v] = number of common neighbors of u and v in color c
        # S[c] = sum of binom(k[c][u][v], 2)
        
        # Initialize k and S
        # k is a list of 5 matrices n x n
        # To save memory and time, we can use a flat list or just recalculate cost?
        # Recalculating cost is O(n^2 * 5).
        # Updating k is O(n) per move.
        # Let's maintain k.
        
        k = [[[0]*n for _ in range(n)] for _ in range(5)]
        S = [0]*5
        
        # Build initial k and S
        # For each color c, for each edge (u,v), 
        # it contributes to k[c][u][w] if (v,w) is edge? No.
        # k[c][u][v] is count of w.
        # Let's build adjacency first.
        adj = [set() for _ in range(5)]
        for i in range(n):
            for j in range(i+1, n):
                c = colors[get_idx(i, j)]
                adj[c].add((i, j))
                adj[c].add((j, i)) # Undirected, but sets of neighbors
                
        # Calculate k
        for c in range(5):
            # For each pair (u,v), count common neighbors
            # This is O(n^3) if naive.
            # n=20, n^3=8000. 5*8000 = 40000. Fast.
            for u in range(n):
                for v in range(u+1, n):
                    # Count w in adj[c][u] that are also in adj[c][v]
                    # adj[c] is set of edges? No, let's use neighbor sets.
                    # Let's create neighbor sets.
                    pass
            
            # Better:
            neigh = [set() for _ in range(n)]
            for (u, v) in adj[c]:
                neigh[u].add(v)
                neigh[v].add(u)
            
            for u in range(n):
                for v in range(u+1, n):
                    cnt = len(neigh[u].intersection(neigh[v]))
                    k[c][u][v] = cnt
                    k[c][v][u] = cnt
                    S[c] += cnt * (cnt - 1) // 2
        
        total_S = sum(S)
        
        if total_S == 0:
            return True, colors
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j)
            old_c = colors[idx]
            
            # Try all other colors
            best_delta = float('inf')
            best_new_c = -1
            
            for new_c in range(5):
                if new_c == old_c:
                    continue
                
                # Calculate delta S for removing from old_c and adding to new_c
                delta_S = 0
                
                # Remove from old_c
                # Affected pairs: (i, w) for w in neigh_old[i] ?
                # k[old_c][i][w] changes if (j, w) in E_old
                # k[old_c][j][w] changes if (i, w) in E_old
                
                # We need the current neighbor sets for old_c and new_c
                # Maintaining neighbor sets dynamically is complex.
                # Alternative: Calculate delta by checking all w.
                
                # For each w in 0..n-1, w != i, j
                # Let's iterate w
                
                # We need to know if (j,w) is in old_c and if (i,w) is in old_c.
                # This requires looking up colors.
                # colors is a list.
                
                # To speed up, we can precompute a 2D array of colors?
                # color_mat[u][v]
                
                # Let's maintain color_mat.
                # But we are in a loop.
                
                # For now, just check colors[idx(i,w)] and colors[idx(j,w)].
                
                for w in range(n):
                    if w == i or w == j:
                        continue
                    
                    # Check removal from old_c
                    # Does (j,w) have color old_c?
                    idx_jw = get_idx(j, w)
                    if colors[idx_jw] == old_c:
                        # k[old_c][i][w] decreases by 1
                        old_k = k[old_c][i][w]
                        new_k = old_k - 1
                        delta_S -= (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                    
                    # Does (i,w) have color old_c?
                    idx_iw = get_idx(i, w)
                    if colors[idx_iw] == old_c:
                        # k[old_c][j][w] decreases by 1
                        old_k = k[old_c][j][w]
                        new_k = old_k - 1
                        delta_S -= (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                        
                    # Check addition to new_c
                    idx_jw = get_idx(j, w)
                    if colors[idx_jw] == new_c:
                        # k[new_c][i][w] increases by 1
                        old_k = k[new_c][i][w]
                        new_k = old_k + 1
                        delta_S += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                        
                    idx_iw = get_idx(i, w)
                    if colors[idx_iw] == new_c:
                        # k[new_c][j][w] increases by 1
                        old_k = k[new_c][j][w]
                        new_k = old_k + 1
                        delta_S += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                        
                if delta_S < best_delta:
                    best_delta = delta_S
                    best_new_c = new_c
            
            # Accept best move if it improves, or with prob exp(-delta/T)
            accept = False
            if best_delta < 0:
                accept = True
            else:
                if T > min_T:
                    if random.random() < math.exp(-best_delta / T):
                        accept = True
            
            if accept:
                # Apply move
                # Update colors
                colors[idx] = best_new_c
                
                # Update k and S
                # We need to actually update the k matrix values we used in delta calc
                # But we calculated delta based on current k.
                # We must update k[i][w], k[j][w] etc.
                
                for w in range(n):
                    if w == i or w == j:
                        continue
                    
                    idx_jw = get_idx(j, w)
                    idx_iw = get_idx(i, w)
                    
                    # Remove from old_c
                    if colors[idx_jw] == old_c:
                        k[old_c][i][w] -= 1
                        k[old_c][w][i] -= 1
                        # Update S[old_c]
                        # We already accounted for delta in best_delta, but we need to keep S consistent
                        # Actually, we can just recalculate S from k at the end?
                        # No, S is used for nothing else except total_S check.
                        # We can just update S[old_c] by the delta part.
                        # But we summed delta_S for all w.
                        # Let's just update k and recalculate S occasionally?
                        # Or update S incrementally.
                        
                        # Let's update S incrementally.
                        # Delta for this w was calculated.
                        # We can just do S[old_c] += delta_part
                        # But we didn't store delta_part per w.
                        # Easier: Just update k, and at the end of the move, 
                        # we know total_S += best_delta.
                        # But S[c] individually?
                        # We need S[c] to be correct for future deltas?
                        # No, delta calculation uses k directly.
                        # So we only need k to be correct.
                        # We can update S[c] at the end of the move by best_delta?
                        # No, best_delta is sum of changes in S[old_c] and S[new_c].
                        # So S[old_c] changes by delta_old, S[new_c] by delta_new.
                        # We can just keep a total_S variable.
                        # But we need to know if total_S == 0.
                        # So we just update total_S += best_delta.
                        pass
                    
                    if colors[idx_iw] == old_c:
                        k[old_c][j][w] -= 1
                        k[old_c][w][j] -= 1
                        
                    # Add to new_c
                    if colors[idx_jw] == new_c:
                        k[new_c][i][w] += 1
                        k[new_c][w][i] += 1
                        
                    if colors[idx_iw] == new_c:
                        k[new_c][j][w] += 1
                        k[new_c][w][j] += 1
                        
                total_S += best_delta
                
                if total_S == 0:
                    return True, colors
                
                T *= cooling
            
            # Cooldown
            if time.time() > time_limit:
                break
        
        return False, colors

    # Try to improve n
    # Start from n=12
    # Use random restarts
    
    for target_n in range(12, 29):
        if time.time() > deadline:
            break
            
        success = False
        attempts = 0
        max_attempts = 10  # Limit attempts per n to avoid wasting time
        
        while not success and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Initialize colors
            # Random
            m = target_n * (target_n - 1) // 2
            colors = [random.randint(0, 4) for _ in range(m)]
            
            # Run local search
            time_limit = deadline - 5.0  # Leave 5s for check and next n
            if time_limit < start_time + 10:
                time_limit = start_time + 10
                
            ok, colors = local_search(target_n, colors, time_limit)
            
            if ok:
                # Verify
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    success = True
                else:
                    # Should not happen if local search is correct
                    pass
        
        # If failed for this n, maybe try a few more times or break?
        # If n=12 fails, n=13 will likely fail.
        # But maybe a different random seed works.
        # Let's break if we fail 3 times in a row for a specific n?
        # For now, just break if we fail the first few.
        if not success:
            # Try one more time with a better init?
            # Or just break.
            break
            
    return {'n': best_n, 'colors': best_colors}
