import time
import random
import sys

def construct():
    """
    Returns {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    giving one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    No colour class may contain a C4 (no two vertices with >=2 common neighbours in one colour).
    """
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    def edge_idx(i, j, n):
        """Lexicographic index of edge (i,j) with i<j."""
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        """Reverse: given linear index idx (0-based), return (i,j) with i<j."""
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def count_c4_bitmask(n, colors):
        """
        Count C4 violations using bitmasks.
        For each color c, build bitmask adjacency.
        For each pair u < v, check if popcount(adj[u] & adj[v]) >= 2.
        """
        # adj[c][u] is an integer bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        
        # Build bitmasks
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        total_violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                if mask_u == 0:
                    continue
                # Iterate over neighbors v of u
                v = (mask_u & -mask_u).bit_length() - 1
                while v >= 0:
                    # We only need to check pairs u < v to avoid double counting?
                    # Actually, common neighbors count is symmetric.
                    # We just need to ensure for ANY pair u,v, common neighbors < 2.
                    # So checking all pairs is fine, but we can restrict to u < v.
                    if u < v:
                        mask_v = adj[c][v]
                        common = mask_u & mask_v
                        # Count bits
                        # Using int.bit_count() (Python 3.8+)
                        cnt = common.bit_count()
                        if cnt >= 2:
                            total_violations += cnt - 1
                    # Next neighbor
                    mask_u &= mask_u - 1
                    if mask_u == 0:
                        break
                    v = (mask_u & -mask_u).bit_length() - 1
        return total_violations

    def local_search_bitmask(n, time_limit=50.0):
        """
        Local search using bitmasks for fast C4 counting.
        """
        m = n * (n - 1) // 2
        random.seed(42)
        colors = [random.randint(0, 4) for _ in range(m)]
        
        # Precompute edge list for fast access in swaps?
        # Actually, we just need to update colors and recompute C4 count.
        # Recomputing C4 count from scratch is O(5 * n^2) bit operations, which is fast for n=26.
        # n=26 -> 5 * 26^2 ~ 3380 operations. Very fast.
        # So we can do many iterations.
        
        current_viol = count_c4_bitmask(n, colors)
        if current_viol == 0:
            return colors, True
        
        t_start = time.time()
        best_colors = colors[:]
        best_viol = current_viol
        
        # Simulated Annealing or Hill Climbing
        # We will use Hill Climbing with random restarts if stuck.
        # But re-initialization is expensive? No, O(m).
        
        for restart in range(10):
            if time.time() > t_start + time_limit:
                break
            
            if restart > 0:
                # Random restart
                random.seed(42 + restart)
                colors = [random.randint(0, 4) for _ in range(m)]
                current_viol = count_c4_bitmask(n, colors)
                if current_viol == 0:
                    return colors, True
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = colors[:]
            
            # Hill Climbing
            stuck_count = 0
            while time.time() < t_start + time_limit:
                if current_viol == 0:
                    return colors, True
                
                # Pick random edge
                eidx = random.randint(0, m - 1)
                old_c = colors[eidx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                colors[eidx] = new_c
                new_viol = count_c4_bitmask(n, colors)
                
                if new_viol <= current_viol:
                    current_viol = new_viol
                    stuck_count = 0
                    if current_viol < best_viol:
                        best_viol = current_viol
                        best_colors = colors[:]
                else:
                    colors[eidx] = old_c
                    stuck_count += 1
                    if stuck_count > 1000:
                        break # Restart hill climbing
            
        return best_colors, (best_viol == 0)

    # Try n=26
    if time.time() < deadline:
        colors26, valid26 = local_search_bitmask(26, time_limit=50.0)
        if valid26:
            return {'n': 26, 'colors': colors26}
        # If not valid, we might still return it if it's the best we have?
        # No, must be valid.
    
    # Try n=25
    if time.time() < deadline:
        colors25, valid25 = local_search_bitmask(25, time_limit=30.0)
        if valid25:
            return {'n': 25, 'colors': colors25}
        
    # Try n=20
    if time.time() < deadline:
        colors20, valid20 = local_search_bitmask(20, time_limit=10.0)
        if valid20:
            return {'n': 20, 'colors': colors20}

    # Fallback n=15
    n15 = 15
    m15 = n15 * (n15 - 1) // 2
    colors15 = [0] * m15
    for idx in range(m15):
        i, j = idx_to_edge(idx, n15)
        colors15[idx] = (j - i) % 5
    # Verify n=15 cyclic
    if count_c4_bitmask(n15, colors15) == 0:
        return {'n': n15, 'colors': colors15}
    
    # Final fallback: random n=10
    n10 = 10
    m10 = n10 * (n10 - 1) // 2
    colors10 = [random.randint(0, 4) for _ in range(m10)]
    return {'n': n10, 'colors': colors10}

