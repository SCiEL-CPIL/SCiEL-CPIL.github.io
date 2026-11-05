import time
import random
import math

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    
    Uses efficient simulated annealing with delta-energy updates.
    """
    deadline = time.time() + 115  # 115 seconds, leaving 5s margin
    
    def time_left():
        return deadline - time.time()

    def validate(colors, n):
        """Check if no color class contains a C4."""
        # For each color, build adjacency and check for C4
        for c in range(5):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            # Check for C4: any two vertices with >= 2 common neighbors
            for i in range(n):
                for j in range(i+1, n):
                    if len(adj[i] & adj[j]) >= 2:
                        return False
        return True

    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def solve_n(n):
        """
        Attempt to find a valid 5-coloring for K_n.
        Returns (success, colors)
        """
        num_edges = n * (n - 1) // 2
        
        # Precompute neighbor lists for each color for fast delta calculation
        # neighbors[c][u] = set of vertices v such that edge (u,v) is color c
        # common_count[c][u][v] = number of common neighbors of u and v in color c
        
        # To save memory and time, we can maintain:
        # deg[c][u] = degree of u in color c
        # common[c][u][v] = number of common neighbors
        
        # However, updating common[c][u][v] for all pairs is O(n^2) per move.
        # We only need to update pairs involving the changed edge endpoints.
        
        # Let's maintain:
        # adj[c] = list of sets of neighbors for each vertex in color c
        # common[c][u][v] is not explicitly stored, but we can calculate the delta
        # based on current common neighbor counts.
        
        # Actually, storing common[c][u][v] for all u,v is O(5 * n^2) memory, which is fine for n=26 (5*676=3380 ints).
        # Updating it:
        # When edge (u,v) changes from c_old to c_new:
        # 1. Remove (u,v) from adj[c_old][u] and adj[c_old][v]
        # 2. Add (u,v) to adj[c_new][u] and adj[c_new][v]
        # 3. Update common counts:
        #    For c_old:
        #       For each w in adj[c_old][u] (after removal? No, before removal, w != v):
        #          v was a common neighbor of u and w. So common[c_old][u][w] decreases by 1.
        #       For each w in adj[c_old][v] (before removal, w != u):
        #          u was a common neighbor of v and w. So common[c_old][v][w] decreases by 1.
        #    For c_new:
        #       For each w in adj[c_new][u] (before addition, w != v):
        #          v becomes a common neighbor of u and w. So common[c_new][u][w] increases by 1.
        #       For each w in adj[c_new][v] (before addition, w != u):
        #          u becomes a common neighbor of v and w. So common[c_new][v][w] increases by 1.
        
        # Energy is sum over c, u<v of C(common[c][u][v], 2)
        # We can maintain total_energy and update it incrementally.
        
        # Initialize
        adj = [[set() for _ in range(n)] for _ in range(5)]
        # common[c][u][v] can be a 3D list or dict. For speed, 3D list [5][n][n]
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        total_energy = 0
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build initial structures
        for idx in range(num_edges):
            c = colors[idx]
            # Recover i, j from idx
            # This is slow. Better to iterate i, j directly.
            pass
            
        # Re-init properly
        adj = [[set() for _ in range(n)] for _ in range(5)]
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        total_energy = 0
        
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        # Calculate initial common counts and energy
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    # Count common neighbors of u and v in color c
                    cnt = len(adj[c][u] & adj[c][v])
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    if cnt >= 2:
                        total_energy += cnt * (cnt - 1) // 2
        
        if total_energy == 0:
            return True, colors
        
        # Simulated Annealing
        temperature = 5.0
        cooling_rate = 0.9999
        min_temperature = 0.01
        max_steps = 100000
        
        best_energy = total_energy
        best_colors = colors[:]
        
        step = 0
        while step < max_steps and temperature > min_temperature:
            if time_left() < 2:
                break
                
            step += 1
            
            # Pick random edge
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            ei = get_idx(i, j, n)
            
            old_c = colors[ei]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Calculate Delta Energy
            delta = 0
            
            # Remove edge (i, j) from old_c
            # 1. Update common counts for old_c
            # Neighbors of i in old_c (excluding j)
            # Neighbors of j in old_c (excluding i)
            
            # We need to iterate over neighbors to update common counts.
            # For each w in adj[old_c][i] (w != j):
            #   common[old_c][i][w] decreases by 1
            #   Energy change for pair (i, w):
            #   old_cnt = common[old_c][i][w]
            #   new_cnt = old_cnt - 1
            #   delta += C(new_cnt, 2) - C(old_cnt, 2)
            
            # For each w in adj[old_c][j] (w != i):
            #   common[old_c][j][w] decreases by 1
            #   Energy change for pair (j, w):
            #   old_cnt = common[old_c][j][w]
            #   new_cnt = old_cnt - 1
            #   delta += C(new_cnt, 2) - C(old_cnt, 2)
            
            # Note: The pair (i, j) itself?
            # common[old_c][i][j] is not affected by the edge (i,j) itself.
            # The edge (i,j) is not a common neighbor.
            
            for w in adj[old_c][i]:
                if w == j: continue
                old_cnt = common[old_c][i][w]
                new_cnt = old_cnt - 1
                # Update common
                common[old_c][i][w] = new_cnt
                common[old_c][w][i] = new_cnt
                # Update energy
                if old_cnt >= 2:
                    delta -= old_cnt - 1
                # C(k-1, 2) - C(k, 2) = - (k-1)
                
            for w in adj[old_c][j]:
                if w == i: continue
                old_cnt = common[old_c][j][w]
                new_cnt = old_cnt - 1
                common[old_c][j][w] = new_cnt
                common[old_c][w][j] = new_cnt
                if old_cnt >= 2:
                    delta -= old_cnt - 1
            
            # Remove from adj
            adj[old_c][i].remove(j)
            adj[old_c][j].remove(i)
            
            # Add edge (i, j) to new_c
            # 1. Update common counts for new_c
            for w in adj[new_c][i]:
                if w == j: continue
                old_cnt = common[new_c][i][w]
                new_cnt = old_cnt + 1
                common[new_c][i][w] = new_cnt
                common[new_c][w][i] = new_cnt
                # C(k+1, 2) - C(k, 2) = k
                delta += old_cnt
                
            for w in adj[new_c][j]:
                if w == i: continue
                old_cnt = common[new_c][j][w]
                new_cnt = old_cnt + 1
                common[new_c][j][w] = new_cnt
                common[new_c][w][j] = new_cnt
                delta += old_cnt
            
            # Add to adj
            adj[new_c][i].add(j)
            adj[new_c][j].add(i)
            
            # Accept or Reject
            new_energy = total_energy + delta
            if new_energy < total_energy or random.random() < math.exp(-delta / temperature):
                total_energy = new_energy
                colors[ei] = new_c
                if total_energy < best_energy:
                    best_energy = total_energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        return True, best_colors
            else:
                # Revert
                # Re-add to old_c
                adj[old_c][i].add(j)
                adj[old_c][j].add(i)
                
                # Revert common counts for old_c
                for w in adj[old_c][i]:
                    if w == j: continue
                    old_cnt = common[old_c][i][w]
                    new_cnt = old_cnt + 1
                    common[old_c][i][w] = new_cnt
                    common[old_c][w][i] = new_cnt
                    
                for w in adj[old_c][j]:
                    if w == i: continue
                    old_cnt = common[old_c][j][w]
                    new_cnt = old_cnt + 1
                    common[old_c][j][w] = new_cnt
                    common[old_c][w][j] = new_cnt
                    
                # Remove from new_c
                adj[new_c][i].remove(j)
                adj[new_c][j].remove(i)
                
                # Revert common counts for new_c
                for w in adj[new_c][i]:
                    if w == j: continue
                    old_cnt = common[new_c][i][w]
                    new_cnt = old_cnt - 1
                    common[new_c][i][w] = new_cnt
                    common[new_c][w][i] = new_cnt
                    
                for w in adj[new_c][j]:
                    if w == i: continue
                    old_cnt = common[new_c][j][w]
                    new_cnt = old_cnt - 1
                    common[new_c][j][w] = new_cnt
                    common[new_c][w][j] = new_cnt
                    
                # Revert color
                colors[ei] = old_c
                
            temperature *= cooling_rate
        
        return False, best_colors

    best_n = 1
    best_colors = [0]
    
    # Try to find valid coloring for increasing n
    # Start from high n and go down? Or low to high?
    # High to low is better if we expect to find 26.
    # But finding 26 is hard. Let's try 20, 22, 24, 26.
    
    for target_n in [20, 22, 24, 25, 26]:
        if time_left() < 5:
            break
            
        success, colors = solve_n(target_n)
        if success:
            best_n = target_n
            best_colors = colors
            # If we found 26, we can stop or try 27 if time permits
            if target_n == 26:
                # Try 27 if time allows
                if time_left() > 20:
                    success27, colors27 = solve_n(27)
                    if success27:
                        best_n = 27
                        best_colors = colors27
                break
        else:
            # If we failed, maybe try a slightly smaller n?
            # Or just keep the best found so far.
            # Since we are going up, if we fail 20, we likely fail 22.
            # But random restarts might succeed later.
            # Let's stick to the loop.
            pass

    # Ensure we return a valid result
    if best_n < 7:
        # Fallback to small n
        best_n = 7
        # Generate a valid coloring for n=7
        # Any 5-coloring of K7 is valid? No, C4 is 4 vertices.
        # K7 has 21 edges.
        # Random search for n=7 is instant.
        success, colors = solve_n(7)
        if success:
            best_colors = colors
        else:
            # Hardcode a valid one or just return n=1
            best_n = 1
            best_colors = [0]

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
