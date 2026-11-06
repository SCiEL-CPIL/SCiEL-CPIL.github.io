import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Generate the base n=32 Greenwood-Gleason coloring
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
            if i // 16 == j // 16:
                colors_32.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors_32.append(3)
                
    def colors_to_adj(colors, n):
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        return adj

    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(int(adj[i, j]))
        return colors

    def count_mono_triangles(adj, n):
        """
        Counts monochromatic triangles in a 4-coloring.
        Uses vectorized operations for speed.
        """
        total = 0
        # For each color c, create a binary matrix M_c where M_c[i,j]=1 if adj[i,j]==c
        # The number of triangles in color c is the sum of M_c @ M_c @ M_c / 6? 
        # No, trace(M_c^3) / 6 gives the number of directed triangles.
        # Since the graph is undirected and symmetric, M_c is symmetric.
        # Triangles (i,j,k) with i<j<k.
        # A faster way for small n:
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i, j]
                # Count k > j such that adj[i,k]==c and adj[j,k]==c
                # Vectorized:
                mask_i = (adj[i, j+1:] == c)
                mask_j = (adj[j, j+1:] == c)
                total += np.sum(mask_i & mask_j)
        return total

    def try_extend(adj, n):
        """
        Attempts to add one vertex to the existing valid coloring of size n.
        Uses backtracking CSP.
        Returns new_adj (size n+1) if successful, else None.
        """
        new_n = n + 1
        new_adj = np.zeros((new_n, new_n), dtype=np.int8)
        new_adj[:n, :n] = adj
        
        # We need to assign colors to edges (new_n-1, i) for i in 0..n-1
        # Let the new vertex be index 'n' (0-indexed, so it's the last one)
        # Constraints: For any i < j < n, if adj[i,j] == c, then we cannot have
        # new_edge(i) == c AND new_edge(j) == c.
        
        # Backtracking
        assigned = [-1] * n
        
        # Order vertices by "constraint tightness" - maybe just 0..n-1 is fine for n<=50
        # To speed up, we can check conflicts incrementally
        
        def backtrack(pos):
            if pos == n:
                return True
            
            # Heuristic: try colors that cause fewest conflicts with already assigned vertices
            # For simplicity, try 0,1,2,3
            for c in [0, 1, 2, 3]:
                conflict = False
                # Check against all previously assigned vertices
                for prev in range(pos):
                    if assigned[prev] == -1:
                        continue
                    if assigned[prev] == c and adj[prev, pos] == c:
                        conflict = True
                        break
                if not conflict:
                    assigned[pos] = c
                    if backtrack(pos + 1):
                        return True
                    assigned[pos] = -1
            return False

        if backtrack(0):
            for i in range(n):
                new_adj[n, i] = assigned[i]
                new_adj[i, n] = assigned[i]
            return new_adj
        return None

    def local_search(adj, n, time_limit):
        """
        Runs Simulated Annealing / Local Search to minimize mono-triangles.
        Returns (success, adj)
        """
        start_time = time.time()
        current_energy = count_mono_triangles(adj, n)
        
        if current_energy == 0:
            return True, adj
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute masks for speed? No, dynamic changes make it hard.
        # We will use vectorized delta calculation.
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            c_old = adj[u, v]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                c_new = (c_new + 1) % 4
            
            # Calculate Delta Energy
            # Triangles containing (u,v) are (u,v,w) for w != u,v
            # Old: w where adj[u,w]==c_old and adj[v,w]==c_old
            # New: w where adj[u,w]==c_new and adj[v,w]==c_new
            
            # Exclude u and v from the count
            # adj[u, :] includes u and v.
            # mask_u_old = (adj[u, :] == c_old)
            # mask_v_old = (adj[v, :] == c_old)
            # count_old = np.sum(mask_u_old & mask_v_old) - (1 if adj[u,u]==c_old else 0) - (1 if adj[v,v]==c_old else 0) - (1 if adj[u,v]==c_old and adj[v,u]==c_old else 0)
            # Actually, adj[u,u] is 0. If c_old is 0, it counts.
            # Let's just create masks and zero out u and v.
            
            mask_u_old = (adj[u, :] == c_old)
            mask_v_old = (adj[v, :] == c_old)
            # Create a copy to modify? No, use logical indexing or just sum and subtract.
            # Simpler: 
            # The intersection includes w=u and w=v.
            # At w=u: adj[u,u]=0, adj[v,u]=c_old. So it counts if 0==c_old and c_old==c_old. i.e., if c_old==0.
            # At w=v: adj[u,v]=c_old, adj[v,v]=0. So it counts if c_old==0.
            # So if c_old==0, the sum includes 2 "ghost" counts.
            # We want to exclude w=u and w=v.
            
            # Efficient way:
            # sum_old = np.sum((adj[u, :] == c_old) & (adj[v, :] == c_old))
            # if c_old == 0: sum_old -= 2 # because adj[u,u]=0 and adj[v,v]=0
            # Wait, adj[u,v] is also c_old. The mask is (adj[u,w]==c_old) and (adj[v,w]==c_old).
            # For w=u: adj[u,u]=0, adj[v,u]=c_old. Match if c_old==0.
            # For w=v: adj[u,v]=c_old, adj[v,v]=0. Match if c_old==0.
            # For w=u and w=v, we are counting the "loops".
            # So yes, subtract 2 if c_old == 0.
            
            sum_old = np.sum((adj[u, :] == c_old) & (adj[v, :] == c_old))
            if c_old == 0:
                sum_old -= 2 # Exclude w=u and w=v
                
            sum_new = np.sum((adj[u, :] == c_new) & (adj[v, :] == c_new))
            if c_new == 0:
                sum_new -= 2
                
            delta_e = sum_new - sum_old
            
            # Acceptance
            if delta_e < 0 or (delta_e == 0 and random.random() < 0.5):
                adj[u, v] = c_new
                adj[v, u] = c_new
                current_energy += delta_e
                if current_energy == 0:
                    return True, adj
            else:
                if random.random() < np.exp(-delta_e / T):
                    adj[u, v] = c_new
                    adj[v, u] = c_new
                    current_energy += delta_e
                    if current_energy == 0:
                        return True, adj
            
            T *= cooling
            if T < min_T:
                T = min_T

        return False, adj

    # Strategy
    # 1. Start with n=32
    adj = colors_to_adj(colors_32, 32)
    current_n = 32
    
    # 2. Try to extend greedily
    while time.time() < deadline - 20:
        new_adj = try_extend(adj, current_n)
        if new_adj is not None:
            adj = new_adj
            current_n += 1
            # If we reached 50, we are done (or can try higher)
            if current_n >= 50:
                # Verify and return
                if count_mono_triangles(adj, current_n) == 0:
                    return {'n': current_n, 'colors': adj_to_colors(adj, current_n)}
        else:
            break
            
    # 3. If we couldn't extend to 50 via greedy, try Local Search on a larger graph
    # We want to reach at least 50.
    target_n = 50
    if current_n < target_n:
        # Create a graph of size target_n
        # Initialize with the current valid part and random rest
        adj_target = np.zeros((target_n, target_n), dtype=np.int8)
        adj_target[:current_n, :current_n] = adj
        # Fill the rest randomly
        for i in range(current_n, target_n):
            for j in range(i+1, target_n):
                adj_target[i, j] = random.randint(0, 3)
                adj_target[j, i] = adj_target[i, j]
        # Also fill edges between old and new
        for i in range(current_n):
            for j in range(current_n, target_n):
                adj_target[i, j] = random.randint(0, 3)
                adj_target[j, i] = adj_target[i, j]
        
        # Run local search
        time_left = deadline - time.time() - 5
        if time_left > 1:
            success, adj_target = local_search(adj_target, target_n, time_left)
            if success:
                return {'n': target_n, 'colors': adj_to_colors(adj_target, target_n)}
            
        # If local search failed to find 0, maybe it found a valid smaller graph?
        # Unlikely to be valid if it started with random edges.
        # But we can try to extract the best valid subgraph?
        # That's complex. Better to just return the best greedy result if it's valid.
        
    # Fallback: Return the best valid greedy result
    if count_mono_triangles(adj, current_n) == 0:
        return {'n': current_n, 'colors': adj_to_colors(adj, current_n)}
    
    # If even the greedy part is invalid (shouldn't happen), return 32
    return {'n': 32, 'colors': colors_32}

