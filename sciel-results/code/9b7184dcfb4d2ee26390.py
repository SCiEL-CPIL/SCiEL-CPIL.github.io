import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for evaluator kill at 150s
    
    # Helper: Check if a coloring is valid (0 monochromatic triangles)
    def count_mono_triangles(adj, n):
        # adj is n x n numpy array of colors (0-3), symmetric, 0 on diagonal
        # We count triangles (i, j, k) where adj[i,j] == adj[j,k] == adj[k,i]
        # Vectorized approach:
        # For each color c, let A_c be the adjacency matrix of edges with color c.
        # Number of triangles in color c is (1/6) * trace(A_c^3).
        # Since A_c is symmetric and 0-diagonal, trace(A_c^3) counts each triangle 6 times.
        
        total = 0
        for c in range(4):
            A = (adj == c).astype(np.int64)
            # A^3 diagonal sum
            # trace(A^3) = sum_i (A^3)_ii = sum_i sum_j sum_k A_ij A_jk A_ki
            # This is O(n^3) if done naively, but numpy optimizes matrix mult.
            # For n=50, 50x50x50 is 125,000 ops, very fast.
            A2 = A @ A
            A3 = A2 @ A
            total += np.trace(A3)
        return total // 6

    # Helper: Compute delta in mono-triangles for flipping edge (u, v) from c_old to c_new
    def flip_delta(adj, n, u, v, c_old, c_new):
        if c_old == c_new:
            return 0
        
        # We need to count how many triangles are broken and how many are formed.
        # A triangle (u, v, w) is monochromatic of color c if adj[u,w]==c and adj[v,w]==c.
        
        # Broken triangles:
        # Triangles that were mono-c_old and involved edge (u,v).
        # These are vertices w such that adj[u,w] == c_old AND adj[v,w] == c_old.
        # Note: w cannot be u or v.
        
        # Formed triangles:
        # Triangles that become mono-c_new and involve edge (u,v).
        # These are vertices w such that adj[u,w] == c_new AND adj[v,w] == c_new.
        
        # Vectorized count:
        # Create a mask for w
        # adj_u = adj[u], adj_v = adj[v]
        # broken = count(w where adj_u[w]==c_old and adj_v[w]==c_old)
        # formed = count(w where adj_u[w]==c_new and adj_v[w]==c_new)
        
        # We must exclude w=u and w=v.
        # adj[u,u] is 0, adj[v,v] is 0. c_old/c_new are 1..4 (or 0..3). 
        # If c_old is 0, and adj[u,u]=0, it might interfere? 
        # In our representation, diagonal is 0. Edge colors are 0,1,2,3.
        # So if c_old=0, adj[u,u]==0 is true. We need to be careful.
        # Let's set diagonal to a value not in {0,1,2,3}, e.g., -1.
        
        # But wait, the problem says colors in {0,1,2,3}.
        # If we use 0 for diagonal, and 0 is a valid color, we have a conflict.
        # Solution: Use colors 1,2,3,4 internally? Or just mask out u and v.
        
        # Let's mask out u and v explicitly.
        mask_u = np.ones(n, dtype=bool)
        mask_v = np.ones(n, dtype=bool)
        mask_u[u] = False
        mask_u[v] = False
        mask_v[u] = False
        mask_v[v] = False
        
        # Actually, simpler:
        # adj_u = adj[u].copy(); adj_u[u]= -1; adj_u[v]=-1
        # adj_v = adj[v].copy(); adj_v[u]= -1; adj_v[v]=-1
        
        # Let's do it without copy for speed if possible, or just simple logic.
        # Since n is small, let's just use numpy boolean indexing on slices.
        
        # adj[u, :] gives colors of edges from u to all w.
        # We want w != u, w != v.
        
        # Vectorized:
        # cond_broken = (adj[u, :] == c_old) & (adj[v, :] == c_old)
        # cond_formed = (adj[u, :] == c_new) & (adj[v, :] == c_new)
        
        # Set positions u and v to False in these masks.
        cond_broken = (adj[u, :] == c_old) & (adj[v, :] == c_old)
        cond_broken[u] = False
        cond_broken[v] = False
        broken = np.sum(cond_broken)
        
        cond_formed = (adj[u, :] == c_new) & (adj[v, :] == c_new)
        cond_formed[u] = False
        cond_formed[v] = False
        formed = np.sum(cond_formed)
        
        # Each broken triangle reduces count by 1.
        # Each formed triangle increases count by 1.
        return formed - broken

    # Helper: Local Search (Hill Climbing)
    # Returns (new_adj, new_cost, improved)
    def local_search(adj, n, max_iters=5000, time_limit=5.0):
        cost = count_mono_triangles(adj, n)
        if cost == 0:
            return adj, cost, True
        
        current_adj = adj.copy()
        start_ls = time.time()
        
        # Precompute edge list for random selection? 
        # Or just random u, v.
        
        for _ in range(max_iters):
            if time.time() - start_ls > time_limit:
                break
            if cost == 0:
                return current_adj, cost, True
            
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            c_old = current_adj[u, v]
            
            # Try all other colors
            best_delta = 0
            best_c = -1
            for c_new in range(4):
                if c_new == c_old:
                    continue
                delta = flip_delta(current_adj, n, u, v, c_old, c_new)
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            if best_c != -1 and best_delta < 0:
                current_adj[u, v] = best_c
                current_adj[v, u] = best_c
                cost += best_delta
            else:
                # Stuck in local minimum? 
                # For pure hill climbing, we stop. 
                # But we can add a small random perturbation to escape?
                # Let's stick to strict hill climbing for now, and rely on restarts.
                # Or, allow "sideways" moves (delta=0) to traverse plateaus?
                # Let's try to find a sideways move if stuck.
                found_sideways = False
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    delta = flip_delta(current_adj, n, u, v, c_old, c_new)
                    if delta == 0:
                        current_adj[u, v] = c_new
                        current_adj[v, u] = c_new
                        found_sideways = True
                        break
                if not found_sideways:
                    # No improving or sideways move for this edge.
                    # In strict HC, we might just continue to next random edge.
                    # If we do this for many iterations without progress, we are stuck.
                    # Let's just break if we've tried many edges and no improvement.
                    # To keep it simple, we just continue the loop.
                    pass
        
        return current_adj, cost, (cost == 0)

    # Helper: Generate initial K32 coloring
    def get_initial_32():
        # Greenwood-Gleason construction for K32
        # Vertices are elements of GF(16). 
        # Color of edge (x, y) is determined by x/y in GF(16)*.
        # GF(16)* is cyclic of order 15.
        # We map the 15 non-zero elements to 3 colors (5 elements each) and 0 to color 3?
        # Actually, the standard construction uses the fact that 15 is divisible by 3.
        # Let's use the explicit Clebsch graph logic or a known valid K32.
        # Simpler: Use a known valid K32 from literature or generate via random + LS.
        # Since K32 is small, we can just random init and LS to 0.
        # But to be safe and fast, let's use a deterministic seed.
        
        # Actually, the Clebsch graph is a 4-coloring of K32?
        # The Clebsch graph is a 16-vertex graph.
        # The Greenwood-Gleason construction gives K(3,3,3) > 32? No, R(3,3,3) = 17.
        # Wait, the task is 4-coloring. R(3,3,3,3) > 50.
        # A known valid 4-coloring of K32 exists.
        # Let's just random initialize and run LS. It's fast for n=32.
        
        adj = np.zeros((32, 32), dtype=np.int8)
        for i in range(32):
            for j in range(i + 1, 32):
                c = random.randint(0, 3)
                adj[i, j] = c
                adj[j, i] = c
        
        adj, cost, _ = local_search(adj, 32, max_iters=10000, time_limit=2.0)
        # If it didn't reach 0, it's not valid. But K32 is known to be colorable.
        # If LS fails, we might return an invalid state. 
        # We should verify.
        if cost != 0:
             # Retry with different seed
             for _ in range(5):
                 adj = np.zeros((32, 32), dtype=np.int8)
                 for i in range(32):
                     for j in range(i + 1, 32):
                         c = random.randint(0, 3)
                         adj[i, j] = c
                         adj[j, i] = c
                 adj, cost, _ = local_search(adj, 32, max_iters=10000, time_limit=2.0)
                 if cost == 0:
                     break
        
        return adj, cost

    # Main Logic
    # Start with K32
    adj, cost = get_initial_32()
    if cost != 0:
        # Fallback: return K1 or K2 if we can't even get K32 (unlikely)
        return {'n': 1, 'colors': []}
    
    best_n = 32
    best_adj = adj.copy()
    
    current_n = 32
    current_adj = adj
    
    while current_n < 62:
        if time.time() > deadline - 10:
            break
            
        # Try to extend to current_n + 1
        new_n = current_n + 1
        new_adj = np.zeros((new_n, new_n), dtype=np.int8)
        
        # Copy old edges
        new_adj[:current_n, :current_n] = current_adj
        
        # Initialize new vertex (index current_n)
        # Heuristic: Assign colors to minimize immediate threats
        # For each existing vertex i, we choose color c for edge (i, new)
        # such that it minimizes the number of monochromatic triangles involving new.
        # Triangle (i, j, new) is mono-c if adj[i,j]==c and adj[i,new]==c and adj[j,new]==c.
        # This is complex to optimize greedily in one pass.
        # Simple heuristic: Random, then LS.
        
        for i in range(current_n):
            c = random.randint(0, 3)
            new_adj[i, new_n - 1] = c
            new_adj[new_n - 1, i] = c
        
        # Run Local Search on the new graph
        # We need to be careful: LS might break the validity of the old K_n part.
        # However, if the old part was valid (0 triangles), and we only add edges to new vertex,
        # any new triangle must involve the new vertex.
        # If LS changes an old edge, it might create a triangle in the old part.
        # To prevent this, we could restrict moves to edges involving the new vertex.
        # But sometimes changing an old edge helps resolve conflicts with the new vertex.
        # Let's allow all moves, but if cost goes up, we reject (HC).
        # If we get stuck with cost > 0, we fail.
        
        # To improve success rate, let's try multiple random initializations for the new vertex.
        success = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Re-init new vertex edges if attempt > 0
            if attempt > 0:
                new_adj = np.zeros((new_n, new_n), dtype=np.int8)
                new_adj[:current_n, :current_n] = current_adj
                for i in range(current_n):
                    c = random.randint(0, 3)
                    new_adj[i, new_n - 1] = c
                    new_adj[new_n - 1, i] = c
            
            new_adj_ls, new_cost, ok = local_search(new_adj, new_n, max_iters=2000, time_limit=1.0)
            
            if ok and new_cost == 0:
                # Check if the old part is still valid?
                # LS on the whole graph ensures total cost is 0.
                # So the old part is also valid (subset of a valid graph is valid).
                current_n = new_n
                current_adj = new_adj_ls
                best_n = current_n
                best_adj = current_adj.copy()
                success = True
                break
        
        if not success:
            # Failed to extend.
            # Try to perturb current_adj to find a different valid coloring that might be easier to extend?
            # This is expensive.
            # Instead, we can try "Simulated Annealing" on the new vertex edges only?
            # Or just break.
            # Given the time constraint, breaking is safer than risking timeout.
            break

    # Convert best_adj to flat list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(int(best_adj[i, j]))
    
    return {'n': best_n, 'colors': colors}

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    # Verify
    n = res['n']
    if n > 2:
        # Reconstruct adj
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = res['colors'][idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        cost = count_mono_triangles(adj, n)
        print(f"Verification: {cost} mono triangles")
