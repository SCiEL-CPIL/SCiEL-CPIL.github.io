import time
import random
import math

def construct():
    # Deadline guard: 140 seconds to be safe within 150s limit
    deadline = time.time() + 140.0
    random.seed(42) # Deterministic for reproducibility in this context, though random restarts help
    
    # Seed solution for n=11
    def get_seed(n):
        # K11 split into five Hamiltonian cycles by circular distance 1..5
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]

    best_n = 11
    best_colors = get_seed(11)
    
    # Helper to check if a coloring is valid and count violations
    def count_violations(colors, n):
        """
        Returns the number of monochromatic C4s.
        colors is a list of length n*(n-1)//2
        """
        # Build adjacency lists for each color
        # adj[c][u] = set of v such that (u,v) is color c
        adj = [dict() for _ in range(5)]
        
        # Map edge index to (u, v)
        # Index k corresponds to (i, j) where i < j
        # We can just iterate through all pairs
        
        # For efficiency, we can precompute pairs
        # But for small n, direct iteration is fine
        
        # Reconstruct adjacency
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if i not in adj[c]:
                    adj[c][i] = set()
                if j not in adj[c]:
                    adj[c][j] = set()
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        total_violations = 0
        for c in range(5):
            # Count C4s in color c
            # For each pair (u, v), count common neighbors
            # Sum binom(common, 2)
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors of u and v in color c
                    nu = adj[c].get(u, set())
                    nv = adj[c].get(v, set())
                    common = len(nu & nv)
                    if common >= 2:
                        total_violations += math.comb(common, 2)
        
        return total_violations

    def local_search(n, time_limit):
        """
        Attempt to find a valid 5-coloring of K_n with no monochromatic C4.
        Returns colors list if found, else None.
        """
        start_time = time.time()
        m = n * (n - 1) // 2
        
        # Initial random coloring
        colors = [random.randint(0, 4) for _ in range(m)]
        current_violations = count_violations(colors, n)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors.copy()
        best_violations = current_violations
        
        # Simulated Annealing / Hill Climbing
        # We will use a simple hill climbing with random restarts
        # Step: Pick random edge, change color, evaluate delta.
        
        # To speed up evaluation, we can just recalculate. 
        # For n up to 20, count_violations is O(5 * n^3) ~ 40k ops. 
        # 1000 steps * 40k = 40M ops, which is fine in Python if optimized? 
        # Python might be slow for 40M ops. Let's try to optimize count_violations or reduce steps.
        
        # Optimization: Only check if violations decreased.
        # We can also cache the adjacency sets.
        
        # Let's implement a faster delta calculation? 
        # Changing color of (u,v) from c_old to c_new:
        # Affects C4 counts in c_old and c_new.
        # This is complex. Given the small n, let's try the full recalculation first.
        
        steps = 0
        while time.time() < time_limit:
            steps += 1
            # Pick a random edge
            e_idx = random.randint(0, m - 1)
            
            # Find u, v for e_idx
            # e_idx = i * (n - 1) - i * (i + 1) / 2 + (j - i - 1) ?? 
            # Easier: precompute edge list
            # But we can just iterate to find it? No, too slow.
            # Let's precompute edge_map: index -> (u, v)
            
            # We need edge_map. Let's create it once outside the loop or inside setup.
            # Since local_search is called multiple times, we should pass it or create inside.
            
            # Let's create edge_map inside local_search setup
            # Actually, I'll modify the structure to have edge_map global or passed.
            
            # For now, let's assume we can map index to (u,v) quickly.
            # i is roughly sqrt(2*idx)
            # Let's just use a precomputed list in the outer scope or here.
            
            # To avoid recreating edge_map every time, let's do it here.
            # It's O(n^2).
            
            # Wait, I should define edge_map in the setup phase of local_search.
            
            # Let's rewrite local_search to be self-contained but efficient.
            
            break # Placeholder, will implement properly below

    # Proper Local Search Implementation
    def attempt_solution(n, time_limit):
        m = n * (n - 1) // 2
        # Precompute edge endpoints
        edge_map = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
                idx += 1
        
        def get_violations(colors):
            # Fast check
            adj = [dict() for _ in range(5)]
            for k in range(m):
                u, v = edge_map[k]
                c = colors[k]
                if u not in adj[c]:
                    adj[c][u] = set()
                if v not in adj[c]:
                    adj[c][v] = set()
                adj[c][u].add(v)
                adj[c][v].add(u)
            
            total = 0
            for c in range(5):
                for u in range(n):
                    nu = adj[c].get(u, set())
                    if not nu: continue
                    for v in range(u + 1, n):
                        nv = adj[c].get(v, set())
                        if not nv: continue
                        common = len(nu & nv)
                        if common >= 2:
                            total += math.comb(common, 2)
            return total

        # Random restarts
        restarts = 0
        max_restarts = 100
        while time.time() < time_limit and restarts < max_restarts:
            restarts += 1
            colors = [random.randint(0, 4) for _ in range(m)]
            curr_v = get_violations(colors)
            
            if curr_v == 0:
                return colors
            
            # Hill Climbing
            no_improve_count = 0
            max_no_improve = 1000
            temp = 1.0
            cooling = 0.999
            
            while time.time() < time_limit and no_improve_count < max_no_improve:
                # Pick random edge
                e_idx = random.randint(0, m - 1)
                old_color = colors[e_idx]
                if old_color == 4:
                    new_color = random.randint(0, 3)
                else:
                    new_color = random.randint(old_color + 1, 4)
                
                colors[e_idx] = new_color
                new_v = get_violations(colors)
                
                if new_v < curr_v:
                    curr_v = new_v
                    no_improve_count = 0
                    if curr_v == 0:
                        return colors
                else:
                    # Accept worse with probability exp(-delta/temp)
                    delta = new_v - curr_v
                    if random.random() < math.exp(-delta / temp):
                        curr_v = new_v
                    else:
                        colors[e_idx] = old_color # Revert
                    no_improve_count += 1
                
                temp *= cooling
            
            # If we exit inner loop without success, restart
            pass
        
        return None

    # Try to extend n
    for n in range(12, 21):
        if time.time() > deadline:
            break
        
        # Allocate time for this n
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        # Give more time to smaller n? Or equal time?
        # Let's give 10 seconds per attempt, or proportional.
        # Since 12 is easier than 20, we can spend less time on 12.
        # But if 12 fails, we might need to try 13.
        # Let's try 15 seconds per n.
        alloc_time = min(15.0, time_left - 2.0)
        
        if alloc_time <= 0:
            break
            
        res = attempt_solution(n, time.time() + alloc_time)
        
        if res is not None:
            best_n = n
            best_colors = res
            # If we found a valid n, we can try to go higher
            # But we should verify it first? 
            # attempt_solution returns None if not valid. 
            # It only returns colors if violations == 0.
            # So it is valid.
            
    return {'n': best_n, 'colors': best_colors}
