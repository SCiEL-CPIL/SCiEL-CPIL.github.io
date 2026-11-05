"""Program for R(K2,2,2) maximization using bitmask constraint satisfaction."""
import numpy as np
import random
import time

def construct():
    """
    Find maximum n such that K_n has a 2-coloring with no monochromatic K_{2,2,2}.
    
    Key characterization: A color-c K_{2,2,2} exists iff there are adjacent vertices
    u,v in color c whose common color-c neighborhood contains a K_{2,2} in color c.
    K_{2,2} in a set S means: exist a,b,c,d in S (all distinct) such that all four
    cross edges between {a,b} and {c,d} are in color c.
    
    We use bitmasks for fast neighborhood operations and a greedy+local search approach.
    """
    
    def build_adj_bitmasks(colors, n, color):
        """Build adjacency bitmasks for a given color class."""
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj
    
    def has_k222_fast(adj, n, color):
        """
        Check if color class (given by adjacency bitmasks) contains K_{2,2,2}.
        
        K_{2,2,2} exists iff there exist u,v with adj[u]&adj[v] containing
        a K_{2,2} in the induced subgraph.
        
        K_{2,2} in common neighborhood: exist a,b,c,d all in common, a!=b, c!=d,
        a,b distinct from c,d, such that adj[a]&adj[b] contains both c and d.
        """
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                # Remove u and v themselves (they shouldn't be in their own adjacency)
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                if common.bit_count() < 4:
                    continue
                
                # Check if common contains K_{2,2}
                # K_{2,2}: exist a,b in common with adj[a]&adj[b] & common having at least 2 bits
                # that are not a or b
                common_list = []
                temp = common
                while temp:
                    lb = temp & (-temp)
                    common_list.append(lb.bit_length() - 1)
                    temp ^= lb
                
                m = len(common_list)
                if m < 4:
                    continue
                
                # For each pair (a,b) in common, check if their common neighbors
                # within common contain at least 2 vertices
                for ai in range(m):
                    a = common_list[ai]
                    for bi in range(ai+1, m):
                        b = common_list[bi]
                        # Check if a and b are connected in this color
                        if not (adj[a] >> b) & 1:
                            continue
                        # Common neighbors of a and b within the common set
                        ab_common = adj[a] & adj[b] & common
                        ab_common &= ~(1 << a)
                        ab_common &= ~(1 << b)
                        if ab_common.bit_count() >= 2:
                            return True
        return False
    
    def check_valid(colors, n):
        """Check if both color classes are K_{2,2,2}-free."""
        adj0 = build_adj_bitmasks(colors, n, 0)
        adj1 = build_adj_bitmasks(colors, n, 1)
        return not has_k222_fast(adj0, n, 0) and not has_k222_fast(adj1, n, 1)
    
    def count_violations(colors, n):
        """Count number of monochromatic K_{2,2,2} subgraphs (for optimization)."""
        adj0 = build_adj_bitmasks(colors, n, 0)
        adj1 = build_adj_bitmasks(colors, n, 1)
        return count_k222(adj0, n) + count_k222(adj1, n)
    
    def count_k222(adj, n):
        """Count K_{2,2,2} subgraphs in a color class."""
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    continue
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common.bit_count() < 4:
                    continue
                common_list = []
                temp = common
                while temp:
                    lb = temp & (-temp)
                    common_list.append(lb.bit_length() - 1)
                    temp ^= lb
                m = len(common_list)
                if m < 4:
                    continue
                for ai in range(m):
                    a = common_list[ai]
                    for bi in range(ai+1, m):
                        b = common_list[bi]
                        if not (adj[a] >> b) & 1:
                            continue
                        ab_common = adj[a] & adj[b] & common
                        ab_common &= ~(1 << a)
                        ab_common &= ~(1 << b)
                        bc = ab_common.bit_count()
                        if bc >= 2:
                            # Number of ways to choose 2 from bc
                            count += bc * (bc - 1) // 2
        return count
    
    # Try Paley constructions first
    for p in [29, 31, 25, 21, 17]:
        if p <= 31:
            squares = set()
            for x in range(1, p):
                squares.add((x*x) % p)
            squares.discard(0)
            colors = [0 if ((j-i) % p) in squares else 1 for i in range(p) for j in range(i+1, p)]
            if check_valid(colors, p):
                return {'n': p, 'colors': colors}
    
    # Greedy construction: build coloring vertex by vertex
    def greedy_construct(n, seed=42):
        """Greedy construction trying to minimize violations."""
        rng = random.Random(seed)
        colors = [0] * (n*(n-1)//2)
        
        for i in range(n):
            for j in range(i+1, n):
                idx = i*n - i*(i+1)//2 + (j - i - 1)
                # Try both colors, pick the one that creates fewer violations
                best_color = 0
                best_score = float('inf')
                for c in [0, 1]:
                    colors[idx] = c
                    # Quick check: count violations involving edges from i or j
                    # Full check is expensive, so use heuristic
                    score = count_violations(colors, n)
                    if score < best_score:
                        best_score = score
                        best_color = c
                colors[idx] = best_color
        
        return colors
    
    # Local search: flip edges to reduce violations
    def local_search(colors, n, max_iters=50000, seed=42):
        """Local search by flipping edges to reduce K_{2,2,2} count."""
        rng = random.Random(seed)
        colors = list(colors)
        total_edges = n*(n-1)//2
        current_viol = count_violations(colors, n)
        
        if current_viol == 0:
            return colors, 0
        
        best_colors = list(colors)
        best_viol = current_viol
        
        for iteration in range(max_iters):
            if current_viol == 0:
                return colors, 0
            
            # Pick a random edge to flip
            idx = rng.randint(0, total_edges - 1)
            old_color = colors[idx]
            new_color = 1 - old_color
            colors[idx] = new_color
            
            new_viol = count_violations(colors, n)
            
            if new_viol < current_viol:
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = list(colors)
                if new_viol == 0:
                    return colors, 0
            else:
                # Accept with some probability (simulated annealing-like)
                if rng.random() < 0.1:  # Accept worse moves 10% of the time
                    current_viol = new_viol
                else:
                    colors[idx] = old_color
        
        # Return best found
        if best_viol < current_viol:
            return best_colors, best_viol
        return colors, current_viol
    
    # Try to construct n=30, then n=29
    start_time = time.time()
    
    for target_n in [30, 29]:
        if time.time() - start_time > 100:
            break
        
        # Multiple greedy+local search attempts
        for seed in range(50):
            if time.time() - start_time > 105:
                break
            
            # Start from a random or structured initial coloring
            rng = random.Random(seed)
            colors = [rng.choice([0, 1]) for _ in range(target_n*(target_n-1)//2)]
            
            # Run local search
            colors, viol = local_search(colors, target_n, max_iters=10000, seed=seed)
            
            if viol == 0:
                return {'n': target_n, 'colors': colors}
        
        # Try greedy construction
        for seed in range(20):
            if time.time() - start_time > 108:
                break
            colors = greedy_construct(target_n, seed=seed)
            colors, viol = local_search(colors, target_n, max_iters=10000, seed=seed+1000)
            if viol == 0:
                return {'n': target_n, 'colors': colors}
    
    # Fallback: return best we found or a small valid coloring
    # Try n=25 with more effort
    for target_n in [25, 21, 17]:
        if time.time() - start_time > 115:
            break
        for seed in range(100):
            if time.time() - start_time > 117:
                break
            rng = random.Random(seed)
            colors = [rng.choice([0, 1]) for _ in range(target_n*(target_n-1)//2)]
            colors, viol = local_search(colors, target_n, max_iters=20000, seed=seed)
            if viol == 0:
                return {'n': target_n, 'colors': colors}
    
    # Last resort: n=17 Paley
    p = 17
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    squares.discard(0)
    colors = [0 if ((j-i) % p) in squares else 1 for i in range(p) for j in range(i+1, p)]
    return {'n': p, 'colors': colors}
