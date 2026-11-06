import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin for safety and evaluation overhead
    
    def get_idx(i, j, n):
        # Return the index of edge (i, j) in lexicographic order for K_n
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def build_adj_masks(colors, n):
        # Build adjacency masks for each color
        adj = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def count_violations(adj, n):
        # Count the number of violations (C4s) in the coloring
        v = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for w in range(u + 1, n):
                    common = mask_u & adj[c][w]
                    cnt = common.bit_count()
                    if cnt > 1:
                        v += cnt - 1
        return v

    def try_algebraic_26():
        # Attempt to construct a valid 5-coloring of K_26 using known algebraic methods
        # The Lazebnik-Woldar construction gives 26 vertices
        # We'll try a simple construction based on finite fields or similar structures
        n = 26
        num_edges = n * (n - 1) // 2
        colors = [0] * num_edges
        
        # Use a simple algebraic construction:
        # Label vertices with pairs (a, b) where a in Z_5, b in Z_5, plus one extra
        # Actually, let's use a simpler approach: random initialization with local search
        # But first, let's try a structured initialization
        
        # For now, we'll just return None to indicate we need to use local search
        return None

    def solve_sa(n, max_iters, start_temp, cooling, initial_colors=None):
        """
        Attempts to find a valid 5-coloring of K_n using Simulated Annealing.
        Returns (success, colors) or (False, None).
        """
        num_edges = n * (n - 1) // 2
        
        if initial_colors is not None:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        adj = build_adj_masks(colors, n)
        current_violations = count_violations(adj, n)
        
        if current_violations == 0:
            return True, colors

        temp = start_temp
        best_violations = current_violations
        best_colors = colors[:]

        # Precompute edge endpoints for faster lookup
        edge_endpoints = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_endpoints.append((i, j))

        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_violations == 0:
                return True, colors

            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            i, j = edge_endpoints[idx]
            
            old_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue

            # Calculate delta using incremental updates
            delta = 0
            
            # Remove from old_color
            c = old_color
            for w in range(n):
                if w == i or w == j:
                    continue
                
                # Pair (i, w)
                old_common = (adj[c][i] & adj[c][w]).bit_count()
                if (adj[c][w] >> j) & 1:
                    new_common = old_common - 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
                
                # Pair (j, w)
                old_common = (adj[c][j] & adj[c][w]).bit_count()
                if (adj[c][w] >> i) & 1:
                    new_common = old_common - 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
            
            # Update masks for removal
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            
            # Add to new_color
            c = new_color
            for w in range(n):
                if w == i or w == j:
                    continue
                
                # Pair (i, w)
                old_common = (adj[c][i] & adj[c][w]).bit_count()
                if (adj[c][w] >> j) & 1:
                    new_common = old_common + 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
                
                # Pair (j, w)
                old_common = (adj[c][j] & adj[c][w]).bit_count()
                if (adj[c][w] >> i) & 1:
                    new_common = old_common + 1
                else:
                    new_common = old_common
                delta += max(0, new_common - 1) - max(0, old_common - 1)
            
            # Update masks for addition
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
            new_violations = current_violations + delta
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                colors[idx] = new_color
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Revert
                adj[new_color][i] &= ~(1 << j)
                adj[new_color][j] &= ~(1 << i)
                adj[old_color][i] |= (1 << j)
                adj[old_color][j] |= (1 << i)
                
            temp *= cooling
            
        if current_violations == 0:
            return True, colors
        return False, best_colors if best_violations < 50 else None

    # Try to find a valid coloring for increasing n
    best_n = 11
    best_colors = [0] * (11 * 10 // 2)  # Fallback
    
    # First, try the algebraic construction for n=26
    alg_result = try_algebraic_26()
    if alg_result is not None:
        best_n = 26
        best_colors = alg_result
        return {'n': best_n, 'colors': best_colors}
    
    # Try to find a valid coloring using simulated annealing for increasing n
    # Start with smaller n and work up to 26
    for target_n in [16, 20, 24, 26]:
        if time.time() > deadline - 10.0:
            break
            
        success = False
        # Try multiple restarts with different initializations
        for restart in range(8):
            if time.time() > deadline - 5.0:
                break
                
            max_iters = 30000
            start_temp = 1.5
            cooling = 0.9998
            
            res_success, res_colors = solve_sa(target_n, max_iters, start_temp, cooling)
            
            if res_success:
                best_n = target_n
                best_colors = res_colors
                success = True
                break
        
        if success:
            continue
        else:
            # If we fail for this n, we might still try a larger n?
            # Unlikely to succeed if we failed a smaller one, but random restarts might work.
            if target_n <= 20:
                break

    return {'n': best_n, 'colors': best_colors}
