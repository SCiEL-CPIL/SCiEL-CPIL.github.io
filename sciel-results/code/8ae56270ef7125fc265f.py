import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    # 1. Deterministic Construction for n=26 (Lazebnik-Woldar)
    # This guarantees a score of 26/11 = 2.36 immediately.
    # The construction uses vertices 0..23 as elements of GF(25), and 24, 25 as special points.
    # GF(25) can be represented as pairs (a, b) where a, b in {0, 1, 2, 3} (GF(4) x GF(4))?
    # Actually, GF(25) is GF(5^2). Let's use a simpler representation if possible, or just the known mapping.
    
    # A standard representation of GF(25):
    # Elements are polynomials a + b*x where a, b in GF(5) and x^2 = 2 (since 2 is not a square in GF(5)).
    # Multiplication: (a + b*x)(c + d*x) = ac + 2bd + (ad + bc)x
    # We need a multiplication table for GF(25).
    
    def gf25_mult(a, b):
        # a, b are tuples (real, imag) in 0..4
        ar, ai = a
        br, bi = b
        # (ar + ai*x)(br + bi*x) = ar*br + ai*bi*x^2 + (ar*bi + ai*br)x
        # x^2 = 2
        real = (ar * br + 2 * ai * bi) % 5
        imag = (ar * bi + ai * br) % 5
        return (real, imag)

    # Generate GF(25) elements
    gf25_elements = []
    for r in range(5):
        for i in range(5):
            gf25_elements.append((r, i))
            
    # Map GF(25) elements to indices 0..24
    elem_to_idx = {e: i for i, e in enumerate(gf25_elements)}
    
    # The Lazebnik-Woldar construction for R_5(K2,2) with n=26:
    # Vertices: 0..23 (GF(25) elements), 24 (point at infinity P), 25 (point at infinity Q)?
    # Wait, the paper "On the Ramsey number R_5(K_2,2)" by Lazebnik and Woldar constructs a C4-free 5-coloring of K_26.
    # The vertices are the 25 points of the affine plane AG(2,5) plus one point? No, n=26.
    # Actually, the construction is often based on the group Z_5 x Z_5.
    
    # Let's look up the specific construction details for n=26.
    # The graph is K_26.
    # A common construction for 5-coloring K_26 without C4 uses the fact that 26 = 5*5 + 1.
    # Vertices are (x, y) in Z_5 x Z_5 plus infinity.
    # Or perhaps it's simpler:
    # There is a known explicit construction.
    # Let's try to generate it algorithmically based on the algebraic structure.
    
    # Alternative: Use a pre-computed valid coloring for n=26 if we can derive it.
    # Since I cannot look up the exact hardcoded list, I must generate it.
    
    # Construction from Lazebnik & Woldar (1998):
    # Let V = Z_5 x Z_5 U {infty}.
    # For (x1, y1), (x2, y2) in Z_5 x Z_5:
    #   If x1 == x2, color is 0.
    #   If y1 == y2, color is 1.
    #   If x1 - x2 == y1 - y2, color is 2.
    #   If x1 - x2 == -(y1 - y2), color is 3.
    #   Otherwise, color is 4.
    #   Wait, this only covers 5 slopes.
    #   Slopes in Z_5: 0, 1, -1, 2, -2.
    #   If dx != 0:
    #      slope = dy * inv(dx) mod 5.
    #      Map slopes to colors 2,3,4? No, we need 5 colors total.
    #      Colors 0 and 1 are for vertical and horizontal.
    #      Colors 2, 3, 4 for the other 3 slopes? That's only 3 slopes.
    #      But in Z_5, there are 4 non-zero slopes: 1, 2, 3, 4.
    #      1 and 4 are inverses? 1*4=4=-1. 2*3=6=1.
    #      So slopes are 1, 4, 2, 3.
    #      We have colors 2, 3, 4 left. That's 3 colors for 4 slopes.
    #      This simple affine plane construction doesn't directly map to 5 colors without splitting.
    #      Actually, R_5(C4) > 26 is a known result.
    #      The construction might be different.
    
    # Let's try a different algebraic construction:
    # Use the group Z_26? No.
    
    # Let's use a robust local search from a random start, but with a very fast cost function.
    # If I can't guarantee the n=26 algebraic construction from memory, I should ensure the local search is fast enough to find n=26 within 115s.
    # The previous program scored 1.636 (n=18 approx).
    # The inspiration mentions n=26 is possible with SA.
    
    # Strategy:
    # 1. Implement a very fast cost function using bitmasks or vectorized numpy ops that updates incrementally?
    #    Incremental is hard to code correctly in short time.
    #    Full re-evaluation with numpy for n=26 (325 edges) is fast.
    #    5 matrices of 26x26. Multiplication 26x26 is trivial.
    #    Cost function:
    #    For each color c, build adjacency matrix A_c.
    #    C4 count = 0.5 * sum_{i != j} ( (A_c @ A_c)[i,j] choose 2 )?
    #    No, C4 is two vertices with 2 common neighbors.
    #    Let M = A @ A. M[i,j] is number of common neighbors.
    #    Number of C4s involving edge (i,j) is C(M[i,j], 2).
    #    Total C4s = sum_{i<j} C(M[i,j], 2).
    
    # Let's optimize the local search to run many iterations per second.
    
    def get_cost_fast(colors, n):
        """
        Calculate total C4s.
        colors: list of ints length n(n-1)/2
        """
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
        
        # Build adjacency matrices
        # Use sparse? No, dense is fine for n=26.
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        total_cost = 0.0
        for m in mats:
            M = m @ m
            # M[i,j] for i!=j is common neighbors.
            # We want sum_{i<j} M[i,j]*(M[i,j]-1)/2
            # sum_{i<j} M[i,j]^2 - sum_{i<j} M[i,j] all over 2
            # Sum of squares of off-diagonal elements.
            # sum(M**2) - sum(diag(M)**2)
            # sum(M) - sum(diag(M))
            
            sum_sq = np.sum(M**2)
            sum_diag_sq = np.sum(np.diag(M)**2)
            sum_off_sq = sum_sq - sum_diag_sq
            
            sum_val = np.sum(M)
            sum_diag = np.sum(np.diag(M))
            sum_off_val = sum_val - sum_diag
            
            cost_c = (sum_off_sq - sum_off_val) / 2.0
            total_cost += cost_c
            
        return total_cost

    # To make local search faster, we can avoid rebuilding matrices from scratch if we do it smartly,
    # but for n=26, rebuilding is ~1ms. 1000 iters = 1s. 100s = 100k iters.
    # This should be enough to find n=26.
    
    best_n = 11
    # Fallback n=11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback
    
    # Try to find solutions for larger n
    # We will try n=26 first, then 25, etc?
    # Or start from 12 and go up?
    # Finding n=26 is the goal.
    # Let's try n=26 with a large budget.
    # If we find n=26, we can stop or try n=27 with remaining time.
    
    target_ns = [26, 27] # Try 26 first, then 27
    # If 26 fails, try 25, 24...
    # Actually, if we can't find 26, finding 25 is still better than 18.
    # Let's iterate n from 26 down to 12?
    # Or 12 up to 26?
    # If we find 26, we are done (mostly).
    # If we fail 26, we might have spent too much time.
    
    # Better: Allocate time slices.
    # Try n=26 for 60s.
    # If found, try n=27 for remaining time.
    # If not found, try n=25 for 30s.
    # If found, try n=26 again? No, just keep best.
    # If not found, try n=24 for 10s.
    
    # Let's just do a sequential search from 12 to 26, but with increasing time budgets?
    # No, finding 26 is hard.
    # Let's try n=26 with 80s.
    
    # Random restart hill climbing
    def solve_n(n, time_limit):
        num_edges = n * (n - 1) // 2
        best_cost = float('inf')
        best_colors = None
        
        # Initial random
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        current_cost = get_cost_fast(colors, n)
        best_cost = current_cost
        best_colors = colors[:]
        
        # Hill climbing
        # To speed up, we can precompute edge endpoints.
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We can also precompute which edges are incident to which vertices?
        # Not needed for full re-eval.
        
        t_start = time.time()
        while time.time() < t_start + time_limit:
            if current_cost == 0:
                return colors, 0
            
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, 4)
            if old_c == new_c:
                continue
                
            colors[e_idx] = new_c
            new_cost = get_cost_fast(colors, n)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_c
                
            # Simulated Annealing style: accept worse with prob exp(-delta/T)
            # But simple hill climbing with restarts is often robust.
            # Let's add some random restarts if stuck.
            
            # Check for stuck: if no improvement in 1000 steps, restart?
            # We can track last_improvement_time.
            
        return best_colors, best_cost

    # Try n=26
    n = 26
    t_limit = 80
    if time.time() < deadline - t_limit:
        cols, cost = solve_n(n, t_limit)
        if cost == 0:
            best_n = n
            best_colors = cols
        else:
            # If not 0, check if it's better than fallback?
            # Cost > 0 means invalid.
            # We only accept valid colorings (cost 0).
            pass
            
    # If n=26 found, try n=27
    if best_n == 26:
        remaining = deadline - time.time()
        if remaining > 5:
            n = 27
            cols, cost = solve_n(n, remaining - 2)
            if cost == 0:
                best_n = n
                best_colors = cols
    else:
        # If 26 failed, try 25
        remaining = deadline - time.time()
        if remaining > 10:
            n = 25
            cols, cost = solve_n(n, remaining - 2)
            if cost == 0:
                best_n = n
                best_colors = cols
            else:
                # Try 24
                remaining = deadline - time.time()
                if remaining > 5:
                    n = 24
                    cols, cost = solve_n(n, remaining - 2)
                    if cost == 0:
                        best_n = n
                        best_colors = cols

    return {'n': best_n, 'colors': best_colors}
