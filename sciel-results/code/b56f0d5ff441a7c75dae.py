"""
Strategy:
1. Use GF(256) arithmetic to create a structured initial coloring for n=40.
   Vertices 0..39 are mapped to distinct non-zero elements of GF(256).
   A triple (a,b,c) is colored Red (0) if a+b+c is a non-zero square in GF(256), else Blue (1).
   This algebraic structure tends to distribute colors more evenly than simple modular arithmetic,
   potentially reducing the density of monochromatic K4s and K5s.
2. Use a fast local search (simulated annealing style) to minimize the number of violations
   (Red K4s + Blue K5s).
3. If violations are zero, we have a valid coloring. We try to maximize n.
   Since we start with n=40, if we can't get 0 violations, we try smaller n (39, 38...) 
   by restricting the search to subsets or restarting.
   However, the task asks to maximize n. The evaluator checks validity.
   If we find a valid coloring for n=40, score is high.
   If not, we return the best n for which we found a valid coloring, or the seed n=9 if none.
   
   Actually, the seed returns n=9. If we can't beat 9, we return 9.
   But finding a valid coloring for n > 9 is the goal.
   
   The local search will be run for a fixed time budget (e.g., 100s).
   We will try to find a valid coloring for n=40. If successful, return n=40.
   If not, try n=39, etc.
   
   To be safe and fast, we will primarily focus on n=40. If it fails, we fall back to a smaller n 
   where we might have found a solution in a previous iteration or just return the seed.
   
   Given the 120s limit, we allocate ~110s for the search.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # --- GF(256) Setup ---
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    poly = 0x11D
    
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return p

    def gf_inv(a):
        if a == 0:
            return 0
        return gf_pow(a, 254)

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mul(result, a)
            a = gf_mul(a, a)
            n >>= 1
        return result

    # Precompute squares in GF(256)
    squares = set()
    for i in range(1, 256):
        squares.add(gf_mul(i, i))
    
    # Map vertices 0..39 to distinct non-zero field elements
    # We can just use 1..40 as their field representatives
    # But to make it more "random" in field space, we can use a permutation.
    # Let's stick to 1..40 for simplicity and determinism.
    
    # --- Initial Coloring for n=40 ---
    n = 40
    vertices = list(range(1, n + 1))  # Field elements 1..40
    
    def get_color(a, b, c):
        s = (a + b + c) % 255  # Wait, addition in GF(256) is XOR!
        # Correction: Addition in GF(2^k) is bitwise XOR.
        s = a ^ b ^ c
        if s == 0:
            # 0 is not a square (usually defined as such, or handled separately)
            # Let's color 0 as Blue (1)
            return 1
        return 0 if s in squares else 1

    # Generate all triples
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
                
    # Map vertex index to field element
    v_to_f = {i: vertices[i] for i in range(n)}
    
    # Initial colors
    colors = [get_color(v_to_f[i], v_to_f[j], v_to_f[k]) for i, j, k in triples]
    
    # Precompute triple indices for vertices to speed up violation counting
    # For each vertex v, list of triples containing v
    v_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        v_triples[i].append(idx)
        v_triples[j].append(idx)
        v_triples[k].append(idx)
        
    # Precompute pairs for K4 and K5 counting?
    # A Red K4 is 4 vertices with all 4 triples Red.
    # A Blue K5 is 5 vertices with all 10 triples Blue.
    
    # Efficient violation counting is tricky. 
    # Let's use a heuristic: count monochromatic K4s and K5s.
    
    def count_violations(colors):
        red_k4 = 0
        blue_k5 = 0
        
        # Count Red K4
        # For each pair (i,j), count common neighbors k such that (i,j,k) is Red?
        # No, K4 requires all 4 triples.
        
        # Iterate over all 4-combinations? C(40,4) = 91390.
        # For each 4-set, check 4 triples.
        # Iterate over all 5-combinations? C(40,5) = 658008.
        # For each 5-set, check 10 triples.
        
        # This might be too slow for 100s of iterations.
        # Let's optimize: 
        # Precompute for each pair (i,j) the list of k > j such that (i,j,k) is Red.
        # Then for each triple (i,j,k) Red, check if (i,j,l), (i,k,l), (j,k,l) are Red for l > k?
        
        # Simpler approach for local search: 
        # Only evaluate delta when flipping a single triple.
        # Flipping triple (a,b,c) affects:
        # - K4s containing (a,b,c): The other vertex d must form a K4 with a,b,c.
        #   So for each d not in {a,b,c}, if (a,b,d), (a,c,d), (b,c,d) are all same color as new (a,b,c), 
        #   then a K4 is formed/broken.
        # - K5s containing (a,b,c): The other 2 vertices d,e must form a K5 with a,b,c.
        #   This is more complex.
        
        # Given the complexity, let's use a simpler metric: 
        # Sum of squares of monochromatic 3-sets? No, that's not K4/K5.
        
        # Let's stick to the full count but optimize it.
        # We will maintain the count incrementally.
        
        # State:
        # red_k4_count
        # blue_k5_count
        # For incremental update:
        # When flipping triple T=(a,b,c) from color C_old to C_new:
        
        # Impact on Red K4s:
        # If C_new == 0 (Red):
        #   New K4s formed: For each d, if (a,b,d), (a,c,d), (b,c,d) are all Red, then {a,b,c,d} is a new Red K4.
        #   Old K4s broken: For each d, if (a,b,d), (a,c,d), (b,c,d) are all Red, then {a,b,c,d} was a Red K4 (since T was Red? No, T was Blue).
        #   Wait. If T was Blue, it couldn't be part of a Red K4. So no Red K4s broken.
        #   If T was Red, it was part of Red K4s. Flipping to Blue breaks them.
        
        # Let's define:
        # count_red_k4
        # count_blue_k5
        
        # Delta for flipping (a,b,c) from old to new:
        
        # 1. Check K4s involving (a,b,c).
        #    A K4 is {a,b,c,d}. The triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d).
        #    If new color is Red (0):
        #       If (a,b,d), (a,c,d), (b,c,d) are all Red, then this K4 becomes Red.
        #       If old color was Blue, this K4 was NOT Red. So we ADD 1 to red_k4.
        #       If old color was Red, this K4 WAS Red. So we SUBTRACT 1 from red_k4.
        #       (Actually, if old was Red, it was already counted. If we flip to Blue, we lose it.)
        
        #    If new color is Blue (1):
        #       Red K4s: If old was Red, we lose any K4 where the other 3 were Red.
        #       Blue K4s: Not directly penalized, but might contribute to Blue K5?
        
        # 2. Check K5s involving (a,b,c).
        #    A K5 is {a,b,c,d,e}. Triples: (a,b,c) + 3 pairs of (d,e) with a,b,c + 3 pairs of (d,e) with d,e + (d,e,?)
        #    Total 10 triples.
        #    If new color is Blue (1):
        #       We need the other 9 triples to be Blue.
        #       For each pair (d,e) not in {a,b,c}, check if all triples in {a,b,c,d,e} except (a,b,c) are Blue.
        #       This is expensive.
        
        # Alternative: Use a simpler penalty function that correlates with K4/K5 violations.
        # Penalty = Sum over all 4-sets of (1 if all 4 triples same color else 0)
        #          + Sum over all 5-sets of (1 if all 10 triples same color else 0)
        
        # This is the exact objective.
        # Incremental update for 4-sets is easy.
        # Incremental update for 5-sets is harder.
        
        # Let's approximate the 5-set penalty by counting "almost monochromatic" 5-sets?
        # Or just ignore 5-sets in the local search and check at the end?
        # Risk: Local search might minimize K4s but create K5s.
        
        # Let's try a hybrid:
        # Local search minimizes Red K4s + Blue K4s (as a proxy for structure).
        # Then we check for K5s.
        
        # Actually, let's just do the full count every N steps or use a fast approximate metric.
        # Given 120s, we can do maybe 1000-5000 iterations of full count if optimized in C?
        # Python is slow.
        
        # Let's use NumPy for vectorized violation counting.
        # Represent colors as a 3D array? No, sparse.
        
        # Let's try a different strategy:
        # 1. Generate random colorings for n=40.
        # 2. Use a fast local search that ONLY checks K4s (4-cliques).
        # 3. If K4 count is 0, check K5s.
        # 4. If K5 count is 0, success.
        
        # This is a gamble.
        
        # Let's implement the K4-only local search.
        
        # Precompute for each triple (a,b,c) the list of d such that (a,b,d), (a,c,d), (b,c,d) exist.
        # Actually, for a fixed triple (a,b,c), the potential 4th vertices d are all other vertices.
        # Condition for {a,b,c,d} to be a mono K4:
        # All 4 triples have same color.
        
        # Let's maintain:
        # red_k4_count
        # blue_k4_count
        
        # When flipping (a,b,c) from old to new:
        # For each d in range(n) where d != a,b,c:
        #   c_abd = colors[idx(a,b,d)]
        #   c_acd = colors[idx(a,c,d)]
        #   c_bcd = colors[idx(b,c,d)]
        
        #   If new == 0 (Red):
        #      If c_abd==0 and c_acd==0 and c_bcd==0:
        #         If old == 0: red_k4_count -= 1 (was red, now not)
        #         Else: red_k4_count += 1 (was not red, now is)
        #         # Note: If old was 1, it wasn't a red K4. If new is 0 and others are 0, it becomes one.
        #      # Blue K4 impact?
        #      # If new is Red, it cannot be part of a Blue K4.
        #      # If old was Blue, and others were Blue, we lost a Blue K4.
        #      If c_abd==1 and c_acd==1 and c_bcd==1:
        #         If old == 1: blue_k4_count -= 1
        #         # If old was 0, it wasn't a blue K4.
        
        #   If new == 1 (Blue):
        #      If c_abd==1 and c_acd==1 and c_bcd==1:
        #         If old == 1: blue_k4_count -= 1
        #         Else: blue_k4_count += 1
        #      If c_abd==0 and c_acd==0 and c_bcd==0:
        #         If old == 0: red_k4_count -= 1
        
        # This allows O(n) update per flip.
        
        # We need a mapping from (a,b,c) to index.
        # idx_map[(a,b,c)] = index
        
        idx_map = {}
        for idx, (i, j, k) in enumerate(triples):
            idx_map[(i, j, k)] = idx
            # Ensure sorted
            if i > j: i, j = j, i
            # etc. The triples list is generated in sorted order.
            
        # Wait, the triples list is generated with i<j<k.
        # So idx_map key is (i,j,k) with i<j<k.
        
        # Initialize counts
        red_k4_count = 0
        blue_k4_count = 0
        
        # We need to calculate initial counts.
        # Iterate all 4-combinations? C(40,4) = 91390.
        # For each 4-set {a,b,c,d}, check 4 triples.
        
        from itertools import combinations
        
        def is_mono_4(a,b,c,d, colors):
            c1 = colors[idx_map[(a,b,c)]]
            c2 = colors[idx_map[(a,b,d)]]
            c3 = colors[idx_map[(a,c,d)]]
            c4 = colors[idx_map[(b,c,d)]]
            return c1 == c2 == c3 == c4
            
        for quad in combinations(range(n), 4):
            if is_mono_4(*quad, colors):
                c = colors[idx_map[(quad[0], quad[1], quad[2])]]
                if c == 0:
                    red_k4_count += 1
                else:
                    blue_k4_count += 1
                    
        # Local Search
        best_score = red_k4_count + blue_k4_count
        best_colors = colors[:]
        
        # Temperature
        temp = 1.0
        cooling = 0.999
        
        steps = 0
        while time.time() < deadline:
            # Pick random triple
            idx = random.randint(0, len(triples) - 1)
            a, b, c = triples[idx]
            old_color = colors[idx]
            new_color = 1 - old_color
            
            # Calculate delta
            delta_red = 0
            delta_blue = 0
            
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                
                # Get indices for (a,b,d), (a,c,d), (b,c,d)
                # Ensure sorted order for key
                ab_d = tuple(sorted((a, b, d)))
                ac_d = tuple(sorted((a, c, d)))
                bc_d = tuple(sorted((b, c, d)))
                
                c_abd = colors[idx_map[ab_d]]
                c_acd = colors[idx_map[ac_d]]
                c_bcd = colors[idx_map[bc_d]]
                
                if new_color == 0: # New is Red
                    if c_abd == 0 and c_acd == 0 and c_bcd == 0:
                        if old_color == 0:
                            delta_red -= 1
                        else:
                            delta_red += 1
                    if c_abd == 1 and c_acd == 1 and c_bcd == 1:
                        if old_color == 1:
                            delta_blue -= 1
                        # If old was 0, no blue K4 was formed/broken by this flip in a way that matters for BLUE count?
                        # Wait. If old was 0, the K4 was not Blue. If new is 0, it's still not Blue.
                        # But if old was 1, and others were 1, it WAS a Blue K4. Now it's not. So -1.
                        
                else: # New is Blue
                    if c_abd == 1 and c_acd == 1 and c_bcd == 1:
                        if old_color == 1:
                            delta_blue -= 1
                        else:
                            delta_blue += 1
                    if c_abd == 0 and c_acd == 0 and c_bcd == 0:
                        if old_color == 0:
                            delta_red -= 1
                            
            delta = delta_red + delta_blue
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / temp):
                colors[idx] = new_color
                red_k4_count += delta_red
                blue_k4_count += delta_blue
                
                # Check if we found a solution with 0 K4s
                if red_k4_count == 0 and blue_k4_count == 0:
                    # Check K5s
                    # This is expensive, so do it rarely or if we have time
                    if time.time() < deadline - 10:
                        if not has_blue_k5(colors, n, idx_map, triples):
                            # Valid coloring!
                            return {'n': n, 'colors': colors}
            
            temp *= cooling
            steps += 1
            
            # Periodic check for K5 if K4s are low
            if steps % 1000 == 0 and red_k4_count < 5:
                 if not has_blue_k5(colors, n, idx_map, triples):
                     return {'n': n, 'colors': colors}

        # If no valid coloring found for n=40, try n=39?
        # Given time constraints, we might just return the best n we can validate quickly.
        # Or fall back to seed.
        
        # Let's try to validate the best_colors found for n=40
        # If it has 0 violations, return it.
        # The loop above checks this.
        
        # If we exit the loop, we likely didn't find a valid coloring.
        # We can try a smaller n by restricting the search.
        # But re-initializing for n=39 takes time.
        
        # Let's just return the seed result if we fail, to be safe.
        # Or return the current colors if they happen to be valid? 
        # We check validity in the loop.
        
        # Fallback: Return seed
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}

    def has_blue_k5(colors, n, idx_map, triples):
        # Check for Blue K5
        # Iterate all 5-combinations
        # C(40,5) = 658008
        # For each, check 10 triples.
        # This is ~6.5M operations. Might be too slow to do frequently.
        # We only call this when K4 count is low.
        
        for quint in combinations(range(n), 5):
            # Get all 10 triples
            # Check if all are Blue (1)
            is_blue = True
            for i in range(5):
                for j in range(i + 1, 5):
                    for k in range(j + 1, 5):
                        t = (quint[i], quint[j], quint[k])
                        if colors[idx_map[t]] != 1:
                            is_blue = False
                            break
                    if not is_blue: break
                if not is_blue: break
            if is_blue:
                return True
        return False

# Helper function defined outside or inside? Inside is fine.
# But the function definition inside construct is slow to re-define? No, it's fine.
# However, `has_blue_k5` uses `combinations` and `idx_map` from closure.

# One issue: The `has_blue_k5` function is defined AFTER the loop that uses it?
# No, Python resolves names at runtime. But it's better to define it before.
# Let's restructure slightly to ensure definitions are available.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # --- GF(256) Setup ---
    poly = 0x11D
    
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return p

    squares = set()
    for i in range(1, 256):
        squares.add(gf_mul(i, i))
    
    n = 40
    vertices = list(range(1, n + 1))
    
    def get_color(a, b, c):
        s = a ^ b ^ c
        if s == 0:
            return 1
        return 0 if s in squares else 1

    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
                
    v_to_f = {i: vertices[i] for i in range(n)}
    colors = [get_color(v_to_f[i], v_to_f[j], v_to_f[k]) for i, j, k in triples]
    
    idx_map = {}
    for idx, (i, j, k) in enumerate(triples):
        idx_map[(i, j, k)] = idx
        
    from itertools import combinations
    
    def is_mono_4(a,b,c,d, colors):
        c1 = colors[idx_map[(a,b,c)]]
        c2 = colors[idx_map[(a,b,d)]]
        c3 = colors[idx_map[(a,c,d)]]
        c4 = colors[idx_map[(b,c,d)]]
        return c1 == c2 == c3 == c4
        
    red_k4_count = 0
    blue_k4_count = 0
    
    for quad in combinations(range(n), 4):
        if is_mono_4(*quad, colors):
            c = colors[idx_map[(quad[0], quad[1], quad[2])]]
            if c == 0:
                red_k4_count += 1
            else:
                blue_k4_count += 1
                
    def has_blue_k5(colors, n, idx_map):
        for quint in combinations(range(n), 5):
            is_blue = True
            # Unroll for speed?
            # 10 triples
            # (0,1,2), (0,1,3), (0,1,4), (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4), (2,3,4)
            # Using indices in quint
            q = quint
            triples_to_check = [
                (q[0], q[1], q[2]), (q[0], q[1], q[3]), (q[0], q[1], q[4]),
                (q[0], q[2], q[3]), (q[0], q[2], q[4]), (q[0], q[3], q[4]),
                (q[1], q[2], q[3]), (q[1], q[2], q[4]), (q[1], q[3], q[4]),
                (q[2], q[3], q[4])
            ]
            for t in triples_to_check:
                if colors[idx_map[t]] != 1:
                    is_blue = False
                    break
            if is_blue:
                return True
        return False

    temp = 1.0
    cooling = 0.999
    steps = 0
    
    while time.time() < deadline:
        idx = random.randint(0, len(triples) - 1)
        a, b, c = triples[idx]
        old_color = colors[idx]
        new_color = 1 - old_color
        
        delta_red = 0
        delta_blue = 0
        
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            
            ab_d = tuple(sorted((a, b, d)))
            ac_d = tuple(sorted((a, c, d)))
            bc_d = tuple(sorted((b, c, d)))
            
            c_abd = colors[idx_map[ab_d]]
            c_acd = colors[idx_map[ac_d]]
            c_bcd = colors[idx_map[bc_d]]
            
            if new_color == 0:
                if c_abd == 0 and c_acd == 0 and c_bcd == 0:
                    if old_color == 0:
                        delta_red -= 1
                    else:
                        delta_red += 1
                if c_abd == 1 and c_acd == 1 and c_bcd == 1:
                    if old_color == 1:
                        delta_blue -= 1
            else:
                if c_abd == 1 and c_acd == 1 and c_bcd == 1:
                    if old_color == 1:
                        delta_blue -= 1
                    else:
                        delta_blue += 1
                if c_abd == 0 and c_acd == 0 and c_bcd == 0:
                    if old_color == 0:
                        delta_red -= 1
                        
        delta = delta_red + delta_blue
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            colors[idx] = new_color
            red_k4_count += delta_red
            blue_k4_count += delta_blue
            
            if red_k4_count == 0 and blue_k4_count == 0:
                if time.time() < deadline - 5:
                    if not has_blue_k5(colors, n, idx_map):
                        return {'n': n, 'colors': colors}
            
        temp *= cooling
        steps += 1
        
        if steps % 2000 == 0 and red_k4_count < 10:
            if time.time() < deadline - 5:
                if not has_blue_k5(colors, n, idx_map):
                    return {'n': n, 'colors': colors}

    # Fallback
    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}

# Re-define construct to avoid nested function scope issues if any, though Python handles it.
# The code above is a single function.
