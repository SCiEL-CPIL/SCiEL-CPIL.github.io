import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    def get_violations(colors, n, idx_map):
        """
        Count Red K4s and Blue K5s.
        Returns (red_k4_count, blue_k5_count, total_violations)
        Uses vectorized numpy operations where possible, but for n<=40, 
        explicit loops with precomputed indices are often faster in Python 
        if the number of combinations is moderate. 
        C(40,4) = 91390, C(40,5) = 658008.
        For n=32: C(32,4)=35960, C(32,5)=201376.
        """
        red_k4 = 0
        blue_k5 = 0
        
        # Check Red K4s
        # A K4 is a set of 4 vertices {a,b,c,d} such that all 4 triples are Red (0).
        # We can precompute the list of 4-combinations and their corresponding triple indices.
        # To be fast, we'll just iterate.
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            if (colors[idx_map[(a,b,c)]] == 0 and 
                colors[idx_map[(a,b,d)]] == 0 and 
                colors[idx_map[(a,c,d)]] == 0 and 
                colors[idx_map[(b,c,d)]] == 0):
                red_k4 += 1
                
        # Check Blue K5s
        # A K5 is a set of 5 vertices {a,b,c,d,e} such that all 10 triples are Blue (1).
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            # 10 triples
            # (a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)
            if (colors[idx_map[(a,b,c)]] == 1 and 
                colors[idx_map[(a,b,d)]] == 1 and 
                colors[idx_map[(a,b,e)]] == 1 and 
                colors[idx_map[(a,c,d)]] == 1 and 
                colors[idx_map[(a,c,e)]] == 1 and 
                colors[idx_map[(a,d,e)]] == 1 and 
                colors[idx_map[(b,c,d)]] == 1 and 
                colors[idx_map[(b,c,e)]] == 1 and 
                colors[idx_map[(b,d,e)]] == 1 and 
                colors[idx_map[(c,d,e)]] == 1):
                blue_k5 += 1
                
        return red_k4, blue_k5, red_k4 + blue_k5

    def generate_initial_coloring(n):
        """
        Generate a structured coloring using GF(32) multiplicative properties.
        Vertices 0..31 map to non-zero elements of GF(32).
        GF(32) has 31 non-zero elements, which form a cyclic group of order 31.
        We can map vertex i to element alpha^i (i=0..30) and handle n=32 carefully.
        Actually, let's just use a simpler heuristic: 
        Color (a,b,c) based on (a+b+c) mod 31 being in a specific set?
        Or better: Use the fact that R(4,5;3) is related to finite geometries.
        
        Let's try a random start but with a bias towards balance, 
        or a specific algebraic start.
        
        Algebraic Start:
        Map vertices 0..30 to GF(32)*. Vertex 31 is a special "infinity" point?
        For n=32, this is tricky.
        
        Let's stick to a high-quality random start or a modular start.
        Modular start: color = (a*b + b*c + c*a) % 2 ?
        Let's try: color = 0 if (a+b+c) % 31 in squares_of_GF31 else 1.
        """
        if n == 0:
            return []
        
        # Precompute squares in GF(31)
        # GF(31) is prime field. Squares are quadratic residues.
        squares_31 = set()
        for i in range(1, 31):
            squares_31.add((i*i) % 31)
        
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Use modular arithmetic with 31
                    # If n > 31, we wrap around or use a different modulus?
                    # For n=32, vertex 31 maps to 0?
                    # Let's map vertex v to v % 31.
                    a = i % 31
                    b = j % 31
                    c = k % 31
                    
                    # Heuristic: Check if sum is a quadratic residue
                    s = (a + b + c) % 31
                    if s in squares_31:
                        colors.append(0)
                    else:
                        colors.append(1)
        return colors

    def local_search(colors, n, idx_map, max_time=2.0):
        """
        Fast local search.
        We use a greedy hill-climbing approach with random restarts if stuck.
        To be fast, we evaluate the full violation count periodically, 
        but for moves, we use a delta evaluation if possible.
        Given the complexity of K5 deltas, we might just do full counts 
        if the number of steps is low, or use a cheaper proxy.
        
        Actually, let's use a "simulated annealing" style but with 
        a very fast acceptance criterion based on a subset of violations 
        or just random flips with full check every K steps.
        
        Better: 
        1. Calculate initial violations.
        2. Loop:
           - Pick random triple to flip.
           - Calculate new violations (expensive).
           - If better, accept.
        
        To speed this up, we can precompute for each triple the list of 
        K4s and K5s it participates in.
        
        Precomputation:
        For each triple T, list of 4-sets containing T.
        For each triple T, list of 5-sets containing T.
        
        Let's do this precomputation once for a given n.
        """
        # Precompute incidence lists
        # k4_incidence[T] = list of 4-tuples (as frozensets or sorted tuples) that contain T
        # k5_incidence[T] = list of 5-tuples that contain T
        
        # This might be memory heavy? 
        # C(40,3) = 9880 triples.
        # Each triple is in C(37,1) = 37 K4s? No.
        # A triple {a,b,c} is in K4s {a,b,c,d} for d not in {a,b,c}. So n-3 K4s.
        # A triple {a,b,c} is in K5s {a,b,c,d,e} for d,e not in {a,b,c}. So C(n-3, 2) K5s.
        # For n=32: 29 K4s, C(29,2)=406 K5s per triple.
        # Total K4 incidences: 9880 * 29 ~ 286k.
        # Total K5 incidences: 9880 * 406 ~ 4M. This is large but manageable in memory if stored as lists of ints.
        
        # However, building these lists takes time.
        # Let's try a simpler approach: 
        # Just do random flips and check full violation count every 10 flips?
        # No, that's slow.
        
        # Let's implement the delta calculation for K4s and K5s.
        
        # For a flip of triple T=(a,b,c) from color c_old to c_new:
        # Delta in Red K4s:
        #   For each d not in T:
        #     Let K = {a,b,c,d}.
        #     If c_new == 0 (Red):
        #        If other 3 triples in K are Red:
        #           If c_old == 0: -1 (was Red, now not)
        #           Else: +1 (was not Red, now is)
        #     If c_new == 1 (Blue):
        #        If other 3 triples in K are Blue:
        #           If c_old == 1: -1 (was Blue K4, now not... wait, we only count RED K4s)
        #           # We don't care about Blue K4s for the objective, only Red K4s and Blue K5s.
        #           # But flipping a triple from Blue to Red might break a Blue K4, which doesn't matter.
        #           # Flipping from Red to Blue might break a Red K4.
        
        # So, we only need to track:
        # 1. Red K4s: All 4 triples Red.
        # 2. Blue K5s: All 10 triples Blue.
        
        # Delta for Red K4s when flipping T:
        # Only K4s containing T are affected.
        # For each d:
        #   Check if (a,b,d), (a,c,d), (b,c,d) are all Red.
        #   If yes:
        #      If c_old == 0: delta_red -= 1
        #      Else: delta_red += 1
        #   (If c_new is 1, we definitely break any Red K4 that was there, so if c_old==0 and others Red, -1. If c_old==1, no Red K4 was there.)
        #   Wait, if c_new is 1, it can never form a Red K4.
        #   If c_new is 0, it can form a Red K4 if others are Red.
        
        # So:
        # If c_new == 0:
        #   For each d: if others are Red:
        #      if c_old == 0: delta_red -= 1
        #      else: delta_red += 1
        # If c_new == 1:
        #   For each d: if others are Red:
        #      if c_old == 0: delta_red -= 1
        #      # else: no change (was not Red K4, still not)
        #      # Note: If c_old was 1, it wasn't a Red K4.
        
        # Similarly for Blue K5s:
        # Only K5s containing T are affected.
        # For each pair (d,e) not in T:
        #   Let K = {a,b,c,d,e}.
        #   Check if the other 9 triples in K are Blue.
        #   If yes:
        #      If c_new == 1:
        #         if c_old == 1: delta_blue -= 1
        #         else: delta_blue += 1
        #      If c_new == 0:
        #         if c_old == 1: delta_blue -= 1
        #         # else: no change
        
        # This is O(n) for K4s and O(n^2) for K5s per flip.
        # For n=32, n^2 = 1024. 
        # 1024 operations per flip. 
        # If we do 1000 flips, that's 1M operations. Very fast.
        
        # We need to quickly check if "other 9 triples are Blue".
        # We can precompute for each K5 the count of Blue triples?
        # Or just check the 9 specific indices.
        
        # Let's precompute for each triple T, the list of (d, e, indices_of_other_9_triples)
        # This is static.
        
        # Precomputation:
        # triple_to_k4s[T] = [d for d in range(n) if d not in T]
        # triple_to_k5s[T] = [(d, e, [idx1..idx9]) for d,e not in T, d<e]
        
        # To make it faster, we can precompute the "other indices" for each T and each potential K4/K5.
        
        # Let's build these structures.
        
        # Map triple to index
        # We already have idx_map.
        
        # For each triple index ti, with vertices (a,b,c):
        # k4_neighbors[ti] = []
        # for d in range(n):
        #   if d in (a,b,c): continue
        #   # Indices of the other 3 triples
        #   # (a,b,d), (a,c,d), (b,c,d)
        #   # Ensure sorted order for idx_map
        #   t1 = tuple(sorted((a,b,d)))
        #   t2 = tuple(sorted((a,c,d)))
        #   t3 = tuple(sorted((b,c,d)))
        #   k4_neighbors[ti].append((idx_map[t1], idx_map[t2], idx_map[t3]))
        
        # k5_neighbors[ti] = []
        # for d in range(n):
        #   if d in (a,b,c): continue
        #   for e in range(n):
        #     if e in (a,b,c) or e == d: continue
        #     if d > e: continue # avoid duplicates
        #     # Indices of other 9 triples
        #     # Vertices: a,b,c,d,e
        #     # Triples:
        #     # (a,b,c) is T.
        #     # (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)
        #     # 9 triples.
        #     t1 = tuple(sorted((a,b,d)))
        #     t2 = tuple(sorted((a,b,e)))
        #     t3 = tuple(sorted((a,c,d)))
        #     t4 = tuple(sorted((a,c,e)))
        #     t5 = tuple(sorted((a,d,e)))
        #     t6 = tuple(sorted((b,c,d)))
        #     t7 = tuple(sorted((b,c,e)))
        #     t8 = tuple(sorted((b,d,e)))
        #     t9 = tuple(sorted((c,d,e)))
        #     k5_neighbors[ti].append((idx_map[t1], idx_map[t2], idx_map[t3], idx_map[t4], idx_map[t5], idx_map[t6], idx_map[t7], idx_map[t8], idx_map[t9]))
        
        # This precomputation is O(n^3) or O(n^4)? 
        # Number of triples is n^3/6.
        # For each triple, we loop d (n) and e (n). So n^2 per triple.
        # Total n^5/6. For n=32, 32^5/6 ~ 5M. This is too slow in Python.
        
        # Alternative: 
        # Don't precompute. Just calculate on the fly.
        # For a flip of T=(a,b,c):
        # Loop d for K4: n iterations. Each does 3 dict lookups. Fast.
        # Loop d,e for K5: n^2 iterations. Each does 9 dict lookups. 
        # n=32 -> 1024 * 9 = 9216 lookups. 
        # 1000 flips -> 9M lookups. 
        # Python can do ~10M-50M simple ops/sec. 
        # So 1000 flips might take 1-2 seconds. 
        # We have 115s. We can do ~50k-100k flips.
        
        # Let's implement this on-the-fly delta calculation.
        
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    triples.append((i, j, k))
                    
        # Rebuild idx_map for this n
        idx_map_local = {}
        for idx, (i, j, k) in enumerate(triples):
            idx_map_local[(i, j, k)] = idx
            
        # Initial violation count
        red_k4, blue_k5, total = get_violations(colors, n, idx_map_local)
        
        best_total = total
        best_colors = colors[:]
        
        # If already 0, we are done with this n
        if total == 0:
            return colors, 0, 0
        
        # Local search
        # We will use a greedy approach with some randomness to escape local minima.
        # Accept if delta < 0.
        # Occasionally accept delta > 0 to escape.
        
        t_start = time.time()
        max_time_local = max_time
        temp = 1.0
        cooling = 0.9999
        
        steps = 0
        while time.time() - t_start < max_time_local:
            # Pick random triple
            ti = random.randint(0, len(triples) - 1)
            a, b, c = triples[ti]
            old_color = colors[ti]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # K4s
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # Check other 3 triples
                t1 = tuple(sorted((a, b, d)))
                t2 = tuple(sorted((a, c, d)))
                t3 = tuple(sorted((b, c, d)))
                
                c1 = colors[idx_map_local[t1]]
                c2 = colors[idx_map_local[t2]]
                c3 = colors[idx_map_local[t3]]
                
                if c1 == 0 and c2 == 0 and c3 == 0:
                    if new_color == 0:
                        if old_color == 0:
                            delta_red -= 1
                        else:
                            delta_red += 1
                    else: # new_color == 1
                        if old_color == 0:
                            delta_red -= 1
                        # else: no change (was not red K4, still not)
                
                # Note: We only count RED K4s. So we only care if the K4 becomes/breaks RED.
                # If new_color is 1, it can't be a Red K4.
                # If old_color was 0 and others 0, it WAS a Red K4. So -1.
                # If old_color was 1, it WAS NOT a Red K4. So 0.
                # If new_color is 0 and others 0, it IS a Red K4.
                # If old_color was 1, it WAS NOT. So +1.
                # If old_color was 0, it WAS. So -1 (net 0 change? No, we are flipping. 
                # Wait. 
                # State 1: Old=0, Others=0 -> Red K4 exists. Count=1.
                # State 2: New=1, Others=0 -> Red K4 does not exist. Count=0.
                # Delta = -1.
                
                # State 1: Old=1, Others=0 -> Red K4 does not exist. Count=0.
                # State 2: New=0, Others=0 -> Red K4 exists. Count=1.
                # Delta = +1.
                
                # State 1: Old=0, Others!=0 -> No Red K4.
                # State 2: New=1, Others!=0 -> No Red K4.
                # Delta = 0.
                
                # So the logic above is correct.
            
            # K5s
            # Loop over pairs (d,e)
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                for e in range(d + 1, n):
                    if e == a or e == b or e == c:
                        continue
                    # Check other 9 triples
                    # Vertices: a,b,c,d,e
                    # Triples:
                    t1 = tuple(sorted((a, b, d)))
                    t2 = tuple(sorted((a, b, e)))
                    t3 = tuple(sorted((a, c, d)))
                    t4 = tuple(sorted((a, c, e)))
                    t5 = tuple(sorted((a, d, e)))
                    t6 = tuple(sorted((b, c, d)))
                    t7 = tuple(sorted((b, c, e)))
                    t8 = tuple(sorted((b, d, e)))
                    t9 = tuple(sorted((c, d, e)))
                    
                    c1 = colors[idx_map_local[t1]]
                    c2 = colors[idx_map_local[t2]]
                    c3 = colors[idx_map_local[t3]]
                    c4 = colors[idx_map_local[t4]]
                    c5 = colors[idx_map_local[t5]]
                    c6 = colors[idx_map_local[t6]]
                    c7 = colors[idx_map_local[t7]]
                    c8 = colors[idx_map_local[t8]]
                    c9 = colors[idx_map_local[t9]]
                    
                    if (c1 == 1 and c2 == 1 and c3 == 1 and c4 == 1 and c5 == 1 and 
                        c6 == 1 and c7 == 1 and c8 == 1 and c9 == 1):
                        
                        if new_color == 1:
                            if old_color == 1:
                                delta_blue -= 1
                            else:
                                delta_blue += 1
                        else: # new_color == 0
                            if old_color == 1:
                                delta_blue -= 1
                            # else: no change
            
            delta = delta_red + delta_blue
            
            # Acceptance
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / temp)):
                colors[ti] = new_color
                red_k4 += delta_red
                blue_k5 += delta_blue
                total = red_k4 + blue_k5
                
                if total < best_total:
                    best_total = total
                    best_colors = colors[:]
                    
                if total == 0:
                    return colors, 0, 0
            
            temp *= cooling
            steps += 1
            
            # If stuck, do a random restart? Or just continue.
            # Given the time, we'll just continue.
            
        return best_colors, best_total, 0

    # Main Loop: Try n from 32 down to 1
    # We want to maximize n.
    # Start with n=32.
    # If we find a valid coloring, return it.
    # If not, try n=31, etc.
    
    # Actually, the task says "Maximize n".
    # The published construction is n=34.
    # Our seed is n=9.
    # We should try to get as close to 34 as possible.
    # Let's try n=34, 33, 32...
    
    # However, finding a valid coloring for n=34 is hard.
    # Let's start with n=32 where the algebraic structure is strong.
    # If we succeed at 32, can we extend to 33?
    # Or just return 32?
    # Score is n. So 32 is much better than 9.
    
    # Let's try n=32 first.
    n = 32
    colors = generate_initial_coloring(n)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    idx_map = {}
    for idx, (i, j, k) in enumerate(triples):
        idx_map[(i, j, k)] = idx
        
    # Run local search
    colors, violations, _ = local_search(colors, n, idx_map, max_time=100)
    
    if violations == 0:
        return {'n': n, 'colors': colors}
    
    # If n=32 fails, try n=31?
    # Or try a different start for n=32?
    # Let's try a few random restarts for n=32 if we have time.
    
    # Actually, let's just try n=32 with a few different random seeds.
    # The initial coloring is deterministic. Let's add random noise.
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
            
        # Random start
        colors = [random.randint(0, 1) for _ in range(len(triples))]
        
        # Run local search with less time
        max_t = min(15, deadline - time.time() - 5)
        if max_t <= 0:
            break
            
        colors, violations, _ = local_search(colors, n, idx_map, max_time=max_t)
        
        if violations == 0:
            return {'n': n, 'colors': colors}
            
    # If n=32 fails, try n=31
    n = 31
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    idx_map = {}
    for idx, (i, j, k) in enumerate(triples):
        idx_map[(i, j, k)] = idx
        
    colors = generate_initial_coloring(n)
    colors, violations, _ = local_search(colors, n, idx_map, max_time=20)
    
    if violations == 0:
        return {'n': n, 'colors': colors}
        
    # Fallback to seed
    n_seed = 9
    colors_seed = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]
    return {'n': n_seed, 'colors': colors_seed}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
