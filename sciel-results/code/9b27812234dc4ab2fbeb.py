"""
Strategy:
1. Use a deterministic, structured initial coloring based on GF(31) arithmetic.
   Vertices 0..30 are mapped to elements of GF(31).
   A triple (a,b,c) is colored Red (0) if a+b+c is a quadratic residue mod 31, else Blue (1).
   This algebraic structure tends to distribute colors evenly.
2. Use a fast local search (simulated annealing style) to minimize the number of violations
   (Red K4s + Blue K5s).
3. If violations are zero, we have a valid coloring. We try to maximize n.
   Since we start with n=31, if we can get 0 violations, we return n=31.
   If not, we try smaller n (30, 29...) by restarting the search with a smaller subset.
   
   The local search will be run for a fixed time budget (e.g., 100s).
   We will try to find a valid coloring for n=31. If successful, return n=31.
   If not, try n=30, etc.
   
   To be safe and fast, we will primarily focus on n=31. If it fails, we fall back to a smaller n 
   where we might have found a solution in a previous iteration or just return the seed.
   
   Given the 120s limit, we allocate ~110s for the search.
"""

import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # --- GF(31) Setup ---
    # 31 is prime. Quadratic residues mod 31:
    residues = set()
    for i in range(1, 31):
        residues.add((i * i) % 31)
    
    def get_color(a, b, c):
        s = (a + b + c) % 31
        if s == 0:
            return 1  # 0 is not a non-zero square, so Blue
        return 0 if s in residues else 1

    # We will try n from 31 down to 9
    for n in range(31, 8, -1):
        if time.time() > deadline - 10:
            break
            
        # Initial coloring for this n
        vertices = list(range(n))
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    triples.append((i, j, k))
                    
        colors = [get_color(vertices[i], vertices[j], vertices[k]) for i, j, k in triples]
        
        idx_map = {}
        for idx, (i, j, k) in enumerate(triples):
            idx_map[(i, j, k)] = idx
            
        # Precompute for each triple (a,b,c) the list of other vertices d
        # and the indices of the other triples in the K4.
        # Actually, we just iterate d in the local search.
        
        # Initial violation count
        def count_violations(colors):
            red_k4 = 0
            blue_k5 = 0
            
            # Count Red K4
            for quad in combinations(range(n), 4):
                a,b,c,d = quad
                c1 = colors[idx_map[(a,b,c)]]
                c2 = colors[idx_map[(a,b,d)]]
                c3 = colors[idx_map[(a,c,d)]]
                c4 = colors[idx_map[(b,c,d)]]
                if c1 == c2 == c3 == c4 == 0:
                    red_k4 += 1
                    
            # Count Blue K5
            for quint in combinations(range(n), 5):
                q = quint
                is_blue = True
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
                    blue_k5 += 1
                    
            return red_k4, blue_k5

        red_k4_count, blue_k5_count = count_violations(colors)
        
        if red_k4_count == 0 and blue_k5_count == 0:
            return {'n': n, 'colors': colors}
            
        # Local Search
        temp = 1.0
        cooling = 0.999
        steps = 0
        
        # We only do local search if n is small enough or we have time
        # For n=31, C(31,5) is ~169911. Counting violations is ~1.7M ops.
        # Doing this every step is too slow.
        # We will use an incremental update for K4s and periodic check for K5s.
        
        # Let's simplify: just do a few random restarts and checks.
        # Given the complexity of incremental K5 updates, we'll rely on the initial
        # structure being good and use local search to fix K4s, then check K5s.
        
        # Incremental K4 update
        def flip_triple(idx, colors, red_k4_count):
            a, b, c = triples[idx]
            old_color = colors[idx]
            new_color = 1 - old_color
            
            delta_red = 0
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
                else:
                    if c_abd == 0 and c_acd == 0 and c_bcd == 0:
                        if old_color == 0:
                            delta_red -= 1
                            
            colors[idx] = new_color
            return red_k4_count + delta_red

        # Run local search
        while time.time() < deadline - 5:
            idx = random.randint(0, len(triples) - 1)
            a, b, c = triples[idx]
            old_color = colors[idx]
            new_color = 1 - old_color
            
            delta_red = 0
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
                else:
                    if c_abd == 0 and c_acd == 0 and c_bcd == 0:
                        if old_color == 0:
                            delta_red -= 1
                            
            delta = delta_red
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                colors[idx] = new_color
                red_k4_count += delta_red
                
                if red_k4_count == 0:
                    # Check K5s
                    _, blue_k5_count = count_violations(colors)
                    if blue_k5_count == 0:
                        return {'n': n, 'colors': colors}
            
            temp *= cooling
            steps += 1
            
            # Periodic check
            if steps % 5000 == 0 and red_k4_count < 5:
                _, blue_k5_count = count_violations(colors)
                if blue_k5_count == 0:
                    return {'n': n, 'colors': colors}
            
            # If n is small, we might try a different initial coloring?
            # For now, just continue.
            
        # If we exit the loop, we didn't find a valid coloring for this n.
        # Try smaller n.
        
    # Fallback: Return seed result
    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
