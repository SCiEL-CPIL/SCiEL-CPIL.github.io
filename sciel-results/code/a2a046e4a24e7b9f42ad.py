"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_n = 9
    # Fallback valid coloring: color by (a+b+c) mod 2
    best_colors = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # We'll try to find valid colorings for increasing n
    # Use local search with exact delta calculation

    for target_n in range(10, 35):
        if time.time() > deadline - 15:
            break

        found = False
        triples = list(combinations(range(target_n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute lists of other vertices for faster delta calculation?
        # For each triple (a,b,c), the other vertices are all v not in {a,b,c}.
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break

            # Initialize random coloring
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Initial violation count
            rk4 = 0
            for quad in combinations(range(target_n), 4):
                a, b, c, d = quad
                if (colors[triple_to_idx[(a,b,c)]] == 0 and
                    colors[triple_to_idx[(a,b,d)]] == 0 and
                    colors[triple_to_idx[(a,c,d)]] == 0 and
                    colors[triple_to_idx[(b,c,d)]] == 0):
                    rk4 += 1
            
            bk5 = 0
            for quint in combinations(range(target_n), 5):
                # Check if all 10 are blue
                is_blue = True
                for t in combinations(quint, 3):
                    if colors[triple_to_idx[t]] != 1:
                        is_blue = False
                        break
                if is_blue:
                    bk5 += 1

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Local search
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                # Pick a random triple to flip
                t_idx = random.randint(0, num_triples - 1)
                t = triples[t_idx]
                a, b, c = t
                
                # Calculate delta rk4
                delta_rk4 = 0
                # Iterate over 4th vertex v
                for v in range(target_n):
                    if v == a or v == b or v == c:
                        continue
                    # Quad {a,b,c,v}
                    # Other triples: (a,b,v), (a,c,v), (b,c,v)
                    # Ensure sorted order for lookup
                    t1 = tuple(sorted((a, b, v)))
                    t2 = tuple(sorted((a, c, v)))
                    t3 = tuple(sorted((b, c, v)))
                    
                    c1 = colors[triple_to_idx[t1]]
                    c2 = colors[triple_to_idx[t2]]
                    c3 = colors[triple_to_idx[t3]]
                    
                    if colors[t_idx] == 0: # Currently Red
                        if c1 == 0 and c2 == 0 and c3 == 0:
                            # Was Red K4, now not
                            delta_rk4 -= 1
                    else: # Currently Blue
                        if c1 == 0 and c2 == 0 and c3 == 0:
                            # Was not Red K4, now is
                            delta_rk4 += 1

                # Calculate delta bk5
                delta_bk5 = 0
                # Iterate over pairs of 4th, 5th vertices u, v
                # u, v not in {a,b,c}, u < v
                other_verts = [x for x in range(target_n) if x != a and x != b and x != c]
                for i in range(len(other_verts)):
                    u = other_verts[i]
                    for j in range(i + 1, len(other_verts)):
                        v = other_verts[j]
                        # Quint {a,b,c,u,v}
                        # Other 9 triples
                        # We need to check if all 9 are Blue (1)
                        # Triples are combinations of 3 from {a,b,c,u,v} excluding (a,b,c)
                        # This is 10 - 1 = 9 triples.
                        
                        # Generate them
                        quint = tuple(sorted((a, b, c, u, v)))
                        all_blue = True
                        for tt in combinations(quint, 3):
                            if tt == (a, b, c):
                                continue
                            if colors[triple_to_idx[tt]] != 1:
                                all_blue = False
                                break
                        
                        if colors[t_idx] == 1: # Currently Blue
                            if all_blue:
                                # Was Blue K5, now not
                                delta_bk5 -= 1
                        else: # Currently Red
                            if all_blue:
                                # Was not Blue K5, now is
                                delta_bk5 += 1

                new_rk4 = rk4 + delta_rk4
                new_bk5 = bk5 + delta_bk5
                
                # Accept if it reduces total violations
                if new_rk4 + new_bk5 < rk4 + bk5:
                    colors[t_idx] = 1 - colors[t_idx]
                    rk4 = new_rk4
                    bk5 = new_bk5
                elif new_rk4 + new_bk5 == rk4 + bk5 and random.random() < 0.1:
                    # Accept neutral moves with low probability to escape local minima
                    colors[t_idx] = 1 - colors[t_idx]
                    rk4 = new_rk4
                    bk5 = new_bk5

            if found:
                break

        if found:
            best_n = target_n
            best_colors = colors.tolist()

    # Final verification
    if best_n > 9:
        color_map = {}
        for i, (a, b, c) in enumerate(combinations(range(best_n), 3)):
            color_map[(a, b, c)] = best_colors[i]
        
        valid = True
        for quad in combinations(range(best_n), 4):
            a, b, c, d = quad
            if all(color_map[t] == 0 for t in [(a,b,c),(a,b,d),(a,c,d),(b,c,d)]):
                valid = False
                break
        if valid:
            for quint in combinations(range(best_n), 5):
                if all(color_map[t] == 1 for t in combinations(quint, 3)):
                    valid = False
                    break

        if not valid:
            # Fallback to n=9
            best_n = 9
            best_colors = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
