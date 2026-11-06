import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Try to find a valid coloring for n=34
    # If not found, fall back to smaller n
    # We will try n from 34 down to 10
    # But 34 is the target.
    
    def solve_for_n(n):
        if n < 4:
            return {'n': n, 'colors': []}
        
        # Generate all triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map (a,b,c) to index
        triple_to_idx = {}
        for i, t in enumerate(triples):
            triple_to_idx[t] = i
            
        # Precompute neighbors for fast delta calculation
        # red_neighs[i] = list of (d, idx1, idx2, idx3)
        # blue_neighs[i] = list of ((d,e), idx1..idx9)
        
        red_neighs = [[] for _ in range(num_triples)]
        blue_neighs = [[] for _ in range(num_triples)]
        
        for i, (a, b, c) in enumerate(triples):
            # Red K4 neighbors
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # Triples: (a,b,d), (a,c,d), (b,c,d)
                # Sort vertices for key
                t1 = tuple(sorted((a, b, d)))
                t2 = tuple(sorted((a, c, d)))
                t3 = tuple(sorted((b, c, d)))
                idx1 = triple_to_idx[t1]
                idx2 = triple_to_idx[t2]
                idx3 = triple_to_idx[t3]
                red_neighs[i].append((idx1, idx2, idx3))
                
            # Blue K5 neighbors
            other_verts = [v for v in range(n) if v not in (a, b, c)]
            for d_idx in range(len(other_verts)):
                for e_idx in range(d_idx + 1, len(other_verts)):
                    d = other_verts[d_idx]
                    e = other_verts[e_idx]
                    # Vertices: a, b, c, d, e
                    # Triples involving (a,b,c) are the one being flipped.
                    # Other 9 triples:
                    # (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e)
                    # (b,c,d), (b,c,e), (b,d,e)
                    # (c,d,e)
                    
                    t1 = tuple(sorted((a, b, d)))
                    t2 = tuple(sorted((a, b, e)))
                    t3 = tuple(sorted((a, c, d)))
                    t4 = tuple(sorted((a, c, e)))
                    t5 = tuple(sorted((a, d, e)))
                    t6 = tuple(sorted((b, c, d)))
                    t7 = tuple(sorted((b, c, e)))
                    t8 = tuple(sorted((b, d, e)))
                    t9 = tuple(sorted((c, d, e)))
                    
                    idxs = [triple_to_idx[t] for t in [t1, t2, t3, t4, t5, t6, t7, t8, t9]]
                    blue_neighs[i].append(idxs)
        
        # Initial random coloring
        # Bias towards red? Red K4 is smaller (4 verts) than Blue K5 (5 verts).
        # Constraint: No Red K4, No Blue K5.
        # Usually, it's easier to avoid smaller cliques?
        # Let's try p_red = 0.5 first.
        
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.uint8)
        
        def count_violations(cols):
            red_viol = 0
            blue_viol = 0
            # Count Red K4
            # Iterate over all 4-sets? No, iterate over triples and check neighbors?
            # Better: iterate over all 4-sets and check if all 4 triples are red.
            # Number of 4-sets is C(n,4). For n=34, C(34,4) = 46376.
            # For each 4-set, 4 lookups. 46376 * 4 = 185k ops. Fast.
            
            # Count Blue K5
            # Iterate over all 5-sets. C(34,5) = 278256.
            # For each 5-set, 10 lookups. 278256 * 10 = 2.7M ops.
            # This might be slow in Python if done repeatedly.
            # But we only need to calculate total score occasionally or use delta.
            
            # For initial score:
            for comb in combinations(range(n), 4):
                a, b, c, d = comb
                if (cols[triple_to_idx[(a,b,c)]] == 0 and
                    cols[triple_to_idx[(a,b,d)]] == 0 and
                    cols[triple_to_idx[(a,c,d)]] == 0 and
                    cols[triple_to_idx[(b,c,d)]] == 0):
                    red_viol += 1
                    
            for comb in combinations(range(n), 5):
                a, b, c, d, e = comb
                # Check all 10 triples
                t1 = triple_to_idx[(a,b,c)]
                t2 = triple_to_idx[(a,b,d)]
                t3 = triple_to_idx[(a,b,e)]
                t4 = triple_to_idx[(a,c,d)]
                t5 = triple_to_idx[(a,c,e)]
                t6 = triple_to_idx[(a,d,e)]
                t7 = triple_to_idx[(b,c,d)]
                t8 = triple_to_idx[(b,c,e)]
                t9 = triple_to_idx[(b,d,e)]
                t10 = triple_to_idx[(c,d,e)]
                
                if (cols[t1] == 1 and cols[t2] == 1 and cols[t3] == 1 and
                    cols[t4] == 1 and cols[t5] == 1 and cols[t6] == 1 and
                    cols[t7] == 1 and cols[t8] == 1 and cols[t9] == 1 and
                    cols[t10] == 1):
                    blue_viol += 1
                    
            return red_viol + blue_viol

        current_score = count_violations(colors)
        best_score = current_score
        best_colors = colors.copy()
        
        if best_score == 0:
            return {'n': n, 'colors': best_colors.tolist()}
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        
        while T > min_T:
            # Do some steps at this temperature
            # Estimate steps per temp based on time
            # We have ~140s.
            # Let's just loop until time runs out or T drops.
            
            for _ in range(1000): # 1000 steps per temp level?
                if time.time() - start_time > time_limit:
                    break
                
                i = random.randint(0, num_triples - 1)
                current_color = colors[i]
                new_color = 1 - current_color
                
                delta = 0
                if current_color == 0: # Turning Red
                    # Count new Red K4
                    for idx1, idx2, idx3 in red_neighs[i]:
                        if colors[idx1] == 0 and colors[idx2] == 0 and colors[idx3] == 0:
                            delta += 1
                    # Count broken Blue K5
                    for idxs in blue_neighs[i]:
                        # All must be 1 (Blue)
                        # Using all() might be slow, manual check
                        if (colors[idxs[0]] == 1 and colors[idxs[1]] == 1 and
                            colors[idxs[2]] == 1 and colors[idxs[3]] == 1 and
                            colors[idxs[4]] == 1 and colors[idxs[5]] == 1 and
                            colors[idxs[6]] == 1 and colors[idxs[7]] == 1 and
                            colors[idxs[8]] == 1):
                            delta -= 1
                else: # Turning Blue
                    # Count new Blue K5
                    for idxs in blue_neighs[i]:
                        if (colors[idxs[0]] == 1 and colors[idxs[1]] == 1 and
                            colors[idxs[2]] == 1 and colors[idxs[3]] == 1 and
                            colors[idxs[4]] == 1 and colors[idxs[5]] == 1 and
                            colors[idxs[6]] == 1 and colors[idxs[7]] == 1 and
                            colors[idxs[8]] == 1):
                            delta += 1
                    # Count broken Red K4
                    for idx1, idx2, idx3 in red_neighs[i]:
                        if colors[idx1] == 0 and colors[idx2] == 0 and colors[idx3] == 0:
                            delta -= 1
                            
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[i] = new_color
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()
                        if best_score == 0:
                            return {'n': n, 'colors': best_colors.tolist()}
                
                steps += 1
                
            T *= cooling
            if time.time() - start_time > time_limit:
                break
                
        if best_score == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        else:
            return None

    # Try n=34
    res = solve_for_n(34)
    if res:
        return res
        
    # Try n=33
    res = solve_for_n(33)
    if res:
        return res
        
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

