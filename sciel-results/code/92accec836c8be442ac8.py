import random
import time
import math
from itertools import combinations

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses Simulated Annealing with delta-energy updates.
    """
    target_n = 34
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead

    # We will try to find a solution for n=34.
    # If it fails, we might try n=33, etc.
    
    def solve_for_n(n):
        # Precomputation
        num_triples = n * (n-1) * (n-2) // 6
        triple_indices = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_indices[(a, b, c)] = idx
                    idx += 1
        
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Build masks
        red_mask = [[0 for _ in range(n)] for _ in range(n)]
        blue_mask = [[0 for _ in range(n)] for _ in range(n)]
        
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if colors[idx] == 0: # Red
                        red_mask[a][b] |= (1 << c)
                        red_mask[b][a] |= (1 << c)
                        red_mask[a][c] |= (1 << b)
                        red_mask[c][a] |= (1 << b)
                        red_mask[b][c] |= (1 << a)
                        red_mask[c][b] |= (1 << a)
                    else: # Blue
                        blue_mask[a][b] |= (1 << c)
                        blue_mask[b][a] |= (1 << c)
                        blue_mask[a][c] |= (1 << b)
                        blue_mask[c][a] |= (1 << b)
                        blue_mask[b][c] |= (1 << a)
                        blue_mask[c][b] |= (1 << a)
                    idx += 1

        def count_violations():
            """
            Count total Red K4s and Blue K5s.
            Used for initial energy and final verification.
            """
            red_k4_count = 0
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        # Candidates d > c
                        candidates = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                        candidates &= ~((1 << (c+1)) - 1)
                        red_k4_count += bin(candidates).count('1')
            
            blue_k5_count = 0
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        candidates = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                        candidates &= ~((1 << (c+1)) - 1)
                        cands = []
                        temp = candidates
                        while temp:
                            lsb = temp & -temp
                            v = lsb.bit_length() - 1
                            cands.append(v)
                            temp ^= lsb
                        # Check pairs in cands
                        for i in range(len(cands)):
                            d = cands[i]
                            for j in range(i+1, len(cands)):
                                e = cands[j]
                                if (blue_mask[a][d] & (1 << e)) and \
                                   (blue_mask[b][d] & (1 << e)) and \
                                   (blue_mask[c][d] & (1 << e)):
                                    blue_k5_count += 1
            return red_k4_count + blue_k5_count

        def get_delta_energy(a, b, c, old_color):
            """
            Calculate change in energy (violations) if we flip (a,b,c) from old_color to 1-old_color.
            """
            new_color = 1 - old_color
            delta = 0
            
            if old_color == 0: # Red -> Blue
                # 1. Red K4s destroyed
                # A Red K4 containing {a,b,c} and d was valid if (a,b,d),(a,c,d),(b,c,d) were Red.
                # Now (a,b,c) is Blue, so those K4s are broken.
                # Count d such that (a,b,d), (a,c,d), (b,c,d) are Red.
                # d can be any vertex not in {a,b,c}.
                # We can iterate d from 0 to n-1, d != a,b,c.
                # Optimization: Use masks.
                # Intersection of red_mask[a][b], red_mask[a][c], red_mask[b][c]
                # This gives all d such that (a,b,d), (a,c,d), (b,c,d) are Red.
                # We must exclude d=a, b, c. But masks don't contain a,b,c in those positions usually?
                # red_mask[a][b] contains c if (a,b,c) is red.
                # Let's just iterate d. n=34 is small.
                red_k4s_destroyed = 0
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    # Check if (a,b,d), (a,c,d), (b,c,d) are Red
                    if (red_mask[a][b] & (1 << d)) and \
                       (red_mask[a][c] & (1 << d)) and \
                       (red_mask[b][c] & (1 << d)):
                        red_k4s_destroyed += 1
                delta -= red_k4s_destroyed # Red K4 count decreases
                
                # 2. Blue K5s created
                # A Blue K5 containing {a,b,c}, d, e is created if (a,b,d),(a,c,d),(b,c,d),(a,b,e),(a,c,e),(b,c,e),(d,e,a? no, (a,d,e) etc) are Blue.
                # Specifically, the 10 triples in {a,b,c,d,e} must be Blue.
                # We are flipping (a,b,c) to Blue.
                # So we need (a,b,d), (a,c,d), (b,c,d), (a,b,e), (a,c,e), (b,c,e) to be Blue.
                # AND (a,d,e), (b,d,e), (c,d,e) to be Blue.
                # Note: (a,d,e) etc involve d,e and one of a,b,c.
                
                blue_k5s_created = 0
                # Iterate pairs (d,e) from V \ {a,b,c}
                # To avoid double counting and ensure d<e, we can just iterate all pairs and check conditions.
                # Since n is small, O(n^2) is fine.
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    # Check if (a,b,d), (a,c,d), (b,c,d) are Blue
                    if not ((blue_mask[a][b] & (1 << d)) and 
                            (blue_mask[a][c] & (1 << d)) and 
                            (blue_mask[b][c] & (1 << d))):
                        continue
                    for e in range(n):
                        if e <= d or e == a or e == b or e == c:
                            continue
                        # Check if (a,b,e), (a,c,e), (b,c,e) are Blue
                        if not ((blue_mask[a][b] & (1 << e)) and 
                                (blue_mask[a][c] & (1 << e)) and 
                                (blue_mask[b][c] & (1 << e))):
                            continue
                        # Check if (a,d,e), (b,d,e), (c,d,e) are Blue
                        if (blue_mask[a][d] & (1 << e)) and \
                           (blue_mask[b][d] & (1 << e)) and \
                           (blue_mask[c][d] & (1 << e)):
                            blue_k5s_created += 1
                delta += blue_k5s_created # Blue K5 count increases
                
            else: # Blue -> Red
                # 1. Red K4s created
                red_k4s_created = 0
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    # Check if (a,b,d), (a,c,d), (b,c,d) are Red
                    if (red_mask[a][b] & (1 << d)) and \
                       (red_mask[a][c] & (1 << d)) and \
                       (red_mask[b][c] & (1 << d)):
                        red_k4s_created += 1
                delta += red_k4s_created
                
                # 2. Blue K5s destroyed
                blue_k5s_destroyed = 0
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    # Check if (a,b,d), (a,c,d), (b,c,d) are Blue
                    if not ((blue_mask[a][b] & (1 << d)) and 
                            (blue_mask[a][c] & (1 << d)) and 
                            (blue_mask[b][c] & (1 << d))):
                        continue
                    for e in range(n):
                        if e <= d or e == a or e == b or e == c:
                            continue
                        # Check if (a,b,e), (a,c,e), (b,c,e) are Blue
                        if not ((blue_mask[a][b] & (1 << e)) and 
                                (blue_mask[a][c] & (1 << e)) and 
                                (blue_mask[b][c] & (1 << e))):
                            continue
                        # Check if (a,d,e), (b,d,e), (c,d,e) are Blue
                        if (blue_mask[a][d] & (1 << e)) and \
                           (blue_mask[b][d] & (1 << e)) and \
                           (blue_mask[c][d] & (1 << e)):
                            blue_k5s_destroyed += 1
                delta -= blue_k5s_destroyed
                
            return delta

        def flip_triple(a, b, c):
            idx = triple_indices[(a, b, c)]
            old_color = colors[idx]
            new_color = 1 - old_color
            colors[idx] = new_color
            
            if old_color == 0: # Red -> Blue
                red_mask[a][b] &= ~(1 << c)
                red_mask[b][a] &= ~(1 << c)
                red_mask[a][c] &= ~(1 << b)
                red_mask[c][a] &= ~(1 << b)
                red_mask[b][c] &= ~(1 << a)
                red_mask[c][b] &= ~(1 << a)
                
                blue_mask[a][b] |= (1 << c)
                blue_mask[b][a] |= (1 << c)
                blue_mask[a][c] |= (1 << b)
                blue_mask[c][a] |= (1 << b)
                blue_mask[b][c] |= (1 << a)
                blue_mask[c][b] |= (1 << a)
            else: # Blue -> Red
                blue_mask[a][b] &= ~(1 << c)
                blue_mask[b][a] &= ~(1 << c)
                blue_mask[a][c] &= ~(1 << b)
                blue_mask[c][a] &= ~(1 << b)
                blue_mask[b][c] &= ~(1 << a)
                blue_mask[c][b] &= ~(1 << a)
                
                red_mask[a][b] |= (1 << c)
                red_mask[b][a] |= (1 << c)
                red_mask[a][c] |= (1 << b)
                red_mask[c][a] |= (1 << b)
                red_mask[b][c] |= (1 << a)
                red_mask[c][b] |= (1 << a)

        # Simulated Annealing
        current_energy = count_violations()
        best_energy = current_energy
        best_colors = colors.copy()
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Pick random triple
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            c = random.randint(0, n-1)
            # Ensure a < b < c
            if a == b or a == c or b == c:
                continue
            if a > b: a, b = b, a
            if a > c: a, c = c, a
            if b > c: b, c = c, b
            
            old_color = colors[triple_indices[(a,b,c)]]
            delta = get_delta_energy(a, b, c, old_color)
            new_energy = current_energy + delta
            
            if new_energy < current_energy or random.random() < math.exp(-delta / T):
                flip_triple(a, b, c)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return {'n': n, 'colors': best_colors}
            
            T *= cooling_rate
            if T < T_min:
                break
        
        # If we found a valid coloring, return it
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
        
        return None

    # Try to solve for n=34
    # We might need multiple restarts
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        res = solve_for_n(34)
        if res:
            return res

    # Fallback: Return a small valid coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
