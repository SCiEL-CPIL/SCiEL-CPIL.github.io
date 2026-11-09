import random
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses Simulated Annealing with efficient delta energy calculation.
    """
    target_n = 34
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead

    # We will try to find a valid coloring for n=34.
    # If it fails, we fall back to smaller n.
    
    def run_sa(n):
        num_triples = n * (n-1) * (n-2) // 6
        # colors: 0 for Red, 1 for Blue
        colors = np.random.randint(0, 2, num_triples)
        
        # Precompute triple indices for fast access
        # Map (a,b,c) to index
        # We can use a dictionary or a formula.
        # For n=34, num_triples = 5984. Dictionary is fine.
        triple_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_idx[(a,b,c)] = idx
                    idx += 1

        # Initialize masks
        # red_mask[i][j] is a bitmask of k such that (i,j,k) is Red
        # We use Python integers for bitmasks as they are fast for bitwise ops
        red_mask = [[0 for _ in range(n)] for _ in range(n)]
        blue_mask = [[0 for _ in range(n)] for _ in range(n)]
        
        for (a,b,c), i in triple_idx.items():
            if colors[i] == 0: # Red
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

        # Initial Energy Calculation
        # Red K4 count
        red_violations = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Count d > c such that (a,b,d), (a,c,d), (b,c,d) are Red
                    cands = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                    cands &= ~((1 << (c+1)) - 1)
                    red_violations += bin(cands).count('1')
                    
        # Blue K5 count
        blue_violations = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    cands = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                    cands &= ~((1 << (c+1)) - 1)
                    # Get list of candidates
                    cands_list = []
                    temp = cands
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        cands_list.append(v)
                        temp ^= lsb
                    # Check pairs
                    for i in range(len(cands_list)):
                        d = cands_list[i]
                        for j in range(i+1, len(cands_list)):
                            e = cands_list[j]
                            if (blue_mask[a][d] & (1 << e)) and \
                               (blue_mask[b][d] & (1 << e)) and \
                               (blue_mask[c][d] & (1 << e)):
                                blue_violations += 1

        total_violations = red_violations + blue_violations
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        min_T = 1e-4
        steps = 0
        max_steps = 50000
        
        while T > min_T and steps < max_steps:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick random triple
            idx = random.randint(0, num_triples - 1)
            # Find (a,b,c) for idx? 
            # This is slow if we don't have reverse map.
            # Let's create a reverse map or just iterate?
            # Better: Precompute list of triples
            if not hasattr(run_sa, 'triples_list'):
                run_sa.triples_list = [ (a,b,c) for a in range(n) for b in range(a+1,n) for c in range(b+1,n) ]
            
            a, b, c = run_sa.triples_list[idx]
            
            old_color = colors[idx]
            new_color = 1 - old_color
            
            # Calculate Delta Energy
            delta_red = 0
            delta_blue = 0
            
            if old_color == 0: # Red -> Blue
                # Remove Red (a,b,c)
                # Red K4s destroyed: d such that (a,b,d),(a,c,d),(b,c,d) are Red
                cands = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                # Note: We don't need to mask out <=c because the mask only contains valid k for the pair.
                # Wait, red_mask[a][b] contains k where (a,b,k) is Red.
                # If k < b, (a,k,b) is the triple, but we store it in red_mask[a][b] as bit k?
                # Yes, we store symmetrically.
                # So cands contains all d such that the 3 triples with d are Red.
                # Each such d forms a Red K4 with (a,b,c).
                # So we destroy bin(cands).count('1') Red K4s.
                delta_red = -bin(cands).count('1')
                
                # Add Blue (a,b,c)
                # Blue K5s created: pairs (d,e) such that other 9 are Blue.
                # d,e must be in D = { v | (a,b,v),(a,c,v),(b,c,v) are Blue }
                # Note: The masks blue_mask currently DO NOT include (a,b,c) as Blue yet.
                # So D is calculated from current state.
                D_mask = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                D_list = []
                temp = D_mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    D_list.append(v)
                    temp ^= lsb
                    
                for i in range(len(D_list)):
                    d = D_list[i]
                    for j in range(i+1, len(D_list)):
                        e = D_list[j]
                        if (blue_mask[a][d] & (1 << e)) and \
                           (blue_mask[b][d] & (1 << e)) and \
                           (blue_mask[c][d] & (1 << e)):
                            delta_blue += 1

            else: # Blue -> Red
                # Remove Blue (a,b,c)
                # Blue K5s destroyed
                D_mask = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                D_list = []
                temp = D_mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    D_list.append(v)
                    temp ^= lsb
                    
                for i in range(len(D_list)):
                    d = D_list[i]
                    for j in range(i+1, len(D_list)):
                        e = D_list[j]
                        if (blue_mask[a][d] & (1 << e)) and \
                           (blue_mask[b][d] & (1 << e)) and \
                           (blue_mask[c][d] & (1 << e)):
                            delta_blue -= 1

                # Add Red (a,b,c)
                # Red K4s created
                cands = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                delta_red = bin(cands).count('1')

            delta_E = delta_red + delta_blue
            new_energy = total_violations + delta_E
            
            # Acceptance
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept move
                total_violations = new_energy
                colors[idx] = new_color
                
                # Update masks
                a, b, c = run_sa.triples_list[idx]
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
                    
                if total_violations == 0:
                    return {'n': n, 'colors': colors.tolist()}
            
            T *= cooling
            steps += 1

        # If not zero, return best found or None
        return None

    # Try n=34
    result = run_sa(34)
    if result:
        return result

    # Fallback to smaller n if 34 fails
    for n in range(33, 9, -1):
        if time.time() - start_time > time_limit:
            break
        result = run_sa(n)
        if result:
            return result

    # Final fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
