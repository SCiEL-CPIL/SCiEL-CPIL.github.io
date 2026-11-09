import random
import time
import math

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses Simulated Annealing with efficient energy delta calculation.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead
    
    def get_masks(n):
        """Initialize random coloring and masks."""
        num_triples = n * (n - 1) * (n - 2) // 6
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Precompute indices for triples to allow fast access
        # We need a way to go from (a,b,c) to index and vice versa.
        # For n=34, C(34,3) = 5984.
        # We can precompute a list of triples and a dict for index lookup.
        triples = []
        triple_index = {}
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    triples.append((a, b, c))
                    triple_index[(a, b, c)] = idx
                    idx += 1
        
        # Build masks
        # red_mask[i][j] is a bitmask of k such that (i,j,k) is Red
        # Note: i < j < k is not required for the mask, just that the triple exists.
        # We store bits for all k.
        red_mask = [[0] * n for _ in range(n)]
        blue_mask = [[0] * n for _ in range(n)]
        
        for idx, (a, b, c) in enumerate(triples):
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
                
        return colors, red_mask, blue_mask, triples, triple_index

    def calculate_energy(n, red_mask, blue_mask, triples):
        """Calculate total number of violations."""
        energy = 0
        
        # Count Red K4
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    # Candidates d > c
                    candidates = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                    candidates &= ~((1 << (c + 1)) - 1)
                    energy += bin(candidates).count('1')
                    
        # Count Blue K5
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    # Candidates d, e > c
                    # S_blue = {v > c | (a,b,v), (a,c,v), (b,c,v) are Blue}
                    s_blue = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                    s_blue &= ~((1 << (c + 1)) - 1)
                    
                    # Iterate d in s_blue
                    temp = s_blue
                    while temp:
                        lsb = temp & -temp
                        d = lsb.bit_length() - 1
                        temp ^= lsb
                        
                        # For this d, find e > d in s_blue such that (a,d,e), (b,d,e), (c,d,e) are Blue
                        # Mask for e: blue_mask[a][d] & blue_mask[b][d] & blue_mask[c][d]
                        mask_e = blue_mask[a][d] & blue_mask[b][d] & blue_mask[c][d]
                        mask_e &= s_blue
                        mask_e &= ~((1 << (d + 1)) - 1) # e > d
                        energy += bin(mask_e).count('1')
                        
        return energy

    def delta_energy(n, red_mask, blue_mask, a, b, c, new_color):
        """
        Calculate change in energy if triple (a,b,c) is set to new_color.
        new_color: 0 for Red, 1 for Blue.
        Current color is assumed to be 1 - new_color.
        """
        old_color = 1 - new_color
        delta = 0
        
        if new_color == 0: # Changing from Blue to Red
            # Remove Blue K5s containing (a,b,c)
            # Add Red K4s containing (a,b,c)
            
            # Remove Blue K5s
            # S_blue = {v | (a,b,v), (a,c,v), (b,c,v) are Blue}
            # Note: (a,b,c) was Blue, so it is part of the "base" for K5s.
            # A Blue K5 containing (a,b,c) is {a,b,c,d,e} where all 10 triples are Blue.
            # Since (a,b,c) is currently Blue, we are removing it.
            # So we subtract the number of pairs (d,e) such that the other 9 triples are Blue.
            # The other 9 triples involve d and e.
            # Condition: (a,b,d), (a,b,e), (a,c,d), (a,c,e), (b,c,d), (b,c,e) are Blue.
            # AND (a,d,e), (b,d,e), (c,d,e) are Blue.
            
            s_blue = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
            # Exclude a,b,c from s_blue? 
            # blue_mask[a][b] contains c if (a,b,c) is blue.
            # We need d,e distinct from a,b,c.
            # The masks blue_mask[a][b] etc. contain all vertices connected by blue edge/triple.
            # We must ensure d,e are not a,b,c.
            # Since a<b<c, and masks are symmetric, we just need to mask out a,b,c.
            s_blue &= ~(1 << a)
            s_blue &= ~(1 << b)
            s_blue &= ~(1 << c)
            
            temp = s_blue
            while temp:
                lsb = temp & -temp
                d = lsb.bit_length() - 1
                temp ^= lsb
                
                # Find e in s_blue, e != d
                # Condition for e: (a,d,e), (b,d,e), (c,d,e) are Blue.
                mask_e = blue_mask[a][d] & blue_mask[b][d] & blue_mask[c][d]
                mask_e &= s_blue
                mask_e &= ~(1 << d) # e != d
                # We count pairs (d,e). To avoid double counting, assume d < e?
                # The energy function counted pairs (d,e) with d < e implicitly by iterating d and checking e>d?
                # In calculate_energy, I iterated d from s_blue and checked e > d.
                # So here I should also count e > d.
                mask_e &= ~((1 << (d + 1)) - 1)
                delta -= bin(mask_e).count('1') # Removing Blue K5s reduces energy (good)
                
            # Add Red K4s
            # A Red K4 containing (a,b,c) is {a,b,c,d} where (a,b,d), (a,c,d), (b,c,d) are Red.
            s_red = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
            s_red &= ~(1 << a)
            s_red &= ~(1 << b)
            s_red &= ~(1 << c)
            # Count d in s_red.
            delta += bin(s_red).count('1') # Adding Red K4s increases energy (bad)

        else: # Changing from Red to Blue
            # Remove Red K4s containing (a,b,c)
            # Add Blue K5s containing (a,b,c)
            
            # Remove Red K4s
            s_red = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
            s_red &= ~(1 << a)
            s_red &= ~(1 << b)
            s_red &= ~(1 << c)
            delta -= bin(s_red).count('1')
            
            # Add Blue K5s
            s_blue = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
            s_blue &= ~(1 << a)
            s_blue &= ~(1 << b)
            s_blue &= ~(1 << c)
            
            temp = s_blue
            while temp:
                lsb = temp & -temp
                d = lsb.bit_length() - 1
                temp ^= lsb
                
                mask_e = blue_mask[a][d] & blue_mask[b][d] & blue_mask[c][d]
                mask_e &= s_blue
                mask_e &= ~(1 << d)
                mask_e &= ~((1 << (d + 1)) - 1)
                delta += bin(mask_e).count('1')

        return delta

    def flip_triple(n, colors, red_mask, blue_mask, a, b, c, new_color):
        """Update masks and colors after flipping triple (a,b,c) to new_color."""
        idx = triple_index[(a, b, c)]
        old_color = colors[idx]
        colors[idx] = new_color
        
        if old_color == 0: # Was Red, now Blue
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
        else: # Was Blue, now Red
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

    # Try n from 34 downwards
    for n in range(34, 8, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Run Simulated Annealing
        colors, red_mask, blue_mask, triples, triple_index = get_masks(n)
        current_energy = calculate_energy(n, red_mask, blue_mask, triples)
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        max_iters = 50000
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            if current_energy == 0:
                return {'n': n, 'colors': colors}
                
            # Pick random triple
            idx = random.randint(0, len(triples) - 1)
            a, b, c = triples[idx]
            
            # Try flipping
            new_color = 1 - colors[idx]
            dE = delta_energy(n, red_mask, blue_mask, a, b, c, new_color)
            
            if dE < 0:
                # Always accept
                flip_triple(n, colors, red_mask, blue_mask, a, b, c, new_color)
                current_energy += dE
            else:
                # Accept with probability exp(-dE / T)
                if random.random() < math.exp(-dE / T):
                    flip_triple(n, colors, red_mask, blue_mask, a, b, c, new_color)
                    current_energy += dE
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]]}
