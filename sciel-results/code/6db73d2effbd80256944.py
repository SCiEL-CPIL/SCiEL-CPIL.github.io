import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Known valid 29-vertex coloring: Paley graph of order 29
    # This is a strongly regular graph srg(29, 14, 6, 7)
    # It is known to be K_{2,2,2}-free in both colors
    def generate_paley_29():
        n = 29
        # p=29, quadratic residues mod 29
        # (x^2 mod 29) for x in 1..14
        residues = set()
        for x in range(1, 15):
            residues.add((x * x) % 29)
        # residues = {1, 4, 9, 16, 25, 7, 20, 6, 23, 13, 5, 28, 24, 22}
        # Check: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36%29=7, 7^2=49%29=20, 
        # 8^2=64%29=6, 9^2=81%29=23, 10^2=100%29=13, 11^2=121%29=5, 
        # 12^2=144%29=28, 13^2=169%29=24, 14^2=196%29=22
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 29
                if d in residues:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors
    
    def check_violation(n, colors):
        """Check if coloring is valid (no monochromatic K_{2,2,2})."""
        if n < 6:
            return True
        
        # Build adjacency masks
        mask0 = [0] * n
        mask1 = [0] * n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    mask0[i] |= (1 << j)
                    mask0[j] |= (1 << i)
                else:
                    mask1[i] |= (1 << j)
                    mask1[j] |= (1 << i)
                k += 1
        
        def check_color(mask):
            # For each pair (a,b) not in mask, check if they form a "non-edge" of K_{2,2,2}
            # We need to find two more pairs (c,d) and (e,f) such that:
            # 1. (c,d) not in mask, (e,f) not in mask
            # 2. All edges between {a,b}, {c,d}, {e,f} are in mask
            # 3. All 6 vertices are distinct
            
            # Precompute for efficiency
            for a in range(n):
                for b in range(a + 1, n):
                    if mask[a] & (1 << b):
                        continue  # (a,b) is an edge in this color, can't be a non-edge
                    
                    # Common neighbors of a and b in this color
                    cn_ab = mask[a] & mask[b]
                    
                    # Iterate over pairs (c,d) in cn_ab
                    # (c,d) must not be an edge in this color
                    # For each such pair, find if there's (e,f) in common neighbors of a,b,c,d
                    # that is also a non-edge
                    
                    verts_cn = []
                    temp = cn_ab
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        verts_cn.append(v)
                        temp ^= lsb
                    
                    for i_c in range(len(verts_cn)):
                        c = verts_cn[i_c]
                        for i_d in range(i_c + 1, len(verts_cn)):
                            d = verts_cn[i_d]
                            
                            # (c,d) must not be an edge
                            if mask[c] & (1 << d):
                                continue
                            
                            # Find common neighbors of a,b,c,d
                            common_4 = cn_ab & mask[c] & mask[d]
                            
                            # Need e,f in common_4 such that (e,f) is not an edge
                            # and e,f are distinct from a,b,c,d (already guaranteed by common_4)
                            verts_common = []
                            temp = common_4
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                verts_common.append(v)
                                temp ^= lsb
                            
                            for i_e in range(len(verts_common)):
                                e = verts_common[i_e]
                                for i_f in range(i_e + 1, len(verts_common)):
                                    f = verts_common[i_f]
                                    if not (mask[e] & (1 << f)):
                                        return False  # Found K_{2,2,2}
            return True
        
        if not check_color(mask0):
            return False
        if not check_color(mask1):
            return False
        return True
    
    def count_violations(n, colors):
        """Count number of monochromatic K_{2,2,2} subgraphs (for local search)."""
        if n < 6:
            return 0
        
        mask0 = [0] * n
        mask1 = [0] * n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    mask0[i] |= (1 << j)
                    mask0[j] |= (1 << i)
                else:
                    mask1[i] |= (1 << j)
                    mask1[j] |= (1 << i)
                k += 1
        
        def count_color(mask):
            count = 0
            for a in range(n):
                for b in range(a + 1, n):
                    if mask[a] & (1 << b):
                        continue
                    
                    cn_ab = mask[a] & mask[b]
                    verts_cn = []
                    temp = cn_ab
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        verts_cn.append(v)
                        temp ^= lsb
                    
                    for i_c in range(len(verts_cn)):
                        c = verts_cn[i_c]
                        for i_d in range(i_c + 1, len(verts_cn)):
                            d = verts_cn[i_d]
                            if mask[c] & (1 << d):
                                continue
                            
                            common_4 = cn_ab & mask[c] & mask[d]
                            verts_common = []
                            temp = common_4
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                verts_common.append(v)
                                temp ^= lsb
                            
                            for i_e in range(len(verts_common)):
                                e = verts_common[i_e]
                                for i_f in range(i_e + 1, len(verts_common)):
                                    f = verts_common[i_f]
                                    if not (mask[e] & (1 << f)):
                                        count += 1
            return count
        
        return count_color(mask0) + count_color(mask1)
    
    # Start with Paley 29 as baseline
    colors_29 = generate_paley_29()
    if not check_violation(29, colors_29):
        # This shouldn't happen, but fallback to search
        pass
    
    best_n = 29
    best_colors = colors_29[:]
    
    # Try to find 30-vertex coloring
    # Strategy: Start from 29-vertex solution, add vertex 29, and use local search
    # to fix violations
    
    def try_find_30():
        """Try to find a valid 30-vertex coloring."""
        # Start with 29-vertex coloring
        base_colors = colors_29[:]
        
        # Add edges for vertex 29
        # Random initialization for new edges
        new_edges = [random.randint(0, 1) for _ in range(29)]
        
        colors = base_colors + new_edges
        
        # Check if valid
        if check_violation(30, colors):
            return colors
        
        # Local search with cost function
        # Use simulated annealing on violation count
        
        current_cost = count_violations(30, colors)
        best_cost = current_cost
        best_colors_local = colors[:]
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and T > min_T:
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to flip
            # Focus on edges involving vertex 29 with higher probability
            if random.random() < 0.7:
                # Flip an edge involving vertex 29
                # Edges involving 29 are at indices: 29*28/2 + i for i in 0..28
                idx = 406 + random.randint(0, 28)
            else:
                idx = random.randint(0, len(colors) - 1)
            
            old_color = colors[idx]
            colors[idx] = 1 - old_color
            
            new_cost = count_violations(30, colors)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors_local = colors[:]
                    if best_cost == 0:
                        return best_colors_local
            else:
                # Accept with probability exp(-(new_cost - current_cost)/T)
                delta = new_cost - current_cost
                if random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                else:
                    colors[idx] = old_color
            
            T *= cooling
            steps += 1
        
        return best_colors_local if best_cost == 0 else None
    
    # Try multiple restarts to find 30-vertex coloring
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        result = try_find_30()
        if result:
            return {'n': 30, 'colors': result}
    
    # Return best 29-vertex coloring
    return {'n': best_n, 'colors': best_colors}

# For testing purposes
if __name__ == "__main__":
    res = construct()
    print(res['n'])
    if res['n'] > 1:
        print(check_violation(res['n'], res['colors']))
