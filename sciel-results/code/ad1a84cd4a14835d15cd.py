import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 57  # Safety margin (90s kill, but be conservative)
    
    N = 512
    
    # Precompute LCM compatibility
    # neighbors[i] is a bitmask of all j where lcm(i,j) <= 512
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
    
    initial_P = (1 << N) - 1
    
    # Precompute divisor-based seeds: for each highly composite number,
    # its divisors form a valid set (lcm of any two divisors divides the number)
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return [d for d in divs if d <= N]
    
    # Find good seed sets from highly composite numbers
    seed_sets = []
    for hcn in [252, 360, 420, 480, 504, 336, 240, 180, 120, 84, 72, 60, 48, 36, 24, 12, 8, 6, 4, 2, 1]:
        divs = get_divisors(hcn)
        if len(divs) > 1:
            seed_sets.append(divs)
    
    # Also try: all numbers that divide some common number <= 512
    # The best single number to divide is the one with most divisors <= 512
    # 360 has 24 divisors, 252 has 18, 420 has 16, 480 has 16, 504 has 20
    # Actually let's check which number in [1,512] has the most divisors
    best_div_count = 0
    best_divs = []
    for n in range(1, N + 1):
        divs = get_divisors(n)
        if len(divs) > best_div_count:
            best_div_count = len(divs)
            best_divs = divs
    if best_divs and len(best_divs) > 1:
        seed_sets.insert(0, best_divs)
    

    # Greedy clique for initial lower bound - try multiple strategies
    def greedy_clique_from_seed(seed):
        """Greedy expand from a seed set."""
        R = list(seed)
        P = 0
        for v in R:
            P |= (1 << (v - 1))
        # Now P contains the seed; intersect neighbors for all in R
        common = (1 << N) - 1
        for v in R:
            common &= neighbors[v]
        P = common
        # Remove seed members from P (they're already in R)
        for v in R:
            P &= ~(1 << (v - 1))
        # Greedy expand
        while P:
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                deg = (P & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
        return R
    
    def greedy_clique_by_degree():
        P = initial_P
        R = []
        while P:
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()  # 1-based
                deg = (P & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
        return R
    
    def greedy_clique_by_value():
        P = initial_P
        R = []
        while P:
            best_v = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                if best_v == -1 or v < best_v:
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
        return R
    
    def greedy_clique_by_value_desc():
        P = initial_P
        R = []
        while P:
            best_v = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                if best_v == -1 or v > best_v:
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
        return R
    
    # Try all seed sets and greedy strategies
    best_clique = greedy_clique_by_degree()
    for seed in seed_sets:
        if time.time() > deadline:
            break
        alt = greedy_clique_from_seed(seed)
        if len(alt) > len(best_clique):
            best_clique = alt
    alt = greedy_clique_by_value()
    if len(alt) > len(best_clique):
        best_clique = alt
    alt = greedy_clique_by_value_desc()
    if len(alt) > len(best_clique):
        best_clique = alt
    # Try more degree-based runs with different tie-breaking
    for _ in range(5):
        if time.time() > deadline:
            break
        alt = greedy_clique_by_degree()
        if len(alt) > len(best_clique):
            best_clique = alt
    
    # Greedy coloring bound - improved with proper vertex ordering
    def greedy_color_bound(P):
        if P == 0:
            return 0
        # Order vertices by degree in P (ascending) for tighter bound
        vertices = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            deg = (P & neighbors[v]).bit_count()
            vertices.append((deg, v))
            temp ^= lsb
        vertices.sort()  # ascending degree
        
        color_count = 0
        uncolored = P
        for deg, v in vertices:
            if not (uncolored & (1 << (v - 1))):
                continue
            color_count += 1
            # Find all uncolored vertices non-adjacent to v and to each other
            # Actually for coloring bound, we need a proper coloring of the subgraph
            # Greedy: assign colors one by one
            pass
        
        # Simpler but effective: use the standard greedy coloring for upper bound
        # The bound is the chromatic number of the subgraph induced by P
        # Greedy coloring gives an upper bound on chromatic number
        color_count = 0
        uncolored = P
        while uncolored:
            color_count += 1
            # Find maximal independent set in uncolored
            independent = 0
            # Order by degree for better independent set
            temp = uncolored
            verts = []
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                deg = (uncolored & neighbors[v]).bit_count()
                verts.append((deg, v))
                temp ^= lsb
            verts.sort()  # lowest degree first
            for deg, v in verts:
                if not (uncolored & (1 << (v - 1))):
                    continue
                if independent & neighbors[v] == 0:
                    independent |= (1 << (v - 1))
                    uncolored &= ~(1 << (v - 1))
        return color_count
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Simple bound
        if len(R) + P.bit_count() <= len(best_clique):
            return
            
        # Coloring bound - key pruning mechanism
        c_bound = greedy_color_bound(P)
        if len(R) + c_bound <= len(best_clique):
            return
            
        PX = P | X
        if PX == 0:
            return
            
        # Find pivot: choose vertex in PX with max overlap with P
        max_overlap = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length()
            overlap = P & neighbors[u]
            count = overlap.bit_count()
            if count > max_overlap:
                max_overlap = count
                pivot = u
            temp ^= lsb
            
        if pivot == -1:
            return

        candidates = P & ~neighbors[pivot]
        
        # Sort candidates by degree in P (descending) for better search order
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            deg = (P & neighbors[v]).bit_count()
            cand_list.append((deg, v))
            temp ^= lsb
        cand_list.sort(reverse=True)
        
        for deg, v in cand_list:
            lsb = 1 << (v - 1)
            if not (P & lsb):
                continue  # already processed
            
            bron_kerbosch(R + [v], P & neighbors[v], X & neighbors[v])
            
            P &= ~lsb
            X |= lsb
            
            if time.time() > deadline:
                return

    bron_kerbosch([], initial_P, 0)
    
    # Local improvement: try to add elements to the best clique
    if time.time() < deadline - 5:
        best_set = set(best_clique)
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if v in best_set:
                    continue
                # Check if v can be added
                can_add = True
                for u in best_set:
                    g = gcd(v, u)
                    if (v // g) * u > N:
                        can_add = False
                        break
                if can_add:
                    best_set.add(v)
                    improved = True
        if len(best_set) > len(best_clique):
            best_clique = list(best_set)
    
    return best_clique
