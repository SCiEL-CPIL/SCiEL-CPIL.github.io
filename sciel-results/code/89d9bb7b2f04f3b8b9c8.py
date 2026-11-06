import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 512
    # 1. Precompute LCM compatibility
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
    
    # Initial P = all vertices
    initial_P = (1 << N) - 1
    
    # Greedy clique for initial lower bound
    def greedy_clique():
        P = initial_P
        R = []
        while P:
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1 + 1
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

    best_clique = greedy_clique()
    
    # Try multiple greedy orderings to get a better initial lower bound
    for _ in range(5):
        alt_clique = greedy_clique()
        if len(alt_clique) > len(best_clique):
            best_clique = alt_clique
    
    # Divisor-core strategy: for each M, start with divisors of M, then greedily expand
    def expand_from_core(core_set):
        """Greedily expand a core set by adding compatible vertices."""
        R = list(core_set)
        R_mask = 0
        for v in R:
            R_mask |= (1 << (v - 1))
        
        # Find all vertices compatible with all in R
        compatible = initial_P
        for v in R:
            compatible &= neighbors[v]
        compatible &= ~R_mask
        
        # Greedy expansion
        while compatible:
            best_v = -1
            max_deg = -1
            temp = compatible
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                deg = (compatible & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            R_mask |= (1 << (best_v - 1))
            compatible &= neighbors[best_v]
            compatible &= ~(1 << (best_v - 1))
        
        return R
    
    # Try divisor cores for all M from 1 to 512
    for M in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        
        # Get all divisors of M
        divisors = []
        for d in range(1, M + 1):
            if M % d == 0:
                divisors.append(d)
        
        if not divisors:
            continue
        
        # Try expanding from this core
        expanded = expand_from_core(divisors)
        if len(expanded) > len(best_clique):
            best_clique = expanded
    
    # Local search polish: try 1-swap and 2-swap moves to improve clique
    def local_search_polish(clique):
        """Improve clique via 1-swap and 2-swap moves."""
        if time.time() > deadline:
            return clique
        
        R = list(clique)
        R_set = set(R)
        improved = True
        max_iters = 100
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            if time.time() > deadline:
                break
            
            # 1-swap: remove one vertex, try to add a better one
            for i, v_remove in enumerate(R):
                R_set.remove(v_remove)
                
                # Find candidates not in R_set that are compatible with all in R_set
                best_add = -1
                best_score = -1
                temp = initial_P & ~(1 << (v_remove - 1))
                # Remove all vertices in R_set
                r_mask = 0
                for u in R_set:
                    r_mask |= (1 << (u - 1))
                temp &= ~r_mask
                
                # Must be compatible with all in R_set
                compatible = temp
                for u in R_set:
                    compatible &= neighbors[u]
                
                # Pick the one with max degree in compatible
                temp2 = compatible
                while temp2:
                    lsb = temp2 & -temp2
                    v = lsb.bit_length()
                    deg = (compatible & neighbors[v]).bit_count()
                    if deg > best_score:
                        best_score = deg
                        best_add = v
                    temp2 ^= lsb
                
                if best_add != -1:
                    R[i] = best_add
                    R_set.add(best_add)
                    improved = True
                    break
                else:
                    R_set.add(v_remove)
            
            # 2-swap: remove two vertices, try to add two better ones
            if not improved and len(R) >= 2:
                for i in range(len(R)):
                    for j in range(i + 1, len(R)):
                        v1, v2 = R[i], R[j]
                        R_set.remove(v1)
                        R_set.remove(v2)
                        
                        r_mask = 0
                        for u in R_set:
                            r_mask |= (1 << (u - 1))
                        
                        temp = initial_P & ~r_mask
                        compatible = temp
                        for u in R_set:
                            compatible &= neighbors[u]
                        
                        # Find best pair in compatible
                        best_pair = None
                        best_pair_score = -1
                        temp2 = compatible
                        while temp2:
                            lsb1 = temp2 & -temp2
                            v_a = lsb1.bit_length()
                            # v_a must be compatible with all remaining
                            temp3 = compatible & neighbors[v_a]
                            while temp3:
                                lsb2 = temp3 & -temp3
                                v_b = lsb2.bit_length()
                                # v_b must be compatible with v_a and all in R_set
                                score = (compatible & neighbors[v_a] & neighbors[v_b]).bit_count()
                                if score > best_pair_score:
                                    best_pair_score = score
                                    best_pair = (v_a, v_b)
                                temp3 ^= lsb2
                            temp2 ^= lsb1
                        
                        if best_pair and best_pair_score > 0:
                            R[i] = best_pair[0]
                            R[j] = best_pair[1]
                            R_set.add(best_pair[0])
                            R_set.add(best_pair[1])
                            improved = True
                            break
                        else:
                            R_set.add(v1)
                            R_set.add(v2)
                    
                    if improved:
                        break
            
            if time.time() > deadline:
                break
        
        return R
    
    best_clique = local_search_polish(best_clique)
    
    # Greedy coloring bound - optimized with early termination
    def greedy_color_bound(P):
        if P == 0:
            return 0
        color_count = 0
        uncolored = P
        while uncolored:
            color_count += 1
            independent = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()  # 1-based
                if independent & neighbors[v] == 0:
                    independent |= lsb
                    uncolored ^= lsb
                    temp ^= lsb
                else:
                    temp ^= lsb
        return color_count

    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
                # Apply local search polish to the new best
                polished = local_search_polish(best_clique)
                if len(polished) > len(best_clique):
                    best_clique = polished
            return
            
        # Simple bound
        if len(R) + P.bit_count() <= len(best_clique):
            return
            
        # Always apply coloring bound - it's the key pruning mechanism
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
        
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            temp ^= lsb
            
            bron_kerbosch(R + [v], P & neighbors[v], X & neighbors[v])
            
            P &= ~lsb
            X |= lsb
            
            if time.time() > deadline:
                return

    bron_kerbosch([], initial_P, 0)
    
    # Final polish
    best_clique = local_search_polish(best_clique)
    
    return best_clique
