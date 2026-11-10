import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 512
    
    # Precompute compatibility bitsets
    # compat[x] is a bitmask where bit (y-1) is set if lcm(x,y) <= N
    compat = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        for y in range(1, N + 1):
            g = gcd(x, y)
            lcm_val = x // g * y
            if lcm_val <= N:
                mask |= (1 << (y - 1))
        compat[x] = mask
    
    # all_mask has bits 0..N-1 set
    all_mask = (1 << N) - 1
    
    # Helper: compute a bitset for a set of vertices
    def set_to_mask(s):
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        return mask
    
    # Helper: compute a set from a bitset
    def mask_to_set(mask):
        s = set()
        m = mask
        pos = 0
        while m:
            if m & 1:
                s.add(pos + 1)
            m >>= 1
            pos += 1
        return s
    
    # Greedy clique construction with ordering
    def greedy_clique(order=None):
        """Build a clique greedily. If order is None, use best-degree heuristic."""
        clique = []
        remaining = all_mask
        
        if order is None:
            # Best-degree greedy: at each step pick vertex with most neighbors in remaining
            while remaining and time.time() < deadline:
                best_v = -1
                best_count = -1
                m = remaining
                pos = 0
                while m:
                    if m & 1:
                        v = pos + 1
                        count = bin(compat[v] & remaining).count('1')
                        if count > best_count:
                            best_count = count
                            best_v = v
                    m >>= 1
                    pos += 1
                if best_v == -1:
                    break
                clique.append(best_v)
                remaining = remaining & compat[best_v]
                remaining &= ~(1 << (best_v - 1))
        else:
            # Fixed order greedy
            for v in order:
                if time.time() > deadline:
                    break
                if remaining & (1 << (v - 1)):
                    clique_mask = set_to_mask(set(clique))
                    if (compat[v] & clique_mask) == clique_mask:
                        clique.append(v)
                        remaining = remaining & compat[v]
                        remaining &= ~(1 << (v - 1))
        
        return clique
    
    # Greedy expansion: add any compatible element
    def greedy_expand(s):
        """Repeatedly add any element compatible with all in s."""
        changed = True
        s_mask = set_to_mask(s)
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x not in s:
                    if (compat[x] & s_mask) == s_mask:
                        s.add(x)
                        s_mask |= (1 << (x - 1))
                        changed = True
        return s
    
    # 1-out, multiple-in local search
    def local_search_1out(s):
        """Try removing one element and greedily adding multiple new ones."""
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # First try to add any compatible element
            s_mask = set_to_mask(s)
            for x in range(1, N + 1):
                if x not in s:
                    if (compat[x] & s_mask) == s_mask:
                        s.add(x)
                        s_mask |= (1 << (x - 1))
                        improved = True
            if improved:
                continue
            
            # Try 1-out, multiple-in
            for remove_x in list(s):
                if time.time() > deadline:
                    break
                trial_set = s - {remove_x}
                trial_mask = set_to_mask(trial_set)
                
                for x in range(1, N + 1):
                    if x not in trial_set:
                        if (compat[x] & trial_mask) == trial_mask:
                            trial_set.add(x)
                            trial_mask |= (1 << (x - 1))
                
                if len(trial_set) > len(s):
                    s = trial_set
                    improved = True
                    break
        
        return s
    
    # Branch and bound for maximum clique with coloring bound
    def max_clique_bb(candidates_mask, current_clique, best_size_ref):
        """
        Branch and bound maximum clique.
        candidates_mask: bitmask of candidate vertices
        current_clique: list of vertices in current clique
        best_size_ref: mutable list [best_size] for tracking
        """
        if time.time() > deadline:
            return
        
        # If no candidates, update best
        if candidates_mask == 0:
            if len(current_clique) > best_size_ref[0]:
                best_size_ref[0] = len(current_clique)
            return
        
        # Upper bound: current clique size + number of candidates
        # If this can't beat best, prune
        if len(current_clique) + bin(candidates_mask).count('1') <= best_size_ref[0]:
            return
        
        # Greedy coloring upper bound
        # Color candidates so that vertices of same color form an independent set
        # Max color number is an upper bound on clique size in candidates
        candidates_list = []
        m = candidates_mask
        pos = 0
        while m:
            if m & 1:
                candidates_list.append(pos + 1)
            m >>= 1
            pos += 1
        
        # Simple greedy coloring
        colors = {}
        max_color = 0
        for v in candidates_list:
            # Find the smallest color not used by any neighbor of v in candidates
            neighbor_colors = set()
            for u in candidates_list:
                if u != v and (compat[v] >> (u-1)) & 1:
                    if u in colors:
                        neighbor_colors.add(colors[u])
            c = 1
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            max_color = max(max_color, c)
        
        if len(current_clique) + max_color <= best_size_ref[0]:
            return
        
        # Branch: pick a vertex to include
        # Order candidates by color (process highest color first)
        ordered = sorted(candidates_list, key=lambda v: -colors[v])
        
        for v in ordered:
            if time.time() > deadline:
                return
            
            new_clique = current_clique + [v]
            if len(new_clique) > best_size_ref[0]:
                best_size_ref[0] = len(new_clique)
            
            # New candidates: intersection of old candidates with neighbors of v
            new_candidates = candidates_mask & compat[v]
            new_candidates &= ~(1 << (v - 1))
            
            max_clique_bb(new_candidates, new_clique, best_size_ref)
            
            # Remove v from candidates for next branch
            candidates_mask &= ~(1 << (v - 1))
            
            # Pruning: if remaining candidates can't beat best
            if len(current_clique) + bin(candidates_mask).count('1') <= best_size_ref[0]:
                return
    
    # --- Main search ---
    
    # Step 1: Get initial solution from best divisor set
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs
    
    best_div_set = set()
    best_div_size = 0
    for L in range(1, N + 1):
        divs = get_divisors(L)
        if len(divs) > best_div_size:
            best_div_size = len(divs)
            best_div_set = divs
    
    current_set = greedy_expand(set(best_div_set))
    current_size = len(current_set)
    
    # Step 2: Local search from this solution
    current_set = local_search_1out(current_set)
    current_size = len(current_set)
    
    # Step 3: Branch and bound to try to find a larger clique
    # We'll run BB from a few different starting points
    
    best_size_ref = [current_size]
    best_clique_ref = [list(current_set)]
    
    # Run BB with candidates being all vertices not in current solution
    # Actually, let's run BB on the full graph to find the global max clique
    # But that's expensive. Instead, let's use BB to verify/improve.
    
    # A better approach: use BB on a reduced candidate set
    # Start with the current solution's complement as candidates
    # and try to find a clique larger than current_size
    
    # Actually, let's just run a limited BB from scratch with good ordering
    # Use the greedy coloring bound for pruning
    
    # First, let's try BB starting from the best-degree greedy solution
    greedy_sol = greedy_clique()
    if len(greedy_sol) > current_size:
        current_set = set(greedy_sol)
        current_size = len(current_set)
        current_set = local_search_1out(current_set)
        current_size = len(current_set)
    
    best_size_ref[0] = current_size
    best_clique_ref[0] = list(current_set)
    
    # Run BB with a time limit
    # We'll branch on the full candidate set
    # But to make it tractable, let's use a smarter approach:
    # Start BB from the current best clique and try to extend it
    
    # Alternative: use BB to search for cliques of size > current_size
    # This is a decision problem: is there a clique of size k+1?
    
    # Let's try a different structural approach:
    # Use the fact that the compatibility graph has structure
    # Numbers that are divisors of a common L form a clique
    # The maximum such clique is the max number of divisors of any L <= 512
    
    # But there might be larger cliques that aren't all divisors of one L
    # Let's use BB to find the true maximum
    
    # For BB to be efficient, we need good bounds
    # Let's use a modified approach: start with the current best and try to improve
    
    # Try random restarts with BB
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Random ordering for greedy
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        sol = greedy_clique(order)
        sol = local_search_1out(set(sol))
        
        if len(sol) > current_size:
            current_set = sol
            current_size = len(current_set)
            best_size_ref[0] = current_size
            best_clique_ref[0] = list(current_set)
    
    # Final local search
    current_set = local_search_1out(current_set)
    current_size = len(current_set)
    
    # Try a few more BB attempts from the current best
    # Use BB to see if we can find a larger clique
    # Start with candidates = all vertices, but use the current best as a lower bound
    
    # Let's try BB with the current best as the initial lower bound
    # and search for a clique larger than that
    
    # For efficiency, let's limit the depth of BB
    def bb_limited(candidates_mask, current_clique, depth, max_depth, best_size_ref):
        """Limited depth BB."""
        if time.time() > deadline:
            return
        
        if depth > max_depth:
            return
        
        if candidates_mask == 0:
            if len(current_clique) > best_size_ref[0]:
                best_size_ref[0] = len(current_clique)
            return
        
        if len(current_clique) + bin(candidates_mask).count('1') <= best_size_ref[0]:
            return
        
        # Pick a vertex
        m = candidates_mask
        pos = 0
        first_v = -1
        while m:
            if m & 1:
                first_v = pos + 1
                break
            m >>= 1
            pos += 1
        
        if first_v == -1:
            return
        
        # Branch 1: include first_v
        new_clique = current_clique + [first_v]
        new_candidates = candidates_mask & compat[first_v]
        new_candidates &= ~(1 << (first_v - 1))
        bb_limited(new_candidates, new_clique, depth + 1, max_depth, best_size_ref)
        
        # Branch 2: exclude first_v
        candidates_mask &= ~(1 << (first_v - 1))
        if len(current_clique) + bin(candidates_mask).count('1') > best_size_ref[0]:
            bb_limited(candidates_mask, current_clique, depth + 1, max_depth, best_size_ref)
    
    # Try limited BB from a few different starting points
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Start with a random subset as initial clique
        init_clique = []
        candidates = all_mask
        for _ in range(5):  # Build a small initial clique
            if candidates == 0:
                break
            m = candidates
            pos = 0
            v = -1
            while m:
                if m & 1:
                    v = pos + 1
                    break
                m >>= 1
                pos += 1
            if v == -1:
                break
            init_clique.append(v)
            candidates = candidates & compat[v]
            candidates &= ~(1 << (v - 1))
        
        best_size_ref[0] = current_size
        bb_limited(candidates, init_clique, 0, 20, best_size_ref)
        
        if best_size_ref[0] > current_size:
            # We found a better size, but need to reconstruct the clique
            # This is tricky with the limited BB, so let's just note the size
            # and try to find it via local search
            pass
    
    # Final answer
    return list(current_set)
