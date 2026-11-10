import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Precompute compatibility bitsets
    # compat_bits[i] has bit j set if lcm(i, j) <= 512
    compat_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                bits |= (1 << (j - 1))
        compat_bits[i] = bits
    
    # Precompute degrees for faster sorting
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = compat_bits[i].bit_count()
    
    # Greedy initialization with multiple strategies, take best
    def greedy_seed_strategy1():
        """Sort by degree descending"""
        S = set()
        S_bits = 0
        nodes_sorted = sorted(range(1, n + 1), key=lambda x: degrees[x], reverse=True)
        for x in nodes_sorted:
            if (compat_bits[x] & S_bits) == S_bits:
                S.add(x)
                S_bits |= (1 << (x - 1))
        return list(S)
    
    def greedy_seed_strategy2():
        """Sort by value descending (prefer larger numbers)"""
        S = set()
        S_bits = 0
        nodes_sorted = sorted(range(1, n + 1), key=lambda x: -x)
        for x in nodes_sorted:
            if (compat_bits[x] & S_bits) == S_bits:
                S.add(x)
                S_bits |= (1 << (x - 1))
        return list(S)
    
    def greedy_seed_strategy3():
        """Sort by degree ascending (prefer low-degree nodes first)"""
        S = set()
        S_bits = 0
        nodes_sorted = sorted(range(1, n + 1), key=lambda x: degrees[x])
        for x in nodes_sorted:
            if (compat_bits[x] & S_bits) == S_bits:
                S.add(x)
                S_bits |= (1 << (x - 1))
        return list(S)
    
    # Try multiple greedy strategies, keep best
    candidates = [greedy_seed_strategy1(), greedy_seed_strategy2(), greedy_seed_strategy3()]
    best_clique = max(candidates, key=len)
    best_size = len(best_clique)
    
    # Bron-Kerbosch with Pivot and Bitsets
    
    def bron_kerbosch(R, P_bits, X_bits):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        # Pruning
        P_count = P_bits.bit_count()
        if len(R) + P_count <= best_size:
            return
            
        if P_bits == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # --- Pivot Selection ---
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        PX_bits = P_bits | X_bits
        pivot = 0
        max_neighbors = -1
        temp_px = PX_bits
        while temp_px:
            lsb = temp_px & (-temp_px)
            u = lsb.bit_length() - 1 + 1
            temp_px ^= lsb
            neigh_count = (P_bits & compat_bits[u]).bit_count()
            if neigh_count > max_neighbors:
                max_neighbors = neigh_count
                pivot = u
        
        # We only need to branch on vertices in P \ N(pivot)
        if pivot:
            branch_bits = P_bits & ~compat_bits[pivot]
        else:
            branch_bits = P_bits
        
        if branch_bits == 0:
            return
            
        # --- Simple Upper Bound: just use |P| ---
        # For speed, skip coloring and use |P| as upper bound
        if len(R) + P_count <= best_size:
            return
            
        # Extract vertices from branch_bits for branching
        branch_vertices = []
        temp_b = branch_bits
        while temp_b:
            lsb = temp_b & (-temp_b)
            v = lsb.bit_length() - 1 + 1
            temp_b ^= lsb
            branch_vertices.append(v)
        
        # Branch on each vertex in branch_vertices
        for v in branch_vertices:
            if time.time() > deadline:
                return
            
            # Add v to R
            R.append(v)
            
            # New P = P ∩ N(v)
            new_P_bits = P_bits & compat_bits[v]
            new_P_bits &= ~(1 << (v - 1))  # Remove v
            
            # New X = X ∩ N(v)
            new_X_bits = X_bits & compat_bits[v]
            
            bron_kerbosch(R, new_P_bits, new_X_bits)
            
            R.pop()
            
            # Move v from P to X
            P_bits &= ~(1 << (v - 1))
            X_bits |= (1 << (v - 1))
            
    # Initial call
    initial_P_bits = 0
    for i in range(1, n + 1):
        initial_P_bits |= (1 << (i - 1))
    initial_X_bits = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P_bits, initial_X_bits)
    
    # Final polish: try to add any remaining compatible nodes
    if time.time() < deadline - 2.0:
        S_bits = 0
        for s in best_clique:
            S_bits |= (1 << (s - 1))
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for x in range(1, n + 1):
                if (1 << (x - 1)) & S_bits:
                    continue
                if (compat_bits[x] & S_bits) == S_bits:
                    best_clique.append(x)
                    S_bits |= (1 << (x - 1))
                    improved = True
                    break
    
    return best_clique
