import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # 1. Build adjacency bitsets
    adj = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
        
        if time.time() > deadline:
            return [1, 2]

    # 2. Generate initial best clique from divisors of highly composite numbers
    def get_divisors(L):
        divs = []
        for k in range(1, int(L**0.5) + 1):
            if L % k == 0:
                divs.append(k)
                if k != L // k:
                    divs.append(L // k)
        return divs
    
    # Try several candidate L values with many divisors
    candidates_L = [480, 504, 420, 360, 336, 288, 240, 216, 180, 168, 144, 120, 96, 72, 60, 48, 40, 36, 30, 24, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    # Also scan for max divisor count
    best_div_count = 0
    best_L = 1
    for L in range(1, n + 1):
        d = get_divisors(L)
        if len(d) > best_div_count:
            best_div_count = len(d)
            best_L = L
    
    best_seed = get_divisors(best_L)
    best_clique_size = len(best_seed)
    best_clique = best_seed
    
    # 3. Bron-Kerbosch with bitsets and improved pivot
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_clique_size
        
        if time.time() > deadline:
            return
            
        # Pruning
        if len(R) + P.bit_count() <= best_clique_size:
            return

        if P == 0:
            if X == 0:
                if len(R) > best_clique_size:
                    best_clique_size = len(R)
                    best_clique = list(R)
            return

        # Pivot selection: choose u in P | X that maximizes (P & adj[u]).bit_count()
        pivot = -1
        max_count = -1
        
        # Iterate through bits in P | X
        px = P | X
        while px:
            lsb = px & -px
            u = lsb.bit_length()
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            px ^= lsb
            
        if pivot == -1:
            return

        # Candidates: P & ~adj[pivot]
        candidates = P & ~adj[pivot]
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length()
            
            # Recurse
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            R.append(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()
            
            # Update P and X
            P ^= lsb
            X |= lsb
            candidates ^= lsb
            
            if time.time() > deadline:
                return

    # Initial call
    initial_P = (1 << n) - 1
    initial_X = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    # 4. Greedy improvement: try to add nodes to best_clique using bitsets
    if time.time() < deadline - 2.0:
        improved_set = set(best_clique)
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for v in range(1, n + 1):
                if v in improved_set:
                    continue
                # Check if v is compatible with all in improved using bitset
                compat_with_all = True
                for u in improved_set:
                    if not (adj[v] >> (u - 1)) & 1:
                        compat_with_all = False
                        break
                if compat_with_all:
                    improved_set.add(v)
                    changed = True
                    if len(improved_set) > best_clique_size:
                        best_clique_size = len(improved_set)
                        best_clique = list(improved_set)
                    break  # Restart scan after adding
    
    # 5. Local search: try removing one element and adding two compatible ones
    if time.time() < deadline - 3.0:
        improved_set = set(best_clique)
        best_local_size = len(improved_set)
        
        for _ in range(50):  # Limited iterations
            if time.time() > deadline - 1.0:
                break
            
            progress = False
            # Try each removal
            for rem in list(improved_set):
                if len(improved_set) - 1 + 2 <= best_local_size:
                    continue  # Can't improve
                
                candidate_set = improved_set - {rem}
                
                # Find two compatible elements to add
                addable = []
                for v in range(1, n + 1):
                    if v in candidate_set:
                        continue
                    # Check compatibility with all in candidate_set
                    compat = True
                    for u in candidate_set:
                        if not (adj[v] >> (u - 1)) & 1:
                            compat = False
                            break
                    if compat:
                        addable.append(v)
                
                if len(addable) < 2:
                    continue
                
                # Try pairs from addable (limit to first few for speed)
                found_pair = False
                for i in range(min(len(addable), 30)):
                    for j in range(i + 1, min(len(addable), 30)):
                        a, b = addable[i], addable[j]
                        # Check a and b are compatible with each other
                        if (adj[a] >> (b - 1)) & 1:
                            new_set = candidate_set | {a, b}
                            if len(new_set) > best_local_size:
                                improved_set = new_set
                                best_local_size = len(new_set)
                                progress = True
                                found_pair = True
                                break
                    if found_pair:
                        break
            
            if not progress:
                break
        
        if best_local_size > best_clique_size:
            best_clique_size = best_local_size
            best_clique = list(improved_set)
    
    # 6. Additional local search: try swapping one element for a different one
    if time.time() < deadline - 2.0:
        improved_set = set(best_clique)
        best_swap_size = len(improved_set)
        
        for _ in range(30):  # Limited iterations
            if time.time() > deadline - 1.0:
                break
            
            progress = False
            # Try each removal
            for rem in list(improved_set):
                candidate_set = improved_set - {rem}
                
                # Find elements compatible with all in candidate_set
                addable = []
                for v in range(1, n + 1):
                    if v in candidate_set:
                        continue
                    # Check compatibility with all in candidate_set
                    compat = True
                    for u in candidate_set:
                        if not (adj[v] >> (u - 1)) & 1:
                            compat = False
                            break
                    if compat:
                        addable.append(v)
                
                if not addable:
                    continue
                
                # Try each addable element
                for v in addable:
                    new_set = candidate_set | {v}
                    if len(new_set) > best_swap_size:
                        improved_set = new_set
                        best_swap_size = len(new_set)
                        progress = True
                        break
            
            if not progress:
                break
        
        if best_swap_size > best_clique_size:
            best_clique_size = best_swap_size
            best_clique = list(improved_set)
    
    return best_clique
