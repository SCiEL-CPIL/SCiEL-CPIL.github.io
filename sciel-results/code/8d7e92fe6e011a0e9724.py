import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    def check_pair(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def check_all(classes):
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                if not check_pair(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                    return False
        return True
    
    # Seed solution: powers of 2 with residues 2^k - 1
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_count = len(seed)
    
    # All even moduli in [2, 128]
    even_moduli = [m for m in range(2, 129) if m % 2 == 0]
    n_mod = len(even_moduli)
    mod_to_idx = {m: i for i, m in enumerate(even_moduli)}
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(even_moduli[i], even_moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    def valid_residue(r, idx, assigned):
        """Check if residue r for modulus idx is valid given assigned residues."""
        for j, rj in assigned.items():
            g = gcd_mat[idx][j]
            if (r - rj) % g == 0:
                return False
        return True
    
    def greedy_build(order, time_limit=5.0):
        """Greedy: assign residues in given order."""
        t0 = time.time()
        assigned = {}  # idx -> residue
        for idx in order:
            if time.time() - t0 > time_limit:
                break
            m = even_moduli[idx]
            for r in range(m):
                if valid_residue(r, idx, assigned):
                    assigned[idx] = r
                    break
        return [[assigned[i], even_moduli[i]] for i in sorted(assigned.keys())]
    
    # Try multiple orderings
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count
    
    orderings = [
        list(range(n_mod)),  # ascending
        list(range(n_mod - 1, -1, -1)),  # descending
        sorted(range(n_mod), key=lambda i: num_divisors(even_moduli[i]), reverse=True),
        sorted(range(n_mod), key=lambda i: even_moduli[i] // 2, reverse=True),  # odd part desc
        sorted(range(n_mod), key=lambda i: even_moduli[i]),  # same as asc
    ]
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        result = greedy_build(order, time_limit=min(3.0, deadline - time.time() - 5))
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Randomized restarts
    random.seed(42)
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        shuffled = list(range(n_mod))
        random.shuffle(shuffled)
        result = greedy_build(shuffled, time_limit=1.0)
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Backtracking search on promising subsets
    def backtrack_search(moduli_list, time_limit=15.0):
        """Backtracking to find max valid assignment."""
        t0 = time.time()
        n = len(moduli_list)
        if n == 0:
            return []
        
        # Build local gcd matrix
        local_gcd = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = math.gcd(moduli_list[i], moduli_list[j])
                local_gcd[i][j] = g
                local_gcd[j][i] = g
        
        # Order: most constrained first (highest max gcd with others)
        def max_gcd_val(idx):
            return max(local_gcd[idx][j] for j in range(n) if j != idx) if n > 1 else 0
        
        order = sorted(range(n), key=max_gcd_val, reverse=True)
        
        best_result = []
        best_count_local = 0
        
        def bt(pos, current):
            nonlocal best_result, best_count_local
            if time.time() - t0 > time_limit:
                return
            if len(current) > best_count_local:
                best_count_local = len(current)
                best_result = list(current)
            
            if pos == n:
                return
            
            idx = order[pos]
            m = moduli_list[idx]
            
            # Try all residues
            for r in range(m):
                ok = True
                for (rj, mj) in current:
                    j = moduli_list.index(mj)
                    g = local_gcd[idx][j]
                    if (r - rj) % g == 0:
                        ok = False
                        break
                if ok:
                    current.append((r, m))
                    bt(pos + 1, current)
                    current.pop()
            
            # Also try skipping
            bt(pos + 1, current)
        
        bt(0, [])
        return [[r, m] for r, m in best_result]
    
    # Try backtracking on curated subsets
    if time.time() < deadline - 10:
        # Powers of 2 + some others
        powers_of_2 = [2, 4, 8, 16, 32, 64, 128]
        
        # Try: powers of 2 + multiples of 6
        subset1 = powers_of_2 + [m for m in even_moduli if m % 6 == 0 and m not in powers_of_2]
        if len(subset1) <= 25:
            result = backtrack_search(subset1, time_limit=min(8.0, deadline - time.time() - 5))
            if len(result) > best_count:
                best = result
                best_count = len(result)
        
        # Try: first 20 even moduli
        if time.time() < deadline - 10:
            subset2 = even_moduli[:20]
            result = backtrack_search(subset2, time_limit=min(8.0, deadline - time.time() - 5))
            if len(result) > best_count:
                best = result
                best_count = len(result)
        
        # Try: moduli with high gcd structure
        # Group by odd part
        if time.time() < deadline - 10:
            # All multiples of 4
            mult4 = [m for m in even_moduli if m % 4 == 0]
            if len(mult4) <= 30:
                result = backtrack_search(mult4[:25], time_limit=min(8.0, deadline - time.time() - 5))
                if len(result) > best_count:
                    best = result
                    best_count = len(result)
    
    # Final multi-start with local improvement
    if time.time() < deadline - 5:
        for seed_val in range(200):
            if time.time() > deadline - 2:
                break
            random.seed(seed_val)
            shuffled = list(range(n_mod))
            random.shuffle(shuffled)
            assigned = {}
            for idx in shuffled:
                m = even_moduli[idx]
                for r in range(m):
                    if valid_residue(r, idx, assigned):
                        assigned[idx] = r
                        break
            result = [[assigned[i], even_moduli[i]] for i in sorted(assigned.keys())]
            if len(result) > best_count:
                best = result
                best_count = len(result)
    
    # Verify final result
    if check_all(best):
        return best
    else:
        return seed
