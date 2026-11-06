"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _prime_power_decomposition(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using gcd criterion."""
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Moduli we can choose from: 2..128
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute prime power decomposition for each modulus
    prime_decomp = []
    for m in moduli:
        prime_decomp.append(_prime_power_decomposition(m))
    
    # Group moduli by their set of prime factors (not exponents)
    # Two moduli with disjoint prime factor sets are automatically disjoint for any residues
    prime_sets = []
    for pd in prime_decomp:
        prime_sets.append(frozenset(pd.keys()))
    
    # Build conflict graph: edge between i,j if gcd(m_i, m_j) > 1
    # If gcd == 1, they're always disjoint regardless of residues
    # If gcd > 1, we need to choose residues carefully
    
    # For each pair with gcd > 1, precompute which residue pairs are disjoint
    # residue_disjoint[i][j] is a set of (r_i, r_j) pairs that are disjoint
    # But this is too large. Instead, for each modulus i, and each other modulus j,
    # we can precompute for each r_i, which r_j values are disjoint
    
    # Better approach: For each modulus m_idx, and for each residue r,
    # precompute a bitmask of which other moduli (among those with gcd>1)
    # have at least one residue that is disjoint from (r, m_idx)
    
    # Actually, let's use a different strategy:
    # 1. First, greedily select moduli that are "easy" (have few conflicts)
    # 2. For the remaining hard ones, use local search
    
    def count_conflicts(m_idx):
        """Count how many other moduli have gcd > 1 with m_idx."""
        count = 0
        for j in range(n_moduli):
            if j != m_idx and gcd_table[m_idx][j] > 1:
                count += 1
        return count
    
    # Sort moduli by number of conflicts (ascending) - easy ones first
    order_by_conflicts = sorted(range(n_moduli), key=lambda i: count_conflicts(i))
    
    def can_add(selected, r_new, m_idx_new):
        """Check if (r_new, moduli[m_idx_new]) is disjoint from all selected."""
        for (r_sel, m_idx_sel) in selected:
            g = gcd_table[m_idx_sel][m_idx_new]
            if (r_new - r_sel) % g == 0:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, pick first valid residue for each."""
        selected = []
        for m_idx in order:
            m = moduli[m_idx]
            best_r = None
            # Try residues in a smart order: prefer residues that conflict with fewer future moduli
            # For simplicity, just try all residues and pick the first valid one
            for r in range(m):
                if can_add(selected, r, m_idx):
                    best_r = r
                    break
            if best_r is not None:
                selected.append((best_r, m_idx))
        return selected
    
    def try_add_to_solution(selected):
        """Try to add any missing modulus with a valid residue. Returns new solution or None."""
        selected_set = set(si[1] for si in selected)
        # Try moduli in order of fewest conflicts first
        candidates = [i for i in range(n_moduli) if i not in selected_set]
        candidates.sort(key=lambda i: count_conflicts(i))
        for m_idx in candidates:
            m = moduli[m_idx]
            for r in range(m):
                if can_add(selected, r, m_idx):
                    return selected + [(r, m_idx)]
        return None
    
    def local_search_swap(selected, max_iter=10000):
        """Try to improve by swapping residues to enable more additions."""
        best = selected[:]
        best_len = len(best)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new modulus
            new_sol = try_add_to_solution(best)
            if new_sol is not None:
                best = new_sol
                best_len += 1
                improved = True
                continue
            
            # Try swapping a residue to enable future additions
            for i, (r_old, m_idx) in enumerate(best):
                m = moduli[m_idx]
                # Try a subset of residues to save time
                step = max(1, m // 20)
                for r_new in range(0, m, step):
                    if r_new == r_old:
                        continue
                    # Check if new residue is disjoint from all others
                    ok = True
                    for j, (r_j, m_idx_j) in enumerate(best):
                        if i == j:
                            continue
                        g = gcd_table[m_idx][m_idx_j]
                        if (r_new - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        # Try this swap and see if it enables an addition
                        test_best = best[:]
                        test_best[i] = (r_new, m_idx)
                        new_sol = try_add_to_solution(test_best)
                        if new_sol is not None:
                            best = new_sol
                            best_len += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best
    
    # Strategy 1: Greedy with moduli sorted by conflict count
    sol1 = greedy_construct(order_by_conflicts)
    sol1 = local_search_swap(sol1, max_iter=5000)
    
    if time.time() > deadline:
        return [[r, moduli[m_idx]] for r, m_idx in sol1]
    
    # Strategy 2: Greedy with moduli in ascending order
    order_asc = list(range(n_moduli))
    sol2 = greedy_construct(order_asc)
    sol2 = local_search_swap(sol2, max_iter=5000)
    
    if time.time() > deadline:
        best_sol = max([sol1, sol2], key=len)
        return [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Strategy 3: Multiple random orderings
    random.seed(42)
    best_sol = max([sol1, sol2], key=len)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        random_order = list(range(n_moduli))
        random.shuffle(random_order)
        sol_r = greedy_construct(random_order)
        sol_r = local_search_swap(sol_r, max_iter=3000)
        if len(sol_r) > len(best_sol):
            best_sol = sol_r
    
    # Final local search on the best solution
    best_sol = local_search_swap(best_sol, max_iter=8000)
    
    # Convert to required format
    result = [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Ensure nonempty
    if not result:
        return [[1, 2]]
    
    return result
# EVOLVE-BLOCK-END
