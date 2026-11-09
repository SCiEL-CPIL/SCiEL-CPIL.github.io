"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(all_moduli[i], all_moduli[j])
            G[i][j] = g
            G[j][i] = g
    
    def is_compatible_idx(r_new, m_new_idx, chosen):
        """Check if (r_new, all_moduli[m_new_idx]) is compatible with chosen list of (idx, residue) pairs."""
        for idx, r in chosen:
            g = G[m_new_idx][idx]
            if r_new % g == r % g:
                return False
        return True
    
    def is_compatible(r_new, m_new, chosen_pairs):
        """Check if (r_new, m_new) is compatible with chosen list of (residue, modulus) pairs."""
        for r, m in chosen_pairs:
            g = gcd(m_new, m)
            if r_new % g == r % g:
                return False
        return True
    
    def verify_solution(pairs):
        """Verify a solution of (residue, modulus) pairs."""
        if not pairs:
            return False
        seen = set()
        for r, m in pairs:
            if m < 2 or m > MAX_MOD:
                return False
            if m in seen:
                return False
            seen.add(m)
            if r < 0 or r >= m:
                return False
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                r1, m1 = pairs[i]
                r2, m2 = pairs[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    def greedy_idx(order, res_orders):
        """Greedy construction using index-based representation."""
        chosen = []
        used = set()
        for i in order:
            if time.time() > deadline - 10:
                break
            if i in used:
                continue
            for r in res_orders[i]:
                if is_compatible_idx(r, i, chosen):
                    chosen.append((i, r))
                    used.add(i)
                    break
        return chosen
    
    def fill_idx(chosen, res_orders):
        """Try to add more moduli to the chosen set."""
        included = set(i for i, _ in chosen)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(n):
                if i in included:
                    continue
                for r in res_orders[i]:
                    if is_compatible_idx(r, i, chosen):
                        chosen.append((i, r))
                        included.add(i)
                        changed = True
                        break
        return chosen
    
    def improve_idx(chosen, res_orders, kicks=100):
        """Local search: try to swap out one element and refill."""
        included = set(i for i, _ in chosen)
        current_len = len(chosen)
        
        for _ in range(kicks):
            if time.time() > deadline - 5:
                break
            
            # Try to add a new modulus
            added = False
            for i in range(n):
                if i in included:
                    continue
                for r in res_orders[i]:
                    if is_compatible_idx(r, i, chosen):
                        chosen.append((i, r))
                        included.add(i)
                        added = True
                        break
                if added:
                    break
            
            if added:
                continue
            
            # Try removing one and refilling
            if len(chosen) < 2:
                break
            
            # Randomly pick an element to remove
            idx_to_remove = random.choice(list(included))
            chosen2 = [(i, r) for i, r in chosen if i != idx_to_remove]
            included2 = set(i for i, _ in chosen2)
            
            # Refill
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(n):
                    if i in included2:
                        continue
                    for r in res_orders[i]:
                        if is_compatible_idx(r, i, chosen2):
                            chosen2.append((i, r))
                            included2.add(i)
                            changed = True
                            break
            
            if len(chosen2) > len(chosen):
                chosen = chosen2
                included = included2
                current_len = len(chosen)
        
        return chosen
    
    def idx_to_pairs(chosen):
        """Convert index-based representation to (residue, modulus) pairs."""
        return [(r, all_moduli[i]) for i, r in chosen]
    
    # Natural residue orders
    nat_order = [list(range(m)) for m in all_moduli]
    
    best_result = []
    best_count = 0
    
    random.seed(42)
    
    # Phase 1: Greedy with multiple orderings
    orderings = []
    orderings.append(list(range(n)))
    orderings.append(list(range(n - 1, -1, -1)))
    orderings.append(sorted(range(n), key=lambda i: (-all_moduli[i].bit_count(), all_moduli[i])))
    orderings.append(sorted(range(n), key=lambda i: (all_moduli[i].bit_count(), all_moduli[i])))
    
    for _ in range(200):
        order = list(range(n))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        
        # Use natural residue order
        chosen = greedy_idx(order, nat_order)
        chosen = fill_idx(chosen, nat_order)
        chosen = improve_idx(chosen, nat_order, kicks=50)
        
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = idx_to_pairs(chosen)
    
    # Phase 2: More random restarts with shuffled residue orders
    while time.time() < deadline - 10:
        order = list(range(n))
        random.shuffle(order)
        
        # Shuffle residue orders
        res_orders = []
        for m in all_moduli:
            ro = list(range(m))
            random.shuffle(ro)
            res_orders.append(ro)
        
        chosen = greedy_idx(order, res_orders)
        chosen = fill_idx(chosen, res_orders)
        chosen = improve_idx(chosen, res_orders, kicks=80)
        
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = idx_to_pairs(chosen)
    
    # Phase 3: Final fill attempt on best result
    if time.time() < deadline - 3 and best_result:
        chosen_pairs = best_result[:]
        chosen_mod_set = set(m for _, m in chosen_pairs)
        remaining = [m for m in all_moduli if m not in chosen_mod_set]
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for m in remaining[:]:
                if time.time() > deadline - 3:
                    break
                for r in range(m):
                    if is_compatible(r, m, chosen_pairs):
                        chosen_pairs.append((r, m))
                        remaining.remove(m)
                        improved = True
                        break
        
        if len(chosen_pairs) > best_count:
            best_result = chosen_pairs
            best_count = len(chosen_pairs)
    
    # Final verification
    if best_result and verify_solution(best_result):
        return best_result
    
    # Fallback
    return [[1, 2]]
# EVOLVE-BLOCK-END
