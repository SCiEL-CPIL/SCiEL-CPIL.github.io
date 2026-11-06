import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    N = 192
    universe = list(range(1, N + 1))
    universe_set = set(universe)
    
    def is_valid(S):
        """Check if set S is valid: no element divides sum of any 3 distinct others."""
        S_list = sorted(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            K_others = len(others)
            if K_others < 3:
                continue
            
            # For fixed a, check if there exist distinct x,y,z in others with (x+y+z) % a == 0
            # Use pair sum approach with careful distinctness check
            pair_map = {}
            for i in range(K_others):
                xi = others[i]
                for j in range(i + 1, K_others):
                    xj = others[j]
                    s = (xi + xj) % a
                    if s not in pair_map:
                        pair_map[s] = []
                    pair_map[s].append((i, j))
            
            for z_idx, z in enumerate(others):
                target = (-z) % a
                if target in pair_map:
                    # Check if any pair (i, j) in pair_map[target] is distinct from z_idx
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True
    
    def can_add_incremental(a, selected):
        """
        Check if adding a to selected maintains validity.
        Only need to check:
        1. a does not divide sum of any 3 distinct elements in selected
        2. For each x in selected, x does not divide (a + b + c) for any distinct b,c in selected\{x,a}
        """
        sel = selected
        n = len(sel)
        
        # Check 1: a does not divide sum of any 3 distinct elements in sel
        if n >= 3:
            # Build: for each residue r mod a, list of elements in sel with that residue
            res_to_elems = {}
            for v in sel:
                r = v % a
                if r not in res_to_elems:
                    res_to_elems[r] = []
                res_to_elems[r].append(v)
            
            # Now for each pair (b,c) in sel:
            for i in range(n):
                b = sel[i]
                for j in range(i + 1, n):
                    c = sel[j]
                    target_res = (-(b + c)) % a
                    if target_res in res_to_elems:
                        # Check if any element d in res_to_elems[target_res] is distinct from b and c
                        for d in res_to_elems[target_res]:
                            if d != b and d != c:
                                return False
        
        # Check 2: For each x in sel, x does not divide (a + b + c) for distinct b,c in sel\{x,a}
        for x in sel:
            if x == a:
                continue
            # Need b,c in sel, b != c, b != x, c != x, b != a, c != a
            # Such that (a + b + c) % x == 0
            # i.e., (b + c) % x == (-a) % x
            target_res = (-a) % x
            
            filtered = [v for v in sel if v != x and v != a]
            m = len(filtered)
            if m < 2:
                continue
            
            # Build residue counts and element lists for filtered
            res_to_elems2 = {}
            for v in filtered:
                r = v % x
                if r not in res_to_elems2:
                    res_to_elems2[r] = []
                res_to_elems2[r].append(v)
            
            for b in filtered:
                r_b = b % x
                r_c_needed = (target_res - r_b) % x
                if r_c_needed in res_to_elems2:
                    for c in res_to_elems2[r_c_needed]:
                        if c != b:
                            return False
        
        return True
    
    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected = []
        for x in order:
            if time.time() - start_time >= time_limit:
                break
            if can_add_incremental(x, selected):
                selected.append(x)
        return set(selected)
    
    def local_search(S, max_rounds=100):
        """Improve set S via local search (additions and swaps)."""
        S = set(S)
        for _ in range(max_rounds):
            if time.time() - start_time >= time_limit:
                break
            
            U_not_S = universe_set - S
            if not U_not_S:
                break
            
            # Try to add one element
            added = False
            for x in sorted(U_not_S, reverse=True):
                if time.time() - start_time >= time_limit:
                    break
                if can_add_incremental(x, list(S)):
                    S.add(x)
                    added = True
                    break
            
            if added:
                continue
            
            # If we cannot add, try a single swap
            if not S:
                break
            
            U_not_S_list = list(U_not_S)
            if not U_not_S_list:
                break
            
            # Try several random swaps
            attempts = 30
            swapped = False
            for _ in range(attempts):
                if time.time() - start_time >= time_limit:
                    break
                remove_candidate = random.choice(list(S))
                add_candidate = random.choice(U_not_S_list)
                new_S = (S - {remove_candidate}) | {add_candidate}
                if is_valid(new_S):
                    S = new_S
                    swapped = True
                    # After a swap, immediately try to add again
                    for x in sorted(universe_set - S, reverse=True):
                        if time.time() - start_time >= time_limit:
                            break
                        if can_add_incremental(x, list(S)):
                            S.add(x)
                            break
                    break
        
        return S
    
    # Phase 1: Multiple greedy starts with different orders
    best_S = set()
    best_size = 0
    
    # Greedy 1: Descending order
    S1 = greedy_construct(range(N, 0, -1))
    best_S = S1
    best_size = len(S1)
    
    # Greedy 2: Ascending order
    if time.time() - start_time < time_limit:
        S2 = greedy_construct(range(1, N + 1))
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
    
    # Greedy 3-8: Random permutations
    for seed in range(6):
        if time.time() - start_time >= time_limit:
            break
        random.seed(seed * 42 + 7)
        perm = list(universe)
        random.shuffle(perm)
        S = greedy_construct(perm)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Phase 2: Local search from best greedy result
    if time.time() - start_time < time_limit:
        S_improved = local_search(best_S)
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
    
    # Phase 3: More local search iterations from multiple starting points
    for seed in range(5):
        if time.time() - start_time >= time_limit:
            break
        random.seed(seed * 123 + 11)
        perm = list(universe)
        random.shuffle(perm)
        S = greedy_construct(perm)
        S_improved = local_search(S)
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
    
    # Final: return the best set found
    # Ensure we return a valid set
    if not is_valid(best_S):
        # Fallback: rebuild greedily
        S_fb = greedy_construct(range(N, 0, -1))
        best_S = S_fb
    
    return list(best_S)
