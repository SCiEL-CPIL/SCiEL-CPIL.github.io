import time
import random
from itertools import combinations
from collections import defaultdict
import numpy as np

def construct():
    start = time.time()
    DEADLINE = start + 55.0

    # All 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    n_all = len(all_sets)  # 1820
    
    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)
    
    def build_P(F):
        """Build 2D list P where P[i][j] = F[i] & F[j]"""
        K = len(F)
        P = [[0]*K for _ in range(K)]
        for i in range(K):
            for j in range(i+1, K):
                v = F[i] & F[j]
                P[i][j] = v
                P[j][i] = v
        return P

    def can_add_fast(C, F, P):
        """Check if C can be added. Returns (valid, conflict_pair)."""
        K = len(F)
        if K == 0:
            return True, None
        V = [C & f for f in F]
        val_to_idx = {}
        for i, v in enumerate(V):
            if v not in val_to_idx:
                val_to_idx[v] = []
            val_to_idx[v].append(i)
        
        for v, indices in val_to_idx.items():
            if len(indices) < 2:
                continue
            # Check pairs in indices
            # Optimization: if len(indices) is large, we might need to be careful
            for a in range(len(indices)):
                i = indices[a]
                Pi = P[i]
                for b in range(a+1, len(indices)):
                    j = indices[b]
                    if Pi[j] == v:
                        return False, (i, j)
        return True, None

    def greedy_pass(order, deadline):
        family = []
        P = []
        for C in order:
            if time.time() > deadline:
                break
            valid, _ = can_add_fast(C, family, P)
            if valid:
                family.append(C)
                K_old = len(family) - 1
                # Update P
                new_row = [0] * (K_old + 1)
                for i in range(K_old):
                    v = family[i] & C
                    P[i][K_old] = v
                    new_row[i] = v
                new_row[K_old] = 0
                P.append(new_row)
        return family, P

    best_family = []
    best_P = []
    best_size = 0
    
    # Phase 1: Multiple greedy passes
    # Natural order
    fam, P = greedy_pass(all_sets, DEADLINE - 15.0)
    if len(fam) > best_size:
        best_family = fam
        best_P = P
        best_size = len(fam)
    
    # Reverse order
    if time.time() < DEADLINE - 15.0:
        fam, P = greedy_pass(all_sets[::-1], DEADLINE - 15.0)
        if len(fam) > best_size:
            best_family = fam
            best_P = P
            best_size = len(fam)
    
    # Random orderings
    for _ in range(10):
        if time.time() > DEADLINE - 15.0:
            break
        order = all_sets[:]
        random.shuffle(order)
        fam, P = greedy_pass(order, DEADLINE - 15.0)
        if len(fam) > best_size:
            best_family = fam
            best_P = P
            best_size = len(fam)
            
    # Phase 2: Local Search
    def local_search(family, P, deadline):
        family = list(family)
        P = build_P(family) # Rebuild to be safe
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        random.shuffle(candidates)
        
        no_improve_count = 0
        max_no_improve = 100
        best_local = len(family)
        
        while time.time() < deadline and candidates:
            if no_improve_count > max_no_improve:
                random.shuffle(candidates)
                no_improve_count = 0
                max_no_improve = max(50, max_no_improve // 2)
            
            idx = random.randint(0, len(candidates) - 1)
            C = candidates[idx]
            
            valid, block_pair = can_add_fast(C, family, P)
            
            if valid:
                # Add it
                family.append(C)
                in_family.add(C)
                P = build_P(family) # Rebuild for simplicity, O(K^2)
                candidates.pop(idx)
                no_improve_count = 0
                if len(family) > best_local:
                    best_local = len(family)
                    max_no_improve = 100
            else:
                if block_pair is None:
                    no_improve_count += 1
                    continue
                    
                A_idx, B_idx = block_pair
                A = family[A_idx]
                B = family[B_idx]
                
                improved = False
                
                # Try 1-for-1 swap
                # Swap A with C
                # Check if C can be added if A is removed
                # This is equivalent to checking if C conflicts with any pair in (Family \ {A})
                # We can simulate by creating a temporary list?
                # Or just use can_add_fast on a modified list?
                # Since K is small, let's just build a temp family without A and check.
                
                def check_swap(fam, P, rem_idx, new_C):
                    # Create temp fam
                    temp_fam = [x for i, x in enumerate(fam) if i != rem_idx]
                    # We don't need P for temp_fam if we use can_add_fast which rebuilds V
                    # But can_add_fast expects P.
                    # Let's just use a simpler check for swap:
                    # C must not form sunflower with any pair in temp_fam.
                    # Pairs in temp_fam are the same as P, except those involving rem_idx.
                    # We can just iterate temp_fam pairs?
                    # Or just call can_add_fast on temp_fam?
                    # can_add_fast needs P.
                    # Let's write a helper that takes fam and C and checks.
                    K_temp = len(temp_fam)
                    V = [new_C & f for f in temp_fam]
                    val_to_idx = {}
                    for i, v in enumerate(V):
                        if v not in val_to_idx:
                            val_to_idx[v] = []
                        val_to_idx[v].append(i)
                    
                    # We need intersections of temp_fam pairs.
                    # We can compute them on the fly or precompute.
                    # On the fly is O(K^2) which is fine.
                    for i in range(K_temp):
                        for j in range(i+1, K_temp):
                            v_intersect = temp_fam[i] & temp_fam[j]
                            if V[i] == v_intersect and V[j] == v_intersect:
                                return False
                    return True

                if check_swap(family, P, A_idx, C):
                    family[A_idx] = C
                    in_family.discard(A)
                    in_family.add(C)
                    P = build_P(family)
                    candidates.pop(idx)
                    candidates.append(A)
                    improved = True
                    no_improve_count = 0
                    if len(family) > best_local:
                        best_local = len(family)
                else:
                    # Try swapping B with C
                    if check_swap(family, P, B_idx, C):
                        family[B_idx] = C
                        in_family.discard(B)
                        in_family.add(C)
                        P = build_P(family)
                        candidates.pop(idx)
                        candidates.append(B)
                        improved = True
                        no_improve_count = 0
                        if len(family) > best_local:
                            best_local = len(family)
                    else:
                        # Try 2-for-1 swap (reduce size by 1, might help later)
                        if len(family) >= 3:
                            # Check if C can be added if A and B are removed
                            # This is check_swap twice?
                            # Remove A, then check swap B with C in the resulting list?
                            # Or just check if C is valid in Family \ {A, B}
                            def check_swap2(fam, rem1, rem2, new_C):
                                temp_fam = [x for i, x in enumerate(fam) if i != rem1 and i != rem2]
                                K_temp = len(temp_fam)
                                V = [new_C & f for f in temp_fam]
                                val_to_idx = {}
                                for i, v in enumerate(V):
                                    if v not in val_to_idx:
                                        val_to_idx[v] = []
                                    val_to_idx[v].append(i)
                                
                                for i in range(K_temp):
                                    for j in range(i+1, K_temp):
                                        v_intersect = temp_fam[i] & temp_fam[j]
                                        if V[i] == v_intersect and V[j] == v_intersect:
                                            return False
                                return True

                            if check_swap2(family, A_idx, B_idx, C):
                                # Remove A and B, add C
                                # Rebuild family
                                new_fam = []
                                for i, x in enumerate(family):
                                    if i != A_idx and i != B_idx:
                                        new_fam.append(x)
                                new_fam.append(C)
                                in_family.discard(A)
                                in_family.discard(B)
                                in_family.add(C)
                                family = new_fam
                                P = build_P(family)
                                candidates.pop(idx)
                                candidates.append(A)
                                candidates.append(B)
                                improved = True
                                no_improve_count = 0
                            
                if not improved:
                    no_improve_count += 1
                    
        return family, P

    # Run local search on best family
    if best_family:
        improved_family, improved_P = local_search(best_family, best_P, DEADLINE - 5.0)
        if len(improved_family) > len(best_family):
            best_family = improved_family
            best_P = improved_P
            best_size = len(improved_family)
    
    # Phase 3: More rounds
    if time.time() < DEADLINE - 8.0:
        for _ in range(10):
            if time.time() > DEADLINE - 8.0:
                break
            order = all_sets[:]
            random.shuffle(order)
            fam, P = greedy_pass(order, DEADLINE - 8.0)
            if len(fam) > len(best_family):
                best_family = fam
                best_P = P
                best_size = len(fam)
            
            if time.time() < DEADLINE - 5.0:
                improved_fam, improved_P = local_search(best_family, best_P, DEADLINE - 5.0)
                if len(improved_fam) > len(best_family):
                    best_family = improved_fam
                    best_P = improved_P
                    best_size = len(improved_fam)
                    
    return [mask_to_list(m) for m in best_family]
