"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 128
    
    # U is fixed to {65, ..., 128}
    U = list(range(65, n + 1))
    candidates_L = list(range(1, 65))
    L_size = len(candidates_L)
    
    # Precompute conflict structure using bitsets
    # For each pair (i,j) in L candidates, determine what conflicts exist
    # We'll use integer bitsets where bit k represents candidate k (0-indexed)
    
    # Conflict types:
    # 1. Pair conflict: candidates i,j cannot both be in L if max(i+1,j+1) <= 63 and (i+1+j+1 >= 65 or i+1+j+1 is a candidate)
    #    Actually, the condition is: for pair (b,c) in L with max(b,c) <= 63, we need b+c < 65 AND b+c not in L
    #    So if max(b,c) <= 63 and b+c >= 65, then b and c conflict
    #    If max(b,c) <= 63 and b+c < 65, then b and c conflict only if b+c is also in L
    # 2. Triple conflict: candidates i,j,k cannot all be in L if all three pair sums are in S
    
    # Let's precompute for each pair (i,j) whether they have a "hard" conflict
    # Hard conflict: max(b,c) <= 63 and b+c >= 65
    # Soft conflict: max(b,c) <= 63 and b+c < 65 (conflicts only if b+c is in L)
    
    # For triples: (i,j,k) is forbidden if all three sums are in S
    # Sum s is in S iff s >= 65 or s is in L (i.e., s-1 is a candidate index and that candidate is selected)
    
    # Let's build a conflict matrix and triple conflict list
    # conflict[i][j] = True if i and j have a hard conflict
    # soft_conflict[i][j] = value v such that if v is in L, then i and j conflict
    # triple_conflicts[i][j][k] = True if (i,j,k) is a forbidden triple
    
    # Index: candidate value v has index v-1 (so value 1 -> index 0, value 64 -> index 63)
    
    conflict = [[False] * L_size for _ in range(L_size)]
    soft_conflict = [[None] * L_size for _ in range(L_size)]
    
    for i in range(L_size):
        b = i + 1
        for j in range(i + 1, L_size):
            c = j + 1
            M = max(b, c)
            if M <= 63:
                s = b + c
                if s >= 65:
                    conflict[i][j] = True
                    conflict[j][i] = True
                else:
                    # s < 65, so s is in S only if s is in L
                    # s in L means candidate s-1 is selected
                    if 1 <= s <= 64:
                        soft_conflict[i][j] = s - 1
                        soft_conflict[j][i] = s - 1
    
    # Precompute triple conflicts
    # triple_conflict[i][j] = set of k > j such that (i,j,k) is forbidden
    # Forbidden if: (b+c in S) AND (b+k in S) AND (c+k in S)
    # where b=i+1, c=j+1, k_val=k+1
    
    triple_conflicts = [set() for _ in range(L_size)]
    for i in range(L_size):
        b = i + 1
        for j in range(i + 1, L_size):
            c = j + 1
            bc = b + c
            bc_hard = bc >= 65
            bc_soft_idx = (bc - 1) if (1 <= bc <= 64) else None
            
            for k in range(j + 1, L_size):
                k_val = k + 1
                bk = b + k_val
                ck = c + k_val
                bk_hard = bk >= 65
                ck_hard = ck >= 65
                bk_soft_idx = (bk - 1) if (1 <= bk <= 64) else None
                ck_soft_idx = (ck - 1) if (1 <= ck <= 64) else None
                
                # Check if all three sums are in S
                # bc in S: bc_hard or (bc_soft_idx is not None and bc_soft_idx is selected)
                # bk in S: bk_hard or (bk_soft_idx is not None and bk_soft_idx is selected)
                # ck in S: ck_hard or (ck_soft_idx is not None and ck_soft_idx is selected)
                
                # We'll store the conditions for later evaluation
                # For now, mark as potentially conflicting
                triple_conflicts[i].add(k)
                triple_conflicts[j].add(k)
    
    # Bitset-based branch and bound
    # State: selected bitset, available bitset
    # We want to maximize |selected|
    
    best_size = 0
    best_selected = 0  # bitset
    
    # Precompute popcount for efficiency (use int.bit_count() in Python 3.10+)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    # Greedy initial solution
    def greedy_initial():
        selected = 0
        available = (1 << L_size) - 1
        while available:
            # Pick the candidate with fewest conflicts
            best = -1
            best_conflicts = L_size + 1
            for i in range(L_size):
                if not (available & (1 << i)):
                    continue
                # Count conflicts with already selected
                conf = 0
                for j in range(L_size):
                    if selected & (1 << j):
                        if conflict[i][j]:
                            conf += 1
                            break
                if conf == 0 and conf < best_conflicts:
                    # Check soft conflicts
                    soft_conf = False
                    for j in range(L_size):
                        if selected & (1 << j):
                            sc = soft_conflict[i][j]
                            if sc is not None and (selected & (1 << sc)):
                                soft_conf = True
                                break
                    if not soft_conf:
                        best = i
                        best_conflicts = -1
                        break
            if best == -1:
                # Find any non-conflicting candidate
                for i in range(L_size):
                    if not (available & (1 << i)):
                        continue
                    ok = True
                    for j in range(L_size):
                        if selected & (1 << j):
                            if conflict[i][j]:
                                ok = False
                                break
                            sc = soft_conflict[i][j]
                            if sc is not None and (selected & (1 << sc)):
                                ok = False
                                break
                    if ok:
                        best = i
                        break
            if best == -1:
                break
            selected |= (1 << best)
            available &= ~(1 << best)
        return selected
    
    initial = greedy_initial()
    best_size = popcount(initial)
    best_selected = initial
    
    # Branch and bound
    def branch_and_bound(selected, available, depth, time_limit):
        nonlocal best_size, best_selected
        
        if time.time() > time_limit:
            return
        
        current_size = popcount(selected)
        max_possible = current_size + popcount(available)
        if max_possible <= best_size:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_selected = selected
            return
        
        # Upper bound: current + available
        # Try to find a good candidate to branch on
        
        # Heuristic: pick candidate that appears in most potential conflicts
        best_i = -1
        best_score = -1
        for i in range(L_size):
            if not (available & (1 << i)):
                continue
            # Score: number of available candidates that conflict with i
            score = 0
            for j in range(L_size):
                if available & (1 << j):
                    if conflict[i][j]:
                        score += 1
            if score > best_score:
                best_score = score
                best_i = i
        
        if best_i == -1:
            # No available candidates
            if current_size > best_size:
                best_size = current_size
                best_selected = selected
            return
        
        # Branch: include best_i
        new_selected = selected | (1 << best_i)
        new_available = available & ~(1 << best_i)
        
        # Check if best_i conflicts with any selected
        valid_include = True
        for j in range(L_size):
            if selected & (1 << j):
                if conflict[best_i][j]:
                    valid_include = False
                    break
                sc = soft_conflict[best_i][j]
                if sc is not None and (selected & (1 << sc)):
                    valid_include = False
                    break
        
        if valid_include:
            # Check triple conflicts with newly added best_i
            triple_ok = True
            for k in range(L_size):
                if selected & (1 << k):
                    # Check if (best_i, k, ?) creates a forbidden triple
                    # Actually, we need to check if adding best_i creates any forbidden triple
                    # with two existing selected elements
                    pass  # This is complex; we'll do a full check if needed
            
            branch_and_bound(new_selected, new_available, depth + 1, time_limit)
        
        # Branch: exclude best_i
        branch_and_bound(selected, new_available, depth + 1, time_limit)
    
    # Run B&B with time limit
    time_limit = start_time + 80.0
    branch_and_bound(0, (1 << L_size) - 1, 0, time_limit)
    
    # Local search to improve
    def can_add(selected, x_idx):
        x = x_idx + 1
        # Check pair conflicts
        for j in range(L_size):
            if selected & (1 << j):
                if conflict[x_idx][j]:
                    return False
                sc = soft_conflict[x_idx][j]
                if sc is not None and (selected & (1 << sc)):
                    return False
        # Check triple conflicts
        # For each pair (j,k) in selected, check if (x, j, k) is forbidden
        for j in range(L_size):
            if not (selected & (1 << j)):
                continue
            b = x
            c = j + 1
            bc = b + c
            bc_in_S = (bc >= 65) or (1 <= bc <= 64 and (selected & (1 << (bc - 1))))
            if not bc_in_S:
                continue
            for k in range(L_size):
                if k == j or not (selected & (1 << k)):
                    continue
                d = k + 1
                bd = b + d
                cd = c + d
                bd_in_S = (bd >= 65) or (1 <= bd <= 64 and (selected & (1 << (bd - 1))))
                if not bd_in_S:
                    continue
                cd_in_S = (cd >= 65) or (1 <= cd <= 64 and (selected & (1 << (cd - 1))))
                if cd_in_S:
                    return False
        return True
    
    # Try to add elements
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for i in range(L_size):
            if best_selected & (1 << i):
                continue
            if can_add(best_selected, i):
                best_selected |= (1 << i)
                best_size += 1
                improved = True
    
    # Try swaps
    for _ in range(50):
        if time.time() > time_limit:
            break
        improved_swap = False
        for i in range(L_size):
            if not (best_selected & (1 << i)):
                continue
            for j in range(L_size):
                if best_selected & (1 << j):
                    continue
                # Try swapping i out and j in
                new_sel = (best_selected & ~(1 << i)) | (1 << j)
                # Check validity
                valid = True
                # Check all triples in new_sel
                for a in range(L_size):
                    if not (new_sel & (1 << a)):
                        continue
                    for b_idx in range(L_size):
                        if b_idx <= a or not (new_sel & (1 << b_idx)):
                            continue
                        for c_idx in range(L_size):
                            if c_idx <= b_idx or not (new_sel & (1 << c_idx)):
                                continue
                            av = a + 1
                            bv = b_idx + 1
                            cv = c_idx + 1
                            ab = av + bv
                            ac = av + cv
                            bc = bv + cv
                            ab_in_S = (ab >= 65) or (1 <= ab <= 64 and (new_sel & (1 << (ab - 1))))
                            if not ab_in_S:
                                continue
                            ac_in_S = (ac >= 65) or (1 <= ac <= 64 and (new_sel & (1 << (ac - 1))))
                            if not ac_in_S:
                                continue
                            bc_in_S = (bc >= 65) or (1 <= bc <= 64 and (new_sel & (1 << (bc - 1))))
                            if bc_in_S:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Also check pair conditions
                pair_ok = True
                for a in range(L_size):
                    if not (new_sel & (1 << a)):
                        continue
                    for b_idx in range(L_size):
                        if b_idx <= a or not (new_sel & (1 << b_idx)):
                            continue
                        av = a + 1
                        bv = b_idx + 1
                        M = max(av, bv)
                        if M <= 63:
                            s = av + bv
                            if (s >= 65) or (1 <= s <= 64 and (new_sel & (1 << (s - 1)))):
                                pair_ok = False
                                break
                    if not pair_ok:
                        break
                if pair_ok:
                    best_selected = new_sel
                    improved_swap = True
                    break
            if improved_swap:
                break
        if not improved_swap:
            break
    
    # Convert back to list
    L_list = []
    for i in range(L_size):
        if best_selected & (1 << i):
            L_list.append(i + 1)
    
    # Final verification
    L_set = set(L_list)
    valid = True
    # Check Condition 4: Three in L
    for i in range(len(L_list)):
        a = L_list[i]
        for j in range(i + 1, len(L_list)):
            b = L_list[j]
            ab = a + b
            ab_in_S = (ab >= 65) or (ab in L_set)
            if not ab_in_S:
                continue
            for k in range(j + 1, len(L_list)):
                c = L_list[k]
                ac = a + c
                ac_in_S = (ac >= 65) or (ac in L_set)
                if not ac_in_S:
                    continue
                bc = b + c
                bc_in_S = (bc >= 65) or (bc in L_set)
                if bc_in_S:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # Check Condition 3: One in U, two in L
        for i in range(len(L_list)):
            b = L_list[i]
            for j in range(i + 1, len(L_list)):
                c = L_list[j]
                M = max(b, c)
                if M <= 63:
                    bc = b + c
                    if (bc >= 65) or (bc in L_set):
                        valid = False
                        break
            if not valid:
                break
    
    if not valid:
        L_list = []
    
    S = U + L_list
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
