import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 90s limit, 15s safety margin
    
    n = 128
    target_start = 128
    target_end = 256
    num_targets = target_end - target_start + 1  # 129 targets
    
    # Precompute for each element a, which targets it affects and how
    # When we add/remove element a, the multiplicities change for targets t where a+b=t or b+a=t
    # For unordered pairs: adding a affects target t if there exists b in A such that a+b=t
    # The multiplicity of target t changes by: number of b in A with b = t-a (if a != b) or 1 if a==b and 2a=t
    
    # Precompute: for each a in [0,128] and each target t in [128,256],
    # what is the "partner" b = t-a, and is it valid?
    partner = np.full((129, num_targets), -1, dtype=np.int32)
    for a in range(129):
        for i, t in enumerate(range(target_start, target_end + 1)):
            b = t - a
            if 0 <= b <= 128:
                partner[a, i] = b
    
    def compute_mults(A_set):
        """Compute multiplicity array for all targets. O(|A|^2) but only for active elements."""
        mults = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for j in range(i, len(A_list)):
                b = A_list[j]
                t = a + b
                if target_start <= t <= target_end:
                    idx = t - target_start
                    mults[idx] += 1
        return mults
    
    def get_cost(mults):
        """Compute (max_mult, num_uncovered) from mults array."""
        unc = np.sum(mults == 0)
        mx = np.max(mults) if len(mults) > 0 else 0
        return int(mx), int(unc)
    
    def evaluate(A_set):
        """Returns (max_mult, num_uncovered, total_pairs)"""
        if not A_set:
            return 1000, 100, 0
        mults = compute_mults(A_set)
        mx, unc = get_cost(mults)
        total_pairs = len(A_set) * (len(A_set) + 1) // 2
        return mx, unc, total_pairs
    
    def incremental_add(mults, A_set, a):
        """Update mults after adding element a. Returns new mults (copy)."""
        new_mults = mults.copy()
        for b in A_set:
            t = a + b
            if target_start <= t <= target_end:
                idx = t - target_start
                new_mults[idx] += 1
        # Also a+a
        t2 = 2 * a
        if target_start <= t2 <= target_end:
            idx2 = t2 - target_start
            new_mults[idx2] += 1
        return new_mults
    
    def incremental_remove(mults, A_set, a):
        """Update mults after removing element a. Returns new mults (copy)."""
        new_mults = mults.copy()
        for b in A_set:
            if b == a:
                continue
            t = a + b
            if target_start <= t <= target_end:
                idx = t - target_start
                new_mults[idx] -= 1
        # a+a
        t2 = 2 * a
        if target_start <= t2 <= target_end:
            idx2 = t2 - target_start
            new_mults[idx2] -= 1
        return new_mults
    
    def local_search(A_set, max_iters=2000, T_init=3.0, T_min=0.05, cooling=0.995):
        """Fast local search with incremental evaluation."""
        current_A = set(A_set)
        if 128 not in current_A:
            current_A.add(128)
        
        cur_mults = compute_mults(current_A)
        cur_max, cur_unc = get_cost(cur_mults)
        cur_cost = cur_unc * 10000 + cur_max
        best_A = set(current_A)
        best_cost = cur_cost
        best_mults = cur_mults.copy()
        
        T = T_init
        it = 0
        stuck_count = 0
        
        while it < max_iters and time.time() < deadline:
            it += 1
            move_type = random.random()
            new_A = set(current_A)
            
            if move_type < 0.35 and len(current_A) > 2:
                # Remove
                x = random.choice(list(current_A))
                new_A.remove(x)
                if not new_A:
                    continue
                new_mults = incremental_remove(cur_mults, current_A, x)
            elif move_type < 0.75:
                # Add
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A.add(x)
                new_mults = incremental_add(cur_mults, current_A, x)
            else:
                # Swap
                if len(current_A) < 1:
                    continue
                x = random.choice(list(current_A))
                y = random.randint(0, 128)
                if y == x or y in current_A:
                    continue
                new_A.remove(x)
                new_A.add(y)
                # Remove x then add y
                tmp_mults = incremental_remove(cur_mults, current_A, x)
                new_mults = incremental_add(tmp_mults, new_A - {y}, y)
            
            new_max, new_unc = get_cost(new_mults)
            new_cost = new_unc * 10000 + new_max
            delta = new_cost - cur_cost
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_A = new_A
                cur_mults = new_mults
                cur_cost = new_cost
                cur_max, cur_unc = get_cost(cur_mults)
                stuck_count = 0
                
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_A = set(current_A)
                    best_mults = cur_mults.copy()
            else:
                stuck_count += 1
            
            T = max(T * cooling, T_min)
        
        return best_A, best_cost
    
    def greedy_construct():
        """Greedy construction: start with {128}, add elements to cover uncovered targets."""
        A = {128}
        mults = compute_mults(A)
        
        # Priority: targets with 0 representations first, then minimize max
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Find best element to add
            best_elem = None
            best_cost = None
            cur_max, cur_unc = get_cost(mults)
            cur_cost = cur_unc * 10000 + cur_max
            
            # Try each candidate
            candidates = list(range(0, 129))
            random.shuffle(candidates)
            
            for a in candidates:
                if a in A:
                    continue
                new_mults = incremental_add(mults, A, a)
                new_max, new_unc = get_cost(new_mults)
                new_cost = new_unc * 10000 + new_max
                if best_cost is None or new_cost < best_cost:
                    best_cost = new_cost
                    best_elem = a
                    best_mults = new_mults
            
            if best_elem is None or best_cost >= cur_cost:
                break
            
            A.add(best_elem)
            mults = best_mults
        
        return A
    
    def polish_remove(A_set):
        """Try to remove elements that don't hurt coverage or increase max."""
        A = set(A_set)
        mults = compute_mults(A)
        cur_max, cur_unc = get_cost(mults)
        cur_cost = cur_unc * 10000 + cur_max
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in sorted(A, reverse=True):
                if time.time() > deadline - 3:
                    break
                new_mults = incremental_remove(mults, A, x)
                new_max, new_unc = get_cost(new_mults)
                new_cost = new_unc * 10000 + new_max
                if new_cost <= cur_cost:
                    A.remove(x)
                    mults = new_mults
                    cur_cost = new_cost
                    cur_max, cur_unc = get_cost(mults)
                    changed = True
        
        return A, cur_cost
    
    # Main search
    best_overall_A = set()
    best_overall_cost = float('inf')
    
    # Phase 1: Greedy construction + polish
    greedy_A = greedy_construct()
    polished_A, polished_cost = polish_remove(greedy_A)
    
    if polished_cost < best_overall_cost:
        best_overall_cost = polished_cost
        best_overall_A = polished_A
    
    # Phase 2: Local search from greedy solution
    ls_A, ls_cost = local_search(polished_A, max_iters=3000)
    if ls_cost < best_overall_cost:
        best_overall_cost = ls_cost
        best_overall_A = ls_A
    
    # Phase 3: Multiple random restarts with local search
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        init_A = {128}
        # Add some structured elements
        for _ in range(random.randint(10, 25)):
            init_A.add(random.randint(0, 128))
        
        # Quick greedy to ensure coverage
        greedy_from_init = greedy_construct()
        # Merge with init
        merged = set(init_A) | set(greedy_from_init)
        
        # Polish
        pol_A, pol_cost = polish_remove(merged)
        
        # Local search
        ls_A2, ls_cost2 = local_search(pol_A, max_iters=1500)
        
        if ls_cost2 < best_overall_cost:
            best_overall_cost = ls_cost2
            best_overall_A = ls_A2
    
    # Phase 4: Final intensive polish on best solution
    if time.time() < deadline - 5:
        final_A, final_cost = local_search(best_overall_A, max_iters=5000, T_init=1.0, T_min=0.01, cooling=0.998)
        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_overall_A = final_A
        
        # Remove redundant elements
        final_A, final_cost = polish_remove(best_overall_A)
        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_overall_A = final_A
    
    # Fallback
    if not best_overall_A:
        best_overall_A = set(range(64, 129))
        best_overall_A.add(128)
    
    # Ensure 128 is in the set
    if 128 not in best_overall_A:
        best_overall_A.add(128)
    
    return sorted(best_overall_A)
