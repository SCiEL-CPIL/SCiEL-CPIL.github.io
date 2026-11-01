import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        ball_indices[i].append(i)
        temp = i
        for coord in range(n):
            orig = temp % q
            temp //= q
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster access
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    # Also create sets for fast intersection
    ball_indices_set = [set(b) for b in ball_indices]

    # Greedy Construction
    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = set()
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            
            for c in range(total):
                if c in code:
                    continue
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
                        
            if best_c == -1:
                break
                
            code.add(best_c)
            covered[ball_indices_np[best_c]] = True
            
        return code

    # Initial Greedy
    current_code = greedy_build()
    best_code = list(current_code)
    best_size = len(best_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed

    def find_criticals(code_set, cnt_arr):
        """Find codewords where min coverage in ball is 1."""
        criticals = []
        for c in code_set:
            ball = ball_indices_np[c]
            min_cov = np.min(cnt_arr[ball])
            if min_cov == 1:
                criticals.append(c)
        return criticals

    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """
        Try to swap c_remove with c_add.
        Assumes validity is checked externally or guaranteed by construction.
        """
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_indices_np[c_remove]] -= 1
        cnt_arr[ball_indices_np[c_add]] += 1
        return True

    # Local Search Loop
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        while run_delete_phase(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        # 2. Swap Phase
        swaps_attempted = 0
        max_swaps = 10000 # Can be higher now
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            
            if len(current_code) == 0:
                break
                
            # Find critical codewords
            criticals = find_criticals(current_code, cnt)
            
            if not criticals:
                break
                
            # Pick a random critical codeword
            c_remove = random.choice(criticals)
            
            # Get unique points for c_remove
            ball_rem = ball_indices_np[c_remove]
            unique_mask = cnt[ball_rem] == 1
            unique_points = ball_rem[unique_mask]
            
            if len(unique_points) == 0:
                # Should not happen if it's critical, but just in case
                continue
                
            # Find valid c_add candidates
            # c_add must be in intersection of balls of all unique_points
            # And not in current_code
            
            # Start with ball of first unique point
            candidates = ball_indices_set[unique_points[0]].copy()
            
            # Intersect with balls of other unique points
            for p in unique_points[1:]:
                candidates.intersection_update(ball_indices_set[p])
                
            # Filter out codewords already in code
            valid_adds = [c for c in candidates if c not in current_code]
            
            if not valid_adds:
                # No valid swap for this critical.
                # Try next critical? Or break?
                # Let's break to avoid wasting time.
                break
                
            # Pick a random valid add
            c_add = random.choice(valid_adds)
            
            # Perform swap
            try_swap(c_remove, c_add, current_code, cnt)
            
            # Run delete phase immediately to capture gains
            while run_delete_phase(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    
                    if best_size <= 160:
                        break
                        
            if best_size <= 160:
                break

        if best_size <= 160:
            break

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
