import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    D = k * (k + 1) // 2
    LB = n_rows * D
    
    # Generate rulers
    # We generate up to LB + 100
    M_gen_limit = LB + 100
    
    rulers = [] # List of (max_diff, bitset, positions)
    
    def backtrack(pos, current_pos_list, current_diffs):
        if pos == k:
            max_d = current_pos_list[-1]
            if max_d <= M_gen_limit:
                bits = 0
                for d in current_diffs:
                    bits |= (1 << d)
                rulers.append((max_d, bits, list(current_pos_list)))
            return
        
        start_val = current_pos_list[-1] + 1
        # Pruning: even if we pick the smallest possible next values, 
        # can we reach k?
        # Not needed, we just iterate.
        
        for next_val in range(start_val, M_gen_limit + 1):
            new_diffs = []
            valid = True
            for i in range(pos):
                d = next_val - current_pos_list[i]
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            current_diffs.update(new_diffs)
            current_pos_list.append(next_val)
            
            backtrack(pos + 1, current_pos_list, current_diffs)
            
            current_pos_list.pop()
            for d in new_diffs:
                current_diffs.discard(d)
        
    backtrack(1, [0], set())
    
    # Sort rulers by max_diff
    rulers.sort(key=lambda x: x[0])
    
    # Extract unique max_diffs
    unique_max_diffs = sorted(set(r[0] for r in rulers))
    
    best_solution = None
    
    for M in unique_max_diffs:
        if M < LB:
            continue
        
        if time.time() > deadline - 10:
            break
        
        # Filter candidates
        candidates = []
        for r in rulers:
            if r[0] <= M:
                candidates.append(r)
            else:
                break
        
        if len(candidates) < n_rows:
            continue
        
        # DFS
        found = False
        solution = []
        
        def dfs(used_bits, idx, rows_found):
            nonlocal found, solution
            if time.time() > deadline - 5:
                return False
            
            if rows_found == n_rows:
                found = True
                return True
            
            # Prune
            if M - used_bits.bit_count() < (n_rows - rows_found) * D:
                return False
            
            for i in range(idx, len(candidates)):
                cand = candidates[i]
                max_d, bits, positions = cand
                
                if used_bits & bits:
                    continue
                
                new_used = used_bits | bits
                solution.append(positions)
                
                if dfs(new_used, i + 1, rows_found + 1):
                    return True
                
                solution.pop()
            
            return False
        
        if dfs(0, 0, 0):
            return solution
    
    # Fallback: Greedy
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
