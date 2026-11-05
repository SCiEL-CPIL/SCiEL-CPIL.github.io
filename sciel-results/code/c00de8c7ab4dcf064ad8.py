# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Precompute comparability masks
    # comp_mask[i] has bit j set if smooth[i] and smooth[j] are comparable (one divides the other)
    comp_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(n_smooth):
            if i == j:
                mask |= (1 << j)
            elif smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                mask |= (1 << j)
        comp_mask[i] = mask
        
    def find_min_rep_bfs(n):
        """Find representation of n with minimum number of terms forming an antichain using BFS."""
        if n in smooth_set:
            return [n]
            
        # State: (current_sum, last_idx, forbidden_mask, path)
        # We use a queue for BFS to ensure minimum terms
        q = deque()
        q.append((0, -1, 0, ()))
        visited = set()
        
        while q:
            curr_sum, last_idx, forb_mask, path = q.popleft()
            
            # Try adding next term with index > last_idx
            for idx in range(last_idx + 1, n_smooth):
                # Check if this index is forbidden (comparable with any previously chosen term)
                if forb_mask & (1 << idx):
                    continue
                
                new_sum = curr_sum + smooth[idx]
                if new_sum > n:
                    break
                
                new_forb = forb_mask | comp_mask[idx]
                new_path = path + (idx,)
                
                if new_sum == n:
                    # Found a solution. Since BFS, this is the minimum number of terms.
                    return [smooth[i] for i in new_path]
                
                # Prune: if we've reached this state (sum, last_idx, forbidden_mask) before, 
                # we don't need to explore it again because the previous path was shorter or equal.
                key = (new_sum, idx, new_forb)
                if key not in visited:
                    visited.add(key)
                    q.append((new_sum, idx, new_forb, new_path))
                    
        return None

    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback if time runs out (should not happen with BFS for n<=128)
            # Try a simple greedy or return None (invalid)
            # But we need valid answers.
            # Let's try a quick DFS fallback just in case
            found = None
            if n in smooth_set:
                found = [n]
            else:
                # Simple backtracking fallback
                def backtrack(target, start_idx, current_sum, current_list, forb_mask):
                    if current_sum == target:
                        return current_list[:]
                    if current_sum > target:
                        return None
                    if len(current_list) >= 30:
                        return None
                    for idx in range(start_idx, n_smooth):
                        if forb_mask & (1 << idx):
                            continue
                        term = smooth[idx]
                        if current_sum + term > target:
                            break
                        new_forb = forb_mask | comp_mask[idx]
                        current_list.append(term)
                        res = backtrack(target, idx + 1, current_sum + term, current_list, new_forb)
                        if res:
                            return res
                        current_list.pop()
                    return None
                found = backtrack(n, 0, 0, [], 0)
                if found is None:
                    found = [1] # Should not happen
            answers.append(found)
            continue
            
        found = find_min_rep_bfs(n)
        if found is None:
            # Fallback if BFS fails (should not happen for 1..128)
            found = [n] if n in smooth_set else [1]
        answers.append(found)
        
    return answers
# EVOLVE-BLOCK-END
