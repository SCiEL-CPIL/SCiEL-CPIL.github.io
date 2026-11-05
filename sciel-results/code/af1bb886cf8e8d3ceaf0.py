import time
import itertools
from math import comb

def construct(n_rows, k):
    # Greedy fallback
    def greedy():
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

    rows_greedy = greedy()
    best_m = max(max(row) for row in rows_greedy)
    
    # Lower bound
    lb = n_rows * comb(k + 1, 2)
    
    # We will try to find a solution with M < best_m
    # Search range: lb to best_m - 1
    # If we find one, we return it.
    # If not, we return greedy.
    
    # Time management
    start_time = time.time()
    time_limit = 110.0 # 10s margin
    
    # Bitset size needs to be at least best_m
    # We will search for M from lb upwards
    # To be efficient, we can jump or just iterate.
    # Given the small n and k, iterating M is fine.
    
    # We use a recursive backtracking function
    # State: 
    #   row_idx: current row index (0 to n_rows-1)
    #   current_row: list of ints
    #   used_mask: integer bitset of used differences
    #   M: max value allowed
    
    # Optimization: 
    # Precompute nothing? 
    # We can prune if the number of available diffs is insufficient.
    # Total diffs needed = n_rows * comb(k+1, 2)
    # Diffs used so far = popcount(used_mask)
    # Diffs available in [1, M] = M
    # If (Total - Used) > (M - Used) i.e. Total > M, prune.
    # This is already covered by M >= LB.
    
    # Another prune: 
    # The current row has len(r) elements. It needs (k+1 - len(r)) more.
    # The max possible value for the next element is M.
    # The differences generated must be within [1, M].
    
    # Let's implement the search
    def search(M):
        # Returns a list of rows or None
        # We use a helper to check time
        def dfs(row_idx, current_row, used_mask):
            if time.time() - start_time > time_limit:
                return "TIMEOUT"
            
            if row_idx == n_rows:
                # All rows constructed
                return [current_row] # This is wrong, we need all rows
                # We need to store the rows.
                # Let's pass a list of rows.
            
            # This structure is awkward. Let's use a list `rows` passed by reference or closure.
            pass
        
        # Better approach:
        rows = []
        
        def dfs(row_idx, current_row, used_mask):
            if time.time() - start_time > time_limit:
                return "TIMEOUT"
            
            if row_idx == n_rows:
                # Success
                return True
            
            # If current row is complete, move to next row
            if len(current_row) == k + 1:
                rows.append(current_row)
                res = dfs(row_idx + 1, [0], used_mask)
                if res == "TIMEOUT":
                    return "TIMEOUT"
                if res:
                    return True
                rows.pop()
                return False
            
            # Determine next value x
            # x must be > current_row[-1] and <= M
            start_x = current_row[-1] + 1
            
            # Pruning: 
            # How many diffs do we need for the rest of this row?
            # Current len = L. Need k+1-L more elements.
            # Each new element adds L diffs.
            # Total new diffs for this row completion = sum_{i=L}^{k} i = (k+1-L)(k+L)/2 ? No.
            # If we add 1 element, we add L diffs.
            # If we add 2, we add L + (L+1) diffs.
            # ...
            # Total remaining diffs for this row = sum_{i=L}^{k} i where i is the number of existing elements when adding the new one?
            # When len is L, adding 1 makes len L+1. Diffs added = L.
            # When len is L+1, adding 1 makes len L+2. Diffs added = L+1.
            # ...
            # When len is k, adding 1 makes len k+1. Diffs added = k.
            # So total diffs to be added = L + (L+1) + ... + k.
            
            rem_diffs = 0
            for i in range(len(current_row), k):
                rem_diffs += i
            # Wait, if len is L, we need to add elements until len is k+1.
            # Next add makes len L+1, adds L diffs.
            # ...
            # Last add makes len k+1, adds k diffs.
            # So sum from i=L to i=k of i.
            
            # Check if enough diffs are available
            # Available diffs in [1, M] not in used_mask.
            # This is hard to count quickly with bitsets without popcount.
            # M - popcount(used_mask) is not exactly available because used_mask might have bits > M? No, we only add diffs <= M.
            # So available = M - popcount(used_mask).
            # But we also need to ensure the *specific* diffs we need can be formed.
            # Simple count prune:
            # if rem_diffs > M - bin(used_mask).count('1'): return False
            
            # Also, the smallest possible new difference is 1 (if 1 is not used and x - last = 1).
            # The largest is M.
            
            for x in range(start_x, M + 1):
                # Calculate new diffs
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d > M: break # Should not happen if x<=M and a>=0
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                new_mask = used_mask
                for d in new_diffs:
                    new_mask |= (1 << d)
                    
                res = dfs(row_idx, current_row + [x], new_mask)
                if res == "TIMEOUT":
                    return "TIMEOUT"
                if res:
                    return True
            
            return False
        
        res = dfs(0, [0], 0)
        if res == "TIMEOUT":
            return None
        if res:
            return rows
        return None

    # Try to find a better solution
    # We know best_m from greedy.
    # We try M from lb to best_m - 1.
    # If we find one, we return it.
    
    # To save time, we can try a few M values.
    # If lb is close to best_m, we try all.
    # If not, we might skip some? No, we want the minimum.
    
    # However, searching for M=112 might take forever if no solution exists.
    # We should limit the total search time.
    
    found = None
    for M in range(lb, best_m):
        if time.time() - start_time > time_limit - 10: # Stop 10s before limit
            break
        
        res = search(M)
        if res is not None:
            found = res
            break # Found the smallest M
    
    if found:
        return found
    else:
        return rows_greedy
