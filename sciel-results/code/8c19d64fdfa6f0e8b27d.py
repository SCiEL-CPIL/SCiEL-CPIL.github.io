import time
import sys

def construct():
    # 1. Greedy Solution
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    greedy_len = len(values)
    best_seq = list(values)
    
    # 2. Optimized DFS with In-Place Backtracking
    # Try to find a sequence of length greedy_len + 1
    
    def find_sequence(target_len, time_limit):
        vals = []
        sums = set()
        calls = [0] # Mutable counter for time checks
        
        def dfs(depth, last_val):
            calls[0] += 1
            # Check time periodically to avoid overhead on every call
            if calls[0] % 1000 == 0 and time.time() > time_limit:
                return None
            
            if depth == target_len:
                return list(vals)
            
            # Pruning: Not enough numbers left in [1, 256]
            if 256 - last_val < (target_len - depth):
                return None
            
            for x in range(last_val + 1, 257):
                # Check collisions for the new tails
                tails = []
                total = x
                valid = True
                tails.append(total)
                
                if total in sums:
                    valid = False
                else:
                    # Check tails formed by adding previous elements in reverse order
                    for i in range(len(vals) - 1, -1, -1):
                        total += vals[i]
                        tails.append(total)
                        if total in sums:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Add to state (In-place)
                vals.append(x)
                sums.update(tails)
                
                res = dfs(depth + 1, x)
                if res:
                    return res
                
                # Backtrack (In-place)
                vals.pop()
                sums.difference_update(tails)
                
            return None
        
        return dfs(0, 0)

    # Set a time limit, leaving buffer for overhead
    time_limit = time.time() + 55
    
    # Try to extend the greedy length by 1
    target = greedy_len + 1
    res = find_sequence(target, time_limit)
    
    if res:
        return res
    
    # If search failed (timeout or no solution found), return greedy
    return best_seq
