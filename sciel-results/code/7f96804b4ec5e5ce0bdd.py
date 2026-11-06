import time

def construct():
    # Generate smooth numbers <= 128
    smooths = []
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128: break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128: break
            for c in range(0, 4):
                p5 = 5 ** c
                val = p2 * p3 * p5
                if val <= 128:
                    smooths.append(val)
    smooths = sorted(list(set(smooths)))
    
    # Precompute index map for quick lookup? Not strictly needed if we check values.
    
    def dfs(target, k, start_idx, current_sum, current_combo, visited_vals):
        # target: remaining sum to reach
        # k: max terms allowed (we search for min k, so we call with increasing k)
        # start_idx: index in smooths to start from (to ensure distinct and ordered)
        # current_sum: sum of current_combo
        # current_combo: list of values chosen so far
        # visited_vals: set of values in current_combo
        
        if target == 0:
            if len(current_combo) == k:
                # Check antichain property
                for i in range(len(current_combo)):
                    for j in range(i+1, len(current_combo)):
                        a, b = current_combo[i], current_combo[j]
                        if a % b == 0 or b % a == 0:
                            return None
                return list(current_combo)
            return None
        
        if target < 0:
            return None
            
        # Pruning: if we can't fill the remaining spots with min possible values
        # Min value is 1. But 1 divides everything, so if k>1, we can't use 1.
        # Actually, if we pick 1, no other number can be picked. So if k>1, 1 is forbidden.
        
        remaining_slots = k - len(current_combo)
        
        # Heuristic: we need to pick 'remaining_slots' numbers from smooths[start_idx:]
        # that sum to 'target'.
        
        for i in range(start_idx, len(smooths)):
            val = smooths[i]
            if val > target:
                break
            
            # Check if val divides any in current_combo or is divided by any
            # If so, skip
            compatible = True
            for cv in current_combo:
                if cv % val == 0 or val % cv == 0:
                    compatible = False
                    break
            if not compatible:
                continue
                
            # Also, if we add val, can we still reach the target with remaining_slots-1 numbers?
            # The max sum we can get from remaining_slots-1 numbers (largest smooths) should be >= target - val
            # The min sum we can get from remaining_slots-1 numbers (smallest smooths > val) should be <= target - val
            # This is a bit complex to bound tightly. Simple DFS is probably fast enough for N=128.
            
            current_combo.append(val)
            res = dfs(target - val, k, i + 1, current_sum + val, current_combo, visited_vals | {val})
            if res:
                return res
            current_combo.pop()
            
        return None

    results = []
    for n in range(1, 129):
        found = None
        for k in range(1, 11):
            # If k==1, just check if n is smooth
            if k == 1:
                if n in smooths:
                    found = [n]
                    break
                else:
                    continue
            
            # For k > 1, 1 cannot be used because it divides everything else.
            # So we can start from index 1 (value 2) if we want to avoid 1.
            # Actually, if we include 1, we can't include anything else.
            # So if k > 1, we must not pick 1.
            start = 1 # Index of 2, assuming smooths[0]=1
            # Ensure smooths[0] is 1
            if smooths[0] == 1:
                start_idx = 1
            else:
                start_idx = 0
                
            found = dfs(n, k, start_idx, 0, [], set())
            if found:
                break
        
        if found is None:
            # Fallback if not found (should not happen)
            found = []
            
        results.append(found)
        
    return results
