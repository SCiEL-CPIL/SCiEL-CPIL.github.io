def _local_search(selected, time_limit):
    start = time.time()
    best = list(selected)
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, 193) if x not in best]
        
        # 1. Try adding (increase cardinality)
        if len(best) < 48:
            for x in candidates:
                if time.time() - start > time_limit:
                    break
                if _can_add(x, best):
                    best.append(x)
                    improved = True
                    break
        
        # 2. If full, try to maximize sum by swapping smaller for larger
        if not improved and len(best) >= 48:
            # Sort best to iterate from smallest to largest
            best_sorted = sorted(best)
            for a in best_sorted:
                if time.time() - start > time_limit:
                    break
                # Try to replace a with a larger x
                # Iterate candidates in descending order
                for x in sorted(candidates, reverse=True):
                    if x <= a:
                        continue
                    # Check if swapping a with x is valid
                    temp = [v for v in best if v != a]
                    if _can_add(x, temp):
                        # Replace a with x
                        best = [v for v in best if v != a]
                        best.append(x)
                        improved = True
                        break
                if improved:
                    break # Restart outer loop with updated best and candidates
            
        if not improved:
            break
    
    return best
