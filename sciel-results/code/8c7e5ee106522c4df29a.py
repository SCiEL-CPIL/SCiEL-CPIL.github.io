import time

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    def can_add(S, x):
        # Check if adding x to S creates a 4-AP
        # A 4-AP is a, a+d, a+2d, a+3d with d != 0
        # If x is part of the AP, it can be at position 1, 2, 3, or 4.
        
        # Max possible d is N // 3
        max_d = N // 3
        
        for d in range(1, max_d + 1):
            # Case 1: x is the first term (a)
            # AP: x, x+d, x+2d, x+3d
            # Need x+d, x+2d, x+3d in S
            if (x + d <= N and x + d in S and 
                x + 2*d <= N and x + 2*d in S and 
                x + 3*d <= N and x + 3*d in S):
                return False

            # Case 2: x is the second term (a+d)
            # AP: x-d, x, x+d, x+2d
            # Need x-d, x+d, x+2d in S
            if (x - d >= 1 and x - d in S and 
                x + d <= N and x + d in S and 
                x + 2*d <= N and x + 2*d in S):
                return False

            # Case 3: x is the third term (a+2d)
            # AP: x-2d, x-d, x, x+d
            # Need x-2d, x-d, x+d in S
            if (x - 2*d >= 1 and x - 2*d in S and 
                x - d >= 1 and x - d in S and 
                x + d <= N and x + d in S):
                return False

            # Case 4: x is the fourth term (a+3d)
            # AP: x-3d, x-2d, x-d, x
            # Need x-3d, x-2d, x-d in S
            if (x - 3*d >= 1 and x - 3*d in S and 
                x - 2*d >= 1 and x - 2*d in S and 
                x - d >= 1 and x - d in S):
                return False

        return True

    def greedy_construction(order):
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected

    # Initial solution from greedy with natural order
    current_S = greedy_construction(range(1, N + 1))
    best_S = set(current_S)
    best_size = len(current_S)
    
    # Local Search
    # We will try to add elements that were rejected, or swap elements.
    # A simple strategy: 
    # 1. Keep a list of "rejected" elements.
    # 2. Periodically, try to add a rejected element. If it fails, try removing a random element from S and see if the rejected element can be added.
    
    rejected = [x for x in range(1, N + 1) if x not in current_S]
    
    import random
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        if not rejected:
            break
            
        # Shuffle rejected to avoid bias
        random.shuffle(rejected)
        
        made_progress = False
        
        for x in rejected:
            if time.time() - start_time > time_limit:
                break
                
            if can_add(current_S, x):
                current_S.add(x)
                rejected.remove(x)
                made_progress = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
                    
        if not made_progress:
            # Stuck. Try a swap.
            # Pick a random element y from current_S to remove.
            # Then try to add a random element x from rejected.
            # If successful, keep the change.
            
            # To be efficient, we can try a few swaps.
            # But removing an element might allow multiple additions.
            
            # Let's pick a random y to remove.
            if current_S:
                y = random.choice(list(current_S))
                current_S.remove(y)
                # Add y to rejected (it's already there if we managed to remove it from S? 
                # No, y was in S, so it was NOT in rejected. We need to add it back to rejected.)
                rejected.append(y)
                
                # Now try to add elements from rejected (which now includes y and others)
                # We should prioritize adding something OTHER than y to improve size, 
                # but if we can't, we might just be swapping.
                # Actually, if we just remove y, size decreases. We want to increase or maintain size.
                # So we try to add a new element z from rejected (z != y ideally, but could be y if we swap).
                # If we add z, size is same as before removal. If we add z and keep y out, size is same.
                # If we can add MORE than one, size increases.
                
                added_any = False
                # Try to add elements from rejected
                # We should probably try to add elements that are "easy" or just random.
                for z in rejected[:]: # Iterate over a copy
                    if z == y:
                        continue # We don't want to just put y back immediately unless necessary, but even then, it's a no-op.
                    if can_add(current_S, z):
                        current_S.add(z)
                        rejected.remove(z)
                        added_any = True
                        if len(current_S) > best_size:
                            best_size = len(current_S)
                            best_S = set(current_S)
                        break # Only add one at a time to keep control? Or add all? 
                        # Adding all might be better. Let's add all possible.
                        # But `can_add` depends on S. So we must re-check after each add.
                        # Let's loop again.
                
                # If we didn't add anything, we are at a local minimum or dead end.
                # We might need to restart or accept the smaller size temporarily to escape.
                # For now, let's just continue. The next iteration will try again.
                # If we removed y and added nothing, size is smaller. 
                # This is risky. We should only remove if we can add.
                # Let's refine: Only remove y if we can find a z to add.
                # This requires lookahead.
                
                # Simpler: Just try random swaps.
                # If size decreases, it's bad. 
                # Let's revert if we didn't add anything.
                if not added_any:
                    current_S.add(y)
                    rejected.remove(y)
                    
    return sorted(best_S)

# EVOLVE-BLOCK-END
