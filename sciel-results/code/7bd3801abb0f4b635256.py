import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check if y can be added to S
    def can_add(y, S, sums):
        # New sums: 2y, y+s for s in S
        # Check if 2y is in sums
        if 2 * y in sums:
            return False
        # Check if y+s is in sums for any s in S
        # Also check if new sums collide with each other
        # Since S is Sidon, s1+s2 are distinct.
        # We need 2y != y+s1 => y != s1 (true since y not in S)
        # We need y+s1 != y+s2 => s1 != s2 (true)
        # So we just need to check against existing sums.
        for s in S:
            if (y + s) in sums:
                return False
        return True

    def get_addable(S, sums):
        addable = []
        for y in range(1, N + 1):
            if y in S:
                continue
            if can_add(y, S, sums):
                addable.append(y)
        return addable

    def greedy_maximal_trial(seed):
        random.seed(seed)
        S = set()
        sums = set()
        
        while True:
            # Check time
            if time.time() > deadline:
                return None
            
            addable = get_addable(S, sums)
            if not addable:
                return list(S)
            
            # Heuristic: Pick x in addable that blocks the most other addable elements
            # "Blocking" y means that if we add x, y becomes non-addable.
            # We approximate this by checking conflicts directly.
            
            best_x = None
            max_blocks = -1
            # Randomize tie-breaking
            random.shuffle(addable)
            
            for x in addable:
                # Simulate adding x
                new_sums = sums.copy()
                new_sums.add(2 * x)
                for s in S:
                    new_sums.add(x + s)
                
                blocks = 0
                for y in addable:
                    if y == x:
                        continue
                    # Check if y is still addable after adding x
                    # y is not addable if:
                    # 1. 2y in new_sums
                    # 2. y+s in new_sums for some s in S U {x}
                    
                    if 2 * y in new_sums:
                        blocks += 1
                        continue
                        
                    blocked = False
                    for s in S:
                        if (y + s) in new_sums:
                            blocked = True
                            break
                    if not blocked and (y + x) in new_sums:
                        blocked = True
                    if not blocked:
                        # y+x is the new sum. Is it in new_sums?
                        # new_sums contains x+s for s in S, and 2x.
                        # y+x is not in {2x} unless y=x (handled).
                        # y+x is not in {x+s} unless y=s (y not in S).
                        # So y+x is a new sum.
                        # But we must check if y+x collides with existing sums in new_sums?
                        # Wait, the condition for y to be addable to S' = S U {x} is:
                        # All sums involving y in S' must be distinct from sums in S'.
                        # Sums in S' are sums(S) U {2x} U {x+s | s in S}.
                        # New sums for y: 2y, y+x, y+s (s in S).
                        # We already checked 2y.
                        # We already checked y+s against new_sums.
                        # We need to check y+x against new_sums.
                        # y+x is in new_sums if y+x = 2x (y=x, no) or y+x = x+s (y=s, no).
                        # So y+x is always distinct from existing sums in new_sums.
                        # However, we must ensure y+x is distinct from OTHER new sums?
                        # No, the sums of S' are fixed. We just check if the new sums for y collide with S' sums.
                        # So if y+x is not in new_sums, and y+s not in new_sums, and 2y not in new_sums, y is addable.
                        pass 
                    
                    if not blocked:
                        # y is still addable
                        pass
                    else:
                        blocks += 1

                if blocks > max_blocks:
                    max_blocks = blocks
                    best_x = x
            
            if best_x is None:
                best_x = addable[0]
            
            S.add(best_x)
            sums.add(2 * best_x)
            for s in S:
                if s != best_x:
                    sums.add(best_x + s)

    # Run multiple trials
    trial_count = 0
    while time.time() < deadline:
        trial_count += 1
        result = greedy_maximal_trial(trial_count)
        if result is not None:
            if len(result) < best_size:
                best_size = len(result)
                best_set = result
                
        # If we find a very small set, we can stop earlier or continue to be sure
        if best_size <= 12: # Theoretical lower bound is ~ sqrt(N) / 2 ? 
            # Actually, minimal maximal Sidon sets are known to be smaller than greedy.
            # For N=256, greedy is ~16. Minimal might be ~10-14.
            # Let's run a few more if we find a good one.
            if trial_count > 20:
                break

    if best_set is None:
        # Fallback to seed logic if something went wrong
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    return best_set
