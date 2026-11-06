import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    def is_valid(selected):
        """Check if selected set is valid. O(N^3)."""
        n = len(selected)
        for i in range(n):
            a = selected[i]
            # Check if a divides sum of any 3 distinct others
            # Sums of 3 distinct others from selected \ {a}
            others = [x for j, x in enumerate(selected) if j != i]
            # Generate all triple sums of others
            # If n is small, this is fine.
            # Optimization: if a > sum(others), it's safe.
            # Max sum of 3 from others is sum of largest 3.
            others_sorted = sorted(others, reverse=True)
            max_sum_3 = sum(others_sorted[:3]) if len(others) >= 3 else 0
            if a > max_sum_3:
                continue
            
            # Check specific sums
            # We can just iterate combinations of 3 from others
            for i2 in range(len(others)):
                b = others[i2]
                for i3 in range(i2+1, len(others)):
                    c = others[i3]
                    for i4 in range(i3+1, len(others)):
                        d = others[i4]
                        if (b + c + d) % a == 0:
                            return False
        return True

    def can_add_fast(selected, candidate):
        """Check if candidate can be added to selected. O(N^2 + N^3?) No, O(N^2) for pair sums + O(N^3) for triple sums?
        Actually, we just need to check:
        1. candidate divides sum of any 3 from selected.
        2. any a in selected divides sum of candidate + 2 from selected\{a}.
        """
        n = len(selected)
        
        # 1. candidate | (a+b+c) for a,b,c in selected
        # Precompute triple sums of selected? 
        # It's O(N^3) to generate, but we can do it once per check? 
        # Or we can just iterate.
        # Given N<=48, N^3 is ~110k. 
        # If we do this for every candidate in a greedy loop, it's expensive.
        
        # Better: Incremental state.
        # But for simplicity in this rewrite, let's use a slightly slower but more robust check
        # that allows for a simpler local search.
        
        # Check 1: candidate divides sum of 3 from selected
        for i in range(n):
            a = selected[i]
            for j in range(i+1, n):
                b = selected[j]
                for k in range(j+1, n):
                    c = selected[k]
                    if (a + b + c) % candidate == 0:
                        return False
        
        # Check 2: a in selected divides (candidate + b + c)
        for i in range(n):
            a = selected[i]
            # Need b, c from selected \ {a}
            # (candidate + b + c) % a == 0  =>  (b+c) % a == (-candidate) % a
            target = (-candidate) % a
            # Find if there exist b, c in selected \ {a} such that (b+c) % a == target
            # Iterate pairs
            others = [x for j, x in enumerate(selected) if j != i]
            for j in range(len(others)):
                b = others[j]
                for k in range(j+1, len(others)):
                    c = others[k]
                    if (b + c) % a == target:
                        return False
        return True

    best_set = []
    best_len = 0
    
    # Strategy: Randomized Greedy with Restarts
    # Prioritize large numbers as they are "safer"
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
            
        random.seed(12345 + trial)
        # Create an order: Large numbers first, with noise
        candidates = list(range(192, 0, -1))
        # Add noise to break ties and explore
        noisy_candidates = [(x, random.uniform(-10, 10)) for x in candidates]
        noisy_candidates.sort(key=lambda t: -(t[0] + t[1]))
        order = [x for x, _ in noisy_candidates]
        
        selected = []
        
        for x in order:
            if len(selected) >= 48:
                break
            if can_add_fast(selected, x):
                selected.append(x)
                
        if len(selected) > best_len:
            best_set = selected[:]
            best_len = len(selected)
            
        if best_len >= 48:
            break

    # Local Search: 1-swap and 1-add
    if best_len < 48 and time.time() < deadline - 5:
        current = best_set[:]
        current_len = best_len
        improved = True
        
        while improved and current_len < 48 and time.time() < deadline - 5:
            improved = False
            
            # Try 1-add: add any missing number
            missing = [x for x in range(1, 193) if x not in current]
            random.shuffle(missing)
            for x in missing:
                if can_add_fast(current, x):
                    current.append(x)
                    current_len += 1
                    improved = True
                    if current_len >= 48:
                        break
            
            if current_len >= 48:
                break
                
            # Try 1-swap: remove one, add one
            # We want to increase length, so 1-swap doesn't help length directly unless we do 2-in-1-out later?
            # 1-swap keeps length same. It helps escape local minima for 1-add.
            # Let's do a few swaps to shuffle the set.
            for _ in range(50): # Max 50 swaps per epoch
                if time.time() > deadline - 5:
                    break
                if not current:
                    break
                i = random.randint(0, len(current) - 1)
                a = current[i]
                # Remove a
                current[i] = None
                temp_current = [x for x in current if x is not None]
                
                # Find a replacement
                # Prefer large numbers
                candidates = [x for x in range(192, 0, -1) if x not in temp_current]
                random.shuffle(candidates[:20]) # Shuffle top 20
                candidates = candidates[:20] + candidates[20:]
                
                found_swap = False
                for b in candidates:
                    if can_add_fast(temp_current, b):
                        temp_current.append(b)
                        # Check if this swap allows a subsequent add?
                        # For now, just accept the swap if it's valid.
                        # To ensure progress, we might check if it enables an add.
                        # But simple hill climbing on validity is enough.
                        current = temp_current
                        found_swap = True
                        break
                if not found_swap:
                    # Restore
                    current[i] = a
                    # If no swap found, break or continue?
                    # If we can't swap, we are stuck.
                    # Let's try to remove an element and see if we can add 2?
                    # That's 1-out-2-in.
                    pass

    return best_set[:48]
