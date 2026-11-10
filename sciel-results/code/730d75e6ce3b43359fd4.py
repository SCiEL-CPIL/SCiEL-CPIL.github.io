import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_set = []
    best_size = 0
    
    def get_initial_greedy():
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        S = []
        counts = {}
        
        for x in candidates:
            # Check if x can be added
            valid = True
            # Check sums with existing elements
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check double
            if counts.get(2 * x, 0) >= 2:
                continue
            
            # Add x
            S.append(x)
            for y in S:
                if y == x:
                    break # 2x handled separately? No, loop covers y=x if x is in S? 
                    # Wait, S.append(x) happens after check. 
                    # The loop `for y in S` iterates over OLD S.
                    # So we need to handle 2x separately or include x in the loop logic carefully.
                    # My check above: `for y in S` (old S). `counts[x+y]`.
                    # Then `counts[2x]`.
                    # This is correct.
                
            # Update counts
            for y in S:
                if y == x:
                    break
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
            s_2x = 2 * x
            counts[s_2x] = counts.get(s_2x, 0) + 1
        
        return S, counts

    def local_search(S, counts):
        # Try to improve S
        # S is a list, counts is a dict
        # We will modify S and counts in place
        # Return new size
        
        n = len(S)
        S_set = set(S)
        
        # Helper to remove x
        def remove(x):
            nonlocal counts, S, S_set
            # Remove x from S
            S.remove(x)
            S_set.remove(x)
            
            # Update counts
            # x was paired with all y in S (current S, which doesn't have x)
            for y in S:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            
            s_2x = 2 * x
            counts[s_2x] -= 1
            if counts[s_2x] == 0:
                del counts[s_2x]

        # Helper to add x
        def add(x):
            nonlocal counts, S, S_set
            # Check validity first? Assume caller checked.
            S.append(x)
            S_set.add(x)
            
            # Update counts
            for y in S:
                if y == x:
                    break
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
            s_2x = 2 * x
            counts[s_2x] = counts.get(s_2x, 0) + 1

        def can_add(x):
            # Check if x can be added to current S
            if x in S_set:
                return False
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    return False
            if counts.get(2 * x, 0) >= 2:
                return False
            return True

        # Local search loop
        # Strategy: 
        # 1. Try to add any element from 1..1000 not in S.
        # 2. If none, try swaps.
        
        # To speed up, we can maintain a list of "available" candidates?
        # Or just iterate through 1..1000 each time?
        # 1000 * (cost of can_add) is fine.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add
            # Shuffle candidates to avoid bias
            candidates = [x for x in range(1, 1001) if x not in S_set]
            random.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                if can_add(x):
                    add(x)
                    improved = True
                    added_any = True
                    break # Restart phase 1 to check for more additions
            
            if added_any:
                continue # Loop back to try more additions
            
            # Phase 2: Try swaps if no additions
            # Pick a random x in S to remove
            if not S:
                break
                
            x = random.choice(S)
            # Pick a random y not in S to add
            candidates = [y for y in range(1, 1001) if y not in S_set]
            if not candidates:
                break
            y = random.choice(candidates)
            
            # Try swap x -> y
            # Remove x
            remove(x)
            
            # Check if y can be added
            if can_add(y):
                add(y)
                improved = True
                # After swap, size is same. But maybe now we can add more?
                # The loop will restart and try Phase 1 again.
            else:
                # Revert: Add x back
                add(x)
                # Note: add(x) assumes x is not in S. It is not.
                # But we need to ensure the state is exactly restored.
                # remove(x) updated counts. add(x) will restore them.
        
        return len(S)

    # Main loop
    num_restarts = 0
    while time.time() < deadline:
        S, counts = get_initial_greedy()
        current_size = len(S)
        
        # Local search
        final_size = local_search(S, counts)
        
        if final_size > best_size:
            best_size = final_size
            best_set = S[:]
            # print(f"New best: {best_size}") # Uncomment for debugging
            
        num_restarts += 1
        # Limit restarts if we are running very fast? 
        # 120s is a lot. Let it run.
        
    return best_set

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
