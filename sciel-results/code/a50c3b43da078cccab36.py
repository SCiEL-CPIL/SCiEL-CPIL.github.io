"""
Extended Sidon set construction for {1, 2, 5, 11} inside [1, 284].
Maximize size; all pairwise sums a+b (a<=b) must be distinct.
"""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    n = 284

    def get_sums(s):
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def can_add(x, s, used_sums):
        # Check if x can be added to set s without creating duplicate sums
        for y in s:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True

    def add_element(x, s, used_sums):
        """Add x to s and update used_sums. Assumes can_add passed."""
        s.add(x)
        for y in s:
            if y != x:
                used_sums.add(x + y)
        used_sums.add(2 * x)

    # --- Phase 1: Greedy initialization ---
    def greedy_init():
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                add_element(x, current_set, used_sums)
        return sorted(current_set)

    best_set = greedy_init()
    best_size = len(best_set)

    # --- Phase 2: Backtracking search for larger sets ---
    # Try to find a set of size best_size + 1, then best_size + 2, etc.
    
    all_candidates = [x for x in range(1, n + 1) if x not in prescribed_set]
    
    def backtrack(current_set, used_sums, start_idx, target, depth, max_depth=100000):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
        
        if len(current_set) == target:
            best_set = sorted(current_set)
            best_size = target
            return True
        
        # Pruning: can we reach target with remaining candidates?
        remaining = len(all_candidates) - start_idx
        if len(current_set) + remaining < target:
            return False
        
        # Forward checking bound: estimate how many more we can add
        # Simple optimistic bound: each new element uses at least (len(current_set)+1) new sums
        # Total sums needed for target: target*(target+1)//2
        # Current sums: len(used_sums)
        # Each addition adds at least (current_size + 1) new sums (sums with all existing + itself)
        # This is a loose bound; use it cautiously
        
        for i in range(start_idx, len(all_candidates)):
            if time.time() > deadline:
                return False
            
            x = all_candidates[i]
            if not can_add(x, current_set, used_sums):
                continue
            
            # Add x
            saved_sums = set(used_sums)
            add_element(x, current_set, used_sums)
            
            if backtrack(current_set, used_sums, i + 1, target, depth + 1):
                return True
            
            # Undo
            current_set.remove(x)
            used_sums.clear()
            used_sums.update(saved_sums)
        
        return False

    # Try to find larger sets via backtracking
    target_size = best_size + 1
    initial_set = set(prescribed)
    initial_sums = get_sums(initial_set)
    
    # Limit backtracking depth to avoid excessive time
    found = backtrack(initial_set, initial_sums, 0, target_size, 0)
    
    if found:
        # Try for even larger
        target_size += 1
        initial_set = set(prescribed)
        initial_sums = get_sums(initial_set)
        while target_size <= 20 and time.time() < deadline - 20:
            found = backtrack(initial_set, initial_sums, 0, target_size, 0)
            if not found:
                break
            target_size += 1

    # --- Phase 3: Local search refinement ---
    # Use swaps to try to improve the set
    if time.time() < deadline - 10:
        rng = random.Random(42)
        current_set = set(best_set)
        current_sums = get_sums(current_set)
        
        # Try multiple random restarts of local search
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            # Start from best_set or a perturbed version
            if restart == 0:
                cur = set(best_set)
            else:
                cur = set(best_set)
                # Perturb: remove 1-3 random non-prescribed elements
                removable = [x for x in cur if x not in prescribed_set]
                k = min(len(removable), rng.randint(1, 3))
                to_remove = rng.sample(removable, k)
                for x in to_remove:
                    cur.remove(x)
                # Re-add greedily
                cur_sums = get_sums(cur)
                for x in range(1, n + 1):
                    if x not in cur and can_add(x, cur, cur_sums):
                        add_element(x, cur, cur_sums)
            
            cur_sums = get_sums(cur)
            improved = False
            
            # Try to add elements greedily
            for x in range(1, n + 1):
                if x not in cur and can_add(x, cur, cur_sums):
                    add_element(x, cur, cur_sums)
                    improved = True
            
            if len(cur) > best_size:
                best_set = sorted(cur)
                best_size = len(best_set)
                continue
            
            # Try swaps: remove one, add one
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                
                removable = [x for x in cur if x not in prescribed_set]
                if not removable:
                    break
                
                x_remove = rng.choice(removable)
                cur.remove(x_remove)
                cur_sums = get_sums(cur)
                
                # Try to add a new element
                added = False
                for x_add in range(1, n + 1):
                    if x_add not in cur and can_add(x_add, cur, cur_sums):
                        add_element(x_add, cur, cur_sums)
                        added = True
                        break
                
                if added:
                    # Try to add more after the swap
                    cur_sums = get_sums(cur)
                    for x in range(1, n + 1):
                        if x not in cur and can_add(x, cur, cur_sums):
                            add_element(x, cur, cur_sums)
                    
                    if len(cur) > best_size:
                        best_set = sorted(cur)
                        best_size = len(best_set)
                        improved = True
                        break
                else:
                    # Restore
                    cur.add(x_remove)
                    cur_sums = get_sums(cur)
            
            if not improved and len(cur) < best_size:
                # Revert to best if we made things worse
                cur = set(best_set)
                cur_sums = get_sums(cur)

    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        best_set = list(prescribed)

    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
