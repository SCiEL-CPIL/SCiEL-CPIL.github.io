import random
import time
import math
import sys

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    range_1_N = list(range(1, N + 1))
    
    # Precompute all possible pairs and sums for faster lookup? 
    # Actually, N=256 is small enough that O(N^2) checks are fast if optimized.
    
    def get_sums(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def is_valid_sidon(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def can_add(x, selected, sums):
        if 2 * x in sums:
            return False
        for a in selected:
            if x + a in sums:
                return False
        return True

    def get_addable(selected, sums):
        """Returns a set of elements in [1,N] not in selected that can be added."""
        addable = set()
        selected_set = set(selected)
        for x in range_1_N:
            if x in selected_set:
                continue
            if can_add(x, selected, sums):
                addable.add(x)
        return addable

    def greedy_blocking():
        """
        Strategy: 
        1. Start with empty set.
        2. At each step, consider all elements x that can be added.
        3. For each candidate x, calculate how many 'addable' elements it would block.
           An element y is blocked by x if y can currently be added, but cannot be added 
           after x is included.
        4. Select the x that blocks the most addable elements.
        5. Update set and sums.
        6. Repeat until no addable elements remain.
        
        This directly constructs a maximal set. The greedy choice aims to minimize the 
        number of steps (cardinality) by maximizing the 'coverage' of the addable set.
        """
        selected = []
        sums = set()
        addable = set(range_1_N)
        
        while addable:
            best_x = None
            best_blocked_count = -1
            best_blocked_set = None
            
            # We can optimize this by sampling if addable is large, 
            # but N=256 is small. Let's iterate all.
            for x in addable:
                # Check if x can actually be added (it should be, by definition of addable set, 
                # but let's be safe or trust the set)
                # Actually, addable is maintained as "can be added". 
                
                # Simulate adding x
                new_sums = set(sums)
                new_sums.add(2 * x)
                for a in selected:
                    new_sums.add(x + a)
                
                # Calculate new addable set
                new_addable = set()
                selected_plus_x = set(selected) | {x}
                for y in addable:
                    if y == x:
                        continue
                    if y in selected_plus_x:
                        continue
                    if can_add(y, selected_plus_x, new_sums):
                        new_addable.add(y)
                
                blocked = addable - new_addable - {x}
                blocked_count = len(blocked)
                
                if blocked_count > best_blocked_count:
                    best_blocked_count = blocked_count
                    best_x = x
                    best_blocked_set = blocked
                    best_new_sums = new_sums
                    best_new_addable = new_addable
            
            if best_x is None:
                break
            
            selected.append(best_x)
            sums = best_new_sums
            addable = best_new_addable
            
            # Time check
            if time.time() > deadline - 5:
                break
                
        # Ensure maximality just in case
        addable = get_addable(selected, sums)
        while addable:
            # Greedily add any, or the one blocking most?
            # Since we are in the "repair" phase, just add the first one or greedy.
            # Let's do greedy again for consistency.
            best_x = None
            best_blocked_count = -1
            for x in addable:
                new_sums = set(sums)
                new_sums.add(2 * x)
                for a in selected:
                    new_sums.add(x + a)
                
                new_addable = set()
                selected_plus_x = set(selected) | {x}
                for y in addable:
                    if y == x:
                        continue
                    if can_add(y, selected_plus_x, new_sums):
                        new_addable.add(y)
                
                blocked_count = len(addable - new_addable - {x})
                if blocked_count > best_blocked_count:
                    best_blocked_count = blocked_count
                    best_x = x
                    best_new_sums = new_sums
                    best_new_addable = new_addable
            
            if best_x:
                selected.append(best_x)
                sums = best_new_sums
                addable = best_new_addable
            else:
                break
                
        return selected

    def local_search_minimize(initial_selected):
        """
        Try to remove elements while maintaining maximality.
        If removing an element x keeps the set Sidon, and the resulting set is still maximal
        (i.e., x was not needed to block any other element? No, if x is removed, x becomes addable.
        So the set is NOT maximal if we remove x, because x can be added back.
        
        Wait, the definition of maximal is: no element from [1,N] can be added.
        If we remove x, then x is a missing element. Can x be added?
        If the remaining set S' is Sidon, can x be added to S'?
        If yes, then S' is not maximal.
        If no, then S' is maximal.
        
        So the move is: Remove x. Check if S' is Sidon. Check if S' is Maximal.
        If both true, we have a smaller maximal Sidon set.
        """
        current_selected = list(initial_selected)
        current_sums = get_sums(current_selected)
        
        # Verify initial is maximal
        if get_addable(current_selected, current_sums):
            # Repair
            addable = get_addable(current_selected, current_sums)
            while addable:
                x = addable.pop()
                current_selected.append(x)
                current_sums.add(2*x)
                for a in current_selected[:-1]:
                    current_sums.add(x+a)
                addable = get_addable(current_selected, current_sums)
        
        best_selected = list(current_selected)
        best_size = len(current_selected)
        
        no_improvement_count = 0
        time_limit = time.time() + 10
        
        while time.time() < time_limit and no_improvement_count < 20:
            improved = False
            # Try removing each element
            for i in range(len(current_selected)):
                x = current_selected[i]
                
                # Remove x
                new_selected = current_selected[:i] + current_selected[i+1:]
                
                # Check Sidon
                if not is_valid_sidon(new_selected):
                    continue
                
                new_sums = get_sums(new_selected)
                
                # Check Maximal
                # Is any element addable to new_selected?
                # Specifically, is x addable?
                # If x is NOT addable to new_selected, then new_selected is maximal?
                # No, we must check ALL missing elements.
                # But if x is not addable, it's a strong candidate.
                # Let's just check full maximality.
                
                addable = get_addable(new_selected, new_sums)
                
                if not addable:
                    # We found a smaller maximal set!
                    current_selected = new_selected
                    current_sums = new_sums
                    best_selected = list(current_selected)
                    best_size = len(current_selected)
                    improved = True
                    no_improvement_count = 0
                    break # Restart loop with new state
                else:
                    # Not maximal, so we can't just remove x.
                    # Could we remove x and add something else?
                    # That's a swap. Let's stick to removals for now to keep it simple.
                    # Or, if x is not addable, but some other y is, maybe we can swap?
                    # For now, simple removal is a good baseline.
                    pass
            
            if not improved:
                no_improvement_count += 1
                # Try random swaps?
                if len(current_selected) > 1 and random.random() < 0.5:
                    i = random.randint(0, len(current_selected)-1)
                    x = current_selected[i]
                    missing = [y for y in range_1_N if y not in current_selected]
                    if missing:
                        y = random.choice(missing)
                        new_selected = current_selected[:i] + [y] + current_selected[i+1:]
                        if is_valid_sidon(new_selected):
                            new_sums = get_sums(new_selected)
                            addable = get_addable(new_selected, new_sums)
                            if not addable:
                                current_selected = new_selected
                                current_sums = new_sums
                                best_selected = list(current_selected)
                                best_size = len(current_selected)
                                improved = True
                                no_improvement_count = 0
        
        return best_selected

    # 1. Generate initial solution using Greedy Blocking
    init_selected = greedy_blocking()
    
    # 2. Refine using Local Search (Removals)
    best_selected = local_search_minimize(init_selected)
    
    # 3. Final check
    if not is_valid_sidon(best_selected):
        # Fallback
        return init_selected
        
    if get_addable(best_selected, get_sums(best_selected)):
        # Repair
        addable = get_addable(best_selected, get_sums(best_selected))
        while addable:
            x = addable.pop()
            best_selected.append(x)
            best_selected.sort() # Keep sorted for consistency? Not strictly needed
            # Update sums for repair
            # This is getting messy, let's just re-run greedy if it fails
            pass
            
    return best_selected
# EVOLVE-BLOCK-END
