import time
import sys

def construct():
    deadline = time.time() + 45.0
    best_size = 0
    best_elements = []
    
    # We will search for the largest set.
    # The maximum possible size is 30, but likely much smaller.
    # We'll try to find a set of size 10, 9, 8, etc.
    
    # To speed up, we can pre-compute some divisibility info?
    # No, the sums depend on the specific set.
    
    # Backtracking state
    current_elements = []
    current_sums = {0}
    
    # We try numbers in descending order.
    # If we want to be more flexible, we can try different orderings.
    # But let's stick to descending first.
    
    candidates = list(range(256, 0, -1))
    
    def backtrack(index, current_elements, current_sums):
        nonlocal best_size, best_elements
        
        if time.time() > deadline:
            return
            
        # Pruning: if we can't beat best_size, return
        # Max possible size = len(current_elements) + (len(candidates) - index)
        # This is a loose bound. A tighter bound is hard.
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_elements = list(current_elements)
            
        # If we have no more candidates, return
        if index >= len(candidates):
            return
            
        # Try to add candidates[index]
        x = candidates[index]
        
        # Generate new sums
        new_sums = set()
        for s in current_sums:
            ns = s + x
            new_sums.add(ns)
            
        # Check collision with existing sums
        if new_sums & current_sums:
            # Collision, skip x
            backtrack(index + 1, current_elements, current_sums)
            return
        
        # Check divisibility
        # 1. New sums vs New sums
        valid = True
        ns_list = list(new_sums)
        for i in range(len(ns_list)):
            a = ns_list[i]
            if a == 0: continue
            for j in range(i + 1, len(ns_list)):
                b = ns_list[j]
                if b % a == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # 2. New sums vs Old sums
            for ns in ns_list:
                if ns == 0: continue
                for os in current_sums:
                    if os == 0: continue
                    if os == ns: continue # Already checked collision
                    if os > ns:
                        if os % ns == 0:
                            valid = False
                            break
                    else:
                        if ns % os == 0:
                            valid = False
                            break
                if not valid:
                    break
        
        if valid:
            # Add x
            current_elements.append(x)
            current_sums |= new_sums
            backtrack(index + 1, current_elements, current_sums)
            current_elements.pop()
            current_sums -= new_sums
            
        # Skip x
        backtrack(index + 1, current_elements, current_sums)

    backtrack(0, [], {0})
    
    return list(best_elements)

if __name__ == "__main__":
    print(construct())
