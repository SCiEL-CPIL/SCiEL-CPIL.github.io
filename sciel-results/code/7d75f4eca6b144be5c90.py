import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 1000
    max_sum = 2 * N
    
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    counts = [0] * (max_sum + 1)
    in_set = [False] * (N + 1)
    S = []
    
    def can_add(y):
        # Check if adding y keeps all counts <= 2
        for x in S:
            s = x + y
            if counts[s] == 2:
                return False
        s_2y = 2 * y
        if counts[s_2y] == 2:
            return False
        return True

    def do_add(y):
        for x in S:
            counts[x + y] += 1
        counts[2 * y] += 1
        in_set[y] = True
        S.append(y)

    def do_remove(x):
        for y in S:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        in_set[x] = False
        S.remove(x)

    def get_addable_set(S_current, counts_current, in_set_current):
        """
        Returns a list of elements y not in S such that S U {y} is valid.
        """
        addable = []
        # We need to check all y not in S
        # To speed up, we can just iterate 1..N
        for y in range(1, N + 1):
            if in_set_current[y]:
                continue
            valid = True
            for x in S_current:
                s = x + y
                if counts_current[s] == 2:
                    valid = False
                    break
            if valid and counts_current[2 * y] == 2:
                valid = False
            if valid:
                addable.append(y)
        return addable

    # 1. Greedy Initialization
    # Try multiple random orders for greedy to find a good starting point
    best_size = 0
    best_S = []
    best_counts = [0] * (max_sum + 1)
    best_in_set = [False] * (N + 1)
    
    # We'll do a few greedy runs with different random seeds/orders
    for run in range(10):
        S_temp = []
        counts_temp = [0] * (max_sum + 1)
        in_set_temp = [False] * (N + 1)
        
        # Random order
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if can add x to S_temp
            valid = True
            for y in S_temp:
                s = x + y
                if counts_temp[s] == 2:
                    valid = False
                    break
            if valid and counts_temp[2 * x] == 2:
                valid = False
                
            if valid:
                for y in S_temp:
                    counts_temp[x + y] += 1
                counts_temp[2 * x] += 1
                in_set_temp[x] = True
                S_temp.append(x)
                
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = list(S_temp)
            best_counts = list(counts_temp)
            best_in_set = list(in_set_temp)
            
        if time.time() > deadline - 10:
            break
            
    # Use the best greedy result as starting point for local search
    S = best_S
    counts = best_counts
    in_set = best_in_set
    current_size = len(S)
    
    # 2. Local Search
    # We will try to improve the set by adding elements or swapping.
    
    while time.time() < deadline:
        # Phase 1: Try to add any element
        added = False
        # Get addable elements
        # Optimization: Instead of checking all, we can just iterate and check
        # But we need to be fast.
        
        # Find any addable element
        addable = []
        for y in range(1, N + 1):
            if in_set[y]:
                continue
            valid = True
            for x in S:
                s = x + y
                if counts[s] == 2:
                    valid = False
                    break
            if valid and counts[2 * y] == 2:
                valid = False
            if valid:
                addable.append(y)
                break # Just find one
        
        if addable:
            y = addable[0]
            # Add it
            for x in S:
                counts[x + y] += 1
            counts[2 * y] += 1
            in_set[y] = True
            S.append(y)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_S = list(S)
                best_counts = list(counts)
                best_in_set = list(in_set)
            continue
        
        # Phase 2: No addition possible. Try swaps.
        # We want to find a swap (x, y) such that S' is valid and potentially allows more additions.
        # Heuristic: Accept swap if it increases the number of addable elements, or reduces congestion.
        
        # Calculate current number of addable elements (for heuristic)
        # This is expensive, so we might sample.
        # Let's just try random swaps and accept if valid.
        # To avoid getting stuck, we can use a "simulated annealing" like acceptance for valid swaps
        # that don't immediately increase size, but we must ensure we don't decrease size.
        
        # Actually, if we swap, size stays same.
        # We accept a valid swap if:
        # 1. It allows an immediate addition (checked in next loop iteration).
        # 2. It reduces the "max count" or "sum of counts".
        
        # Let's calculate congestion: sum(c*c for c in counts)
        # This is O(N) = 2000 ops. Cheap.
        
        current_congestion = sum(c * c for c in counts)
        
        found_swap = False
        attempts = 0
        max_attempts = 100 # Limit per iteration to avoid wasting time
        
        while attempts < max_attempts and not found_swap:
            attempts += 1
            x = random.choice(S)
            y = random.choice([i for i in range(1, N + 1) if not in_set[i]])
            
            # Check if swap x->y is valid
            # 1. Simulate removal of x
            # We need to check if removing x and adding y keeps counts <= 2.
            
            # Instead of full simulation, we can check:
            # For all z in S \ {x}, count_{S'}(z+y) <= 2.
            # count_{S'}(s) = count_S(s) - delta_remove(s) + delta_add(s).
            
            # Let's just do the full simulation in a lightweight way.
            # We'll use temporary variables or just check the constraints directly.
            
            # Constraint 1: For all z in S, z != x, (z, y) pair sum s=z+y.
            # The new count for s is:
            # old_count[s] 
            # minus 1 if x+z == s (i.e., z = s-x) -> but we are iterating z.
            # Let's just iterate z in S.
            
            valid_swap = True
            
            # We need to check the new counts for sums involving y.
            # Sums: y+z for z in S\{x}, and 2y.
            
            # For each z in S\{x}:
            #   s = y+z
            #   new_count[s] = counts[s] 
            #                  - (1 if x+z == s else 0)  # Remove x
            #                  + (1 if y+z == s else 0)  # Add y -> This is always 1 for this z
            #   Wait, the term "Add y" adds 1 to count[y+z].
            #   The term "Remove x" subtracts 1 from count[x+z].
            #   If x+z == y+z, i.e., x=y, but x in S, y not in S, so x!=y.
            #   So x+z != y+z.
            #   So new_count[y+z] = counts[y+z] - (1 if x+z == y+z? No) 
            #   Actually, we are calculating count for sum s = y+z.
            #   Before swap: count_S(s).
            #   After removing x: count_S(s) - 1 if x is involved in sum s.
            #   x is involved in sum s if x+z' = s for some z' in S.
            #   Here s = y+z. x is involved if x+z' = y+z.
            #   This is getting complicated.
            
            # Simpler: Just do the add/remove on a copy? No, too slow.
            # Let's use the property:
            # We want to know if any count exceeds 2.
            # The only counts that change are those involving x or y.
            # Sums involving x: x+z (z in S\{x}), 2x.
            # Sums involving y: y+z (z in S\{x}), 2y.
            
            # We check the new values for these sums.
            
            # 1. Check 2y
            # new_count[2y] = counts[2y] - (1 if 2x == 2y else 0) + 1
            # x != y, so 2x != 2y.
            # new_count[2y] = counts[2y] + 1
            if counts[2 * y] + 1 > 2:
                valid_swap = False
            
            if valid_swap:
                # 2. Check y+z for z in S\{x}
                for z in S:
                    if z == x:
                        continue
                    s = y + z
                    # new_count[s] = counts[s] 
                    #             - (1 if x+z == s? No, x is removed. Does x contribute to s=y+z?
                    #               x contributes to s if x+w = s for some w in S.
                    #               Here s = y+z. x contributes if x+w = y+z.
                    #               This is not simply "1 if x+z == s".
                    
                    # Let's re-evaluate.
                    # count_{S'}(s) = count_S(s) - count_{S \setminus \{x\}}(s) + count_{\{y\} \cup (S \setminus \{x\})}(s)
                    # This is hard to compute directly.
                    
                    # Let's just simulate the changes on the `counts` array, check, and revert.
                    # It's O(|S|) per swap attempt.
                    # |S| ~ 80. 100 attempts * 80 = 8000 ops. Very fast.
                    
                    pass # We will do the simulation below
                    
            if not valid_swap:
                continue
                
            # Simulation
            # Remove x
            for z in S:
                if z != x:
                    counts[x + z] -= 1
            counts[2 * x] -= 1
            in_set[x] = False
            S.remove(x)
            
            # Check if y can be added
            valid_add_y = True
            for z in S:
                if counts[z + y] == 2:
                    valid_add_y = False
                    break
            if valid_add_y and counts[2 * y] == 2:
                valid_add_y = False
                
            if valid_add_y:
                # Add y
                for z in S:
                    counts[z + y] += 1
                counts[2 * y] += 1
                in_set[y] = True
                S.append(y)
                
                # Check if this swap reduces congestion or allows addition
                new_congestion = sum(c * c for c in counts)
                
                # Heuristic: Accept if congestion decreased, or if we can add more now.
                # We'll let the main loop check for additions.
                # To avoid random walk, only accept if congestion is better or equal.
                if new_congestion <= current_congestion:
                    found_swap = True
                    current_congestion = new_congestion
                else:
                    # Revert
                    for z in S:
                        if z != y:
                            counts[z + y] -= 1
                    counts[2 * y] -= 1
                    in_set[y] = False
                    S.remove(y)
                    
                    for z in S:
                        counts[x + z] += 1
                    counts[2 * x] += 1
                    in_set[x] = True
                    S.append(x)
            else:
                # Revert removal of x
                for z in S:
                    counts[x + z] += 1
                counts[2 * x] += 1
                in_set[x] = True
                S.append(x)

    return best_S

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
