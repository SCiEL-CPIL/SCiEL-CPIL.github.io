import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = []
    best_sum = -1.0
    
    base_set = list(range(129, 257))
    
    # Helper to run greedy given a starting set
    def run_greedy(initial_selected):
        selected = list(initial_selected)
        current_sum = sum(1.0 / x for x in selected)
        
        # Candidates to add
        # We can just iterate 2 to 128
        for z in range(2, 129):
            if time.time() > deadline:
                break
            if z in selected:
                continue
                
            conflicts = []
            for x in selected:
                if math.lcm(z, x) <= 256:
                    conflicts.append(x)
            
            if not conflicts:
                selected.append(z)
                current_sum += 1.0 / z
            else:
                conflict_sum = sum(1.0 / x for x in conflicts)
                gain = (1.0 / z) - conflict_sum
                if gain > 0:
                    conflict_set = set(conflicts)
                    selected = [x for x in selected if x not in conflict_set]
                    selected.append(z)
                    current_sum += gain
        
        return selected, current_sum

    # Multi-start: Force each small number to be the first one added
    for z in range(2, 129):
        if time.time() > deadline:
            break
            
        # Start with base, force add z
        # Check if z is compatible with base?
        # If not, we must remove conflicts first.
        # This is effectively what run_greedy does, but we want to FORCE z.
        # So we manually add z and remove conflicts, then run greedy for others.
        
        initial = list(base_set)
        # Remove conflicts with z
        conflicts_z = [x for x in initial if math.lcm(z, x) <= 256]
        conflict_set_z = set(conflicts_z)
        initial = [x for x in initial if x not in conflict_set_z]
        initial.append(z)
        
        # Now run greedy for other numbers
        # We can just call run_greedy, but it will check z again (skip)
        # and check others.
        res_set, res_sum = run_greedy(initial)
        
        if res_sum > best_sum:
            best_sum = res_sum
            best_set = res_set

    # Also try the case where no small number is forced (standard greedy)
    res_set, res_sum = run_greedy(base_set)
    if res_sum > best_sum:
        best_sum = res_sum
        best_set = res_set

    return best_set
