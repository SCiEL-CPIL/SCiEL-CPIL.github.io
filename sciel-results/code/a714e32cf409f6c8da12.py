"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = []
    best_size = 0
    
    # First, run a deterministic greedy pass to get a good baseline
    # This is fast and provides a strong starting point
    selected = []
    sums_set = set()
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        # Check if x can be added
        valid = True
        # We need to check all triples involving x
        # Triples are (x, x, x), (x, x, y), (x, y, y), (x, y, z) where y, z are in selected
        # Since we maintain sums_set for all triples of current 'selected',
        # we only need to check new sums involving x against sums_set AND against each other.
        
        # New sums involving x:
        # 1. (x, x, x)
        # 2. (x, x, y) for y in selected
        # 3. (x, y, y) for y in selected
        # 4. (x, y, z) for y, z in selected (y <= z)
        
        new_sums = []
        
        # (x, x, x)
        new_sums.append(3 * x)
        
        # (x, x, y)
        for y in selected:
            new_sums.append(2 * x + y)
            
        # (x, y, y)
        for y in selected:
            new_sums.append(x + 2 * y)
            
        # (x, y, z)
        n_sel = len(selected)
        for i in range(n_sel):
            for j in range(i, n_sel):
                new_sums.append(x + selected[i] + selected[j])
                
        # Check for duplicates within new_sums
        if len(new_sums) != len(set(new_sums)):
            valid = False
        else:
            # Check against existing sums_set
            for s in new_sums:
                if s in sums_set:
                    valid = False
                    break
                    
        if valid:
            selected.append(x)
            for s in new_sums:
                sums_set.add(s)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            
    # If we have time left, try randomized restarts to potentially find a larger set
    # The deterministic greedy from 1 upwards might get stuck in a local optimum.
    # Randomizing the order or starting points might allow denser packing.
    
    while time.time() < deadline - 2:  # Keep 2s buffer
        random.shuffle_range = list(range(1, 4097))
        random.shuffle(random.shuffle_range)
        
        selected = []
        sums_set = set()
        
        for x in random.shuffle_range:
            if time.time() > deadline - 1:
                break
                
            new_sums = []
            
            # (x, x, x)
            new_sums.append(3 * x)
            
            # (x, x, y)
            for y in selected:
                new_sums.append(2 * x + y)
                
            # (x, y, y)
            for y in selected:
                new_sums.append(x + 2 * y)
                
            # (x, y, z)
            n_sel = len(selected)
            for i in range(n_sel):
                for j in range(i, n_sel):
                    new_sums.append(x + selected[i] + selected[j])
                    
            if len(new_sums) != len(set(new_sums)):
                continue
                
            valid = True
            for s in new_sums:
                if s in sums_set:
                    valid = False
                    break
                    
            if valid:
                selected.append(x)
                for s in new_sums:
                    sums_set.add(s)
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = list(selected)
                    
    return best_set
# EVOLVE-BLOCK-END
