"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations_with_replacement
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 4096
    max_size = 80
    
    # Vectorized collision check using NumPy
    # Precompute pair sums for the selected set
    # When adding x, new triple sums are:
    #   3x, 2x+y for each y, x+y+z for each pair (y,z)
    
    def check_and_add_fast(selected_arr, sums_set, x):
        """
        Check if x can be added. selected_arr is a numpy array of selected values.
        Returns (bool, new_sums_set) where new_sums_set contains all new triple sums.
        """
        m = len(selected_arr)
        
        # Collect all new triple sums involving x
        new_sums_list = [3 * x]
        
        # (x, x, y) for each y
        if m > 0:
            new_sums_list.extend(2 * x + selected_arr)
        
        # (x, y, z) for y <= z
        if m > 1:
            # Generate all pairs (i, j) with i <= j
            for i in range(m):
                for j in range(i, m):
                    new_sums_list.append(x + selected_arr[i] + selected_arr[j])
        
        new_sums_set = set(new_sums_list)
        
        # Check for collisions with existing sums
        for s in new_sums_set:
            if s in sums_set:
                return False, None
        
        # Check for internal collisions within new_sums
        # This can happen if different triples produce the same sum
        # e.g., 3x == 2x+y => x == y (impossible since x > all y in selected)
        # 2x+y1 == 2x+y2 => y1 == y2 (impossible)
        # 2x+y == x+y+z => x == z (impossible since x > z)
        # x+y1+z1 == x+y2+z2 => y1+z1 == y2+z2 (possible!)
        # So we need to check for duplicate sums within new_sums_list
        
        if len(new_sums_set) != len(new_sums_list):
            return False, None
        
        return True, new_sums_set
    
    def greedy_construct(start_val=1, time_limit=None):
        """Greedy construction from start_val."""
        if time_limit is None:
            time_limit = deadline
        
        selected = []
        selected_arr = np.array([], dtype=np.int64)
        sums_set = set()
        
        for x in range(start_val, n + 1):
            if time.time() > time_limit:
                break
            
            ok, new_sums = check_and_add_fast(selected_arr, sums_set, x)
            if ok:
                selected.append(x)
                selected_arr = np.append(selected_arr, x)
                sums_set = sums_set | new_sums
        
        return selected, sums_set, selected_arr
    
    # First, get a quick greedy solution
    best_solution, best_sums, _ = greedy_construct(1, deadline - 5)
    best_size = len(best_solution)
    
    # Try different starting offsets to see if we can do better
    if best_size < max_size and time.time() < deadline - 10:
        for offset in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
            if time.time() > deadline - 10:
                break
            
            sol, sums, _ = greedy_construct(offset + 1, deadline - 10)
            if len(sol) > best_size:
                best_solution = sol
                best_sums = sums
                best_size = len(sol)
            
            if best_size == max_size:
                break
    
    # Try a more sophisticated approach: use exponential-like spacing
    # For B_3 sets, elements should grow roughly exponentially
    if best_size < max_size and time.time() < deadline - 15:
        # Try geometric sequences with different bases
        for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            if time.time() > deadline - 15:
                break
            
            # Generate a candidate set using powers
            candidates = []
            val = 1
            while val <= n:
                candidates.append(val)
                val *= base
            
            # Now greedily add from candidates, then fill in gaps
            selected = []
            selected_arr = np.array([], dtype=np.int64)
            sums_set = set()
            
            # First add the geometric candidates
            for x in candidates:
                if time.time() > deadline - 15:
                    break
                ok, new_sums = check_and_add_fast(selected_arr, sums_set, x)
                if ok:
                    selected.append(x)
                    selected_arr = np.append(selected_arr, x)
                    sums_set = sums_set | new_sums
            
            # Now try to fill in gaps with other values
            for x in range(1, n + 1):
                if time.time() > deadline - 15:
                    break
                if x in sums_set:  # quick reject (not quite right but fast)
                    continue
                ok, new_sums = check_and_add_fast(selected_arr, sums_set, x)
                if ok:
                    selected.append(x)
                    selected_arr = np.append(selected_arr, x)
                    sums_set = sums_set | new_sums
            
            if len(selected) > best_size:
                best_solution = selected
                best_sums = sums_set
                best_size = len(selected)
            
            if best_size == max_size:
                break
    
    # Try random restarts with different initial seeds
    if best_size < max_size and time.time() < deadline - 20:
        import random
        random.seed(42)
        
        for trial in range(10):
            if time.time() > deadline - 20:
                break
            
            # Random subset of initial values
            initial_size = random.randint(5, 20)
            initial_vals = random.sample(range(1, min(500, n + 1)), initial_size)
            initial_vals.sort()
            
            selected = []
            selected_arr = np.array([], dtype=np.int64)
            sums_set = set()
            
            for x in initial_vals:
                if time.time() > deadline - 20:
                    break
                ok, new_sums = check_and_add_fast(selected_arr, sums_set, x)
                if ok:
                    selected.append(x)
                    selected_arr = np.append(selected_arr, x)
                    sums_set = sums_set | new_sums
            
            # Now greedily extend
            if selected:
                for x in range(selected[-1] + 1, n + 1):
                    if time.time() > deadline - 20:
                        break
                    ok, new_sums = check_and_add_fast(selected_arr, sums_set, x)
                    if ok:
                        selected.append(x)
                        selected_arr = np.append(selected_arr, x)
                        sums_set = sums_set | new_sums
            
            if len(selected) > best_size:
                best_solution = selected
                best_sums = sums_set
                best_size = len(selected)
            
            if best_size == max_size:
                break
    
    # Verify the solution is valid
    if best_solution:
        all_sums = [sum(t) for t in combinations_with_replacement(best_solution, 3)]
        if len(all_sums) != len(set(all_sums)):
            # Fallback: try a simple exponential sequence
            fallback = []
            val = 1
            while val <= n and len(fallback) < 20:
                fallback.append(val)
                val = val * 3 + 1
            best_solution = fallback
    
    return best_solution
# EVOLVE-BLOCK-END
