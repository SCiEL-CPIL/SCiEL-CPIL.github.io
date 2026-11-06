"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    # Precompute residues for fast lookup
    # residues[x][r] = sorted list of values in 1..N with value % x == r
    residues = {}
    for x in range(1, N+1):
        residues[x] = {}
        for v in range(1, N+1):
            r = v % x
            if r not in residues[x]:
                residues[x][r] = []
            residues[x][r].append(v)
    
    def can_add(a, selected):
        """Check if we can add element 'a' to 'selected' (a not in selected).
        Returns True if valid, False otherwise.
        Uses O(n^2) check instead of O(n^3)."""
        sel = selected
        n = len(sel)
        
        # Check 1: a does not divide sum of any 3 distinct elements in selected
        # For each pair (b,c) in sel, need d in sel\{b,c} such that (b+c+d) % a == 0
        # i.e., d % a == (-(b+c)) % a
        if n >= 3:
            # Build residue lookup for selected elements
            res_lookup = {}
            for idx, v in enumerate(sel):
                r = v % a
                if r not in res_lookup:
                    res_lookup[r] = []
                res_lookup[r].append(idx)
            
            for i in range(n):
                b = sel[i]
                for j in range(i+1, n):
                    c = sel[j]
                    need_res = (-(b + c)) % a
                    if need_res in res_lookup:
                        # Check if there's an element with that residue that's not b or c
                        for k in res_lookup[need_res]:
                            if k != i and k != j:
                                return False
        
        # Check 2: For each x in sel, x does not divide (a + b + c) for distinct b,c in sel\{x}
        for idx_x, x in enumerate(sel):
            need_res = (-a) % x
            # Find all elements in sel except x
            # For each pair (b,c) from sel\{x}, check if (b+c) % x == need_res
            # Build residue lookup for sel\{x}
            res_lookup = {}
            for idx, v in enumerate(sel):
                if idx == idx_x:
                    continue
                r = v % x
                if r not in res_lookup:
                    res_lookup[r] = []
                res_lookup[r].append(idx)
            
            # For each b in sel\{x}, check if there exists c in sel\{x,b} with (b+c) % x == need_res
            # i.e., c % x == (need_res - b) % x
            for idx_b, b in enumerate(sel):
                if idx_b == idx_x:
                    continue
                need_c_res = (need_res - b) % x
                if need_c_res in res_lookup:
                    for idx_c in res_lookup[need_c_res]:
                        if idx_c != idx_b:
                            return False
        
        return True
    
    def full_check(candidate_set):
        """Full validation of a set using O(n^4) but with early termination."""
        n = len(candidate_set)
        for i, x in enumerate(candidate_set):
            others = [candidate_set[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
                    return False
        return True
    
    # Phase 1: Greedy from large to small
    selected = []
    for x in range(N, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if can_add(x, selected):
            selected.append(x)
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # Phase 2: Try to add more elements in different orders
    if time.time() < deadline - 10.0:
        # Try adding from small to large on top of what we have
        for x in range(1, N+1):
            if time.time() > deadline - 10.0:
                break
            if x not in best_selected:
                if can_add(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    # Phase 3: Local search with swaps
    if time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            for i in range(len(best_selected)):
                if time.time() > deadline - 15.0:
                    break
                for replacement in range(N, 0, -1):
                    if time.time() > deadline - 15.0:
                        break
                    if replacement in best_selected:
                        continue
                    new_selected = best_selected[:i] + best_selected[i+1:] + [replacement]
                    if full_check(new_selected):
                        best_selected = new_selected
                        best_size = len(best_selected)
                        improved = True
                        # Try to add more after swap
                        for x in range(N, 0, -1):
                            if time.time() > deadline - 15.0:
                                break
                            if x not in best_selected:
                                if can_add(x, best_selected):
                                    best_selected.append(x)
                                    best_size = len(best_selected)
                        break
    
    # Phase 4: Multi-start with random perturbations
    if time.time() < deadline - 5.0:
        random.seed(42)
        num_restarts = 0
        while time.time() < deadline - 5.0 and num_restarts < 30:
            num_restarts += 1
            
            # Create a perturbed version of the best solution
            perturbed = list(best_selected)
            
            # Remove 2-4 random elements
            remove_count = random.randint(2, 4)
            indices_to_remove = random.sample(range(len(perturbed)), min(remove_count, len(perturbed)))
            for idx in sorted(indices_to_remove, reverse=True):
                perturbed.pop(idx)
            
            # Try to greedily fill back up
            for x in range(N, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                if x not in perturbed:
                    if can_add(x, perturbed):
                        perturbed.append(x)
            
            if len(perturbed) > best_size:
                best_selected = perturbed
                best_size = len(best_selected)
    
    # Phase 5: Final attempt to add any remaining elements
    if time.time() < deadline:
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if can_add(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
