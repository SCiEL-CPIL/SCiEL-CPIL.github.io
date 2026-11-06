import time
import random
import numpy as np

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Divisors > 1:
# 3, 5, 7, 9, 11, 15, 21, 25, 27, 33, 35, 45, 55, 63, 75, 77, 99, 105, 135, 165, 175, 225, 231, 275, 315, 385, 495, 525, 693, 825, 945, 1155, 1485, 1575, 2079, 2739, 2925, 3675, 4950 (no, even), ...
# Let's just generate them properly.
divisors = []
for i in range(2, int(BASE**0.5) + 1):
    if BASE % i == 0:
        divisors.append(i)
        other = BASE // i
        if other != i:
            divisors.append(other)
divisors = sorted(set(divisors))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute residues for each divisor
    # For a divisor m, the residues are r, r+m, r+2m, ... < BASE
    # We represent coverage as a boolean array of size BASE
    # But iterating over all residues for each step is O(BASE * num_divisors) which is too slow if done naively in Python loops.
    # We need to use NumPy for speed.
    
    # State:
    # assignments: dict {m: r}
    # covered: np.array of bools, size BASE
    
    # Objective: minimize number of uncovered residues = BASE - sum(covered)
    
    # Strategy:
    # 1. Greedy initialization:
    #    Iterate divisors in some order (e.g., descending size to cover more "structure" or ascending?).
    #    Usually, larger moduli cover fewer residues per class (BASE/m), but they are more "specific".
    #    Smaller moduli cover more residues (BASE/m is large).
    #    Greedy: pick the class (m, r) that maximizes newly covered residues.
    #    This is expensive to compute from scratch every step.
    #    Alternative: Simulated Annealing / Local Search.
    
    # Let's try a local search approach.
    # Initialize with random assignments or greedy.
    # Move: change residue r for a specific modulus m to a new r'.
    # Delta score: (newly covered) - (newly uncovered).
    
    # To make this fast, we need to quickly compute the delta.
    # Let C be the current covered set (bitmask or bool array).
    # Let S_m be the set of residues covered by modulus m with residue r.
    # If we change m's residue from r to r':
    #   Removed: S_m(r) \ C_others  (where C_others = C \ S_m(r))
    #   Added: S_m(r') \ C_others
    
    # Actually, it's easier to just recompute the coverage for that specific modulus's contribution?
    # No, we need the global count.
    
    # Let's use a count array: count[x] = number of classes covering x.
    # Uncovered = count of x where count[x] == 0.
    
    # When we change modulus m from r to r':
    # 1. For all x in S_m(r): count[x] -= 1
    # 2. For all x in S_m(r'): count[x] += 1
    # 3. Update uncovered count:
    #    - If count[x] becomes 0 (step 1), uncovered += 1
    #    - If count[x] becomes 1 (step 2) from 0, uncovered -= 1
    
    # This is O(BASE/m) per move.
    # Sum of BASE/m for all m is BASE * sum(1/m).
    # Sum(1/m) for divisors of 51975 is small.
    # So each move is fast.
    
    # Initialization:
    # Greedy: Start with no classes. Add classes one by one.
    # Which class to add? The one that covers the most currently uncovered residues.
    # This is O(num_divisors * BASE) per step, which is too slow for 150s if done naively in Python.
    # We can use NumPy to speed up the greedy step.
    
    # Let's try a simpler initialization: Random assignment for all divisors.
    # Then local search.
    
    # Divisors to use: All divisors > 1.
    # There are 32 divisors.
    
    # Let's implement the local search.
    
    random.seed(42)
    
    # count array
    counts = np.zeros(BASE, dtype=np.int8)
    uncovered = BASE
    assignments = {}
    
    # Order of divisors for initialization
    # Maybe larger moduli first? Or random?
    # Let's try random initialization first, it's fast.
    
    for m in divisors:
        r = random.randint(0, m - 1)
        assignments[m] = r
        # Add coverage
        residues = np.arange(r, BASE, m)
        # Update counts
        old_counts = counts[residues]
        new_counts = old_counts + 1
        counts[residues] = new_counts
        
        # Update uncovered
        # Where old_counts == 0 and new_counts == 1: uncovered -= 1
        # (new_counts will always be >= 1)
        # So just count how many were 0 before.
        uncovered -= np.sum(old_counts == 0)
    
    best_uncovered = uncovered
    best_assignments = assignments.copy()
    
    # Local Search
    # Simulated Annealing or just Hill Climbing with restarts?
    # Hill Climbing is fast. Let's try it.
    
    # Precompute residue arrays for each divisor?
    # residue_masks[m] = list of np.arange(r, BASE, m) for all r?
    # That's too much memory.
    # Just generate on the fly? np.arange is fast.
    
    # To speed up delta calculation, we can precompute the indices for each (m, r)?
    # No, let's just use np.arange.
    
    # Move: pick random m, pick random new r != old r.
    # Calculate delta.
    # If delta < 0 (uncovered decreases), accept.
    
    # We need to track time.
    
    num_divs = len(divisors)
    
    # To make it faster, we can precompute the lists of indices for each modulus and residue?
    # No, just use slicing.
    # counts[r::m]
    
    # Let's try a few restarts.
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Reset
        counts = np.zeros(BASE, dtype=np.int8)
        uncovered = BASE
        assignments = {}
        
        # Random init
        for m in divisors:
            r = random.randint(0, m - 1)
            assignments[m] = r
            residues = np.arange(r, BASE, m)
            old_counts = counts[residues]
            counts[residues] = old_counts + 1
            uncovered -= np.sum(old_counts == 0)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_assignments = assignments.copy()
        
        # Local search
        # Order of divisors for trying moves
        # Maybe try larger moduli first? They have fewer residues, so cheaper to update?
        # No, update cost is O(BASE/m). Smaller m -> larger cost.
        # So try larger m first?
        
        # Let's just iterate randomly.
        
        stuck_count = 0
        max_stuck = 1000
        
        while time.time() < deadline - 5:
            if stuck_count > max_stuck:
                break
                
            m = divisors[random.randint(0, num_divs - 1)]
            old_r = assignments[m]
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta
            # Remove old
            old_res = np.arange(old_r, BASE, m)
            old_counts = counts[old_res]
            new_counts_old = old_counts - 1
            # Uncovered increases where old_counts == 1 (becomes 0)
            delta_uncovered = np.sum(old_counts == 1)
            
            # Add new
            new_res = np.arange(new_r, BASE, m)
            # Note: new_res and old_res might overlap?
            # No, for a fixed m, residues r and r' are disjoint sets of indices mod m?
            # Wait. The sets {r + km} and {r' + km} are disjoint if r != r'.
            # Yes, because they are different residue classes modulo m.
            # So old_res and new_res are disjoint.
            
            # But we need to check the count at new_res BEFORE adding.
            # Since old_res and new_res are disjoint, counts[new_res] is unaffected by the removal of old_res.
            # So we can just look at current counts[new_res].
            
            current_new_counts = counts[new_res]
            # Uncovered decreases where current_new_counts == 0 (becomes 1)
            delta_uncovered -= np.sum(current_new_counts == 0)
            
            if delta_uncovered < 0:
                # Accept
                counts[old_res] = new_counts_old
                counts[new_res] = current_new_counts + 1
                uncovered += delta_uncovered
                assignments[m] = new_r
                stuck_count = 0
                
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_assignments = assignments.copy()
            else:
                stuck_count += 1
                
    # Convert best_assignments to list of [m, r]
    result = [[m, r] for m, r in best_assignments.items()]
    return result

# Test locally? No, just return.
# The function construct is the entry point.

if __name__ == "__main__":
    # Quick test
    classes = construct()
    # Verify
    covered = bytearray(BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    score = covered.count(0)
    print(f"Score: {score}")
