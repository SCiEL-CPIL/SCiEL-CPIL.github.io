"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Precompute position lists for each divisor/residue pair
    # pos_lists[i][r] = numpy array of positions covered by (divisors[i], r)
    pos_lists = []
    for m in divisors:
        plist = []
        for r in range(m):
            plist.append(np.arange(r, base, m, dtype=np.int32))
        pos_lists.append(plist)
    
    def count_uncovered(covered):
        """Count uncovered positions."""
        return int(np.sum(covered == 0))
    
    def build_coverage(residues):
        """Build coverage array and coverage count from residues."""
        covered = np.zeros(base, dtype=np.uint8)
        cov_count = np.zeros(base, dtype=np.int16)
        for i, m in enumerate(divisors):
            r = residues[i]
            positions = pos_lists[i][r]
            covered[positions] = 1
            cov_count[positions] += 1
        return covered, cov_count
    
    def compute_delta(i, old_r, new_r, cov_count):
        """Compute change in uncovered count when changing modulus i from old_r to new_r.
        Returns delta (new_uncovered - old_uncovered)."""
        m = divisors[i]
        old_positions = pos_lists[i][old_r]
        new_positions = pos_lists[i][new_r]
        
        # Positions in old but not new: lose coverage from i
        # If cov_count was 1, they become uncovered (+1 each)
        # Positions in new but not old: gain coverage from i
        # If cov_count was 0, they become covered (-1 each)
        
        # Use set operations for efficiency
        old_set = set(old_positions.tolist())
        new_set = set(new_positions.tolist())
        
        removed = old_set - new_set
        added = new_set - old_set
        
        delta = 0
        for p in removed:
            if cov_count[p] == 1:
                delta += 1
        for p in added:
            if cov_count[p] == 0:
                delta -= 1
        
        return delta
    
    def apply_move(i, old_r, new_r, covered, cov_count):
        """Apply a move: change modulus i from old_r to new_r."""
        m = divisors[i]
        old_positions = pos_lists[i][old_r]
        new_positions = pos_lists[i][new_r]
        
        old_set = set(old_positions.tolist())
        new_set = set(new_positions.tolist())
        
        removed = old_set - new_set
        added = new_set - old_set
        
        for p in removed:
            cov_count[p] -= 1
            if cov_count[p] == 0:
                covered[p] = 0
        for p in added:
            cov_count[p] += 1
            if cov_count[p] == 1:
                covered[p] = 1
    
    def revert_move(i, old_r, new_r, covered, cov_count):
        """Revert a move: change modulus i from new_r back to old_r."""
        apply_move(i, new_r, old_r, covered, cov_count)
    
    def greedy_init(seed_offset=0):
        """Greedy initialization: add moduli in order of best marginal gain."""
        rng = random.Random(seed_offset)
        residues = [0] * n_div
        covered = np.zeros(base, dtype=np.uint8)
        cov_count = np.zeros(base, dtype=np.int16)
        remaining = list(range(n_div))
        
        # Process in order of decreasing modulus (larger moduli cover fewer positions)
        # This helps cover "hard" positions first
        order = sorted(range(n_div), key=lambda i: -divisors[i])
        
        for idx in order:
            if time.time() > deadline - 30:
                break
            m = divisors[idx]
            # Find best residue for this modulus
            best_r = 0
            best_gain = -1
            # For large moduli, sample residues
            if m <= 30:
                rs_to_try = list(range(m))
            else:
                # Sample residues
                n_samples = min(20, m)
                rs_to_try = [0]
                for _ in range(n_samples - 1):
                    rs_to_try.append(rng.randint(0, m - 1))
                rs_to_try = list(set(rs_to_try))
            
            for r in rs_to_try:
                positions = pos_lists[idx][r]
                gain = int(np.sum(covered[positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[idx] = best_r
            positions = pos_lists[idx][best_r]
            covered[positions] = 1
            cov_count[positions] += 1
        
        return residues, covered, cov_count, int(np.sum(covered == 0))
    
    def local_search(residues, covered, cov_count, current_uncovered, max_iters=100000):
        """Simulated annealing local search with incremental delta computation."""
        best_residues = residues[:]
        best_uncovered = current_uncovered
        temp = 30.0
        cooling = 0.9997
        no_improve = 0
        rng = random.Random(12345)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            if temp < 0.5:
                break
            
            # Pick a random modulus to change
            i = rng.randrange(n_div)
            mi = divisors[i]
            old_ri = residues[i]
            new_ri = rng.randint(0, mi - 1)
            
            if new_ri == old_ri:
                continue
            
            # Compute delta
            delta = compute_delta(i, old_ri, new_ri, cov_count)
            
            # Acceptance criterion
            if delta <= 0 or rng.random() < np.exp(-delta / temp):
                # Accept
                apply_move(i, old_ri, new_ri, covered, cov_count)
                residues[i] = new_ri
                current_uncovered += delta
                no_improve = 0
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues[:]
            else:
                no_improve += 1
            
            temp *= cooling
            
            if no_improve > 3000:
                temp = 30.0
                no_improve = 0
        
        return best_residues, best_uncovered, covered, cov_count
    
    # Main: try multiple greedy inits with different seeds, then local search
    best_uncovered = base  # worst case
    best_residues = [0] * n_div
    best_covered = None
    best_cov_count = None
    
    n_restarts = 5
    for seed in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        residues, covered, cov_count, current_uncovered = greedy_init(seed * 7919 + 42)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]
            best_covered = covered.copy()
            best_cov_count = cov_count.copy()
        
        # Local search from this starting point
        if time.time() < deadline - 10:
            new_res, new_unc, new_cov, new_cc = local_search(residues, covered, cov_count, current_uncovered)
            if new_unc < best_uncovered:
                best_uncovered = new_unc
                best_residues = new_res[:]
                best_covered = new_cov.copy()
                best_cov_count = new_cc.copy()
    
    # Final polish: more local search from the best solution
    if time.time() < deadline - 5 and best_covered is not None:
        new_res, new_unc, new_cov, new_cc = local_search(best_residues, best_covered, best_cov_count, best_uncovered, max_iters=200000)
        if new_unc < best_uncovered:
            best_uncovered = new_unc
            best_residues = new_res[:]
    
    # Return the best solution found
    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
