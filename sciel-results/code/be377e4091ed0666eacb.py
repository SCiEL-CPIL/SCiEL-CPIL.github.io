<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import random

def construct():
    base = 51975
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order
    
    n_divs = len(divisors)
    
    # For each divisor m, precompute the coverage pattern for each residue r
    # cover_idx[m][r] = numpy array of indices covered by class (m, r)
    # This allows fast numpy operations
    
    # Actually, let's use a different approach with numpy for speed
    
    # Greedy initialization
    covered = np.zeros(base, dtype=np.uint8)
    classes = {}
    
    for m in divisors:
        if time.time() > deadline:
            break
        best_r = 0
        best_count = -1
        for r in range(m):
            idx = np.arange(r, base, m)
            cnt = np.sum(covered[idx] == 0)
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes[m] = best_r
        covered[np.arange(best_r, base, m)] = 1
    
    def get_uncovered(covered):
        return int(np.sum(covered == 0))
    
    current_uncovered = get_uncovered(covered)
    
    # Local search using numpy for efficiency
    # For each modulus m, try changing its residue to see if it reduces uncovered
    
    improved = True
    max_passes = 20
    for _ in range(max_passes):
        if not improved or time.time() > deadline:
            break
        improved = False
        
        for m in divisors:
            if time.time() > deadline:
                break
            old_r = classes[m]
            old_idx = np.arange(old_r, base, m)
            
            # Compute coverage without this class
            # We need to know which positions are covered by other classes
            # Let's compute the "other coverage" mask
            
            # Try each residue
            best_r = old_r
            best_uncovered = current_uncovered
            
            for r in range(m):
                if r == old_r:
                    continue
                new_idx = np.arange(r, base, m)
                
                # Positions that are in new_idx but not in old_idx
                gained = np.setdiff1d(new_idx, old_idx, assume_unique=True)
                # Positions that are in old_idx but not in new_idx  
                lost = np.setdiff1d(old_idx, new_idx, assume_unique=True)
                
                # For gained positions: they help if they were uncovered by other classes
                # For lost positions: they hurt if they were only covered by this class
                
                # This is still complex. Let's just do a full recompute for now
                # but use numpy for speed
                
                # Rebuild covered without m, then add new (m, r)
                new_covered = np.zeros(base, dtype=np.uint8)
                for m2, r2 in classes.items():
                    if m2 != m:
                        new_covered[np.arange(r2, base, m2)] = 1
                new_covered[new_idx] = 1
                
                new_uncovered = int(np.sum(new_covered == 0))
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_r = r
            
            if best_r != old_r:
                classes[m] = best_r
                current_uncovered = best_uncovered
                improved = True
                # Update covered
                covered = np.zeros(base, dtype=np.uint8)
                for m2, r2 in classes.items():
                    covered[np.arange(r2, base, m2)] = 1
    
    result = [[m, classes[m]] for m in divisors if m in classes]
    return result

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import random

def construct():
    base = 51975
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order
    
    n_divs = len(divisors)
    
    # Precompute index arrays for each (m, r) pair
    # idx_map[m][r] = np.arange(r, base, m)
    idx_map = {}
    for m in divisors:
        idx_map[m] = {}
        for r in range(m):
            idx_map[m][r] = np.arange(r, base, m)
    
    def compute_uncovered(classes):
        """Compute uncovered count for a given assignment of residues to moduli"""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes.items():
            covered[idx_map[m][r]] = 1
        return int(np.sum(covered == 0))
    
    # Greedy initialization
    classes = {}
    for m in divisors:
        if time.time() > deadline:
            break
        best_r = 0
        best_count = -1
        for r in range(m):
            idx = idx_map[m][r]
            cnt = 0
            # Count how many of these positions are not yet covered
            # We need to track covered positions
            # For greedy, let's just use the full coverage so far
            # Actually, for greedy we need incremental coverage
            # Let's do a simpler greedy: for each m, pick r that maximizes |idx \ covered|
            # But we don't have a global covered array in this formulation
            # Let's switch back to incremental approach for initialization
        
        # Simplified: just pick r=0 for now, then optimize
        classes[m] = 0
    
    current_uncovered = compute_uncovered(classes)
    best_uncovered = current_uncovered
    best_classes = dict(classes)
    
    # Simulated annealing
    random.seed(42)
    temp = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random modulus and try a random new residue
        m = random.choice(divisors)
        old_r = classes[m]
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
        
        classes[m] = new_r
        new_uncovered = compute_uncovered(classes)
        
        delta = new_uncovered - current_uncovered
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_uncovered = new_uncovered
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = dict(classes)
        else:
            classes[m] = old_r
        
        temp *= cooling
    
    # Final local search from best state
    classes = dict(best_classes)
    current_uncovered = best_uncovered
    
    # Hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in divisors:
            if time.time() > deadline:
                break
            old_r = classes[m]
            best_r = old_r
            best_u = current_uncovered
            for r in range(m):
                if r == old_r:
                    continue
                classes[m] = r
                u = compute_uncovered(classes)
                if u < best_u:
                    best_u = u
                    best_r = r
            classes[m] = best_r
            if best_r != old_r:
                current_uncovered = best_u
                improved = True
    
    result = [[m, classes[m]] for m in divisors if m in classes]
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

This is better but `compute_uncovered` is still O(n_divs * base/m) per call, and we're calling it many times. Let me optimize by precomputing and using a faster method.

Actually, let me reconsider the whole approach. The key bottleneck is computing the uncovered count. Let me use a **bitmask** or **numpy** approach that's faster.

Wait, I just realized: with 47 moduli, and each modulus m covering base/m positions, the total work for one full recomputation is sum_{m in divisors} base/m. Let me estimate this:

For 51975 = 3^3 * 5^2 * 7 * 11, the sum of reciprocals of divisors is:
(1 + 1/3 + 1/9 + 1/27) * (1 + 1/5 + 1/25) * (1 + 1/7) * (1 + 1/11)
= (40/27) * (31/25) * (8/7) * (12/11)
≈ 1.481 * 1.24 * 1.143 * 1.091
≈ 2.29

So sum of base/m ≈ 2.29 * 51975 ≈ 119,000 operations per full recomputation. In numpy, this is very fast (microseconds).

So simulated annealing with full recomputation should be feasible. Let me make sure the code is clean and efficient.

Let me also add a better greedy initialization:

