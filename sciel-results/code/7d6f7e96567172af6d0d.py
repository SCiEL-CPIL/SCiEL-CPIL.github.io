import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    divisors.sort()
    n_divs = len(divisors)
    
    cover_count = np.zeros(base, dtype=np.uint8)
    residues = [0] * n_divs
    uncovered_count = base
    
    # Initial Greedy
    # We can just run the local search from a random or zero start?
    # Zero start means all residues are 0.
    # This is a valid starting point.
    # Let's initialize with residues = [0] * n_divs
    # And apply them to cover_count
    
    for i, m in enumerate(divisors):
        r = 0
        residues[i] = r
        idx = np.arange(r, base, m)
        cover_count[idx] += 1
        
    # Recalculate uncovered_count
    uncovered_count = np.sum(cover_count == 0)
    
    improved = True
    # To help escape local minima, we can shuffle the order of divisors in each pass
    # or use random restarts if stuck.
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle divisors to vary the search path
        indices = list(range(n_divs))
        np.random.shuffle(indices)
        
        for idx_i in indices:
            if time.time() > deadline:
                break
            
            i = idx_i
            m = divisors[i]
            old_r = residues[i]
            
            # Save state
            old_uncovered_count = uncovered_count
            
            # 1. Remove old residue
            idx_old = np.arange(old_r, base, m)
            cover_count[idx_old] -= 1
            
            # 2. Find best new residue
            # Get currently uncovered indices
            unc_mask = (cover_count == 0)
            unc_idx = np.nonzero(unc_mask)[0]
            
            if len(unc_idx) == 0:
                best_r = old_r
                gain = 0
            else:
                # Count uncovered residues for each residue class mod m
                counts = np.bincount(unc_idx % m, minlength=m)
                best_r = int(np.argmax(counts))
                gain = int(counts[best_r])
            
            # 3. Apply best residue
            idx_new = np.arange(best_r, base, m)
            cover_count[idx_new] += 1
            
            # Calculate new uncovered count
            # Only residues that went from 0 to 1 are newly covered
            # We can just recompute sum(cover_count == 0)
            # But that is O(N).
            # Alternatively, we know:
            # Newly covered are in idx_new where cover_count became 1.
            # But some might have been >1 before? No, we removed old_r first.
            # If old_r == best_r, we removed and added same. Net change 0.
            
            # Let's just recompute for simplicity and correctness, 
            # as N is small (51975).
            new_uncovered_count = np.sum(cover_count == 0)
            
            # 4. Check improvement
            if new_uncovered_count < old_uncovered_count:
                uncovered_count = new_uncovered_count
                residues[i] = best_r
                improved = True
            else:
                # Revert
                cover_count[idx_new] -= 1
                cover_count[idx_old] += 1
                uncovered_count = old_uncovered_count
                # residues[i] remains old_r

    return [[divisors[i], residues[i]] for i in range(n_divs)]
