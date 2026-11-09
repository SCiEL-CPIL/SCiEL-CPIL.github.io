"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # wall-clock limit with margin; we target finishing well before 150s

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Precompute all divisors > 1 of base
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    num_divisors = len(divisors)
    
    # Precompute coverage masks for each (m, r) pair - but only for small m
    # For large m, we'll compute on the fly
    # Actually, let's use a different approach: represent coverage as a boolean array
    
    def compute_score(covered):
        return base - np.sum(covered)
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, each picks best residue."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            # For each residue, count newly covered positions
            for r in range(m):
                idx = np.arange(r, base, m)
                count = np.sum(covered[idx] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, int(compute_score(covered))
    
    def local_search(classes, max_iter=5000):
        """Local search: try changing one class at a time."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        current_score = int(compute_score(covered))
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Pick a random class to modify
            idx = random.randint(0, len(classes) - 1)
            m_old, r_old = classes[idx]
            
            # Try all residues for this class
            # Remove old class from coverage
            # We need to recompute coverage without this class
            # This is expensive, so let's use a different approach
            
            # Alternative: for each candidate new residue, compute the delta
            # Delta = (newly covered by new residue) - (lost coverage from old residue that isn't covered by others)
            
            # For efficiency, let's only try a subset of residues if m is large
            if m_old > 200:
                # Sample some residues
                candidates = random.sample(range(m_old), min(50, m_old))
            else:
                candidates = list(range(m_old))
            
            best_new_r = r_old
            best_score = current_score
            
            for new_r in candidates:
                if new_r == r_old:
                    continue
                
                # Compute coverage without class idx, then add with new_r
                # This is expensive, so let's estimate
                # Positions covered by old residue: old_r, old_r+m, old_r+2m, ...
                # Positions covered by new residue: new_r, new_r+m, new_r+2m, ...
                
                # Let's do a quick check: if new_r == old_r, skip
                # Otherwise, compute the actual score change
                
                # For now, let's just recompute for small m
                if m_old > 100:
                    continue
                    
                covered_no_idx = np.zeros(base, dtype=np.uint8)
                for k_idx in range(len(classes)):
                    if k_idx == idx:
                        continue
                    mk, rk = classes[k_idx]
                    covered_no_idx[rk::mk] = 1
                
                # Add old residue
                covered_with_old = covered_no_idx.copy()
                covered_with_old[r_old::m_old] = 1
                score_old = int(compute_score(covered_with_old))
                
                # Add new residue
                covered_with_new = covered_no_idx.copy()
                covered_with_new[new_r::m_old] = 1
                score_new = int(compute_score(covered_with_new))
                
                if score_new < best_score:
                    best_score = score_new
                    best_new_r = new_r
            
            if best_new_r != r_old:
                classes[idx][1] = best_new_r
                current_score = best_score
                # Update covered
                covered = np.zeros(base, dtype=np.uint8)
                for m, r in classes:
                    covered[r::m] = 1
        
        return classes, current_score
    
    def simulated_annealing(classes, max_iter=10000):
        """Simulated annealing to escape local optima."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        current_score = int(compute_score(covered))
        best_classes = [c[:] for c in classes]
        best_score = current_score
        
        T = 50.0  # initial temperature
        T_min = 0.1
        cooling_rate = 0.999
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            if T < T_min:
                break
                
            # Pick a random class to modify
            idx = random.randint(0, len(classes) - 1)
            m, r_old = classes[idx]
            
            # Pick a random new residue
            if m > 100:
                new_r = random.randint(0, m - 1)
            else:
                new_r = random.randint(0, m - 1)
            
            if new_r == r_old:
                continue
                
            # Compute score change
            # Remove old, add new
            # For efficiency, compute delta
            old_positions = set(range(r_old, base, m))
            new_positions = set(range(new_r, base, m))
            
            # Positions that lose coverage: in old but not in new, and not covered by others
            # Positions that gain coverage: in new but not in old, and not covered by others
            
            # This is still expensive, so let's just recompute for small m
            if m > 200:
                continue
                
            covered_no_idx = np.zeros(base, dtype=np.uint8)
            for k_idx in range(len(classes)):
                if k_idx == idx:
                    continue
                mk, rk = classes[k_idx]
                covered_no_idx[rk::mk] = 1
            
            covered_with_old = covered_no_idx.copy()
            covered_with_old[r_old::m] = 1
            score_old = int(compute_score(covered_with_old))
            
            covered_with_new = covered_no_idx.copy()
            covered_with_new[new_r::m] = 1
            score_new = int(compute_score(covered_with_new))
            
            delta = score_new - score_old
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                classes[idx][1] = new_r
                current_score = score_new
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
            
            T *= cooling_rate
        
        return best_classes, best_score
    
    # Try multiple initial orderings
    orderings = []
    
    # Ordering 1: increasing
    orderings.append(divisors.copy())
    
    # Ordering 2: decreasing
    orderings.append(divisors[::-1].copy())
    
    # Ordering 3: sort by base//m descending (largest m first)
    orderings.append(sorted(divisors, key=lambda m: -m))
    
    # Ordering 4: sort by number of prime factors descending
    def omega(n):
        count = 0
        temp = n
        for p in [3, 5, 7, 11]:
            while temp % p == 0:
                count += 1
                temp //= p
        return count
    
    orderings.append(sorted(divisors, key=lambda m: (-omega(m), m)))
    
    # Ordering 5: sort by m ascending, then by base//m descending
    orderings.append(sorted(divisors, key=lambda m: (m, -base // m)))
    
    # Random orderings
    for _ in range(10):
        perm = divisors.copy()
        random.shuffle(perm)
        orderings.append(perm)
    
    best_classes = None
    best_score = base  # worst case: nothing covered
    
    # Phase 1: Greedy construction with different orderings
    for i, order in enumerate(orderings):
        if time.time() - start_time > deadline - 30:
            break
        try:
            classes, sc = greedy_construct(order)
            if sc < best_score:
                best_score = sc
                best_classes = classes
        except Exception:
            pass
    
    # Phase 2: Local search on the best result
    if best_classes is not None and time.time() - start_time < deadline - 20:
        try:
            best_classes, best_score = local_search(best_classes, max_iter=10000)
        except Exception:
            pass
    
    # Phase 3: Simulated annealing
    if best_classes is not None and time.time() - start_time < deadline - 10:
        try:
            best_classes, best_score = simulated_annealing(best_classes, max_iter=20000)
        except Exception:
            pass
    
    # Phase 4: Multiple restarts if time allows
    restart_count = 0
    while time.time() - start_time < deadline - 15:
        restart_count += 1
        # Create a new random ordering
        order = divisors.copy()
        random.shuffle(order)
        
        try:
            classes, sc = greedy_construct(order)
            if sc < best_score:
                best_score = sc
                best_classes = classes
            
            # Quick local search
            if time.time() - start_time < deadline - 10:
                classes, sc = local_search(classes, max_iter=2000)
                if sc < best_score:
                    best_score = sc
                    best_classes = classes
        except Exception:
            pass
    
    return best_classes if best_classes is not None else [[3, 0]]
# EVOLVE-BLOCK-END
