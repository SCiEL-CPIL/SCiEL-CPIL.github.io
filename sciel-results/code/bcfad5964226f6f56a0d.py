import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort()
    
    # Precompute indices for each modulus and residue
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            idx = np.arange(r, base, m)
            res_indices.append(idx)
        indices_by_m[m] = res_indices

    # State
    selected_moduli = []
    residues = []
    cover_count = np.zeros(base, dtype=np.int32)
    uncovered_count = base
    
    def get_gains_losses(m, r_curr=None):
        """
        Compute gains and losses for all residues of modulus m.
        gains[r] = number of x in class r where cover_count[x] == 0
        losses[r] = number of x in class r where cover_count[x] == 1
        """
        is_uncovered = (cover_count == 0)
        is_solo = (cover_count == 1)
        
        gains = np.zeros(m, dtype=np.int32)
        losses = np.zeros(m, dtype=np.int32)
        
        for r in range(m):
            idx = indices_by_m[m][r]
            gains[r] = np.sum(is_uncovered[idx])
            losses[r] = np.sum(is_solo[idx])
            
        return gains, losses

    def apply_change(m, r_old, r_new):
        """
        Change residue of modulus m from r_old to r_new.
        Returns net change in uncovered count (negative is improvement).
        """
        nonlocal cover_count, uncovered_count
        
        idx_old = indices_by_m[m][r_old]
        idx_new = indices_by_m[m][r_new]
        
        # Remove old
        # Elements in idx_old that were covered only by this class become uncovered
        # cover_count[idx_old] decreases by 1
        # If it becomes 0, it was 1 before, so it was solo covered.
        solo_old = (cover_count[idx_old] == 1)
        num_solo_old = np.sum(solo_old)
        
        cover_count[idx_old] -= 1
        
        # Add new
        # Elements in idx_new that were uncovered become covered
        # cover_count[idx_new] increases by 1
        # If it becomes 1, it was 0 before, so it was uncovered.
        uncov_new = (cover_count[idx_new] == 0)
        num_uncov_new = np.sum(uncov_new)
        
        cover_count[idx_new] += 1
        
        # Update uncovered count
        # We lost num_solo_old covered residues (they are now uncovered)
        # We gained num_uncov_new covered residues (they are now covered)
        # Wait, if an element is in BOTH idx_old and idx_new?
        # m is the same, r_old != r_new, so they are disjoint.
        
        uncovered_count += num_solo_old
        uncovered_count -= num_uncov_new
        
        return (num_solo_old - num_uncov_new) # Positive means worse (more uncovered)

    def add_class(m, r):
        nonlocal cover_count, uncovered_count
        idx = indices_by_m[m][r]
        uncov = (cover_count[idx] == 0)
        num_uncov = np.sum(uncov)
        
        cover_count[idx] += 1
        uncovered_count -= num_uncov
        
        selected_moduli.append(m)
        residues.append(r)
        
    def remove_class(index):
        nonlocal cover_count, uncovered_count
        m = selected_moduli[index]
        r = residues[index]
        idx = indices_by_m[m][r]
        
        solo = (cover_count[idx] == 1)
        num_solo = np.sum(solo)
        
        cover_count[idx] -= 1
        uncovered_count += num_solo
        
        selected_moduli.pop(index)
        residues.pop(index)

    # 1. Greedy Initialization
    available = set(divisors)
    while True:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in available:
            if time.time() > deadline - 5: break
            gains, _ = get_gains_losses(m)
            g = np.max(gains)
            if g > best_gain:
                best_gain = g
                best_m = m
                best_r = np.argmax(gains)
        
        if best_gain == 0 or best_m is None:
            break
            
        add_class(best_m, best_r)
        available.remove(best_m)
        
        if time.time() > deadline - 5: break

    best_uncovered = uncovered_count
    best_classes = [[m, r] for m, r in zip(selected_moduli, residues)]

    # 2. Local Search with Simulated Annealing
    T = 10.0
    cooling = 0.95
    
    # Precompute list of unselected moduli for swap logic
    # We'll just iterate divisors and check if in selected_moduli
    
    while time.time() < deadline - 5:
        improved = False
        
        # Pass 1: Optimize Residues
        for i in range(len(selected_moduli)):
            if time.time() > deadline - 5: break
            m = selected_moduli[i]
            r_curr = residues[i]
            
            gains, losses = get_gains_losses(m)
            
            # We want to switch to r_new if (gains[r_new] - losses[r_curr]) > 0
            # Net change in uncovered = losses[r_curr] - gains[r_new]
            # We want to minimize this, i.e., maximize (gains[r_new] - losses[r_curr])
            
            best_r_new = r_curr
            best_net_gain = 0 # Current state has net change 0 vs itself
            
            for r_new in range(m):
                if r_new == r_curr: continue
                net_gain = gains[r_new] - losses[r_curr]
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_r_new = r_new
            
            if best_net_gain > 0:
                # Apply change
                apply_change(m, r_curr, best_r_new)
                residues[i] = best_r_new
                improved = True
                # Restart residue optimization for this modulus? 
                # Or just continue. Since we changed cover_count, gains/losses for others changed.
                # For efficiency, we can just continue to next modulus, as the change might be small.
                # But to be safe, we could break and restart the pass.
                # Given time constraints, we'll continue.
        
        # Pass 2: Remove useless classes
        for i in range(len(selected_moduli) - 1, -1, -1):
            if time.time() > deadline - 5: break
            m = selected_moduli[i]
            r = residues[i]
            _, losses = get_gains_losses(m)
            # Loss of removing current class is losses[r]
            if losses[r] == 0:
                remove_class(i)
                improved = True
        
        # Pass 3: Add useful classes
        selected_set = set(selected_moduli)
        for m in divisors:
            if time.time() > deadline - 5: break
            if m in selected_set: continue
            
            gains, _ = get_gains_losses(m)
            g = np.max(gains)
            if g > 0:
                r = np.argmax(gains)
                add_class(m, r)
                selected_set.add(m)
                improved = True
                # Don't break, try to add more if possible? 
                # Adding one changes cover_count, so subsequent gains change.
                # We'll just do one pass.
        
        # Pass 4: Swap
        # For each selected, check if swapping with an unselected yields improvement
        # This is O(K * U * N) which is expensive.
        # Simplified: Only swap if current class has high loss and new class has high gain.
        
        if not improved:
            # Perturbation
            if selected_moduli:
                idx = random.randint(0, len(selected_moduli) - 1)
                m = selected_moduli[idx]
                r_curr = residues[idx]
                r_new = random.randint(0, m - 1)
                if r_new != r_curr:
                    delta = apply_change(m, r_curr, r_new)
                    # Accept if better (delta < 0) or with prob exp(-delta/T)
                    if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
                        residues[idx] = r_new
                    else:
                        # Revert
                        apply_change(m, r_new, r_curr)
            else:
                # Randomly add a class
                m = random.choice(divisors)
                r = random.randint(0, m - 1)
                add_class(m, r)
                
            T *= cooling
            
        if uncovered_count < best_uncovered:
            best_uncovered = uncovered_count
            best_classes = [[m, r] for m, r in zip(selected_moduli, residues)]
            if best_uncovered == 0:
                break

    return best_classes

