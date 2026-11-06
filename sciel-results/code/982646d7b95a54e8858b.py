import time
import random
import math
import numpy as np

BASE = 51975
DEADLINE_MARGIN = 12.0  # Safety margin from 150s hard kill

# Precompute divisors
def get_divisors(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            if i > 1:
                divs.append(i)
            other = n // i
            if other > 1 and other != i:
                divs.append(other)
    return divs

DIVISORS = get_divisors(BASE)
# Filter to odd divisors (all are odd since BASE is odd)
# Ensure distinct and > 1
DIVISORS = list(set(d for d in DIVISORS if d > 1))
DIVISORS.sort()

# Precompute index arrays for each modulus and residue
# indices_cache[m][r] = np.arange(r, BASE, m)
indices_cache = {}
for m in DIVISORS:
    indices_cache[m] = [np.arange(r, BASE, m) for r in range(m)]

def compute_uncovered(counts):
    return int(np.sum(counts == 0))

def greedy_initialization(order='asc', use_noise=False, noise_scale=0.0, deadline=None):
    """
    Greedy initialization.
    order: 'asc' or 'desc'
    """
    counts = np.zeros(BASE, dtype=np.int32)
    active = {}
    
    if order == 'asc':
        divs = sorted(DIVISORS)
    else:
        divs = sorted(DIVISORS, reverse=True)
        
    for m in divs:
        if deadline and time.time() > deadline:
            break
            
        best_r = 0
        max_gain = -1
        
        # Vectorized best residue finding
        # We want to maximize the number of x in indices_cache[m][r] where counts[x] == 0
        # Instead of looping r, we can compute gains for all r at once?
        # gains[r] = sum(counts[indices_cache[m][r]] == 0)
        
        # Efficient way:
        # uncovered_mask = counts == 0
        # For each m, we need to count how many uncovered indices fall into each residue class mod m.
        # indices = np.arange(BASE)
        # residues = indices[uncovered_mask] % m
        # gains = np.bincount(residues, minlength=m)
        
        uncovered_mask = counts == 0
        if np.any(uncovered_mask):
            idxs = np.nonzero(uncovered_mask)[0]
            residues = idxs % m
            gains = np.bincount(residues, minlength=m)
            if use_noise and noise_scale > 0:
                gains = gains + np.random.uniform(0, noise_scale, size=m)
            best_r = int(np.argmax(gains))
            max_gain = int(gains[best_r])
        else:
            max_gain = 0
            
        if max_gain > 0:
            active[m] = best_r
            counts[indices_cache[m][best_r]] += 1
            
    return active, counts, compute_uncovered(counts)

def local_search(active, counts, time_limit, deadline, random_seed=42):
    """
    Local search on the state (active, counts).
    Uses exact best improvement for residue changes.
    """
    rng = random.Random(random_seed)
    best_uncovered = compute_uncovered(counts)
    best_active = active.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Try changing residues of active moduli
        # Iterate through a random subset or all? All is fine if fast.
        # There are ~30-40 divisors.
        # For each m, find best new r.
        
        # To avoid bias, shuffle order
        moduli_list = list(active.keys())
        rng.shuffle(moduli_list)
        
        for m in moduli_list:
            if time.time() > deadline:
                return best_active, best_uncovered
                
            old_r = active[m]
            
            # Compute gains for all residues r for modulus m
            # Gain(r) = (# uncovered in class m,r) - (# singly covered in class m,r that are NOT in new class?)
            # Wait, the delta calculation:
            # If we change m from r_old to r_new:
            # 1. Uncover r_old: counts[idx_old] -= 1. Uncovered increases by count of idx_old where counts was 1.
            # 2. Cover r_new: counts[idx_new] += 1. Uncovered decreases by count of idx_new where counts was 0 (before step 1? No, step 1 happens first).
            
            # Let's be precise.
            # Current state: counts.
            # Move: remove m (r_old), add m (r_new).
            
            # Cost of removing m (r_old):
            # L = number of x in idx_old such that counts[x] == 1. (These become 0, so uncovered increases by L)
            
            # State after removal: counts_temp = counts - 1_{idx_old}
            # Gain of adding m (r_new) to counts_temp:
            # G = number of x in idx_new such that counts_temp[x] == 0. (These become 1, so uncovered decreases by G)
            
            # Net change in uncovered = L - G.
            
            idx_old = indices_cache[m][old_r]
            L = np.sum(counts[idx_old] == 1)
            
            # We need to evaluate G for all r_new.
            # counts_temp = counts - 1_{idx_old}
            # G(r) = sum( (counts_temp)[idx_r] == 0 )
            
            # Instead of materializing counts_temp (which is size N), we can compute G(r) efficiently.
            # counts_temp[x] = counts[x] - 1 if x in idx_old else counts[x]
            
            # For a candidate r_new:
            # G(r_new) = count of x in idx_new where (counts[x] - 1_{x in idx_old}) == 0
            # Case 1: x in idx_old. Then counts_temp[x] = counts[x] - 1.
            #   Condition: counts[x] - 1 == 0 => counts[x] == 1.
            # Case 2: x not in idx_old. Then counts_temp[x] = counts[x].
            #   Condition: counts[x] == 0.
            
            # So G(r_new) = (# x in idx_new ∩ idx_old where counts[x] == 1) + (# x in idx_new \ idx_old where counts[x] == 0)
            
            # This is still a bit complex to vectorize across all r_new simultaneously without loops or large memory.
            # Given N=51975, and m is small (max 51975 but usually much smaller),
            # looping over r_new in range(m) might be okay if m is small.
            # But for large m, it's slow.
            
            # Alternative: Just loop r_new and compute delta directly?
            # For each r_new:
            #   idx_new = indices_cache[m][r_new]
            #   # Intersection logic is expensive in Python.
            
            # Let's stick to the previous approach in the seed:
            # Temporarily remove, find best, add.
            # This is O(N) per m. With ~40 moduli, 40 * 50k = 2M ops per pass.
            # This is fast enough for a few passes.
            
            # Remove m
            counts[idx_old] -= 1
            
            # Find best r_new
            uncovered_mask = counts == 0
            if np.any(uncovered_mask):
                idxs = np.nonzero(uncovered_mask)[0]
                residues = idxs % m
                gains = np.bincount(residues, minlength=m)
                best_new_r = int(np.argmax(gains))
                max_gain = int(gains[best_new_r])
            else:
                best_new_r = old_r
                max_gain = 0
                
            # Add m with best_new_r
            idx_new = indices_cache[m][best_new_r]
            counts[idx_new] += 1
            
            current_uncovered = compute_uncovered(counts)
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_active = active.copy()
                active[m] = best_new_r
                improved = True
                no_improve_count = 0
            else:
                # Revert
                # We added idx_new, need to remove it.
                counts[idx_new] -= 1
                # We removed idx_old, need to add it back.
                counts[idx_old] += 1
                no_improve_count += 1
                
            # Early exit if no improvement in many steps?
            # We are in a loop over moduli.
            
        if not improved:
            # Try adding inactive moduli
            inactive = [m for m in DIVISORS if m not in active]
            rng.shuffle(inactive)
            
            for m in inactive:
                if time.time() > deadline:
                    return best_active, best_uncovered
                    
                # Find best r for m
                uncovered_mask = counts == 0
                if not np.any(uncovered_mask):
                    break
                    
                idxs = np.nonzero(uncovered_mask)[0]
                residues = idxs % m
                gains = np.bincount(residues, minlength=m)
                best_r = int(np.argmax(gains))
                max_gain = int(gains[best_r])
                
                if max_gain > 0:
                    idx_new = indices_cache[m][best_r]
                    counts[idx_new] += 1
                    active[m] = best_r
                    
                    current_uncovered = compute_uncovered(counts)
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_active = active.copy()
                        improved = True
                        no_improve_count = 0
                    else:
                        counts[idx_new] -= 1
                        del active[m]
                        no_improve_count += 1
                        
            if not improved:
                # Try removing active moduli
                moduli_list = list(active.keys())
                rng.shuffle(moduli_list)
                
                for m in moduli_list:
                    if time.time() > deadline:
                        return best_active, best_uncovered
                        
                    r = active[m]
                    idx_old = indices_cache[m][r]
                    counts[idx_old] -= 1
                    del active[m]
                    
                    current_uncovered = compute_uncovered(counts)
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_active = active.copy()
                        improved = True
                        no_improve_count = 0
                    else:
                        counts[idx_old] += 1
                        active[m] = r
                        no_improve_count += 1
                        
            if not improved:
                if no_improve_count > max_no_improve:
                    break
                else:
                    # Random kick
                    # Pick random m (active or inactive) and random r
                    if active:
                        m = rng.choice(list(active.keys()))
                        old_r = active[m]
                        new_r = rng.randint(0, m - 1)
                        if new_r != old_r:
                            idx_old = indices_cache[m][old_r]
                            idx_new = indices_cache[m][new_r]
                            counts[idx_old] -= 1
                            counts[idx_new] += 1
                            active[m] = new_r
                            no_improve_count = 0
                    else:
                        break

    return best_active, best_uncovered

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 150s limit, 12s margin
    
    best_score = BASE
    best_classes = []
    
    # Phase 1: Multiple Greedy Initializations
    initializations = [
        {'order': 'asc', 'noise': 0.0},
        {'order': 'desc', 'noise': 0.0},
        {'order': 'asc', 'noise': 1.0},
        {'order': 'desc', 'noise': 1.0},
        {'order': 'asc', 'noise': 5.0},
        {'order': 'desc', 'noise': 5.0},
    ]
    
    for init_config in initializations:
        if time.time() > deadline - 30: # Leave time for search
            break
            
        active, counts, score = greedy_initialization(
            order=init_config['order'],
            use_noise=init_config['noise'] > 0,
            noise_scale=init_config['noise'],
            deadline=deadline - 30
        )
        
        if score < best_score:
            best_score = score
            best_classes = [[m, r] for m, r in active.items()]
            
    # Phase 2: Local Search on the best initialization
    # Reconstruct state from best_classes
    active = {m: r for m, r in best_classes}
    counts = np.zeros(BASE, dtype=np.int32)
    for m, r in active.items():
        counts[indices_cache[m][r]] += 1
        
    # Run local search
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        active, score = local_search(active, counts, time_limit=time_remaining - 5, deadline=deadline, random_seed=42)
        
        if score < best_score:
            best_score = score
            best_classes = [[m, r] for m, r in active.items()]
            
    # Phase 3: More restarts if time permits
    while time.time() < deadline - 10:
        # Perturb current best
        active = {m: r for m, r in best_classes}
        counts = np.zeros(BASE, dtype=np.int32)
        for m, r in active.items():
            counts[indices_cache[m][r]] += 1
            
        # Random perturbation
        if active:
            keys = list(active.keys())
            for _ in range(min(3, len(keys))):
                m = random.choice(keys)
                old_r = active[m]
                new_r = random.randint(0, m - 1)
                if new_r != old_r:
                    counts[indices_cache[m][old_r]] -= 1
                    counts[indices_cache[m][new_r]] += 1
                    active[m] = new_r
        else:
            # If empty, pick a random divisor
            m = random.choice(DIVISORS)
            r = random.randint(0, m - 1)
            active[m] = r
            counts[indices_cache[m][r]] += 1
            
        # Local search
        time_remaining = deadline - 10 - time.time()
        if time_remaining > 2:
            active_perturbed, score_perturbed = local_search(
                active, counts, 
                time_limit=time_remaining, 
                deadline=deadline - 10, 
                random_seed=random.randint(0, 10000)
            )
            
            if score_perturbed < best_score:
                best_score = score_perturbed
                best_classes = [[m, r] for m, r in active_perturbed.items()]

    return best_classes

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Classes: {len(res)}")
    # Verify
    counts = np.zeros(BASE, dtype=np.int32)
    for m, r in res:
        if m > 1 and BASE % m == 0 and 0 <= r < m:
            counts[indices_cache[m][r]] += 1
        else:
            print("Invalid class:", m, r)
    uncovered = np.sum(counts == 0)
    print(f"Uncovered: {uncovered}")
