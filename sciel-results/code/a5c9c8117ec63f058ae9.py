import time
import random
import sys
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115.0  # Leave 5s buffer for 120s limit
    
    # 1. Precompute all 4-APs as a matrix
    # ap_matrix: shape (num_aps, 4), each row is (a, a+d, a+2d, a+3d) in 0-based
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    ap_matrix = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # 2. Build incidence structure for vectorized delta computation
    # For each position i (0..n-1), we need to know which APs it participates in
    # and what the "other three" positions are.
    # 
    # Key insight for vectorized computation:
    # Let x be the coloring vector (0/1). For an AP (a,b,c,d), it's monochromatic
    # iff all four are equal. 
    # 
    # When we flip bit i, the change in mono count depends on the other 3 bits.
    # For each AP containing i with others (j,k,l):
    #   - If x[j]==x[k]==x[l]==x[i]: flipping i makes it non-mono, delta -= 1
    #   - If x[j]==x[k]==x[l]!=x[i]: flipping i makes it mono, delta += 1
    #   - Otherwise: no change
    #
    # Vectorized approach: Build sparse matrices
    # For each position i, let A_i be the set of APs containing i.
    # For each such AP, store the other 3 positions.
    #
    # We can compute the delta for ALL positions simultaneously using matrix ops.
    # 
    # Let's define:
    # For each AP p = (a,b,c,d) and each position i in p, the "others" are the other 3.
    # 
    # Vectorized delta computation:
    # For position i, delta_i = sum over APs p containing i of:
    #   sign * indicator(others are all same and match/mismatch i)
    #
    # Let's build:
    # M[i, p] = 1 if position i is in AP p, else 0
    # This is n x num_aps binary matrix.
    
    M = np.zeros((n, num_aps), dtype=np.float32)
    for p in range(num_aps):
        for pos in ap_matrix[p]:
            M[pos, p] = 1.0
    
    # For each AP p and position i in p, the other three positions.
    # We need to compute: for each (i, p) where M[i,p]=1, are the other 3 all equal?
    # And do they match i or not?
    #
    # Let x be the coloring as a vector of 0/1.
    # For AP p = (a,b,c,d):
    #   others_for_a = (b,c,d), others_for_b = (a,c,d), etc.
    #
    # Vectorized: For each position i, we want to sum over APs p containing i:
    #   let others = the 3 positions in p other than i
    #   let prod_others = x[o1]*x[o2]*x[o3] + (1-x[o1])*(1-x[o2])*(1-x[o3])
    #     This is 1 if all others are equal (all 0 or all 1), else 0.
    #   let match = 1 if x[i] == x[o1] (they're all equal so any one suffices)
    #   delta contribution = prod_others * (1 if match else -1)
    #     = prod_others * (1 - 2*|x[i] - x[o1]|)
    #     Since if all others equal, x[o1]=x[o2]=x[o3], so |x[i]-x[o1]| tells us match.
    #
    # So delta_i = sum_p M[i,p] * prod_others(i,p) * (1 - 2*|x[i] - x[o1(i,p)]|)
    #
    # This is still complex to fully vectorize without loops over APs.
    # 
    # Alternative: Precompute for each position i, the list of (o1,o2,o3) tuples.
    # Then for a given x, compute delta for all i in a loop over i but vectorized over APs per i.
    # 
    # Actually, let's try a different vectorization:
    # Build a 3D tensor or use the AP matrix directly.
    #
    # For each AP p=(a,b,c,d), define:
    #   all_same = (x[a]==x[b]) & (x[b]==x[c]) & (x[c]==x[d])
    #   This is 1 if mono.
    #
    # When flipping position i, which APs are affected? Those containing i.
    # For each such AP, the new all_same can be computed.
    #
    # Let's precompute for each position i:
    #   ap_indices_i = list of AP indices p where i is in AP p
    #   for each such p, the other 3 positions
    #
    # Then for a batched flip of multiple positions, we need to be careful about
    # interactions. But if we flip one at a time in a smart order, or if we
    # compute the "greedy" flip set iteratively...
    #
    # Let's do: iterative batched local search
    #   1. Compute delta for all 256 positions simultaneously (using precomputed structure)
    #   2. Flip all positions with delta < 0
    #   3. Repeat until no improvement or time runs out
    #   4. Add random perturbations to escape local minima
    
    # Precompute for each position i: list of (o1, o2, o3) for each AP containing i
    neighbors = [[] for _ in range(n)]
    for p in range(num_aps):
        a, b, c, d = ap_matrix[p]
        neighbors[a].append((b, c, d))
        neighbors[b].append((a, c, d))
        neighbors[c].append((a, b, d))
        neighbors[d].append((a, b, c))
    
    # Convert neighbors to numpy arrays for speed
    # neighbors_arr[i] will be a 2D array of shape (num_aps_i, 3)
    neighbors_arr = []
    for i in range(n):
        if neighbors[i]:
            neighbors_arr.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_arr.append(np.empty((0, 3), dtype=np.int32))
    
    def count_mono_vec(x):
        """Count monochromatic 4-APs using vectorized numpy."""
        x = x.astype(np.float32)
        # For each AP, check if all 4 are equal
        # mono_p = (x[a]==x[b]) & (x[b]==x[c]) & (x[c]==x[d])
        # Vectorized: 
        cols = ap_matrix  # shape (num_aps, 4)
        x_ap = x[cols]  # shape (num_aps, 4)
        # All equal iff max-min == 0
        mono = (x_ap.max(axis=1) - x_ap.min(axis=1)) == 0
        return int(mono.sum())
    
    def compute_deltas(x):
        """Compute the delta (change in mono count) if we flip each position i.
        Returns an array of shape (n,) with delta for each position.
        delta[i] < 0 means flipping i reduces mono count (good).
        """
        x = x.astype(np.float32)
        deltas = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            if len(neighbors_arr[i]) == 0:
                continue
            others = neighbors_arr[i]  # shape (k, 3)
            # For each AP containing i, check if the other 3 are all equal
            o1 = x[others[:, 0]]  # shape (k,)
            o2 = x[others[:, 1]]
            o3 = x[others[:, 2]]
            
            # All three others equal?
            all_same = ((o1 == o2) & (o2 == o3)).astype(np.float32)
            
            # Do they match x[i]?
            # If all_same, then x[i] == o1 tells us if flipping helps or hurts
            matches = (x[i] == o1).astype(np.float32)
            
            # delta_i = sum(all_same * (1 if matches else -1))
            #         = sum(all_same * (2*matches - 1))
            deltas[i] = np.sum(all_same * (2.0 * matches - 1.0))
        
        return deltas
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def batched_local_search(x0, time_limit, rng):
        """Iteratively flip all beneficial bits until convergence or time limit."""
        x = x0.copy()
        current_score = count_mono_vec(x)
        start_time = time.time()
        
        max_iter = 10000
        for it in range(max_iter):
            if time.time() > start_time + time_limit:
                break
            
            deltas = compute_deltas(x)
            
            # Find all positions with delta < 0 (flipping reduces mono count)
            beneficial = np.where(deltas < 0)[0]
            
            if len(beneficial) == 0:
                # No single flip helps. Try random perturbation to escape local min.
                # Flip a random subset of positions
                num_flips = rng.randint(1, 8)
                flip_mask = np.zeros(n, dtype=bool)
                flip_indices = rng.sample(range(n), num_flips)
                flip_mask[flip_indices] = True
                x[flip_mask] = 1 - x[flip_mask]
                new_score = count_mono_vec(x)
                if new_score > current_score + 5:
                    # Revert if too bad
                    x[flip_mask] = 1 - x[flip_mask]
                else:
                    current_score = new_score
                continue
            
            # Flip all beneficial positions
            x[beneficial] = 1 - x[beneficial]
            current_score = count_mono_vec(x)
        
        return x, current_score
    
    def run_with_restarts(time_budget):
        """Run batched local search with multiple random restarts."""
        best_x = None
        best_score = float('inf')
        
        rng = random.Random(12345)
        start_time = time.time()
        
        # Multiple restart strategies
        strategies = []
        
        # Strategy 1: Thue-Morse
        strategies.append(np.array(thue_morse(n), dtype=np.float32))
        
        # Strategy 2: Random
        for _ in range(10):
            strategies.append(np.random.RandomState(rng.randint(0, 2**31)).randint(0, 2, n).astype(np.float32))
        
        # Strategy 3: Thue-Morse with random perturbations
        tm = np.array(thue_morse(n), dtype=np.float32)
        for _ in range(10):
            perturbed = tm.copy()
            num_flips = rng.randint(5, 20)
            flip_indices = rng.sample(range(n), num_flips)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            strategies.append(perturbed)
        
        # Strategy 4: Spectral initialization from pair-interaction matrix
        # Build a matrix where entry (i,j) counts how many APs contain both i and j
        pair_count = np.zeros((n, n), dtype=np.float32)
        for p in range(num_aps):
            a, b, c, d = ap_matrix[p]
            for i in range(4):
                for j in range(i+1, 4):
                    pos_i = ap_matrix[p, i]
                    pos_j = ap_matrix[p, j]
                    pair_count[pos_i, pos_j] += 1
                    pair_count[pos_j, pos_i] += 1
        
        # The eigenvalue problem: find x that minimizes sum over APs of (all same)
        # Approximate: minimize sum_{i,j} pair_count[i,j] * x[i]*x[j] + ...
        # Use the eigenvector of pair_count corresponding to the most negative eigenvalue
        # (for a symmetric matrix, we want the ground state)
        try:
            eigvals, eigvecs = np.linalg.eigh(pair_count)
            # Most negative eigenvalue gives the "ground state"
            idx_min = np.argmin(eigvals)
            spectral_vec = eigvecs[:, idx_min]
            # Convert to binary: threshold at 0
            spectral_bin = (spectral_vec > 0).astype(np.float32)
            strategies.append(spectral_bin)
        except:
            pass
        
        # Run each strategy
        for s_idx, x0 in enumerate(strategies):
            remaining = time_budget - (time.time() - start_time)
            if remaining < 2.0:
                break
            
            time_per_run = min(remaining - 1.0, 15.0)  # Cap each run at 15s
            if time_per_run < 1.0:
                break
            
            x, score = batched_local_search(x0, time_per_run, rng)
            if score < best_score:
                best_score = score
                best_x = x.copy()
        
        return best_x, best_score
    
    # Main execution
    x_best, score_best = run_with_restarts(110.0)
    
    # Final polish: one more round of batched local search on the best
    remaining = deadline - time.time()
    if remaining > 2.0 and x_best is not None:
        x_best, score_best = batched_local_search(x_best, remaining - 1.0, random.Random(999))
    
    return x_best.astype(int).tolist()

# EVOLVE-BLOCK-END
