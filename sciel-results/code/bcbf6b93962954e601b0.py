import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay well under 90s kill

    N = 256

    # 1. Precompute ALL APs (length >= 1) and build incidence structure
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    # Iterate over all valid (start, step) pairs
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d

            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)

    num_aps = len(ap_indices)

    # Precompute the initial sums for a given sign vector using matrix-vector product
    # Build a sparse matrix A where A[ap, pos] = 1 if pos is in ap
    # Use numpy for speed: build as list of (row, col) pairs for scipy sparse, or just use the list approach.
    # Since we have many APs, let's build a dense-ish approach using numpy broadcasting where possible.
    # Actually, let's use a different approach: for each AP, we know its indices.
    # We can compute sums by iterating over APs and summing signs[idx_list].

    def thue_morse(N):
        """Generate Thue-Morse sequence as +/-1 signs."""
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            try:
                parity = i.bit_count() & 1
            except AttributeError:
                parity = bin(i).count('1') & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def compute_sums(signs):
        """Compute sums for all APs given a signs array."""
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(signs[idx_list])
        return sums

    def continuous_relaxation(time_budget, rng):
        """
        Use gradient descent on a continuous surrogate to find a good discrete starting point.
        Loss: sum of squared AP sums.
        Regularization: push towards +/-1.
        """
        x = np.array([rng.uniform(-0.5, 0.5) for _ in range(N)], dtype=np.float64)
        lr = 0.05
        lr_reg = 0.1
        
        # Precompute list of AP indices for each position for fast gradient update
        # aps_containing is already a list of lists of AP indices.
        
        last_check = time.time()
        
        while time.time() - start_time < time_limit - (time_limit - time_budget): # Simplified: just check against a deadline
            # Compute S = A x
            S = np.zeros(num_aps, dtype=np.float64)
            for r, idx_list in enumerate(ap_indices):
                S[r] = np.sum(x[idx_list])
            
            # Compute Gradient G = A^T S
            G = np.zeros(N, dtype=np.float64)
            for i in range(N):
                # Sum S[k] for all k containing i
                # This loop is O(total incidences). Total incidences is sum of lengths of all APs.
                # It's roughly N * num_aps / 2? No, it's the sum over all APs of their length.
                # For N=256, this is manageable if we do it efficiently.
                # However, doing this in a Python loop every step might be slow.
                # Let's optimize: 
                # We can just iterate over all APs and add their contribution to G.
                pass # Placeholder for optimization
            
            # Better gradient computation:
            G = np.zeros(N, dtype=np.float64)
            for r, idx_list in enumerate(ap_indices):
                if S[r] != 0:
                    for idx in idx_list:
                        G[idx] += S[r]
            
            # Update x
            x -= lr * G
            
            # Regularization: x = x - lr_reg * (x^3 - x)
            # This creates a potential well at -1, 0, 1. We want to avoid 0.
            # Actually, we want to push away from 0.
            # Let's use a simpler projection or stronger regularization.
            # x = np.tanh(x) is a good squashing function, but gradient vanishes.
            # Let's just clip and add a small pull towards +/-1 if |x| > 0.5
            x = np.clip(x, -1.5, 1.5)
            
            # Check time
            if time.time() - last_check > 0.1:
                last_check = time.time()
                # Discretize and check score occasionally?
                # Or just let it run.
                
            # Early exit if time is up
            if time.time() - start_time > time_limit - 1.0:
                break

        # Discretize
        s = np.sign(x)
        # Handle zeros
        s[s == 0] = 1
        return s

    def greedy_optimize(s_init, time_budget, rng):
        """
        Improved Greedy Coordinate Descent with:
        - Fast sum updates using precomputed affected AP lists
        - Smarter tie-breaking based on number of tight constraints
        - Periodic perturbation to escape local minima
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))

        # Precompute for each position i: the list of AP indices and their current sums
        # We'll maintain sums array and update it incrementally.

        no_improve_sweeps = 0
        max_no_improve = 5
        sweep_count = 0

        # Local time limit for this specific optimization run
        local_deadline = time.time() + time_budget

        while time.time() < local_deadline:
            rng.shuffle(indices_list)
            improved_this_sweep = False
            sweep_count += 1

            for i in indices_list:
                if time.time() > local_deadline:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = aps_containing[i]

                # Save old sums for affected APs
                old_sums = sums[affected].copy()

                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    improved_this_sweep = True
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                elif new_max == current_max:
                    # Count how many APs are at the boundary (|sum| == current_max)
                    # Prefer flips that reduce the number of tight constraints
                    old_tight = np.sum(np.abs(old_sums) == current_max)
                    new_tight = np.sum(np.abs(sums[affected]) == current_max)
                    # Also check if any other APs became tight
                    # This is approximate but fast
                    if new_tight < old_tight:
                        improved_this_sweep = True
                    elif new_tight == old_tight:
                        # Accept with small probability to explore
                        if rng.random() < 0.02:
                            improved_this_sweep = True
                        else:
                            sums[affected] = old_sums
                            s[i] = old_val
                    else:
                        # Revert
                        sums[affected] = old_sums
                        s[i] = old_val
                else:
                    # Revert
                    sums[affected] = old_sums
                    s[i] = old_val

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    # Try a perturbation: flip a random subset of positions
                    if rng.random() < 0.5:
                        num_flips = rng.randint(3, 10)
                        for _ in range(num_flips):
                            j = rng.choice(indices_list)
                            s[j] = -s[j]
                            sums[aps_containing[j]] += -2 * s[j]
                        current_max = int(np.max(np.abs(sums)))
                        no_improve_sweeps = 0
                    else:
                        break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    # 3. Main loop: multiple restarts with different initializations
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Run 1: Continuous Relaxation to get a good start
    if time.time() - start_time < time_limit * 0.5:
        s_cont = continuous_relaxation(time_limit * 0.4, rng_global)
        s1, score1 = greedy_optimize(s_cont, time_limit * 0.2, rng_global)
        if score1 < best_global_score:
            best_global_score = score1
            best_global_s = s1.copy()

    # Run 2: Thue-Morse
    if time.time() - start_time < time_limit * 0.7:
        tm = thue_morse(N)
        s2, score2 = greedy_optimize(tm, time_limit * 0.15, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Run 3: Random init
    if time.time() - start_time < time_limit * 0.85:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s3, score3 = greedy_optimize(rand_init, time_limit * 0.1, rng_global)
        if score3 < best_global_score:
            best_global_score = score3
            best_global_s = s3.copy()

    # Remaining runs: more random inits with shorter budgets
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 2.0:
        num_extra = min(5, int(remaining_budget / 2.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            budget = min(2.0, time_limit - (time.time() - start_time) - 1.0)
            if budget < 0.5:
                break
            s_k, score_k = greedy_optimize(rand_init, budget, rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
