import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 145
    
    # Precompute all 4-APs as numpy arrays
    aps_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps_list.append((a, a+d, a+2*d, a+3*d))
    
    aps_np = np.array(aps_list, dtype=np.int32)  # shape (M, 4)
    M = len(aps_np)
    
    # Precompute for single-bit delta (used for final polish)
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps[a].append((idx, b, c, d_pos))
        pos_aps[b].append((idx, a, c, d_pos))
        pos_aps[c].append((idx, a, b, d_pos))
        pos_aps[d_pos].append((idx, a, b, c))
        
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 4), dtype=np.int32))

    def count_mono(s):
        """s is array of 1/-1. Returns count of mono 4-APs."""
        # Convert to 0/1 for standard count if needed, but we can use spin formula
        # Cost = (M + sum(s_i s_j s_k s_l)) / 2
        s_ap = s[aps_np]
        products = np.prod(s_ap, axis=1)
        return int((M + np.sum(products)) / 2)

    def compute_U(s):
        """Compute U_i = sum of products of APs containing i."""
        s_ap = s[aps_np]
        P = np.prod(s_ap, axis=1)
        U = np.zeros(N, dtype=np.int32)
        np.add.at(U, aps_np[:, 0], P)
        np.add.at(U, aps_np[:, 1], P)
        np.add.at(U, aps_np[:, 2], P)
        np.add.at(U, aps_np[:, 3], P)
        return U

    def batch_sa_step(s, T, rng):
        """
        Attempt a batch flip.
        Returns (new_s, accepted_delta_Q).
        """
        U = compute_U(s)
        # Flip if U_i > 0
        flip_mask = U > 0
        if not np.any(flip_mask):
            return s, 0
        
        # Delta Q for flipping all these bits simultaneously
        # Note: This assumes independence. The true delta is more complex if bits interact.
        # However, since we are minimizing sum of 4-body terms, and each term involves at most 1 flipped bit 
        # if we flip disjoint sets? No, flipping a set F changes all APs that intersect F.
        # The approximation delta_Q = -2 * sum(U_i for i in F) is valid if the flipped bits do not share APs.
        # They do share APs. 
        # But this is a heuristic "gradient" step. It points in the direction of decreasing energy 
        # if we consider the linearized change.
        
        delta_Q_approx = -2 * np.sum(U[flip_mask])
        
        # If delta is negative, always accept.
        # If positive, accept with probability exp(-delta/T).
        if delta_Q_approx < 0:
            accept = True
        else:
            if T > 0.01:
                prob = math.exp(-delta_Q_approx / T)
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            new_s = s.copy()
            new_s[flip_mask] *= -1
            return new_s, delta_Q_approx
        else:
            return s, 0

    def single_bit_polish(s, max_iters=100):
        """
        Greedy single-bit polish with limited iterations to prevent timeout.
        """
        for _ in range(max_iters):
            U = compute_U(s)
            # In single bit flip, delta_Q_i = -2 * U_i
            # So we flip if U_i > 0
            flip_mask = U > 0
            if not np.any(flip_mask):
                break
            s[flip_mask] *= -1
        return s

    def thue_morse_spins():
        """Generate Thue-Morse spins (1-indexed)."""
        spins = np.ones(N, dtype=np.int8)
        for i in range(N):
            n = i + 1
            if bin(n).count('1') % 2 == 1:
                spins[i] = -1
        return spins

    def generate_initial_spins():
        """Generate diverse initial spin configurations."""
        inits = []
        
        # Thue-Morse
        inits.append(thue_morse_spins())
        
        # Inverted Thue-Morse
        inits.append(-thue_morse_spins())
        
        # Random
        for _ in range(5):
            inits.append(np.random.choice([-1, 1], size=N).astype(np.int8))
            
        # Block patterns
        for block in [2, 4, 8]:
            p = np.array([(i // block) % 2 for i in range(N)], dtype=np.int8)
            spins = 1 - 2 * p # 0->1, 1->-1
            inits.append(spins)
            
        return inits

    initial_spins = generate_initial_spins()
    
    # Evaluate initial best
    best_s = initial_spins[0].copy()
    best_cost = count_mono(best_s)
    
    for s in initial_spins[1:]:
        c = count_mono(s)
        if c < best_cost:
            best_cost = c
            best_s = s.copy()
    
    # Batch Simulated Annealing
    # Reduce restarts and time per restart to ensure we finish within 120s
    num_restarts = 10
    time_per_restart = 10.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Choose initial
        if restart < len(initial_spins):
            current_s = initial_spins[restart % len(initial_spins)].copy()
        else:
            # Perturb best
            current_s = best_s.copy()
            # Flip random subset
            flip_prob = 0.2 + 0.1 * (restart % 3)
            mask = np.random.random(N) < flip_prob
            current_s[mask] *= -1
            
        T = 2.0
        cooling = 0.9999
        min_T = 0.01
        rng = random.Random(42 + restart)
        
        end_time = time.time() + time_per_restart * 0.9
        
        while time.time() < end_time:
            current_s, delta = batch_sa_step(current_s, T, rng)
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Polish the result of SA
        polished_s = single_bit_polish(current_s, max_iters=50)
        polished_cost = count_mono(polished_s)
        
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_s = polished_s.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        final_s = single_bit_polish(best_s, max_iters=100)
        final_cost = count_mono(final_s)
        if final_cost < best_cost:
            best_cost = final_cost
            best_s = final_s
    
    # Convert spins back to colors {0, 1}
    # s=1 -> 0, s=-1 -> 1 ? Or s=1 -> 1?
    # Task says colors in {0,1}.
    # My spin convention: s=1, s=-1.
    # Cost calculation: Mono if s_i s_j s_k s_l = 1.
    # If s_i=1 (color 0) and s_j=1 (color 0), product 1.
    # If s_i=-1 (color 1) and s_j=-1 (color 1), product 1.
    # So mapping s=1 -> 0, s=-1 -> 1 is consistent with "same color".
    colors = (1 - best_s) // 2
    return colors.tolist()
