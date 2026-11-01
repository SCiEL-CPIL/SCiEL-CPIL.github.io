import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Number of replicas for Parallel Tempering
    n_reps = 10
    
    # Temperature schedule: Geometric progression
    # T_min allows local search, T_max allows global mixing
    T_min = 0.5
    T_max = 50.0
    # Create a geometric temperature ladder
    # We want n_reps temperatures
    # T_i = T_min * (T_max/T_min)^(i/(n_reps-1))
    if n_reps > 1:
        ratio = (T_max / T_min) ** (1.0 / (n_reps - 1))
        temperatures = T_min * (ratio ** np.arange(n_reps))
    else:
        temperatures = np.array([T_min])
        
    # Initialize replicas with random signs
    # Use different seeds for diversity
    np.random.seed(42)
    replicas = [np.random.choice([-1.0, 1.0], size=N) for _ in range(n_reps)]
    
    # Precompute FFT of the autocorrelation for fast energy calculation?
    # Actually, for updates, we can use the property that changing h[idx] changes R(tau) locally.
    # However, calculating the *change* in the sum of squares (L2 energy) is O(N).
    # E = sum_{tau=1}^{N-1} R(tau)^2
    # Delta E when flipping h[idx]:
    # R_new(tau) = R_old(tau) + delta(tau)
    # E_new = sum (R_old(tau) + delta(tau))^2
    #       = sum R_old(tau)^2 + 2 sum R_old(tau)delta(tau) + sum delta(tau)^2
    # Delta E = 2 sum R_old(tau)delta(tau) + sum delta(tau)^2
    
    # We need to maintain R for each replica.
    # R[tau] = sum_{j=0}^{N-1-tau} h[j] h[j+tau]
    
    # Initialize R and E for each replica
    replica_R = []
    replica_E = []
    replica_best_h = []
    replica_best_psl = []
    
    for i in range(n_reps):
        h = replicas[i]
        # Compute R
        R = np.zeros(N)
        # Use FFT for initial computation? Or direct?
        # Direct O(N^2) is 160k ops, ~1ms. 10 replicas -> 10ms. Fine.
        for tau in range(1, N):
            R[tau] = np.sum(h[:N-tau] * h[tau:])
        # R[0] = N
        R[0] = N
        
        replica_R.append(R)
        
        # Calculate L2 Energy (sum of squared sidelobes)
        E = np.sum(R[1:]**2)
        replica_E.append(E)
        
        # Calculate PSL (L-inf norm) for tracking best
        psl = np.max(np.abs(R[1:]))
        replica_best_psl.append(psl)
        replica_best_h.append(h.copy())
        
    best_global_psl = min(replica_best_psl)
    best_global_h = replica_best_h[np.argmin(replica_best_psl)].copy()
    
    # Parallel Tempering Loop
    # We will perform steps. Estimate time per step.
    # Step:
    # 1. For each replica, pick a random index, flip, calculate delta_E, accept/reject. O(N)
    # 2. Try to swap adjacent replicas based on energy and temperature. O(1) per pair.
    
    # We can do many steps per second.
    # Let's run until deadline.
    
    while time.time() < deadline:
        # 1. Update each replica (Metropolis step)
        for i in range(n_reps):
            h = replicas[i]
            R = replica_R[i]
            E = replica_E[i]
            T = temperatures[i]
            
            # Pick random index to flip
            idx = np.random.randint(N)
            
            # Calculate delta_R for all tau
            # delta_R(tau) = -2 * h[idx] * (h[idx+tau] (if valid) + h[idx-tau] (if valid))
            # We need to compute sum(R[tau] * delta_R(tau)) and sum(delta_R(tau)^2)
            
            # Vectorized calculation of delta_R components?
            # h[idx] is the value being flipped. Let v = h[idx].
            v = h[idx]
            
            # Contributions to R(tau) from h[idx]:
            # 1. h[idx] * h[idx+tau] for tau > 0, idx+tau < N
            # 2. h[idx-tau] * h[idx] for tau > 0, idx-tau >= 0
            
            # Let's construct the delta_R vector (size N-1, for tau 1..N-1)
            # It is mostly zero.
            # We can compute the two terms separately.
            
            # Term 1: tau in 1..N-1-idx
            # delta_R1[tau] = -2 * v * h[idx+tau]
            # Term 2: tau in 1..idx
            # delta_R2[tau] = -2 * v * h[idx-tau]
            
            # We need sum_{tau=1}^{N-1} R[tau] * (delta_R1[tau] + delta_R2[tau])
            # And sum (delta_R1[tau] + delta_R2[tau])^2
            
            # Note: delta_R1 and delta_R2 are non-zero on disjoint sets of tau (except possibly boundaries? No, tau is unique).
            # Actually, for a fixed idx, the set of tau where idx+tau < N is {1, ..., N-1-idx}.
            # The set of tau where idx-tau >= 0 is {1, ..., idx}.
            # These sets are disjoint because tau > 0.
            
            # So sum R*delta = sum_{tau in S1} R[tau]*delta1[tau] + sum_{tau in S2} R[tau]*delta2[tau]
            
            S1_end = N - idx - 1
            S2_end = idx
            
            # Sum 1
            # tau = 1..S1_end
            # delta1[tau] = -2 * v * h[idx+tau]
            # Indices in h: idx+1 .. idx+S1_end = N-1
            # Indices in R: 1 .. S1_end
            if S1_end > 0:
                h_part1 = h[idx+1 : idx+1+S1_end] # length S1_end
                R_part1 = R[1 : 1+S1_end]         # length S1_end
                sum1 = np.sum(R_part1 * (-2 * v * h_part1))
            else:
                sum1 = 0.0
                
            # Sum 2
            # tau = 1..S2_end
            # delta2[tau] = -2 * v * h[idx-tau]
            # Indices in h: idx-1 .. idx-S2_end = 0
            # Note: h[idx-tau] for tau=1 is h[idx-1]. For tau=S2_end=idx, h[0].
            # So h_part2 should be h[0:idx] reversed?
            # h[idx-1], h[idx-2], ..., h[0]
            # This is h[:idx][::-1]
            if S2_end > 0:
                h_part2 = h[:idx][::-1] # length idx
                R_part2 = R[1 : 1+S2_end] # length idx
                sum2 = np.sum(R_part2 * (-2 * v * h_part2))
            else:
                sum2 = 0.0
                
            cross_term = 2.0 * (sum1 + sum2)
            
            # Quadratic term: sum (delta_R)^2
            # Since S1 and S2 are disjoint, sum of squares is sum of squares of each part.
            # sum (delta1)^2 = sum_{tau in S1} (-2 * v * h[idx+tau])^2 = 4 * v^2 * sum h[idx+tau]^2
            # Since h is +/- 1, h^2 = 1.
            # So sum1_sq = 4 * v^2 * S1_end = 4 * S1_end
            # Similarly sum2_sq = 4 * S2_end
            quad_term = 4.0 * (S1_end + S2_end)
            
            delta_E = cross_term + quad_term
            
            # Metropolis Acceptance
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                # Accept
                h[idx] = -v
                # Update R
                # We need to add delta_R to R
                # R[1 : 1+S1_end] += -2 * v * h_part1
                # R[1 : 1+S2_end] += -2 * v * h_part2
                if S1_end > 0:
                    R[1 : 1+S1_end] += -2 * v * h_part1
                if S2_end > 0:
                    R[1 : 1+S2_end] += -2 * v * h_part2
                replica_E[i] += delta_E
                
                # Update best PSL for this replica
                # We only need to update best if the new PSL is lower.
                # Calculating max(abs(R[1:])) is O(N).
                # We can do this every step? 10 reps * 400 ops = 4000 ops. Very fast.
                current_psl = np.max(np.abs(R[1:]))
                if current_psl < replica_best_psl[i]:
                    replica_best_psl[i] = current_psl
                    replica_best_h[i] = h.copy()
                    if current_psl < best_global_psl:
                        best_global_psl = current_psl
                        best_global_h = h.copy()
                        
        # 2. Swap adjacent replicas
        # Iterate over pairs (0,1), (1,2), ...
        # Probability of swap = min(1, exp(-(E_i - E_j)/T_i + (E_i - E_j)/T_j))
        # Let dE = E_i - E_j
        # prob = min(1, exp(-dE * (1/T_i - 1/T_j)))
        
        for i in range(n_reps - 1):
            j = i + 1
            E_i = replica_E[i]
            E_j = replica_E[j]
            T_i = temperatures[i]
            T_j = temperatures[j]
            
            dE = E_i - E_j
            # If T_i > T_j, then 1/T_i < 1/T_j, so (1/T_i - 1/T_j) is negative.
            # dE * negative.
            # If E_i > E_j (dE > 0), exponent is positive -> prob > 1 -> accept.
            # This makes sense: swap high energy state to high T, low energy to low T.
            
            exponent = -dE * (1.0/T_i - 1.0/T_j)
            
            if exponent > 0 or np.random.rand() < np.exp(exponent):
                # Swap
                replicas[i], replicas[j] = replicas[j], replicas[i]
                replica_R[i], replica_R[j] = replica_R[j], replica_R[i]
                replica_E[i], replica_E[j] = replica_E[j], replica_E[i]
                # Note: We do NOT swap best_h or best_psl, as those are global records per replica index?
                # Actually, best_h is associated with the *state* history of that replica slot?
                # Or is it just the best found *so far* by any replica?
                # My current implementation tracks best per replica index.
                # If we swap states, the "best found so far" for that index remains valid?
                # Yes, it's a global record. We don't need to update best_h on swap.
                # However, the *current* state of the replica changes.
                # The next step will operate on the swapped state.
                # This is correct.

    return best_global_h
