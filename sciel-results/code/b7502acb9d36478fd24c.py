import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148.0
    n = 256

    # 1. Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # 2. Construct the "True" QUBO Matrix for the count of monochromatic APs.
    # We want to minimize the number of APs where x0=x1=x2=x3.
    # Indicator of mono AP: I(x0=x1=x2=x3) = I(x0=x1) * I(x1=x2) * I(x2=x3)
    # In binary x in {0,1}:
    # I(xu=xv) = 1 - (xu - xv) = 1 - xu + xv if xu>xv? No.
    # I(xu=xv) = 1 if xu=xv, 0 otherwise.
    # (xu - xv)^2 = 0 if equal, 1 if not.
    # So I(xu=xv) = 1 - (xu - xv)^2 = 1 - (xu + xv - 2 xu xv) = 1 - xu - xv + 2 xu xv.
    
    # This is a cubic/quartic polynomial in x.
    # E_mono = sum_ap (1 - x0 - x1 + 2 x0 x1) (1 - x1 - x2 + 2 x1 x2) (1 - x2 - x3 + 2 x2 x3)
    # This is degree 4.
    # To use QUBO, we need to linearize higher order terms or use a different penalty.
    
    # Alternative Penalty: Minimize the number of "breaks".
    # If we minimize the number of unequal adjacent pairs, we get blocks.
    # But we want to minimize mono APs specifically.
    
    # Let's use a 2nd order approximation that correlates with mono APs.
    # Observation: If x0=x1=x2=x3, then all 3 adjacent pairs are equal.
    # If we penalize adjacent mismatches, we create long runs, which might increase mono APs?
    # Actually, Thue-Morse has few long runs. Random has short runs.
    # Thue-Morse: 0,1,1,0,1,0,0,1...
    # It avoids long runs of the same color.
    # So minimizing "long runs" is the goal.
    # A 4-term AP with d=1 is a run of 4.
    # A 4-term AP with d>1 is not a contiguous run.
    
    # Let's stick to the Ising model but use the TRUE objective in the acceptance criteria.
    # The Ising model with adjacency matrix W is a good heuristic for "smoothing" the coloring.
    # We will use a Multi-Start approach with different initializations.
    # Initializations:
    # 1. Random
    # 2. Thue-Morse
    # 3. Block patterns (e.g., 0011, 01110000, etc.)
    # 4. Random with high density of 0s or 1s (though symmetric, so 0.5 is best).
    
    # To be structurally different from p45 (which was single start + Ising W + SA),
    # we will implement a Vectorized Multi-Start with a mix of Ising W and Full-Count guidance.
    # Specifically, we will run SA on the Ising model, but every K steps, we evaluate
    # the true count and if it's worse than best, we revert or restart.
    # Also, we will use a "kick" strategy: if stuck, flip a random subset.
    
    # Crucially, we will vectorize the delta evaluation for the Ising model to allow
    # many steps per second, enabling more restarts.
    
    W = np.zeros((n, n), dtype=np.float64)
    for ap in aps_arr:
        i0, i1, i2, i3 = ap
        W[i0, i1] += 1; W[i1, i0] += 1
        W[i1, i2] += 1; W[i2, i1] += 1
        W[i2, i3] += 1; W[i3, i2] += 1
        
    # Precompute rows of W for fast access? W @ s is already fast with BLAS.
    
    def count_mono(x):
        # x is numpy array of 0/1
        # Vectorized count
        c0 = x[aps_arr[:,0]]
        c1 = x[aps_arr[:,1]]
        c2 = x[aps_arr[:,2]]
        c3 = x[aps_arr[:,3]]
        return np.sum((c0 == c1) & (c1 == c2) & (c2 == c3))
    
    def thue_morse(n):
        x = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            x[i] = x[i >> 1] ^ (i & 1)
        return x
    
    best_score = 10**9
    best_x = None
    
    # Define several initializations
    inits = []
    
    # 1. Thue-Morse
    tm = thue_morse(n)
    inits.append(tm)
    
    # 2. Random
    for _ in range(5):
        inits.append(np.random.randint(0, 2, size=n).astype(np.int8))
        
    # 3. Block patterns
    # Pattern: 00001111 repeated? No, that has long runs.
    # Pattern: 010101... (alternating) - d=1 has no mono. d=2: 0,0,0,0? No.
    # Alternating: x[i] = i % 2.
    # d=1: 0,1,0,1 (not mono)
    # d=2: 0,0,0,0 (mono!) -> a, a+2, a+4, a+6. All same parity.
    # So alternating is bad for d=2.
    
    # Thue-Morse is good for small d.
    
    # Let's add a "Randomized Thue-Morse"
    tm_rand = tm.copy()
    mask = np.random.rand(n) < 0.1
    tm_rand[mask] = 1 - tm_rand[mask]
    inits.append(tm_rand)
    
    # 4. Greedy Random: Pick x[i] randomly, then for each j, if flipping x[j] reduces current mono count, flip it. Repeat.
    # This is slow to initialize but might find a good local min quickly.
    
    # Strategy:
    # For each initialization:
    #   Run Simulated Annealing on the Ising model (W) for a fixed number of steps or time.
    #   Use the TRUE count (count_mono) to track the best solution.
    #   If the Ising energy decreases but True count increases, we might be in a "bad valley".
    #   We will use a "Penalty Method": 
    #     E_total = E_ising + Lambda * E_true_count
    #     But E_true_count is high-order.
    #     Approximation: E_ising is a lower bound on "disorder".
    #     Let's just use Ising SA, but check True count after each flip (vectorized flip + count is fast enough? 
    #     count_mono is O(M) ~ 20k ops. 10k flips/s -> 200M ops/s. Doable in C, but in Python/Numpy it might be slow.
    #     M = ~20,000. Numpy sum of 20k booleans is ~5-10 microseconds.
    #     So 100,000 steps/s is possible.
    #     Let's try to do 10,000 steps per restart.
    
    steps_per_run = 5000
    T_start = 2.0
    T_end = 0.05
    cooling_rate = (T_end / T_start) ** (1.0 / steps_per_run)
    
    # To allow more structural variety, we will also try a "Pairwise Swap" move.
    # Swap x[i] and x[j].
    # Delta E_ising for swap:
    # This is more complex to calculate.
    # Let's stick to flips but use multiple threads? No, standard library only.
    
    # Optimization: Precompute the effect of flipping each bit on the Ising energy.
    # E_ising = - sum W s s
    # h_i = sum_j W[i,j] s_j
    # Flip i: Delta E = 2 s_i h_i
    
    # We will run the SA loop.
    
    for init_x in inits:
        if time.time() > deadline:
            break
            
        # Convert x to s
        s = 2 * init_x - 1  # s in {-1, 1}
        
        T = T_start
        current_score = count_mono(init_x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = init_x.copy()
            
        # SA Loop
        for step in range(steps_per_run):
            if time.time() > deadline:
                return [int(c) for c in best_x]
                
            h = W @ s
            p = np.random.randint(0, n)
            
            delta_E = 2 * s[p] * h[p]
            
            if delta_E < 0:
                s[p] *= -1
            else:
                prob = np.exp(-delta_E / T)
                if np.random.random() < prob:
                    s[p] *= -1
                    # We only flip if accepted, but we need to update s if delta_E < 0 too?
                    # Yes, handled above.
            
            # Periodically check true score
            # Doing it every step is expensive. Let's do it every 50 steps.
            if step % 50 == 0:
                x = (s + 1) // 2
                score = count_mono(x)
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
                    
            T *= cooling_rate
            
        # End of SA run, final check
        x = (s + 1) // 2
        score = count_mono(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    # If we still have time, do a final polish: 
    # Local search on the best_x found, trying all single flips and keeping the best.
    # Repeat until no improvement.
    
    if best_x is not None and time.time() < deadline - 5:
        x = best_x.copy()
        score = count_mono(x)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try flipping each bit
            for i in range(n):
                if time.time() > deadline - 1:
                    break
                x[i] = 1 - x[i]
                new_score = count_mono(x)
                if new_score < score:
                    score = new_score
                    improved = True
                else:
                    x[i] = 1 - x[i] # revert
            if score < best_score:
                best_score = score
                best_x = x.copy()
                
    return [int(c) for c in best_x]
