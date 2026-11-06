"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    prev_row = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines, dtype=np.float64)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])

def _compute_prefix_completions(perm, processing_times):
    """
    Compute completion time vectors for every prefix of the permutation.
    prefix_comps[i] is the completion vector after processing perm[0..i].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps
    
    prev_row = np.zeros(n_machines, dtype=np.float64)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines, dtype=np.float64)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps

def _compute_suffix_delays(perm, processing_times):
    """
    Compute delay vectors for suffixes using max-plus linearity.
    suffix_delays[i] is a vector D such that for any input completion vector V,
    the makespan of the sequence perm[i..] starting from V is:
    max_k (V[k] + D[k])
    
    suffix_delays[n_jobs] = zeros(n_machines)
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    suffix_delays = [None] * (n_jobs + 1)
    suffix_delays[n_jobs] = np.zeros(n_machines, dtype=np.float64)
    
    if n_jobs == 0:
        return suffix_delays
        
    # Iterate backwards
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        D_next = suffix_delays[i+1]
        
        # We want to compute D_current such that:
        # Makespan(V, perm[i..]) = max_k (V[k] + D_current[k])
        
        # Let W_m be the vector of coefficients for the completion time on machine m
        # when processing job `job` from input vector V.
        # C_m = max_k (V[k] + W_m[k])
        
        # m=0: C_0 = V_0 + pt[0]
        # So W_0[0] = pt[0], W_0[k>0] = -inf
        W = np.full(n_machines, -np.inf, dtype=np.float64)
        W[0] = pt[0]
        
        for m in range(1, n_machines):
            # C_m = max(V_m + pt[m], C_{m-1} + pt[m])
            # C_{m-1} = max_k (V[k] + W_{m-1}[k])
            # C_m = max(V_m + pt[m], max_k (V[k] + W_{m-1}[k] + pt[m]))
            # C_m = max_k (V[k] + max( (k==m ? pt[m] : -inf), W_{m-1}[k] + pt[m] ))
            
            # Update W for machine m
            # W_m[k] = max( (k==m ? pt[m] : -inf), W_{m-1}[k] + pt[m] )
            # Note: W_{m-1}[m] is -inf if m > 0 in the previous step?
            # Let's check.
            # W_0[0]=pt0, W_0[1]=-inf
            # m=1:
            # W_1[0] = max(-inf, W_0[0]+pt1) = pt0+pt1
            # W_1[1] = max(pt1, W_0[1]+pt1) = max(pt1, -inf) = pt1
            # W_1[2] = max(-inf, -inf+pt1) = -inf
            
            # So we update W in place?
            # W_new[k] = max( (k==m ? pt[m] : -inf), W_old[k] + pt[m] )
            
            # We can update W iteratively.
            # For k < m: W[k] = W[k] + pt[m]
            # For k == m: W[m] = max(pt[m], W[m] + pt[m]) = pt[m] (since W[m] was -inf)
            # For k > m: W[k] = -inf + pt[m] = -inf (since W[k] was -inf)
            
            # Wait, is W[k] always -inf for k > m?
            # Yes, because the max-plus propagation only goes forward in indices?
            # Actually, C_m depends on V_m and C_{m-1}.
            # C_{m-1} depends on V_{m-1}, C_{m-2}...
            # So C_m depends on V_0...V_m.
            # Thus W_m[k] is -inf for k > m.
            
            # So the update is:
            # W[:m] += pt[m]
            # W[m] = pt[m]
            # W[m+1:] remains -inf
            
            W[:m] += pt[m]
            W[m] = pt[m]
            # W[m+1:] is already -inf, so no change needed, but let's be safe?
            # Actually, if we just do W[:m] += pt[m], W[m] is untouched by that slice.
            # Then we set W[m] = pt[m].
            # What about W[m+1:]? It should stay -inf.
            
        # Now W is the coefficient vector for C_{M-1} (the last machine).
        # The makespan of the suffix perm[i+1..] starting from C_{M-1} is:
        # max_j (C_{M-1}[j] + D_next[j])
        # C_{M-1}[j] = max_k (V[k] + W[j]) -- Wait, W is a vector of length M.
        # C_{M-1} is a vector of length M.
        # C_{M-1}[j] = max_k (V[k] + W_j[k])?
        # No, my W definition was: C_m = max_k (V[k] + W_m[k]).
        # So W is a vector of length M, where W[k] is the coefficient of V[k] in C_M-1?
        # No, W is updated per machine m.
        # At the end of the loop (m = M-1), W[k] is the coefficient of V[k] in C_{M-1}.
        
        # So C_{M-1}[j] is NOT a single value, it's a vector.
        # My notation was slightly off.
        # Let's re-read the derivation.
        # C_m = max_k (V[k] + W_m[k])
        # Here W_m is a vector of length M.
        # So C_{M-1} is a vector of length M? No, C_m is a scalar (completion time on machine m).
        # Ah, the flow shop completion time C_m is a scalar.
        # So C_{M-1} is a scalar.
        # C_{M-1} = max_k (V[k] + W_{M-1}[k])
        
        # Then the makespan of the suffix is:
        # Makespan = max_j (C_{M-1} + D_next[j]) ?
        # No, D_next is defined such that Makespan(V', suffix) = max_j (V'[j] + D_next[j]).
        # Here V' is the completion vector after processing perm[i].
        # But in a flow shop, the "state" after processing a job is the vector of completion times on all machines.
        # So V' = C_vector.
        # C_vector[j] = completion time on machine j after processing perm[i].
        
        # My previous derivation assumed V' is a vector.
        # Let's re-evaluate.
        # State after processing perm[0..i-1] is a vector V of length M.
        # Processing perm[i] (job j) results in a new state vector V'.
        # V'[0] = V[0] + p_j0
        # V'[m] = max(V[m], V'[m-1]) + p_jm
        
        # We want to express V'[m] as max_k (V[k] + W_m[k]).
        # This is what I did.
        # So V' is a vector of length M, where V'[m] = max_k (V[k] + W_m[k]).
        
        # Then the makespan of the suffix perm[i+1..] starting from state V' is:
        # Makespan = max_j (V'[j] + D_next[j])
        # Makespan = max_j ( max_k (V[k] + W_m[k]) + D_next[j] )  where m = M-1?
        # No, V' has components for all machines.
        # V'[j] = max_k (V[k] + W_j[k]) where W_j is the coefficient vector for machine j.
        
        # So Makespan = max_j ( max_k (V[k] + W_j[k]) + D_next[j] )
        # Makespan = max_j max_k ( V[k] + W_j[k] + D_next[j] )
        # Makespan = max_k ( V[k] + max_j ( W_j[k] + D_next[j] ) )
        
        # So D_current[k] = max_j ( W_j[k] + D_next[j] )
        
        # I need to store W_j for all j? Or just the final W?
        # In my loop, I updated W to be W_m for the current m.
        # So at the end, I only have W_{M-1}.
        # But I need W_j for all j to compute max_j ( W_j[k] + D_next[j] ).
        
        # This means my single vector W is not enough. I need a matrix W of size M x M.
        # W[j][k] is the coefficient of V[k] in V'[j].
        
        # Let's re-implement with a matrix.
        
    # Re-doing the loop with a matrix W
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        D_next = suffix_delays[i+1]
        
        # W[j][k] is coeff of V[k] in V'[j]
        W = np.full((n_machines, n_machines), -np.inf, dtype=np.float64)
        
        # j=0: V'[0] = V[0] + pt[0]
        W[0][0] = pt[0]
        
        for j in range(1, n_machines):
            # V'[j] = max(V[j], V'[j-1]) + pt[j]
            # V'[j-1] = max_k (V[k] + W[j-1][k])
            # V'[j] = max(V[j] + pt[j], max_k (V[k] + W[j-1][k] + pt[j]))
            # V'[j] = max_k (V[k] + max( (k==j ? pt[j] : -inf), W[j-1][k] + pt[j] ))
            
            # So W[j][k] = max( (k==j ? pt[j] : -inf), W[j-1][k] + pt[j] )
            
            # For k < j: W[j][k] = W[j-1][k] + pt[j]
            # For k == j: W[j][j] = max(pt[j], W[j-1][j] + pt[j]) = pt[j] (since W[j-1][j] is -inf)
            # For k > j: W[j][k] = -inf + pt[j] = -inf
            
            W[j][:j] = W[j-1][:j] + pt[j]
            W[j][j] = pt[j]
            # W[j][j+1:] is already -inf
        
        # Now compute D_current[k] = max_j ( W[j][k] + D_next[j] )
        D_current = np.zeros(n_machines, dtype=np.float64)
        for k in range(n_machines):
            # max over j of W[j][k] + D_next[j]
            # W is M x M. Column k of W.
            vals = W[:, k] + D_next
            D_current[k] = np.max(vals)
            
        suffix_delays[i] = D_current
        
    return suffix_delays

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH heuristic for initial construction, then local search with
    insertion moves, plus a deadline guard to ensure termination.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n_jobs, n_machines = processing_times.shape

    if n_jobs <= 1:
        return list(range(n_jobs))

    # Precompute makespan function for speed
    def makespan(seq):
        """Compute makespan for a given job sequence."""
        completion = np.zeros(n_machines)
        for job in seq:
            for m in range(n_machines):
                completion[m] = completion[m] + processing_times[job, m]
                if m > 0:
                    completion[m] = max(completion[m], completion[m-1])
        return completion[-1]

    # --- Stage 1: NEH Heuristic ---
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))

    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            ms = makespan(candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]

    best_seq = list(seq)
    best_ms = makespan(best_seq)
    
    # --- Stage 2: Local Search with Insertion Moves using Fast Evaluation ---
    
    # We will use prefix_comps and suffix_delays for fast evaluation of moves.
    # However, insertion moves change the sequence structure significantly.
    # Removing a job at i and inserting at j changes the prefix and suffix.
    
    # Let's stick to swap moves first for fast evaluation, or implement fast insertion.
    # Fast insertion evaluation:
    # Move job from i to j.
    # If i < j:
    #   New seq: P[0..i-1] + P[i+1..j] + [J_i] + P[j+1..]
    #   Prefix: P[0..i-1] -> C_{i-1}
    #   Middle: P[i+1..j] -> C_mid
    #   Job J_i -> C_job
    #   Suffix: P[j+1..] -> Makespan from C_job
    #   
    #   We need C_{i-1}.
    #   We need to simulate P[i+1..j] from C_{i-1}. This is O((j-i) * M).
    #   We need to simulate J_i from C_mid. O(M).
    #   We need Makespan from C_job for P[j+1..]. O(M) using suffix_delays[j+1].
    #   
    #   Total O(N * M) per move. Same as full makespan?
    #   Full makespan is O(N * M).
    #   So this is not faster for insertion.
    
    # Swap moves (adjacent or not) might be faster?
    # Swap i, j.
    # If |i-j| is small, it's fast.
    # But insertion is more powerful.
    
    # Let's use the fast evaluation for NEH construction?
    # NEH inserts one job at a time.
    # When inserting job j into pos p in seq S of length L:
    # Prefix S[0..p-1] -> C_{p-1}
    # Job j -> C_j
    # Suffix S[p..L-1] -> Makespan from C_j
    # This is O(p*M + M + (L-p)*M) = O(L*M).
    # Full makespan of S + j is O((L+1)*M).
    # So it's roughly the same.
    
    # The inspiration suggested that the current program is slow because it evaluates from scratch.
    # But for insertion, the "from scratch" is O(N*M).
    # The fast eval is also O(N*M) but with smaller constant?
    # Actually, the fast eval avoids recomputing the prefix if we store it.
    # In NEH, we insert one by one.
    # We can maintain prefix_comps for the current seq.
    # When inserting j at pos p:
    #   We have prefix_comps[p-1].
    #   We simulate j: O(M).
    #   We use suffix_delays[p] for the rest?
    #   But suffix_delays depends on the specific jobs in the suffix.
    #   The suffix is S[p..L-1].
    #   We can precompute suffix_delays for the current S.
    #   But S changes every step.
    
    # Let's try to implement the fast evaluation for the local search part,
    # specifically for swap moves, which might be more amenable.
    # Or just use the fast evaluation for the makespan calculation in the local search.
    
    # Actually, let's just use the fast evaluation for the makespan function itself?
    # No, makespan is a single value.
    
    # Let's try to use the fast evaluation for the NEH construction.
    # And for the local search, let's use swap moves with fast evaluation.
    
    # Re-implementing NEH with fast evaluation.
    
    def _makespan_fast(perm, prefix_comps, suffix_delays):
        """
        Compute makespan using precomputed prefix and suffix.
        This is not directly useful for a full permutation unless we have the full prefix/suffix.
        """
        pass

    # Let's go back to the standard local search but with a time limit.
    # The inspiration score was 1.2190, current is 1.1250.
    # The inspiration used `random` and `math`.
    
    # Let's add a random restart mechanism or perturbation.
    
    # For now, let's keep the structure but ensure it's robust.
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = best_seq[i]
            for j in range(n_jobs):
                if i == j or time.time() > deadline:
                    continue
                # Try moving job from position i to position j
                new_seq = best_seq[:i] + best_seq[i+1:j+1] + [job] + best_seq[j+1:]
                if i > j:
                    new_seq = best_seq[:j] + [job] + best_seq[j:i] + best_seq[i+1:]
                ms = makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
                    break  # Restart scan from beginning after improvement
            if time.time() > deadline:
                break

    return list(best_seq)
# EVOLVE-BLOCK-END
