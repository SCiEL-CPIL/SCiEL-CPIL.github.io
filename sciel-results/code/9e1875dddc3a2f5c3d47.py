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
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
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
    
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps


def _compute_suffix_completions(perm, processing_times):
    """
    Compute "backward" completion vectors for every suffix of the permutation.
    suffix_comps[i] represents the state needed to compute the makespan if we start
    from a known completion vector and process perm[i..n-1].
    
    Specifically, if we have a completion vector `C` at step `i-1`, the final makespan
    is determined by the interactions between `C` and the suffix.
    
    To make incremental evaluation easy, we store the "effective start times" or 
    rather, we can just re-evaluate the small middle part.
    
    Actually, a better approach for flow shop:
    The makespan of a sequence S1 + S2 is not simply determined by the completion of S1 
    and the start of S2 because of the max() operation.
    
    However, for a swap at i, the sequence is P[0..i-1] + [J_{i+1}, J_i] + P[i+2..].
    We need the completion vector of P[0..i-1]. Let's call it C_{i-1}.
    Then we simulate J_{i+1} and J_i. Then we need to know the impact on the rest.
    The rest P[i+2..] depends on the completion vector after J_i.
    
    So, we can precompute:
    1. Prefix completions C_k for k=0..N-1.
    2. Suffix "requirements"? No, flow shop is forward.
    
    Alternative Fast Evaluation for Swap(i, i+1):
    Let C_{i-1} be the completion vector after perm[0..i-1]. (If i=0, C_{-1} = 0)
    Let J_a = perm[i], J_b = perm[i+1].
    New sequence segment: J_b, J_a.
    Simulate J_b from C_{i-1} -> C_b.
    Simulate J_a from C_b -> C_a.
    Now we have C_a. The remaining sequence is perm[i+2..].
    We need the makespan of C_a followed by perm[i+2..].
    
    We can precompute for each position k, a function or data structure that allows us to 
    compute the makespan of a sequence starting from a given completion vector.
    This is generally hard to compress without storing the whole vector.
    
    However, note that N is likely small (<= 100?). 
    If N=100, M=10. 
    Full makespan: 100 * 10 = 1000 ops.
    Swap eval with prefix/suffix:
    Prefix C_{i-1} is available.
    Simulate 2 jobs: 2 * 10 = 20 ops.
    Now we have C_a.
    We need to simulate N - i - 2 jobs.
    This is still O(N).
    
    Is there a way to speed up the tail?
    If we store the "critical path" or something? 
    For a fixed suffix, the makespan is a max-plus linear function of the input completion vector?
    Makespan = max_k (C_k + L_k) where L_k depends on the suffix?
    Yes! In max-plus algebra, the makespan of a flow shop sequence is the max over all paths.
    For a fixed suffix S, the makespan given an input vector V is:
    max_m (V_m + D_m)
    where D_m is the "delay" from machine m to the end.
    
    Let's verify this linearity.
    Completion C_m(k) = max(C_m(k-1) + p_mk, C_{m-1}(k))
    This is a max-plus linear system.
    So, C_final = V * A (in max-plus).
    C_final[m] = max_k (V_k + A_{k,m})
    The makespan is max_m C_final[m].
    So Makespan = max_m max_k (V_k + A_{k,m}) = max_k (V_k + max_m A_{k,m}).
    
    So, for a fixed suffix, we can precompute a vector `D` of length M, where 
    D[k] = max_m A_{k,m}.
    Then Makespan(V, Suffix) = max_k (V_k + D[k]).
    
    This allows O(M) evaluation of the tail!
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return [np.zeros(n_machines)] # Dummy
    
    # suffix_comps[i] will store the vector D such that 
    # Makespan(V, perm[i..]) = max_k (V[k] + suffix_comps[i][k])
    
    # For the empty suffix (i=n), D is 0.
    # suffix_comps[n] = zeros(M)
    
    suffix_comps = [None] * (n_jobs + 1)
    suffix_comps[n_jobs] = np.zeros(n_machines)
    
    # Iterate backwards
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        D_next = suffix_comps[i+1]
        
        # We want to compute D_current such that:
        # C_final = max_k (V_k + D_current[k])
        # where C_final is the makespan after processing job `job` and then the suffix.
        
        # Let V be the input completion vector.
        # Processing job `job`:
        # C_0 = V_0 + pt[0]
        # C_m = max(V_m, C_{m-1}) + pt[m] for m > 0
        
        # We need to express C_m in terms of V_k.
        # C_0 = V_0 + pt[0]
        # C_1 = max(V_1, V_0 + pt[0]) + pt[1] = max(V_1 + pt[1], V_0 + pt[0] + pt[1])
        # C_m = max(V_m + pt[m], C_{m-1} + pt[m])
        
        # Let's compute the coefficients for each V_k.
        # Actually, we just need the vector D_current.
        # D_current[k] is the contribution of V_k to the final makespan.
        
        # Let's trace the max-plus propagation.
        # Let W_m be the vector of coefficients for C_m.
        # W_0: C_0 = V_0 + pt[0]. So W_0[0] = pt[0], W_0[1..] = -inf.
        # W_m: C_m = max(V_m + pt[m], C_{m-1} + pt[m]).
        # So W_m[m] = pt[m] (from V_m term? No, V_m enters as V_m + pt[m]).
        # Wait, V_m is an input.
        # C_m = max( V_m + pt[m], max_k (V_k + W_{m-1}[k] + pt[m]) )
        # C_m = max_k ( V_k + max( (k==m ? pt[m] : -inf), W_{m-1}[k] + pt[m] ) )
        
        # So W_m[k] = max( (k==m ? pt[m] : -inf), W_{m-1}[k] + pt[m] )
        
        # Initialize W as a matrix or just update a vector?
        # We want D_current[k] which is the max over all final machines.
        # But the suffix follows.
        # The makespan after the suffix is max_j (C_M_j + D_next[j]) where C_M is the completion vector after job `job`.
        # C_M_j = max_k (V_k + W_M[k])
        # Makespan = max_j max_k (V_k + W_M[k] + D_next[j])
        # Makespan = max_k (V_k + max_j (W_M[k] + D_next[j]))
        
        # So D_current[k] = max_j (W_M[k] + D_next[j])
        
        # We need to compute W_M.
        # W is a vector of length M (coefficients for V_0..V_{M-1}).
        # W_m[k] is the coefficient of V_k in C_m.
        
        W = np.full(n_machines, -np.inf)
        # m=0
        W[0] = pt[0]
        
        for m in range(1, n_machines):
            # W_m[k] = max( (k==m ? pt[m] : -inf), W_{m-1}[k] + pt[m] )
            # Note: W_{m-1} is the vector from the previous step.
            # We update W in place? No, we need W_{m-1} for all k.
            # W_new[k] = max( (k==m ? pt[m] : -inf), W_old[k] + pt[m] )
            
            # Since we iterate m from 1 to M-1, we can update.
            # But careful: W_old is the vector for machine m-1.
            
            # Let's create a new vector or update carefully.
            # W[k] currently holds W_{m-1}[k].
            
            # We need to compute W_m.
            # For k < m: W_m[k] = W_{m-1}[k] + pt[m]
            # For k == m: W_m[m] = max(pt[m], W_{m-1}[m] + pt[m]) = pt[m] + max(0, W_{m-1}[m])? 
            #           Wait, W_{m-1}[m] is -inf initially? 
            #           Let's check base case.
            #           W_0[0] = pt[0]. W_0[1..] = -inf.
            #           m=1.
            #           k=0: W_1[0] = W_0[0] + pt[1] = pt[0] + pt[1].
            #           k=1: W_1[1] = max(pt[1], W_0[1] + pt[1]) = max(pt[1], -inf) = pt[1].
            #           k>1: -inf.
            
            # So we can do:
            # W = W + pt[m]
            # Then set W[m] = max(W[m], pt[m]) ?
            # Let's check:
            # W_old[m] + pt[m]. If W_old[m] is -inf, result is -inf.
            # max(-inf, pt[m]) = pt[m].
            # If W_old[m] is finite, say X. max(X + pt[m], pt[m]).
            # Is X + pt[m] >= pt[m]? 
            # X is a sum of processing times, so X >= 0.
            # So X + pt[m] >= pt[m].
            # So max(X + pt[m], pt[m]) = X + pt[m].
            # So W[m] remains W_old[m] + pt[m].
            
            # Wait, is W_old[m] always >= 0?
            # W_0[0] = pt[0] >= 0.
            # W_1[1] = pt[1] >= 0.
            # Yes, all W are sums of positive processing times.
            
            # So the update is simply:
            # W = W + pt[m]
            # And for k=m, we ensure it's at least pt[m]?
            # Actually, if W_old[m] was -inf, W_new[m] becomes -inf + pt[m] = -inf.
            # But it should be pt[m].
            # So we need to explicitly set W[m] = max(W[m], pt[m]) AFTER adding pt[m]?
            # If W[m] was -inf, it becomes -inf. max(-inf, pt[m]) = pt[m]. Correct.
            # If W[m] was X, it becomes X + pt[m]. max(X+pt[m], pt[m]) = X+pt[m]. Correct.
            
            W += pt[m]
            if W[m] < pt[m]: # This handles the -inf case
                W[m] = pt[m]
        
        # Now W holds W_M (coefficients for the completion vector after job `job`).
        # We need D_current[k] = max_j (W[k] + D_next[j])
        # This is max_j D_next[j] + W[k]
        # Let max_D_next = max(D_next)
        # D_current = W + max_D_next
        
        max_D_next = np.max(suffix_comps[i+1])
        suffix_comps[i] = W + max_D_next
        
    return suffix_comps


def _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of swapping perm[i] and perm[i+1].
    i is the index of the first job in the pair.
    """
    n_machines = processing_times.shape[1]
    
    # Get completion vector before perm[i]
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1]
    
    job_a = perm[i]
    job_b = perm[i+1]
    
    # Simulate job_b then job_a
    pt_b = processing_times[job_b]
    pt_a = processing_times[job_a]
    
    # C_b
    C_b = np.zeros(n_machines)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = max(C_prev[m], C_b[m-1]) + pt_b[m]
        
    # C_a
    C_a = np.zeros(n_machines)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = max(C_b[m], C_a[m-1]) + pt_a[m]
        
    # Now evaluate suffix starting from i+2
    # The suffix is perm[i+2..]
    # We need to combine C_a with suffix_comps[i+2]
    
    if i + 2 >= len(perm):
        return float(np.max(C_a))
    else:
        D_suffix = suffix_comps[i+2]
        # Makespan = max_k (C_a[k] + D_suffix[k])
        return float(np.max(C_a + D_suffix))


def _eval_insert_incremental(perm, i, j, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of moving perm[i] to position j.
    If i < j, the job is moved to j (0-indexed in the new list, which corresponds to 
    inserting after perm[j-1] in the old list? No, standard insertion).
    
    Let's define the move clearly.
    Original: ... A ... B ...
    Remove A at i.
    Insert A at j.
    
    If i < j:
    Sequence: P[0..i-1] + P[i+1..j] + A + P[j+1..]
    If i > j:
    Sequence: P[0..j-1] + A + P[j..i-1] + P[i+1..]
    
    Note: The "position j" usually means the index in the resulting list.
    If we remove i, the list shrinks.
    If we insert at j, it's the index in the list *after* removal?
    Standard convention in these loops:
    `new_perm = _move_job(perm, job, new_pos)`
    `_move_job` removes job, then inserts at `new_pos` in the reduced list.
    
    So if i < j:
    The job moves to the right.
    Prefix: P[0..i-1]. Completion C_{i-1}.
    Middle: P[i+1..j]. (Jobs from i+1 to j in original indices).
    Job A.
    Suffix: P[j+1..].
    
    Wait, if we remove i, the element at index j in the *original* list shifts to j-1.
    The insertion position `j` in the *new* list corresponds to:
    - If j > i: The job is inserted after the element that was at index j-1 in the original list?
      Let's trace `_move_job`:
      `perm = [j for j in perm if j != job]`
      `return perm[:new_pos] + [job] + perm[new_pos:]`
      
      If i=1, j=3.
      Original: [0, 1, 2, 3]
      Remove 1: [0, 2, 3]
      Insert at 3: [0, 2, 3, 1]
      So the sequence is 0, 2, 3, 1.
      Original indices: 0, 2, 3, 1.
      Prefix: 0. (i-1 = 0). C_0.
      Middle: 2, 3. (Original indices i+1 to j).
      Job: 1.
      Suffix: None.
      
      If i=3, j=1.
      Original: [0, 1, 2, 3]
      Remove 3: [0, 1, 2]
      Insert at 1: [0, 3, 1, 2]
      Sequence: 0, 3, 1, 2.
      Original indices: 0, 1, 2, 3.
      Prefix: 0. (j-1 = 0). C_0.
      Job: 3.
      Middle: 1, 2. (Original indices j to i-1).
      Suffix: None.
    """
    n_machines = processing_times.shape[1]
    
    if i == j:
        return _makespan(perm, processing_times) # Should not happen
    
    if i < j:
        # Move right
        # Prefix: 0..i-1
        if i == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[i-1]
            
        # Middle: i+1..j
        C_mid = C_prev
        for k in range(i+1, j+1):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        # Job A (perm[i])
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_mid[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_mid[m], C_a[m-1]) + pt_a[m]
            
        # Suffix: j+1..
        if j + 1 >= len(perm):
            return float(np.max(C_a))
        else:
            D_suffix = suffix_comps[j+1]
            return float(np.max(C_a + D_suffix))
            
    else:
        # Move left (i > j)
        # Prefix: 0..j-1
        if j == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[j-1]
            
        # Job A (perm[i])
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_prev[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_prev[m], C_a[m-1]) + pt_a[m]
            
        # Middle: j..i-1
        C_mid = C_a
        for k in range(j, i):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        # Suffix: i+1..
        if i + 1 >= len(perm):
            return float(np.max(C_mid))
        else:
            D_suffix = suffix_comps[i+1]
            return float(np.max(C_mid + D_suffix))


def _makespan_from_prev(perm, processing_times, prev_completion):
    """
    Compute makespan starting from a known completion vector for a prefix.
    prev_completion[m] is the completion time on machine m after some prefix.
    perm is the full permutation, and we compute from scratch but this is
    just the same as _makespan. Keeping for clarity.
    """
    return _makespan(perm, processing_times)


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # Add slight randomization to break ties and diversify
    if n_jobs > 2:
        # Swap a few adjacent elements randomly
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward pass
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    # Backward pass: remove each job and reinsert at best position
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        # Remove job at position i
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _insert_job(perm, job, pos):
    """Insert job at position pos in perm (removing it from current position if present)."""
    if job in perm:
        perm = [j for j in perm if j != job]
    return perm[:pos] + [job] + perm[pos:]


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _local_search_2opt(perm, processing_times, max_iterations=200, deadline=None):
    """
    Local search with 2-opt (adjacent swap) and insert moves using incremental evaluation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Precompute prefix and suffix vectors for current best_perm
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)
        
        improved = False
        
        # Try adjacent swaps
        for i in range(n_jobs - 1):
            ms = _eval_swap_incremental(best_perm, i, processing_times, prefix_comps, suffix_comps)
            if ms < best_ms - 1e-10:
                # Apply the swap
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms = ms
                improved = True
                # Re-compute prefix/suffix? 
                # A single swap changes the prefix from i onwards and suffix from i onwards.
                # It is safer to recompute for the next iteration or next move if we want correctness.
                # However, if we accept the move, the old prefix/suffix are invalid.
                # We can break out of the swap loop and recompute, or just continue and accept 
                # that subsequent swaps in this iteration might be evaluated with stale data?
                # No, stale data leads to incorrect ms.
                # So, if we improve, we should break and recompute, or just mark improved and break.
                # Given the cost of recomputation is O(NM), and we do this per improvement, 
                # it's better to just break the inner loop and let the outer loop handle it?
                # Or, we can just continue and recompute at the start of the next outer iteration.
                # But we might miss other improvements in the same pass.
                # Let's just break the swap loop if we find an improvement, to keep it simple and safe.
                # Actually, best-improvement vs first-improvement.
                # The original code did first-improvement (updated best_perm immediately).
                # This invalidates the precomputed vectors.
                # So we must break.
                break
        
        if improved:
            continue # Go to next iteration to recompute vectors
        
        # If no swap improvement, try insert moves
        # We need to be careful: if we find an insert improvement, we also break and recompute.
        
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                # pos is the index in the NEW list.
                # Our _eval_insert_incremental expects i (old index) and j (new index).
                # Does our function match the semantics of _move_job?
                # _move_job(perm, job, new_pos)
                # If i < pos: job moves right.
                # If i > pos: job moves left.
                # My _eval_insert_incremental logic:
                # if i < j: ...
                # if i > j: ...
                # This matches.
                
                ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                if ms < best_ms - 1e-10:
                    # Apply the move
                    # We need to actually update best_perm
                    new_perm = _move_job(best_perm, job, pos)
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simple simulated annealing with adjacent swap and insert moves.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Set initial temperature based on problem size
    initial_temp = current_ms * 0.1
    cooling_rate = 0.999
    
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.5:
            # Adjacent swap
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
        else:
            # Insert move
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 10s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    # 1. Simple LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    # 3. NEH with randomization (diversified)
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    # 4. NEH forward-backward
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Local search polish on best solution
    if time.time() < deadline and n_jobs > 1:
        # Determine max iterations based on problem size and remaining time
        # Since each iteration is faster (incremental eval), we can afford more iterations
        remaining = deadline - time.time()
        if n_jobs <= 30:
            max_iter = 1000
        elif n_jobs <= 50:
            max_iter = 500
        elif n_jobs <= 100:
            max_iter = 200
        else:
            max_iter = 100
        
        polished = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Simulated annealing if time remains
    if time.time() < deadline - 5.0 and n_jobs > 1:
        sa_deadline = deadline - 5.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Final local search polish
        if time.time() < deadline - 2.0:
            # Use more iterations here too
            max_iter_final = 200 if n_jobs <= 50 else 100
            final_polish = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter_final, deadline=deadline - 2.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
