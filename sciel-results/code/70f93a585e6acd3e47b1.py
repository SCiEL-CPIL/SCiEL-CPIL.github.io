import time
import numpy as np

N = 256

# Build 3-AP index arrays (each 3-AP stored once: (a, b, c) with a<b<c, d>0)
ap3_a, ap3_b, ap3_c = [], [], []
for d in range(1, N // 2 + 1):
    for s in range(1, N - 2 * d + 1):
        ap3_a.append(s)
        ap3_b.append(s + d)
        ap3_c.append(s + 2 * d)
ap3_a = np.array(ap3_a, dtype=np.int32)
ap3_b = np.array(ap3_b, dtype=np.int32)
ap3_c = np.array(ap3_c, dtype=np.int32)
M3 = len(ap3_a)

# Build 4-AP index arrays
ap4_a, ap4_b, ap4_c, ap4_d = [], [], [], []
for d in range(1, N // 3 + 1):
    for s in range(1, N - 3 * d + 1):
        ap4_a.append(s)
        ap4_b.append(s + d)
        ap4_c.append(s + 2 * d)
        ap4_d.append(s + 3 * d)
ap4_a = np.array(ap4_a, dtype=np.int32)
ap4_b = np.array(ap4_b, dtype=np.int32)
ap4_c = np.array(ap4_c, dtype=np.int32)
ap4_d = np.array(ap4_d, dtype=np.int32)
M4 = len(ap4_a)

def count_3aps(A):
    """A: boolean numpy array of length N+1 (1-indexed). Returns number of 3-APs."""
    if M3 == 0:
        return 0
    return int(np.sum(A[ap3_a] & A[ap3_b] & A[ap3_c]))

def is_4ap_free(A):
    """Check if A contains no nonconstant 4-AP."""
    if M4 == 0:
        return True
    return not np.any(A[ap4_a] & A[ap4_b] & A[ap4_c] & A[ap4_d])

def greedy_initial(rng):
    """Greedy construction: add elements in random order if 4-AP-free."""
    A = np.zeros(N + 1, dtype=bool)
    order = rng.permutation(1, N + 1)
    for x in order:
        A[x] = True
        if not is_4ap_free(A):
            A[x] = False
    return A

def simulated_annealing(A_init, deadline, rng):
    """
    SA search: single-flip moves (add or remove one element).
    Vectorized evaluation of all candidate moves per iteration.
    """
    A = A_init.copy()
    current_score = count_3aps(A)
    best_score = current_score
    best_A = A.copy()
    
    # Temperature schedule
    T0 = 2.0
    T_end = 0.05
    # We'll use a geometric cooling schedule, adjusting per iteration
    
    start_time = time.time()
    iters_per_check = 50  # check time every 50 iterations
    
    # Precompute for fast evaluation
    # For adding element x: delta_3 = number of 3-APs containing x where other 2 are in A
    # For removing element x: delta_3 = number of 3-APs containing x where other 2 are in A (loss)
    
    # Build per-element 3-AP neighbor lists
    # elem_3ap_partners[x] = list of (other1, other2) pairs for each 3-AP containing x
    # Actually, let's just vectorize: for each element x not in A, compute gain by adding x
    # For each element x in A, compute loss by removing x
    
    iteration = 0
    T = T0
    
    while time.time() < deadline:
        iteration += 1
        if iteration % iters_per_check == 0 and time.time() >= deadline:
            break
        
        # Vectorized: evaluate all possible single-flip moves
        # For additions: for each x not in A, check 4-AP constraint and compute 3-AP gain
        # For removals: for each x in A, compute 3-AP loss (always 4-AP safe to remove)
        
        not_in_A = np.where(~A)[0]
        in_A = np.where(A)[0]
        
        best_move = None
        best_delta = -np.inf
        
        # --- Evaluate removals ---
        if len(in_A) > 0:
            # For each x in A, loss = number of 3-APs containing x with other 2 in A
            # Vectorize: for each 3-AP, check if it's fully in A, then attribute to each member
            # loss[x] = count of 3-APs fully in A that contain x
            # We can compute this via: for each 3-AP (a,b,c) fully in A, increment loss for a, b, c
            full_3aps = A[ap3_a] & A[ap3_b] & A[ap3_c]
            # loss for each element = sum over 3-APs containing it that are full
            # Use bincount
            loss = np.zeros(N + 1, dtype=np.int32)
            np.add.at(loss, ap3_a[full_3aps], 1)
            np.add.at(loss, ap3_b[full_3aps], 1)
            np.add.at(loss, ap3_c[full_3aps], 1)
            
            # For removals, delta = -loss[x] (we lose that many 3-APs)
            # But we also need to check: does removing x help? Only if it allows future additions.
            # For SA, we accept based on delta.
            rem_deltas = -loss[in_A]
            # Find best removal
            if len(rem_deltas) > 0:
                best_rem_idx = np.argmax(rem_deltas)
                if rem_deltas[best_rem_idx] > best_delta:
                    best_delta = rem_deltas[best_rem_idx]
                    best_move = ('remove', in_A[best_rem_idx])
        
        # --- Evaluate additions ---
        if len(not_in_A) > 0:
            # For each x not in A: check if adding x creates a 4-AP
            # 4-AP constraint: no 4-AP (a,b,c,d) where x is one of them and the other 3 are in A
            # For each 4-AP, if 3 of its elements are in A, the 4th cannot be added
            # Vectorize: for each 4-AP, count how many of its 4 elements are in A
            # If exactly 3 are in A, the missing one is blocked
            
            # blocked[x] = True if x cannot be added
            blocked = np.zeros(N + 1, dtype=bool)
            
            # For each 4-AP (a,b,c,d):
            #   If A[a]&A[b]&A[c] and not A[d], then d is blocked
            #   If A[a]&A[b]&A[d] and not A[c], then c is blocked
            #   etc.
            # Vectorized:
            ab = A[ap4_a] & A[ap4_b] & A[ap4_c]
            if np.any(ab):
                blocked[ap4_d[ab]] = True
            
            ab = A[ap4_a] & A[ap4_b] & A[ap4_d]
            if np.any(ab):
                blocked[ap4_c[ab]] = True
            
            ab = A[ap4_a] & A[ap4_c] & A[ap4_d]
            if np.any(ab):
                blocked[ap4_b[ab]] = True
            
            ab = A[ap4_b] & A[ap4_c] & A[ap4_d]
            if np.any(ab):
                blocked[ap4_a[ab]] = True
            
            # Now filter not_in_A to only unblocked
            candidates = not_in_A[~blocked[not_in_A]]
            
            if len(candidates) > 0:
                # Compute gain for each candidate: number of 3-APs containing x where other 2 are in A
                # gain[x] = count of 3-APs (a,b,c) containing x where the other 2 are in A
                # Vectorize:
                gain = np.zeros(N + 1, dtype=np.int32)
                
                # For each 3-AP (a,b,c):
                #   If A[b]&A[c] and not A[a], then gain[a] += 1
                #   If A[a]&A[c] and not A[b], then gain[b] += 1
                #   If A[a]&A[b] and not A[c], then gain[c] += 1
                bc = A[ap3_b] & A[ap3_c]
                if np.any(bc):
                    np.add.at(gain, ap3_a[bc], 1)
                
                ac = A[ap3_a] & A[ap3_c]
                if np.any(ac):
                    np.add.at(gain, ap3_b[ac], 1)
                
                ab = A[ap3_a] & A[ap3_b]
                if np.any(ab):
                    np.add.at(gain, ap3_c[ab], 1)
                
                cand_gains = gain[candidates]
                if len(cand_gains) > 0:
                    best_cand_idx = np.argmax(cand_gains)
                    if cand_gains[best_cand_idx] > best_delta:
                        best_delta = cand_gains[best_cand_idx]
                        best_move = ('add', candidates[best_cand_idx])
        
        # Decide whether to make the move
        if best_move is not None:
            action, x = best_move
            # SA acceptance: always accept if delta > 0; otherwise accept with prob exp(delta/T)
            if best_delta > 0 or (T > 0 and rng.random() < np.exp(best_delta / T)):
                if action == 'add':
                    A[x] = True
                else:
                    A[x] = False
                current_score += int(best_delta)
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
        
        # Cool down
        T = max(T * 0.999, T_end)
        
        # Periodic restart from best if stuck
        if iteration % 500 == 0:
            # Check if we've been stuck
            if current_score < best_score - 5:
                # Restart from best
                A = best_A.copy()
                current_score = best_score
                T = T0  # Reheat
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse initial solutions
    best_A = None
    best_score = -1
    
    # How many initial candidates to try
    n_candidates = 8
    
    for i in range(n_candidates):
        if time.time() > deadline - 5:
            break
        
        # Create initial solution
        if i == 0:
            # Deterministic greedy (ordered)
            A_init = np.zeros(N + 1, dtype=bool)
            for x in range(1, N + 1):
                A_init[x] = True
                if not is_4ap_free(A_init):
                    A_init[x] = False
        elif i == 1:
            # Reverse greedy
            A_init = np.zeros(N + 1, dtype=bool)
            for x in range(N, 0, -1):
                A_init[x] = True
                if not is_4ap_free(A_init):
                    A_init[x] = False
        else:
            # Random greedy
            A_init = greedy_initial(rng)
        
        # Run SA on this initial solution
        sa_deadline = deadline - 3
        A_opt, score_opt = simulated_annealing(A_init, sa_deadline, rng)
        
        if score_opt > best_score:
            best_score = score_opt
            best_A = A_opt
    
    if best_A is None:
        # Fallback: simple greedy
        best_A = np.zeros(N + 1, dtype=bool)
        for x in range(1, N + 1):
            best_A[x] = True
            if not is_4ap_free(best_A):
                best_A[x] = False
    
    # Final verification
    if not is_4ap_free(best_A):
        # Shouldn't happen, but safety check
        # Remove elements that create 4-APs
        changed = True
        while changed and is_4ap_free(best_A) == False:
            changed = False
            for x in range(N, 0, -1):
                if best_A[x]:
                    best_A[x] = False
                    if is_4ap_free(best_A):
                        changed = True
                        break
    
    # Convert to list
    result = [i for i in range(1, N + 1) if best_A[i]]
    
    if not result:
        # Absolute fallback
        result = [1]
    
    return result
