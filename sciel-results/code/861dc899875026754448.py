import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We will evaluate up to a sufficiently large L to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    # We cap the search space for evaluation at MAX_EVAL to keep memory/time low.
    # Based on inspirations, 20000 is a good balance. If we exceed this, we can extend, 
    # but for simplicity and speed, we fix the window.
    MAX_EVAL = 20000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        
        first_fail_idx = np.argmax(mask)
        return int(first_fail_idx)

    def get_coverage(residues):
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    # Initial Solution Strategy:
    # To start with a reasonable baseline, we use a "small number bias".
    # We want small x to be covered by many primes.
    # If we set r_p = 1 for all p, x=1 is covered by all.
    # But we need double coverage for ALL x in 1..L.
    # A better start: Random, but then immediately apply a few rounds of greedy repair.
    
    def generate_initial_residues():
        res = np.zeros(n_primes, dtype=np.int64)
        for i, p in enumerate(primes):
            res[i] = random.randint(0, p - 1)
        return res

    residues = generate_initial_residues()
    
    # Pre-compute coverage for initial state
    current_coverage = get_coverage(residues)
    current_score = evaluate(residues)
    
    best_residues = residues.copy()
    best_score = current_score
    
    # Greedy Conflict Resolution with Lookahead
    # We maintain a list of "conflicts" (indices where coverage < 2).
    # We want to minimize the number of conflicts, specifically the earliest one.
    
    # To speed up lookahead, we don't want to recompute full coverage for every candidate prime.
    # Instead, for a candidate move (change r_i to new_r_i), we can compute the delta in coverage.
    # old_hits_i = (xs - old_r_i) % p_i == 0
    # new_hits_i = (xs - new_r_i) % p_i == 0
    # delta_coverage = new_hits_i - old_hits_i
    # new_coverage = current_coverage + delta_coverage
    
    # We define a "cost" for a move:
    # 1. Does it fix the current first conflict?
    # 2. If yes, does it create new conflicts before the current first conflict is resolved?
    #    Actually, if it fixes the first conflict, the new first conflict will be later.
    #    We want to minimize the number of new conflicts created in [1, new_L].
    
    # Let's implement a loop that repeatedly finds the first conflict and fixes it.
    
    # To avoid getting stuck, we will use a "stall" counter. If no improvement in X steps, 
    # we randomize a few residues to escape local minima.
    
    stall_counter = 0
    max_stall = 500
    
    # We also need to handle the case where we hit MAX_EVAL.
    # If we hit MAX_EVAL, we can try to extend the window or just stop.
    # Given the 120s limit, if we reach 20000 quickly, we might have time to extend.
    # But for now, we focus on maximizing L within MAX_EVAL.
    
    # Optimization: Precompute modulo indices? 
    # (xs - r) % p == 0 is equivalent to xs % p == r % p.
    # We can precompute xs % p for each p? That's 168 * 20000 = 3.3M ops. 
    # But r changes. So we need (xs % p) == (r % p).
    # Let's precompute xs_mod_p for each prime.
    # xs_mod_p[i] = xs % primes[i]
    # Then hits for residue r is (xs_mod_p[i] == r % p).
    # This is faster than remainder on the fly.
    
    xs_mods = np.zeros((n_primes, MAX_EVAL), dtype=np.int64)
    for i, p in enumerate(primes):
        xs_mods[i] = np.remainder(xs, p)
        
    # Update evaluate and get_coverage to use precomputed mods?
    # evaluate(residues) -> coverage = sum((xs_mods[i] == residues[i] % p) for i)
    # This is much faster.
    
    def fast_get_coverage(residues):
        # residues is array of shape (n_primes,)
        # residues_mod = residues % primes_arr
        residues_mod = np.remainder(residues, primes_arr)
        # hits[i, j] = (xs_mods[i, j] == residues_mod[i])
        # We can do this vectorized:
        # Compare xs_mods (n_primes, MAX_EVAL) with residues_mod[:, None]
        hits = (xs_mods == residues_mod[:, None])
        return np.sum(hits, axis=0)
        
    def fast_evaluate(residues):
        coverage = fast_get_coverage(residues)
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        return int(np.argmax(mask))

    # Re-evaluate initial state with fast method
    current_coverage = fast_get_coverage(residues)
    current_score = fast_evaluate(residues)
    best_score = current_score
    best_residues = residues.copy()
    
    # Main Loop: Greedy Repair with Random Escapes
    while time.time() < deadline:
        if current_score >= MAX_EVAL:
            # We reached the cap. 
            # We could try to extend MAX_EVAL, but that requires reallocating xs and xs_mods.
            # Let's just do random moves to see if we can find a structure that allows higher L 
            # if we were to extend. But since we can't extend easily without breaking the 
            # precomputed arrays, we'll just stop or do random moves to maintain state.
            # Actually, if we hit MAX_EVAL, the score is capped. 
            # To get a higher score, we MUST extend MAX_EVAL.
            # Let's check if we have time to extend.
            # Extending to 40000 doubles memory and time per eval.
            # If we are at 20000, we might be stuck. 
            # Let's try a simple extension: if we hit 20000, we break and return.
            # Or we can try to extend once.
            # For now, break.
            break
        
        # Find the first conflict
        # current_coverage < 2
        conflict_mask = current_coverage < 2
        first_conflict_idx = np.argmax(conflict_mask)
        x_target = first_conflict_idx + 1  # 1-based index
        
        # We want to fix x_target.
        # We need to increase coverage of x_target by at least 1.
        # Currently c = current_coverage[first_conflict_idx]. c is 0 or 1.
        
        # Candidates: Primes p_i such that they do NOT currently cover x_target.
        # If we change r_i to x_target % p_i, they will cover x_target.
        
        # Compute which primes cover x_target currently
        # covers_target[i] = (xs_mods[i, first_conflict_idx] == residues[i] % primes[i])
        residues_mod = np.remainder(residues, primes_arr)
        covers_target = (xs_mods[:, first_conflict_idx] == residues_mod)
        
        # Indices of primes that do NOT cover x_target
        candidate_indices = np.where(~covers_target)[0]
        
        if len(candidate_indices) == 0:
            # All primes cover x_target. This means c >= n_primes. 
            # But c < 2, so this is impossible unless n_primes < 2, which is false.
            # So this branch should not happen.
            # If it does, random move.
            i = random.randint(0, n_primes - 1)
            new_r = random.randint(0, primes_arr[i] - 1)
            old_r = residues[i]
            residues[i] = new_r
            current_coverage = fast_get_coverage(residues)
            current_score = fast_evaluate(residues)
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
                stall_counter = 0
            else:
                stall_counter += 1
            continue
            
        # For each candidate, calculate the "cost" of changing its residue.
        # Cost = Number of new conflicts created in [1, MAX_EVAL] 
        #       MINUS (Number of old conflicts resolved in [1, MAX_EVAL])
        # We want to minimize Cost.
        # Ideally, we resolve the first conflict and don't create new ones before it.
        
        # To speed this up, we only evaluate a subset of candidates if there are many.
        # But 168 candidates is small. Let's evaluate all.
        
        best_candidate = -1
        best_cost = float('inf')
        best_new_r = 0
        
        # Pre-compute old hits for candidates?
        # We have xs_mods.
        # For candidate i, old_r = residues[i], new_r = x_target % primes[i]
        # old_hits_i = (xs_mods[i] == residues_mod[i])
        # new_hits_i = (xs_mods[i] == new_r % primes[i])
        
        # We can compute delta_coverage for each candidate.
        # But we need to aggregate them? No, we only change ONE prime at a time.
        
        # Let's pick a random subset of candidates to evaluate to save time if needed, 
        # but 168 is small enough to evaluate all.
        
        # Optimization: 
        # We know that changing r_i to new_r_i will increase coverage of x_target by 1.
        # The risk is breaking coverage of other x in 1..x_target-1 or creating new conflicts elsewhere.
        
        # Let's evaluate the top few candidates based on a heuristic:
        # Prefer primes with larger p? Changing a large prime affects fewer numbers.
        # Or prefer primes that are already "loose"?
        
        # Let's just evaluate all candidates.
        # To make it faster, we can vectorize the cost calculation for all candidates at once?
        # No, because each candidate changes a different row in the coverage matrix.
        
        # Let's loop through candidates.
        # To save time, we can limit the number of candidates we check per iteration.
        # Let's check up to 20 random candidates.
        
        if len(candidate_indices) > 20:
            # Randomly sample 20
            sampled_indices = random.sample(candidate_indices.tolist(), 20)
        else:
            sampled_indices = candidate_indices.tolist()
            
        for i in sampled_indices:
            old_r = residues[i]
            new_r = x_target % primes_arr[i]
            
            # Calculate delta in coverage
            # old_hits = (xs_mods[i] == residues_mod[i])
            # new_hits = (xs_mods[i] == new_r)  # Note: new_r < p, so new_r % p == new_r
            # But residues_mod[i] is old_r % p.
            
            # We can compute the vector of changes for prime i
            # old_val = residues_mod[i]
            # new_val = new_r
            # old_hits_i = (xs_mods[i] == old_val)
            # new_hits_i = (xs_mods[i] == new_val)
            # delta_i = new_hits_i.astype(int) - old_hits_i.astype(int)
            
            # new_coverage = current_coverage + delta_i
            # conflicts_new = (new_coverage < 2)
            # conflicts_old = (current_coverage < 2)
            
            # We want to minimize the number of conflicts.
            # Specifically, we want the first conflict to move forward.
            
            # Let's compute the new first conflict index.
            # This requires finding the first True in (new_coverage < 2).
            
            # To speed up, we can just check if the number of conflicts decreases.
            # Or better, check if the first conflict index increases.
            
            old_val = residues_mod[i]
            new_val = new_r
            
            # If old_val == new_val, no change. (Should not happen for candidates not covering x_target? 
            # Wait, if it doesn't cover x_target, then xs_mods[i, first_conflict_idx] != old_val.
            # new_val = x_target % p. xs_mods[i, first_conflict_idx] is x_target % p.
            # So new_hits[i, first_conflict_idx] is True.
            # old_hits[i, first_conflict_idx] is False.
            # So delta is non-zero at first_conflict_idx.
            
            # Compute delta vector
            # We can avoid full vector computation if we just check the first conflict?
            # No, we need to know if we create earlier conflicts.
            
            # Let's compute the new coverage for this specific prime change.
            # This is O(MAX_EVAL) per candidate. 20 * 20000 = 400k ops. Very fast.
            
            # old_hits_i = (xs_mods[i] == old_val)
            # new_hits_i = (xs_mods[i] == new_val)
            # delta_i = new_hits_i.astype(np.int8) - old_hits_i.astype(np.int8)
            
            # To save memory/time, we can just compute the new coverage array for this move.
            # new_cov = current_coverage + (new_hits_i - old_hits_i)
            
            # Let's do it.
            old_hits_i = (xs_mods[i] == old_val)
            new_hits_i = (xs_mods[i] == new_val)
            
            # Convert to int for subtraction
            # Using int8 might overflow if coverage > 127, but max coverage is 168. 
            # So int16 is safe.
            delta_i = new_hits_i.astype(np.int16) - old_hits_i.astype(np.int16)
            
            new_cov = current_coverage + delta_i
            
            # Find new first conflict
            new_conflict_mask = new_cov < 2
            if not np.any(new_conflict_mask):
                new_first_conflict_idx = MAX_EVAL
            else:
                new_first_conflict_idx = np.argmax(new_conflict_mask)
                
            # We want to maximize new_first_conflict_idx.
            # If it's larger than current first_conflict_idx, it's an improvement.
            
            if new_first_conflict_idx > first_conflict_idx:
                # This is a good move.
                # We prefer the one that moves the conflict the furthest.
                # Tie-breaker: fewer total conflicts?
                # Let's just use the index.
                if new_first_conflict_idx > best_cost:
                    best_cost = new_first_conflict_idx
                    best_candidate = i
                    best_new_r = new_r
            
        if best_candidate != -1:
            # Apply the best move
            i = best_candidate
            old_r = residues[i]
            residues[i] = best_new_r
            
            # Update coverage
            # We can recompute fast_get_coverage, or update incrementally.
            # Incremental is faster.
            old_val = residues_mod[i]
            new_val = best_new_r
            old_hits_i = (xs_mods[i] == old_val)
            new_hits_i = (xs_mods[i] == new_val)
            delta_i = new_hits_i.astype(np.int16) - old_hits_i.astype(np.int16)
            current_coverage = current_coverage + delta_i
            
            # Update residues_mod
            residues_mod[i] = new_val
            
            # Update score
            current_score = best_cost  # Since best_cost was the new first conflict index
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
                stall_counter = 0
            else:
                # This shouldn't happen if we only accept improvements in first_conflict_idx
                # But maybe we accepted a move that didn't improve the best?
                # No, we only applied if new_first_conflict_idx > first_conflict_idx.
                # So current_score should be > old current_score.
                # So it should update best_score.
                stall_counter += 1
        else:
            # No improving move found among sampled candidates.
            # We are stuck.
            stall_counter += 1
            
            if stall_counter > max_stall:
                # Random escape
                # Pick 5 random primes and randomize their residues
                for _ in range(5):
                    i = random.randint(0, n_primes - 1)
                    residues[i] = random.randint(0, primes_arr[i] - 1)
                    residues_mod[i] = residues[i] % primes_arr[i]
                
                # Recompute coverage
                current_coverage = fast_get_coverage(residues)
                current_score = fast_evaluate(residues)
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
                stall_counter = 0
                max_stall = random.randint(200, 1000) # Randomize escape frequency

        if time.time() > deadline - 5.0:
            break

    # Final check: if we hit MAX_EVAL, we might be able to extend.
    # But for now, return best_residues.
    
    return best_residues.tolist()
