import time
import random
import sys

N = 5000
DEADLINE = 300  # seconds, safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Precompute primes up to N+1
    # Note: Admissibility requires checking primes p <= len(tuple).
    # Since len(tuple) <= N+1, we need primes up to N+1.
    primes = _primes(N + 1)
    
    # We will maintain the state of the current set S.
    # To efficiently check if a candidate x can be added to S:
    # For each prime p <= len(S) + 1:
    #   The residue r = x % p must NOT be the *only* missing residue in S mod p.
    #   Actually, the condition is: S is admissible if for every p <= |S|, some residue is missing.
    #   If we add x, the new size is |S|+1. We need the new set S' to be admissible.
    #   This means for every p <= |S|+1, S' must have a missing residue mod p.
    
    # Let's track `missed[p]` as the set of residues mod p not in S.
    # Initially S is empty, so missed[p] = {0, 1, ..., p-1} for all p.
    
    # When we add x:
    # 1. For each p, remove (x % p) from missed[p].
    # 2. Check validity:
    #    - For p <= |S| (old size): missed[p] must not be empty. (It wasn't before, and we just removed one, so if it had size 1 and we removed that one, it's now empty -> Invalid).
    #    - For p = |S|+1 (new size): missed[p] must not be empty. (It had size p, we removed one, so size p-1. Since p>=2, p-1>=1, so it's never empty).
    #    - For p > |S|+1: No constraint yet.
    
    # So the only constraint is: For all p <= |S|, len(missed[p]) > 0 after removal.
    # This is equivalent to: For all p <= |S|, x % p is NOT the unique element in missed[p].
    
    # Wait, if p <= |S|, and missed[p] has size > 1, we can remove any element.
    # If missed[p] has size 1, we CANNOT remove that specific element.
    
    # So, x is valid to add to S of size k if:
    # For all primes p <= k, x % p != the unique element in missed[p] (if |missed[p]|==1).
    # If |missed[p]| > 1, x % p can be anything.
    # If |missed[p]| == 0, S is already invalid (should not happen).
    
    # Let's optimize this check.
    # We maintain `unique_missed[p]`:
    #   If |missed[p]| == 1, unique_missed[p] = that element.
    #   Else unique_missed[p] = -1 (None).
    
    # Then x is valid if for all p <= k, x % p != unique_missed[p] (when unique_missed[p] != -1).
    
    # To make the search "Randomized Greedy":
    # 1. Start with S = empty.
    # 2. Candidate pool = [0, ..., N].
    # 3. Repeatedly:
    #    a. Filter candidate pool to valid candidates based on current S.
    #    b. If no valid candidates, stop (or restart with different randomness?).
    #    c. Select a candidate.
    #       - Pure Greedy: Pick the one that "costs" the least? 
    #         Cost could be defined as the number of future candidates it invalidates.
    #         However, calculating this is expensive.
    #       - Randomized: Pick a random valid candidate.
    #       - Hybrid: Pick from top K valid candidates randomly?
    
    # Given the time limit, let's try a simpler heuristic first:
    # Just pick a random valid candidate. This is essentially building a random independent set in the conflict graph, but the graph is dynamic.
    # Actually, "Randomized Greedy" usually implies picking the node with minimum degree in the remaining graph.
    # Degree of x = number of other candidates y such that x and y cannot coexist.
    # Two candidates x, y cannot coexist if adding x then y (or y then x) leads to a non-admissible set?
    # No, admissibility is a property of the set.
    # x and y are "conflicting" if there is no superset S containing {x,y} that is admissible?
    # That's hard to compute.
    
    # Let's stick to the simple validity check.
    # To improve over pure random, we can use a heuristic for selection:
    # Prefer candidates that are "rare" in some sense?
    # Or, we can run multiple random restarts and keep the best set found.
    
    # Let's implement a function that builds a set using random valid choices.
    # We will run this multiple times (restarts) if time permits.
    
    def build_set_randomized(seed):
        random.seed(seed)
        S = []
        in_S = set()
        
        # Initialize missed sets
        # We only care about primes p <= N+1.
        # missed[p] is a set of residues.
        # For efficiency, we can just track the count of missed residues and the unique one if count==1.
        
        # Let's use a list of sets for simplicity first, then optimize if needed.
        # N=5000, primes ~ 669.
        # Sets of size up to 5000.
        
        missed = {}
        unique_missed = {}
        
        for p in primes:
            missed[p] = set(range(p))
            unique_missed[p] = -1 # None
            
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        current_size = 0
        
        while candidates:
            # Check time
            if time.time() - start_time > DEADLINE - 2:
                break
                
            # Find a valid candidate
            # To speed up, we can iterate through candidates and check validity.
            # If we find one, we pick it (or pick randomly from a small subset of valid ones found).
            
            # Optimization: Instead of checking all candidates every time,
            # we can just iterate.
            
            valid_found = False
            # To introduce randomness in selection among valids, 
            # we can shuffle the candidates list periodically or just pick the first valid one we find in a shuffled list.
            # But shuffling 5000 elements every step is O(N^2 log N).
            # Better: Maintain a list of active candidates.
            
            # Let's just iterate through the current candidate list.
            # To avoid bias towards lower numbers, we can shuffle the candidate list initially.
            # But as we remove elements, the order is preserved.
            
            # A better strategy for "Greedy":
            # Pick the candidate that is "least restrictive".
            # How to measure restrictiveness?
            # If we add x, it removes residue r=p% x from missed[p].
            # If missed[p] becomes empty, it's invalid.
            # If missed[p] becomes size 1, it creates a new constraint for future elements.
            # So, adding x is "bad" if it causes many primes p to go from |missed|>1 to |missed|==1.
            
            # Let's calculate a "cost" for each valid candidate.
            # Cost = sum(1 for p in primes if p <= current_size and unique_missed[p] == -1 and (x%p in missed[p]) and len(missed[p]) == 2)
            # This counts how many primes will become "critical" (size 1) after adding x.
            # We want to minimize this cost.
            
            # However, calculating this for all candidates is O(N * #primes).
            # #primes ~ 670. N=5000. 5000*670 = 3.35M operations per step.
            # Steps ~ 5000. Total ~ 16B operations. Too slow in Python.
            
            # So we must use a cheaper heuristic.
            # Cheapest: Random valid candidate.
            # Medium: Pick the first valid candidate in a randomly shuffled list of *all* candidates? No, we need to filter.
            
            # Let's try: 
            # 1. Identify all valid candidates.
            # 2. If many, pick one randomly.
            # 3. If few, pick the one with lowest "potential cost" (approximated).
            
            # To identify valid candidates efficiently:
            # We need to check: for all p <= current_size, x % p != unique_missed[p] (if unique).
            
            # Let's just iterate through candidates and keep the first valid one we find, but to avoid bias, 
            # we can randomly sample a few valid ones and pick the best?
            
            # Given the time constraint, let's try a simple "Random Valid" approach with multiple restarts.
            # The "Greedy" aspect can be that we prioritize numbers that are "spread out"?
            
            # Actually, a known heuristic for this type of problem (Max Independent Set on hypergraph-like structures) 
            # is to pick the vertex with the minimum degree. 
            # Here, "degree" can be approximated by how many residues it "uses up" in critical primes.
            
            # Let's implement a cost function that is cheap:
            # cost(x) = number of primes p <= current_size such that len(missed[p]) == 2 and x % p in missed[p].
            # This is O(#primes) per candidate.
            # If we evaluate cost for all 5000 candidates, it's 5000 * 670 = 3.35M ops.
            # If we do this for 5000 steps, it's 16B ops. Still too slow.
            
            # We need to reduce the number of candidates evaluated per step.
            # Strategy: 
            # 1. Sample a random subset of candidates (e.g., 100).
            # 2. Filter to valid ones.
            # 3. Pick the one with min cost (or random if costs are similar).
            # 4. Add to S.
            
            # This makes the search stochastic and fast.
            
            sample_size = 50
            valid_in_sample = []
            
            # To ensure we don't miss good candidates, we can mix:
            # - Check the first 10 candidates in the list (to handle easy cases).
            # - Check a random sample of 50.
            
            # Actually, just checking a random sample is fine for a randomized algorithm.
            
            for _ in range(sample_size):
                if not candidates:
                    break
                idx = random.randint(0, len(candidates) - 1)
                x = candidates[idx]
                
                # Check validity
                is_valid = True
                for p in primes:
                    if p > current_size:
                        break
                    um = unique_missed[p]
                    if um != -1:
                        if x % p == um:
                            is_valid = False
                            break
                
                if is_valid:
                    # Calculate cheap cost
                    cost = 0
                    for p in primes:
                        if p > current_size:
                            break
                        if len(missed[p]) == 2:
                            if x % p in missed[p]:
                                cost += 1
                    valid_in_sample.append((cost, x))
            
            if not valid_in_sample:
                # No valid candidates found in sample. 
                # This might mean we are stuck, or the sample was unlucky.
                # We can try a larger sample or just break.
                # If we break, we finish this run.
                # To be safe, let's try one more pass with full scan? 
                # No, full scan is expensive. Let's just break.
                # Or, we can try to find *any* valid candidate by scanning the whole list?
                # If the list is small, scan all.
                if len(candidates) < 200:
                    for x in candidates:
                        is_valid = True
                        for p in primes:
                            if p > current_size:
                                break
                            um = unique_missed[p]
                            if um != -1:
                                if x % p == um:
                                    is_valid = False
                                    break
                        if is_valid:
                            valid_in_sample.append((0, x))
                            break # Just find one
                    if not valid_in_sample:
                        break
                else:
                    break
            
            # Pick the best candidate (min cost, random tie-break)
            # Sort by cost, then pick random from the lowest cost group
            min_cost = min(c for c, _ in valid_in_sample)
            best_candidates = [x for c, x in valid_in_sample if c == min_cost]
            chosen = random.choice(best_candidates)
            
            # Add chosen to S
            S.append(chosen)
            in_S.add(chosen)
            
            # Remove chosen from candidates
            # This is O(N) if we use list remove. 
            # To make it faster, we can just mark it as invalid or use a set for candidates?
            # If we use a set, we lose the "list" structure for random sampling without replacement easily.
            # Let's just use a list and `remove`. N=5000, 5000 removals = 25M ops. Acceptable.
            candidates.remove(chosen)
            
            # Update state
            current_size += 1
            for p in primes:
                if p > current_size + 1: # We only need to update up to new size + 1? 
                    # Actually, we need to update missed[p] for all p.
                    # But we only *check* p <= current_size for validity.
                    # However, missed[p] for p > current_size will be checked later when current_size grows.
                    # So we must update all p.
                    pass
                
                r = chosen % p
                if r in missed[p]:
                    missed[p].discard(r)
                    if len(missed[p]) == 1:
                        unique_missed[p] = next(iter(missed[p]))
                    elif len(missed[p]) == 0:
                        # Should not happen if valid
                        unique_missed[p] = -1
                    else:
                        unique_missed[p] = -1
                # If r not in missed[p], it was already removed? No, missed[p] contains residues NOT in S.
                # So r should always be in missed[p] before adding chosen.
                # Wait, if chosen was already in S? No, we removed it from candidates.
                # So r must be in missed[p].
            
        return S

    best_set = []
    
    # Run multiple restarts
    # How many? 
    # Each run takes T seconds. 
    # We have DEADLINE seconds.
    # Let's estimate T. 
    # 5000 steps. Each step: 50 samples * (670 checks + 670 cost calcs) ~ 50 * 1340 = 67k ops.
    # Plus update: 670 ops.
    # Total per step ~ 68k ops.
    # 5000 steps * 68k = 340M ops.
    # Python does ~10M-50M simple ops per second. 
    # So 340M ops might take 10-30 seconds.
    # So we can run ~10-30 restarts in 300s.
    
    run_id = 0
    while time.time() - start_time < DEADLINE - 5:
        run_id += 1
        current_set = build_set_randomized(run_id)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # If we found a very large set, we might want to stop early?
        # Max possible is N+1 = 5001. 
        # If we get close, we can break.
        if len(best_set) > 4500: 
            break

    return best_set

# EVOLVE-BLOCK-END
