import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We want to maximize L.
    # The seed strategy was: residue[i] = ceil(i/2) mod p.
    # This ensures that the first 84 integers are covered at least twice?
    # Let's check the seed score.
    # The inspiration says score 1.0000 for the seed.
    # The current program has score 0.4471.
    # Why is the current program worse?
    # The current program tries to greedily extend L by changing residues.
    # It seems to get stuck or make bad moves.
    # The seed is a fixed heuristic.
    # Let's analyze the seed heuristic:
    # r_i = ceil(i/2) mod p_i.
    # For integer k, which primes cover it?
    # k is covered by prime i if k = r_i mod p_i.
    # r_i = ceil(i/2).
    # So k is covered by prime i if k = ceil(i/2) mod p_i.
    # This implies ceil(i/2) = k + m * p_i for some integer m >= 0.
    # So ceil(i/2) >= k.
    # Also ceil(i/2) < p_i + k (if m=0) or larger.
    
    # Let's just use the seed as a strong baseline and try to improve it with local search.
    # The seed gives L >= 84?
    # Actually, the task says "L >= 1000 would realize the property...".
    # And "Scores are doubly covered lengths for n = 1000".
    # The seed score is 1.0.
    # If the seed returns a specific L, then 1.0 corresponds to that L.
    # Let's calculate the L for the seed.
    
    def evaluate(residues, limit=10000):
        # Returns the largest L such that all 1..L have >= 2 hits
        counts = np.zeros(limit + 2, dtype=np.int32)
        
        for r, p in zip(residues, primes):
            start = r
            if start == 0:
                start = p
            if start <= limit:
                counts[start:limit+1:p] += 1
            
        L = 0
        # Vectorized check for first failure
        # We need the first index i where counts[i] < 2
        # Find indices where counts < 2
        # But we only care about the first one in 1..limit
        # If all are >= 2, L = limit.
        
        # np.argmax returns first occurrence of max.
        # We want first occurrence of (counts < 2).
        # Invert: (counts >= 2) is True. We want first False.
        # ~ (counts >= 2) is True where counts < 2.
        # But we start from 1.
        
        # Check 1..limit
        valid = counts[1:limit+1] >= 2
        # If all True, L = limit
        if np.all(valid):
            return limit, counts
        else:
            # Find first False
            first_false = np.argmax(~valid)
            # L is the index before first_false (1-based)
            # first_false is 0-based index in valid array.
            # So the failing index is first_false + 1.
            # L is first_false.
            L = first_false
            return L, counts

    # Seed initialization
    def seed_residues():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]

    best_L = 0
    best_residues = seed_residues()
    L, counts = evaluate(best_residues)
    best_L = L

    # Local Search / Hill Climbing
    # Try to change one residue to increase L.
    # If L is stuck, try to change a residue to allow a future extension?
    # Or just try to extend L step by step.
    
    # To extend L to L+1, we need counts[L+1] >= 2.
    # Currently counts[L+1] is 0 or 1.
    # We need to add 1 or 2 hits to L+1.
    # We can change a prime's residue to cover L+1.
    # This might break coverage for some y <= L.
    # We need to ensure no y <= L drops below 2.
    
    # Strategy:
    # 1. Start with seed.
    # 2. Try to extend L one by one.
    # 3. At each step, try all primes to see if changing residue to cover L+1 is safe.
    # 4. If safe, apply.
    # 5. If not safe, try to find a sequence of changes? Or random restart?
    # 6. Or, try to "prepare" for L+1 by changing residues that don't break current L but increase counts[L+1]?
    
    # Let's implement a robust greedy extension with backtracking or randomized restarts if stuck.
    
    def try_extend(residues, L, counts, max_attempts=1000):
        target = L + 1
        if target > 10000:
            return False, residues, L, counts
        
        # Shuffle primes to try different orders
        indices = list(range(n_primes))
        random.shuffle(indices)
        
        for _ in range(max_attempts):
            random.shuffle(indices)
            for idx in indices:
                p = primes[idx]
                old_r = residues[idx]
                new_r = target % p
                
                if old_r == new_r:
                    continue
                
                # Check if changing idx to new_r is safe for [1, L]
                # And if it helps target.
                
                # Old coverage in [1, L]:
                old_start = old_r if old_r != 0 else p
                new_start = new_r if new_r != 0 else p
                
                # Indices affected in [1, L]
                # We need to check if any y in [1, L] drops from 2 to 1.
                # This happens if y is in OldCover \ NewCover and counts[y] == 2.
                
                # Generate lists
                old_indices = []
                if old_start <= L:
                    old_indices = range(old_start, L + 1, p)
                
                new_indices = []
                if new_start <= L:
                    new_indices = range(new_start, L + 1, p)
                
                # Create sets for fast lookup?
                # Or just iterate.
                # L is smallish (up to 1000+).
                # p is small.
                
                new_set = set(new_indices)
                
                safe = True
                for y in old_indices:
                    if y not in new_set:
                        if counts[y] == 2:
                            safe = False
                            break
                
                if not safe:
                    continue
                
                # If safe, check if this change helps target.
                # Target is L+1.
                # new_r covers target by definition.
                # Did old_r cover target?
                old_covers_target = (target % p == old_r)
                
                # If old_r didn't cover target, we add 1 hit.
                # If old_r covered target, we swap (net 0 change for target count, but we already had it).
                # Wait, if old_r covered target, then counts[target] was already incremented by this prime.
                # Changing to new_r (which also covers target) means counts[target] stays same.
                # But we need counts[target] to reach 2.
                # If counts[target] was 1, and old_r covered it, it's still 1. No gain.
                # If counts[target] was 0, old_r didn't cover it. New_r covers it. Count becomes 1. Still not 2.
                # So a single prime change can only increase counts[target] by 1 if it wasn't covered before.
                # We need counts[target] >= 2.
                
                # So we need to know current counts[target].
                # If counts[target] == 1, we need to add 1 more.
                # If counts[target] == 0, we need to add 2 more.
                # A single prime change adds at most 1.
                # So if counts[target] == 0, one change is not enough.
                # We need to change two primes?
                # Or maybe one prime change makes it 1, and we try another?
                
                # Let's just apply the safe change if it increases counts[target].
                # And if counts[target] becomes 2, we extend L.
                
                # Calculate new count for target
                if old_covers_target:
                    new_count_target = counts[target]
                else:
                    new_count_target = counts[target] + 1
                
                if new_count_target >= 2:
                    # Apply change
                    residues[idx] = new_r
                    # Update counts
                    for y in old_indices:
                        if y not in new_set:
                            counts[y] -= 1
                    for y in new_indices:
                        if y not in set(old_indices):
                            counts[y] += 1
                            
                    if not old_covers_target:
                        counts[target] += 1
                    # else no change to counts[target]
                    
                    # Success
                    return True, residues, target, counts
                else:
                    # This change is safe for [1,L] but doesn't finish the job for L+1.
                    # We could apply it to "prepare" for the next step?
                    # But that's complex. Let's stick to greedy: only apply if it extends L.
                    # If we can't extend L in one step, we might be stuck.
                    # We can try random restarts or more complex moves.
                    pass
        
        return False, residues, L, counts

    # Main loop
    current_residues = best_residues[:]
    current_L = best_L
    current_counts = counts
    
    # Try to extend
    stuck_counter = 0
    while time.time() < deadline:
        extended, current_residues, current_L, current_counts = try_extend(current_residues, current_L, current_counts)
        
        if extended:
            if current_L > best_L:
                best_L = current_L
                best_residues = current_residues[:]
            stuck_counter = 0
        else:
            # Stuck
            stuck_counter += 1
            if stuck_counter > 5:
                # Try random restarts or perturbations
                # Perturb: change a few random residues that are "safe" to change?
                # Or just restart from seed with different randomization?
                # Let's try a few random restarts from seed.
                restarts = 0
                while restarts < 5 and time.time() < deadline:
                    current_residues = seed_residues()
                    # Add some random noise?
                    # Or just try the same greedy again? It's deterministic if we don't shuffle.
                    # But we shuffle in try_extend.
                    current_L, current_counts = evaluate(current_residues)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = current_residues[:]
                    
                    # Try to extend from this restart
                    while time.time() < deadline:
                        extended, current_residues, current_L, current_counts = try_extend(current_residues, current_L, current_counts)
                        if extended:
                            if current_L > best_L:
                                best_L = current_L
                                best_residues = current_residues[:]
                        else:
                            break
                    restarts += 1
                break # Exit main loop if we are doing restarts
    
    return best_residues
