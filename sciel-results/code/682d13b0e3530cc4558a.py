"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(1000)
    num_primes = len(primes)
    
    # Precompute reciprocals as Fractions for exact budget check
    # We will use a float approximation for the fast SA loop, 
    # and verify with Fraction only when a candidate is accepted or at the end.
    # Actually, for SA, we need very fast budget checks. 
    # Sum of 1/p for all primes <= 1000 is ~2.2. 
    # We can track the float sum and only check Fraction if float sum <= 1.0 + epsilon.
    reciprocals = [1.0 / p for p in primes]
    reciprocals_frac = [Fraction(1, p) for p in primes]
    
    # Coverage representation:
    # We want to cover 1..L.
    # Using a list of booleans is slow for repeated updates in SA.
    # Using bitmasks is fast but Python integers are arbitrary precision.
    # Let's use a list of 64-bit integers as a bitmask array.
    # MAX_L = 100000. 100000 / 64 = 1563 words.
    MAX_L = 100000
    NUM_WORDS = (MAX_L + 64) // 64
    
    def create_mask_array():
        return [0] * NUM_WORDS
        
    def set_bit(arr, pos):
        # pos is 1-indexed (1..L)
        if pos < 1: return
        if pos > MAX_L: return
        word_idx = (pos - 1) // 64
        bit_idx = (pos - 1) % 64
        arr[word_idx] |= (1 << bit_idx)
        
    def get_mask(arr, pos):
        if pos < 1 or pos > MAX_L: return False
        word_idx = (pos - 1) // 64
        bit_idx = (pos - 1) % 64
        return (arr[word_idx] >> bit_idx) & 1

    def add_prime_to_mask(arr, p, r):
        # Adds coverage for k % p == r in range 1..MAX_L
        # k = r, r+p, r+2p, ...
        # Note: if r == 0, k is multiple of p. k >= 1.
        # First k: if r == 0, k=p. Else k=r (if r>=1).
        # Wait, r is in 0..p-1.
        # If r == 0, numbers are p, 2p, 3p...
        # If r > 0, numbers are r, r+p, r+2p... (if r >= 1)
        
        first_k = r if r > 0 else p
        # Iterate and set bits
        # This is O(MAX_L / p). 
        # In SA, we do this for every move. 
        # If p is small (e.g., 2), this is 50,000 operations.
        # This might be too slow for Python SA if we do many steps.
        
        # Optimization: Precompute masks for each (p, r)?
        # Total primes 168. Sum of p is ~76k.
        # Precomputing 76k masks of size 1563 words is 76k * 1563 * 8 bytes ~ 950 MB. Too much.
        
        # Alternative: Instead of updating the global mask array, 
        # maintain a count array or a "coverage depth" array?
        # Or, simply compute the score (contiguous L) from scratch?
        # Computing score from scratch requires knowing which numbers are covered.
        
        # Let's try a different approach for SA:
        # State: List of selected (p, r) pairs.
        # Score: Contiguous L.
        # To compute Score, we need to check 1, 2, 3... until a gap.
        # For each k, check if any (p, r) in selected covers k.
        # If we have ~50 primes selected, checking k takes 50 mod ops.
        # If L is ~1000, that's 50k ops per evaluation.
        # If we do 1000 evaluations/sec, that's 50M ops/sec. Python might struggle.
        
        # Better: Maintain the `covered` boolean array incrementally.
        # But SA needs to undo moves. 
        # We can use a "stack" of changes.
        # When adding (p, r), we mark new covered spots.
        # When removing (p, r), we unmark them? 
        # Problem: If another prime covers the same spot, unmarking is wrong.
        # So we need a "count" array: count[k] = number of primes covering k.
        # covered[k] is True if count[k] > 0.
        
        # Let's use a count array of small ints.
        # MAX_L = 100000. Array of 100001 ints.
        # Update: O(MAX_L / p).
        # For p=2, 50k updates.
        # For p=1000, 100 updates.
        # Average p is ~500. So ~200 updates per prime.
        # If we have 50 primes, total updates to build state is ~10k.
        # SA step: Remove one prime (~200 ops), Add one prime (~200 ops).
        # Then recalculate L?
        # Recalculating L from scratch is O(L * num_primes) if done naively.
        # But we can maintain L incrementally?
        # If we remove a prime, L might shrink. We'd have to scan back.
        # If we add a prime, L might grow. We scan forward.
        
        # Let's try this:
        # State: 
        #   selected_primes: list of indices
        #   residues: dict prime_idx -> residue
        #   count_arr: list of ints length MAX_L+1
        #   current_L: int
        
        # Move:
        #   1. Remove prime i: 
        #      - For k in range(r_i, MAX_L+1, p_i): count_arr[k] -= 1
        #      - If count_arr[k] becomes 0, it's uncovered.
        #      - Update current_L: 
        #         If current_L was covered by i, and now count_arr[current_L] == 0, 
        #         we need to find new L. Scan backwards? 
        #         Actually, if we remove a prime, L can only decrease or stay same.
        #         We can scan from current_L downwards until we find a gap?
        #         Or scan from 1 upwards? Scanning from 1 is O(L).
        #         Scanning from current_L downwards is O(L) worst case.
        #   2. Add prime j with residue r_j:
        #      - For k in range(r_j, MAX_L+1, p_j): count_arr[k] += 1
        #      - Update current_L:
        #         Scan from current_L + 1 upwards while count_arr[k] > 0.
        
        # This seems feasible.
        
        # Initialization:
        # Start with a greedy solution to get a good starting point.
        
        count_arr = [0] * (MAX_L + 1)
        selected_idx = []
        residues = {}
        current_L = 0
        current_sum_float = 0.0
        current_sum_frac = Fraction(0, 1)
        
        def add_prime(idx, r):
            nonlocal current_L, current_sum_float, current_sum_frac
            p = primes[idx]
            # Update counts
            # k = r, r+p, ...
            # If r == 0, start at p.
            start_k = r if r > 0 else p
            for k in range(start_k, MAX_L + 1, p):
                count_arr[k] += 1
            
            # Update L
            # Scan forward from current_L + 1
            while current_L < MAX_L and count_arr[current_L + 1] > 0:
                current_L += 1
                
            current_sum_float += reciprocals[idx]
            current_sum_frac += reciprocals_frac[idx]
            selected_idx.append(idx)
            residues[idx] = r
            
        def remove_prime(idx):
            nonlocal current_L, current_sum_float, current_sum_frac
            p = primes[idx]
            r = residues[idx]
            start_k = r if r > 0 else p
            for k in range(start_k, MAX_L + 1, p):
                count_arr[k] -= 1
            
            # Update L
            # If current_L is now uncovered, we need to find the new L.
            # It might be significantly smaller.
            # We can scan from 1 upwards? That's O(L).
            # Or scan from current_L downwards? 
            # If count_arr[current_L] > 0, L stays same.
            # If count_arr[current_L] == 0, we need to find the last covered index.
            # Since coverage is contiguous from 1, if current_L is uncovered, 
            # all indices > current_L are also uncovered? No, that's not true.
            # Wait, "contiguous coverage from 1" means 1..L are covered.
            # If L is uncovered, then the new L is the largest k < L such that 1..k are covered.
            # We can just scan from 1 upwards again. 
            # Given L is likely small compared to MAX_L, this is fast.
            # But if L is 10000, scanning 10000 is fast.
            new_L = 0
            while new_L < MAX_L and count_arr[new_L + 1] > 0:
                new_L += 1
            current_L = new_L
            
            current_sum_float -= reciprocals[idx]
            current_sum_frac -= reciprocals_frac[idx]
            selected_idx.remove(idx)
            del residues[idx]

        # Phase 1: Greedy Initialization
        # Use the same greedy logic as the original program but with the count array.
        available = set(range(num_primes))
        
        # We will use the original greedy logic to populate the state.
        # It's fast enough.
        
        # Re-implementing greedy with count array for consistency
        # Actually, let's just run the original greedy logic to get the initial list, 
        # then build the state.
        
        # Let's run a simplified greedy to get initial classes
        init_classes = []
        temp_covered = [False] * (MAX_L + 1)
        temp_L = 0
        temp_sum = Fraction(0, 1)
        temp_available = set(primes)
        
        while True:
            if time.time() > deadline - 10: break
            N = temp_L + 1
            if N > MAX_L: break
            best_p = None
            best_r = None
            best_new_L = temp_L
            best_sum = temp_sum
            
            for p in temp_available:
                r = N % p
                new_sum = temp_sum + Fraction(1, p)
                if new_sum > Fraction(1, 1): continue
                
                # Simulate
                k = N
                while k <= MAX_L:
                    if temp_covered[k] or (k % p == r):
                        k += 1
                    else:
                        break
                new_L = k - 1
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_p = p
                    best_r = r
                    best_sum = new_sum
                    
            if best_p is None: break
            
            temp_available.remove(best_p)
            init_classes.append([best_p, best_r])
            temp_sum = best_sum
            temp_L = best_new_L
            for j in range(best_r, MAX_L + 1, best_p):
                if j >= 1:
                    temp_covered[j] = True
        
        # Build SA state from init_classes
        for p, r in init_classes:
            idx = primes.index(p)
            add_prime(idx, r)
            
        best_L = current_L
        best_state = (list(selected_idx), dict(residues))
        
        # Phase 2: Simulated Annealing
        # Temperature schedule
        T_start = 1.0
        T_end = 0.001
        alpha = 0.999 # cooling rate
        
        # We will perform moves:
        # 1. Change residue of a selected prime.
        # 2. Swap a selected prime with an unselected prime (if budget allows).
        # 3. Add a new prime (if budget allows).
        # 4. Remove a prime (if L doesn't drop too much? Or just accept if T is high).
        
        # To keep it simple and effective, let's focus on:
        # - Residue changes (cheap)
        # - Prime swaps (expensive but explores structure)
        
        T = T_start
        steps = 0
        
        while T > T_end and time.time() < deadline:
            steps += 1
            
            # Choose a move type
            # 70% residue change, 30% prime swap/add
            move_type = random.random()
            
            old_L = current_L
            old_sum_float = current_sum_float
            
            if move_type < 0.7:
                # Change residue of a random selected prime
                if not selected_idx: continue
                idx = random.choice(selected_idx)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                if new_r == old_r: continue
                
                # Apply move
                # Remove old coverage, add new coverage
                # This is effectively remove_prime(idx) and add_prime(idx, new_r)
                # But we can do it in place to save time?
                # Let's just do remove and add.
                remove_prime(idx)
                add_prime(idx, new_r)
                
                new_L = current_L
                delta_L = new_L - old_L
                
                # Acceptance criteria
                # If delta_L > 0, always accept.
                # If delta_L < 0, accept with probability exp(delta_L / T) ?
                # Since L is integer, delta_L is integer.
                # We want to maximize L.
                # Let's use probability exp(-delta_L / T) where delta_L is negative.
                # So exp(delta_L / T).
                
                if delta_L >= 0:
                    accepted = True
                else:
                    # delta_L is negative
                    prob = math.exp(delta_L / T)
                    accepted = random.random() < prob
                    
                if not accepted:
                    # Revert
                    remove_prime(idx)
                    add_prime(idx, old_r)
                    
            else:
                # Swap or Add
                if len(selected_idx) < num_primes:
                    # Try to add a new prime or swap
                    # Let's try swapping a random selected prime with a random unselected one
                    if selected_idx:
                        idx_old = random.choice(selected_idx)
                        unselected = [i for i in range(num_primes) if i not in selected_idx]
                        if not unselected: continue
                        idx_new = random.choice(unselected)
                        
                        # Check budget for new prime
                        # current_sum_float + reciprocals[idx_new] - reciprocals[idx_old] <= 1.0
                        new_sum_float = current_sum_float - reciprocals[idx_old] + reciprocals[idx_new]
                        if new_sum_float > 1.0 + 1e-9:
                            continue
                            
                        # Apply move
                        old_r_old = residues[idx_old]
                        # Remove old
                        remove_prime(idx_old)
                        # Add new with random residue
                        p_new = primes[idx_new]
                        new_r = random.randint(0, p_new - 1)
                        add_prime(idx_new, new_r)
                        
                        new_L = current_L
                        delta_L = new_L - old_L
                        
                        if delta_L >= 0:
                            accepted = True
                        else:
                            prob = math.exp(delta_L / T)
                            accepted = random.random() < prob
                            
                        if not accepted:
                            # Revert
                            remove_prime(idx_new)
                            add_prime(idx_old, old_r_old)
                    else:
                        # No primes selected, just add one
                        idx_new = random.randint(0, num_primes - 1)
                        if current_sum_float + reciprocals[idx_new] > 1.0 + 1e-9:
                            continue
                        p_new = primes[idx_new]
                        new_r = random.randint(0, p_new - 1)
                        add_prime(idx_new, new_r)
                        
                        new_L = current_L
                        delta_L = new_L - old_L
                        if delta_L >= 0:
                            accepted = True
                        else:
                            prob = math.exp(delta_L / T)
                            accepted = random.random() < prob
                            
                        if not accepted:
                            remove_prime(idx_new)
                else:
                    # All primes used? Just do residue change
                    if selected_idx:
                        idx = random.choice(selected_idx)
                        p = primes[idx]
                        old_r = residues[idx]
                        new_r = random.randint(0, p - 1)
                        if new_r == old_r: continue
                        
                        remove_prime(idx)
                        add_prime(idx, new_r)
                        
                        new_L = current_L
                        delta_L = new_L - old_L
                        
                        if delta_L >= 0:
                            accepted = True
                        else:
                            prob = math.exp(delta_L / T)
                            accepted = random.random() < prob
                            
                        if not accepted:
                            remove_prime(idx)
                            add_prime(idx, old_r)
                            
            if current_L > best_L:
                best_L = current_L
                best_state = (list(selected_idx), dict(residues))
                
            T *= alpha
            
            # Check time
            if time.time() > deadline - 5:
                break
                
        # Final verification and cleanup
        # Use best_state
        # Clear current state
        # We can just use best_state to construct the result
        
        # Verify budget with Fraction
        final_classes = []
        final_sum = Fraction(0, 1)
        for idx, r in zip(best_state[0], best_state[1].values()):
            p = primes[idx]
            final_classes.append([p, r])
            final_sum += Fraction(1, p)
            
        if final_sum > Fraction(1, 1):
            # Fallback to greedy result if SA broke budget (shouldn't happen with float check + margin)
            return init_classes
            
        return final_classes
# EVOLVE-BLOCK-END
