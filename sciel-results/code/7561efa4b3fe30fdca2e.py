import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Generate candidate pool: 2,3-smooth numbers (Hamming numbers) up to 10000
    # These are numbers of the form 2^i * 3^j
    pool = set()
    i = 0
    while (1 << i) <= 10000:
        j = 0
        while (1 << i) * (3 ** j) <= 10000:
            pool.add((1 << i) * (3 ** j))
            j += 1
        i += 1
    
    # Also add some small primes and their multiples to diversify?
    # The prompt suggests smooth numbers are good for product compression.
    # Let's stick to 2,3-smooth for the main pool, maybe add 5-smooth if needed.
    # Actually, 2,3,5-smooth might be better. Let's generate 2,3,5-smooth.
    pool = set()
    def gen_smooth(max_val, primes):
        curr = [1]
        while True:
            min_val = min(curr)
            if min_val > max_val:
                break
            pool.add(min_val)
            # Remove min_val and add its multiples by primes
            # This is a standard generation, but let's just brute force 2^i 3^j 5^k
            pass
    
    # Simpler: Brute force 2,3,5-smooth
    pool = set()
    for i in range(14): # 2^13 > 10000
        for j in range(9): # 3^8 > 10000
            for k in range(6): # 5^5 > 10000
                val = (2 ** i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    pool.add(val)
    
    # Add 1 explicitly if not there (it is: 2^0 3^0 5^0)
    # Sort pool for deterministic initial pick
    pool_list = sorted(list(pool))
    
    # Initial set: Smallest 20 smooth numbers
    # This minimizes product set size significantly.
    A = list(pool_list[:20])
    
    # State variables for incremental update
    # sum_counts: dict {sum_val: count}
    # prod_counts: dict {prod_val: count}
    
    def compute_initial_counts(A):
        sc = defaultdict(int)
        pc = defaultdict(int)
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                p = A[i] * A[j]
                sc[s] += 1
                pc[p] += 1
        return sc, pc

    sum_counts, prod_counts = compute_initial_counts(A)
    
    current_sum_size = len(sum_counts)
    current_prod_size = len(prod_counts)
    current_score = max(current_sum_size, current_prod_size)
    
    best_A = list(A)
    best_score = current_score
    
    # Pre-compute indices for faster access? 
    # We will iterate over indices of A and candidates in pool.
    
    # Helper to evaluate a swap (i, new_val)
    # Returns (new_score, new_sum_size, new_prod_size)
    # Note: We must revert changes if we don't accept the move.
    # To avoid complex revert logic, we can just compute the delta.
    
    def eval_swap(i, new_val):
        old_val = A[i]
        if old_val == new_val:
            return current_score
        
        # Delta in sum size
        ds = 0
        # Remove old_val contributions
        # Pair (old_val, old_val) -> 2*old_val
        # If i is the index, the diagonal term is 2*old_val.
        # The off-diagonal terms are old_val + A[k] for k != i.
        
        # We need to be careful with counts.
        # Let's just simulate the counts change.
        
        # 1. Remove old_val
        # Diagonal: 2*old_val
        cnt_s_diag = sum_counts.get(2*old_val, 0)
        # Off-diagonal: old_val + A[k] for k in [0..n-1], k != i
        # Note: In the set A, the element at index i is old_val.
        
        # To make this efficient and safe, we will calculate the new sizes 
        # by applying changes to temporary copies of the counts? 
        # No, that's O(M). 
        # We need O(n).
        
        # Let's track the number of "unique" sums and products.
        # A sum value v is "present" if count[v] > 0.
        # When we decrement count[v] to 0, size decreases by 1.
        # When we increment count[v] from 0 to 1, size increases by 1.
        
        new_sum_size = current_sum_size
        new_prod_size = current_prod_size
        
        # Helper to handle decrement
        def dec_sum(val):
            nonlocal new_sum_size
            c = sum_counts.get(val, 0)
            if c == 1:
                new_sum_size -= 1
            # else no change in size
        
        def dec_prod(val):
            nonlocal new_prod_size
            c = prod_counts.get(val, 0)
            if c == 1:
                new_prod_size -= 1
        
        def inc_sum(val):
            nonlocal new_sum_size
            c = sum_counts.get(val, 0)
            if c == 0:
                new_sum_size += 1
        
        def inc_prod(val):
            nonlocal new_prod_size
            c = prod_counts.get(val, 0)
            if c == 0:
                new_prod_size += 1
        
        # Remove old_val
        # Diagonal
        dec_sum(2 * old_val)
        dec_prod(old_val * old_val)
        
        # Off-diagonal with other elements in A
        for k in range(len(A)):
            if k == i:
                continue
            other = A[k]
            dec_sum(old_val + other)
            dec_prod(old_val * other)
            
        # Add new_val
        # Diagonal
        inc_sum(2 * new_val)
        inc_prod(new_val * new_val)
        
        # Off-diagonal with other elements in A
        for k in range(len(A)):
            if k == i:
                continue
            other = A[k]
            inc_sum(new_val + other)
            inc_prod(new_val * other)
            
        return max(new_sum_size, new_prod_size)

    # Search Loop
    # We use a simple hill climbing with random restarts/perturbations
    # Since n=20 and pool is ~1000, one pass is 20k evals.
    # Each eval is ~40 dict lookups. 20k * 40 = 800k ops.
    # In Python, this might take ~0.1s. So we can do ~500 passes in 50s.
    
    # To avoid getting stuck, we allow "worse" moves occasionally (Simulated Annealing)
    # Or just do greedy with random restarts if stuck.
    # Given the small size, greedy with random jumps is often sufficient.
    
    # Let's do:
    # 1. Greedy local search until no improvement.
    # 2. Perturb (replace 1-2 random elements with random pool elements).
    # 3. Repeat.
    
    import math
    
    # Optimize eval_swap by avoiding function call overhead in inner loop?
    # The inner loop runs 20 times. Function calls to dec/inc are 40 per eval.
    # 20 * 40 = 800 calls per eval.
    # 20k evals -> 16 million function calls. This might be slow in Python.
    
    # Let's inline the logic in eval_swap for speed.
    
    def eval_swap_fast(i, new_val):
        old_val = A[i]
        if old_val == new_val:
            return current_score
            
        nss = current_sum_size
        nps = current_prod_size
        
        # Remove old_val
        # Diagonal
        v = 2 * old_val
        if sum_counts[v] == 1: nss -= 1
        v = old_val * old_val
        if prod_counts[v] == 1: nps -= 1
        
        # Off-diagonal
        for k in range(len(A)):
            if k == i: continue
            other = A[k]
            v = old_val + other
            if sum_counts[v] == 1: nss -= 1
            v = old_val * other
            if prod_counts[v] == 1: nps -= 1
            
        # Add new_val
        # Diagonal
        v = 2 * new_val
        if sum_counts.get(v, 0) == 0: nss += 1
        v = new_val * new_val
        if prod_counts.get(v, 0) == 0: nps += 1
        
        # Off-diagonal
        for k in range(len(A)):
            if k == i: continue
            other = A[k]
            v = new_val + other
            if sum_counts.get(v, 0) == 0: nss += 1
            v = new_val * other
            if prod_counts.get(v, 0) == 0: nps += 1
            
        return max(nss, nps)

    # We need to actually apply the move to A and counts if we accept it.
    
    def apply_swap(i, new_val):
        nonlocal sum_counts, prod_counts, current_sum_size, current_prod_size, current_score
        old_val = A[i]
        if old_val == new_val:
            return
        
        # Update A
        A[i] = new_val
        
        # Update counts
        # Remove old
        v = 2 * old_val
        sum_counts[v] -= 1
        if sum_counts[v] == 0:
            del sum_counts[v]
            current_sum_size -= 1
        v = old_val * old_val
        prod_counts[v] -= 1
        if prod_counts[v] == 0:
            del prod_counts[v]
            current_prod_size -= 1
            
        for k in range(len(A)):
            if k == i: continue
            other = A[k]
            v = old_val + other
            sum_counts[v] -= 1
            if sum_counts[v] == 0:
                del sum_counts[v]
                current_sum_size -= 1
            v = old_val * other
            prod_counts[v] -= 1
            if prod_counts[v] == 0:
                del prod_counts[v]
                current_prod_size -= 1
                
        # Add new
        v = 2 * new_val
        if v in sum_counts:
            sum_counts[v] += 1
        else:
            sum_counts[v] = 1
            current_sum_size += 1
        v = new_val * new_val
        if v in prod_counts:
            prod_counts[v] += 1
        else:
            prod_counts[v] = 1
            current_prod_size += 1
            
        for k in range(len(A)):
            if k == i: continue
            other = A[k]
            v = new_val + other
            if v in sum_counts:
                sum_counts[v] += 1
            else:
                sum_counts[v] = 1
                current_sum_size += 1
            v = new_val * other
            if v in prod_counts:
                prod_counts[v] += 1
            else:
                prod_counts[v] = 1
                current_prod_size += 1
                
        current_score = max(current_sum_size, current_prod_size)

    # Main Search
    # To ensure we don't pick the same candidates repeatedly in a bad way,
    # we can shuffle the pool or just iterate.
    
    # We will run for the duration.
    # Since eval is fast, we can do many iterations.
    
    # One optimization: Pre-compute a list of (index, candidate) pairs?
    # No, just nested loops.
    
    # To avoid O(n) loop in eval_swap_fast being too slow, 
    # note that n=20 is small.
    
    # Let's run until deadline
    while time.time() < deadline:
        improved = False
        # Shuffle pool to avoid bias? Or just iterate.
        # Iterating is deterministic. Shuffling helps escape symmetry.
        # But we have random perturbations later.
        
        for i in range(20):
            if time.time() > deadline: break
            # Try all candidates in pool not in A
            # To speed up, we can just check all in pool_list
            # 20 * len(pool_list) ~ 20 * 1000 = 20k checks per pass.
            
            # Optimization: If we find a better move, apply it and restart the i loop?
            # Or just find the best move in the whole neighborhood?
            # Best move search: O(N * M).
            
            best_i = -1
            best_new_val = -1
            best_score_found = current_score
            
            for new_val in pool_list:
                if new_val in A: continue # Skip if already in set
                # Check if new_val is valid (distinct)
                # The 'in A' check is O(n). 
                # To make this O(1), we can maintain a set of A.
                
                s = eval_swap_fast(i, new_val)
                if s < best_score_found:
                    best_score_found = s
                    best_i = i
                    best_new_val = new_val
            
            if best_i != -1:
                apply_swap(best_i, best_new_val)
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
                improved = True
                # Restart inner loop to re-evaluate with new A
                # Break and let the outer while loop handle it?
                # Or just continue to next i?
                # Restarting is safer for hill climbing.
                break # Break i loop, restart full neighborhood search
        
        if not improved:
            # Perturb
            # Replace 1 or 2 random elements with random pool elements
            num_perturb = random.randint(1, 2)
            for _ in range(num_perturb):
                i = random.randint(0, 19)
                # Pick a random candidate not in A
                # Simple rejection sampling
                while True:
                    cand = random.choice(pool_list)
                    if cand not in A:
                        break
                apply_swap(i, cand)
            
            # If we got stuck and perturbed, we might have worsened the score.
            # That's okay, we will try to climb back.

    # Return the best set found
    return best_A
# EVOLVE-BLOCK-END
