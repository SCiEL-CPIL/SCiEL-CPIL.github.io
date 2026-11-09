import time
import random
import bisect
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (90s kill)
    
    # The problem: Find longest subset S of {2, ..., 128} such that 
    # all products of consecutive blocks in sorted(S) are distinct.
    
    # Key insight for speed: We can represent each number by its prime factorization.
    # The product of a block is the sum of exponent vectors.
    # Two blocks are equal iff their exponent vectors are equal.
    # We can use prefix sums of exponent vectors to check equality in O(1) (after O(N^2) setup per check).
    # But actually, we can just use the actual product value. Since max product is 128^127 which is huge,
    # we can't store it directly in a fixed-size integer efficiently in C, but Python handles big ints.
    # However, the number of blocks is N(N+1)/2. For N=60, that's ~1800 blocks.
    # Checking if a new product already exists is O(1) dictionary lookup.
    # The cost is computing the products.
    
    # Optimization: When checking if adding a new element x at position p is valid,
    # we only need to check the new products that involve x.
    # These are x, x*seq[p-1], x*seq[p-1]*seq[p-2], ... and x*seq[p], x*seq[p]*seq[p+1], ...
    # Wait, x is inserted, so it splits the sequence.
    # The new blocks are:
    # 1. [x]
    # 2. [seq[i], x] for i < p
    # 3. [x, seq[j]] for j >= p
    # 4. [seq[i], ..., seq[i+k], x, seq[j], ...] - NO, the blocks are consecutive in the NEW sequence.
    # Let's list them carefully.
    # Original: A = [a_0, ..., a_{p-1}], B = [a_p, ..., a_{n-1}]
    # New: A + [x] + B
    # New blocks:
    # - x
    # - a_i * x (for i in A, ending at x) -> a_{p-1}*x, a_{p-2}*a_{p-1}*x, ...
    # - x * a_j (for j in B, starting at x) -> x*a_p, x*a_p*a_{p+1}, ...
    # - a_i * ... * a_{p-1} * x * a_p * ... * a_j
    # So we need to check:
    # 1. x is not in prods
    # 2. Suffix products of A * x are not in prods
    # 3. x * Prefix products of B are not in prods
    # 4. Suffix products of A * x * Prefix products of B are not in prods
    
    # This is still O(N) new products to check, but we don't recompute everything.
    # However, maintaining the set of prods incrementally is tricky when inserting in the middle.
    # Removing is easy (subtract products involving removed element).
    # Inserting in the middle requires adding all new products that span the insertion point.
    
    # Given the small N (<= 127), a full re-check O(N^2) is ~ 8000 multiplications.
    # In Python, 1000 full checks/sec is feasible (8M ops/sec). 50s -> 400M ops.
    # This suggests we can do many local search moves.
    
    def get_collisions(seq):
        """
        Returns a set of indices involved in any collision.
        """
        n = len(seq)
        if n == 0:
            return set()
        
        prods = {}
        bad = set()
        
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    pi, pj = prods[p]
                    # Collision between [pi..pj] and [i..j]
                    for k in range(pi, pj+1):
                        bad.add(k)
                    for k in range(i, j+1):
                        bad.add(k)
                else:
                    prods[p] = (i, j)
                    
        return bad

    def is_valid(seq):
        return len(get_collisions(seq)) == 0

    # Greedy Initialization
    def greedy_init():
        current_seq = []
        prods = set()
        
        for x in range(2, 129):
            # Check if adding x to the end is valid
            # New products: x, x*last, x*last*second_last...
            valid = True
            if x in prods:
                valid = False
            else:
                prod = x
                for i in range(len(current_seq) - 1, -1, -1):
                    prod *= current_seq[i]
                    if prod in prods:
                        valid = False
                        break
            
            if valid:
                current_seq.append(x)
                # Update prods
                prods.add(x)
                prod = x
                for i in range(len(current_seq) - 2, -1, -1):
                    prod *= current_seq[i]
                    prods.add(prod)
                    
        return current_seq

    best_seq = greedy_init()
    best_len = len(best_seq)
    
    # Local Search
    # Strategy:
    # 1. Start from a valid sequence.
    # 2. Try to insert a missing element.
    #    - Pick a missing element and a position.
    #    - Check validity.
    #    - If valid, accept.
    #    - If not, reject.
    # 3. If stuck, try to swap: Remove one, insert another.
    #    - Or just remove one to escape local optimum, then re-insert.
    
    # To speed up, we can pre-compute the "conflict" of inserting x at pos p
    # without doing a full check? 
    # The new products are those that include x.
    # If any of these are in the existing prods, it's a collision.
    # Also, two new products could collide with each other?
    # E.g. [a, x] and [x, b] -> a*x and x*b. If a*x == x*b, then a==b. But a!=b since distinct.
    # What about [x] and [x]? Impossible, only one x.
    # What about suffix of A * x and x * prefix of B?
    # a_i * x and x * a_j. a_i == a_j? Possible if i=j? No, i in A, j in B.
    # But a_i * x == x * a_k => a_i == a_k.
    # So we just need to check if any NEW product is in the OLD prods.
    # AND if any two NEW products are equal.
    
    def try_insert(seq, x):
        """
        Try to insert x into seq. Returns (new_seq, success) or (None, False).
        seq is sorted.
        """
        n = len(seq)
        pos = bisect.bisect_left(seq, x)
        
        # Calculate all new products
        new_prods = set()
        
        # 1. x
        new_prods.add(x)
        
        # 2. Suffixes of seq[:pos] * x
        p = 1
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p * x)
            
        # 3. x * Prefixes of seq[pos:]
        p = 1
        for j in range(pos, n):
            p *= seq[j]
            new_prods.add(x * p)
            
        # 4. Suffixes of seq[:pos] * x * Prefixes of seq[pos:]
        # This is the most expensive part O(pos * (n-pos))
        # Let's optimize: 
        # Let L be the list of suffix products of seq[:pos] (ending at x)
        # Let R be the list of prefix products of seq[pos:] (starting at x)
        # We need to check if any l * r is in old_prods OR if any two l1*r1 == l2*r2.
        
        # Actually, let's just generate the set of new products and check against old prods.
        # If there's a collision among new products, it's a problem too.
        
        # Let's compute the full set of new products first.
        # Then check if any is in old prods.
        # Then check if len(new_prods) == number_of_new_blocks.
        
        # Number of new blocks involving x:
        # 1 (x)
        # pos (suffix * x)
        # n - pos (x * prefix)
        # pos * (n - pos) (suffix * x * prefix)
        # Total: 1 + pos + n - pos + pos*(n-pos) = n + 1 + pos*(n-pos)
        
        expected_count = n + 1 + pos * (n - pos)
        
        # We already have 1 + pos + (n-pos) in new_prods.
        # Now add the combined ones.
        
        # Pre-compute suffix products of left part
        left_suffixes = [1] * (pos + 1) # left_suffixes[k] is product of seq[pos-k-1 ... pos-1]
        # Actually, let's just use a list
        left_sufs = []
        p = 1
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            left_sufs.append(p)
            
        right_pufs = []
        p = 1
        for j in range(pos, n):
            p *= seq[j]
            right_pufs.append(p)
            
        # Now combine
        for l in left_sufs:
            for r in right_pufs:
                new_prods.add(l * x * r)
                
        # Check for internal collisions in new_prods
        # We can't easily check if len(new_prods) == expected_count because 
        # some new prods might have been added multiple times? No, we use a set.
        # But if two different blocks produce the same product, the set size will be smaller.
        
        # Wait, if l1*x*r1 == l2*x*r2, then l1*r1 == l2*r2.
        # So we can just check if len(new_prods) < expected_count.
        
        if len(new_prods) < expected_count:
            return None, False
            
        # Now check against old prods.
        # We need the set of old prods.
        # To avoid recomputing old prods every time, we should maintain it.
        # But inserting/removing in the middle makes maintenance hard.
        # Given the speedup potential, maybe we should just recompute old prods 
        # only when we accept a move?
        # No, we need it for the check.
        
        # Let's restructure: 
        # Maintain `seq` and `prods` (set of all products).
        # When checking a candidate, we compute the new products and check against `prods`.
        # But `prods` is for the CURRENT seq.
        # If we accept, we update `seq` and `prods`.
        # How to update `prods`?
        # We add the new products.
        # We don't need to remove old ones because the old blocks still exist 
        # UNLESS the insertion splits a block? 
        # No, insertion doesn't remove any old block. It only adds new blocks.
        # Wait, does it?
        # Old: [a, b]. Prods: a, b, ab.
        # New: [a, x, b]. Prods: a, b, x, ax, xb, axb.
        # Old prods are still present? a, b are present. ab is GONE.
        # So insertion REMOVES some old products (those that spanned the gap) and ADDS new ones.
        # This makes incremental update of `prods` complex.
        
        # Therefore, full re-computation is safer.
        # Let's stick to full re-computation for simplicity and correctness.
        # O(N^2) per check.
        
        candidate = seq[:pos] + [x] + seq[pos:]
        return candidate, is_valid(candidate)

    # Local Search Loop
    current_seq = best_seq[:]
    current_len = best_len
    removed = [x for x in range(2, 129) if x not in current_seq]
    removed.sort()
    
    global_best = current_seq[:]
    global_best_len = current_len
    
    random.seed(42)
    
    # We will use a simulated annealing or random walk approach.
    # Since we must return a valid sequence, we only accept valid moves.
    # This is a hill-climbing algorithm.
    
    stuck_count = 0
    
    while time.time() < deadline - 5.0:
        # Try to insert a random element from removed
        if removed:
            # Choose a random element
            idx = random.randrange(len(removed))
            x = removed[idx]
            
            # Try to insert
            new_seq, success = try_insert(current_seq, x)
            
            if success:
                current_seq = new_seq
                current_len += 1
                removed.pop(idx)
                stuck_count = 0
                
                if current_len > global_best_len:
                    global_best_len = current_len
                    global_best = current_seq[:]
            else:
                stuck_count += 1
                
                # If stuck for a while, try a random swap or removal to escape
                if stuck_count > 20:
                    # Remove a random element from current_seq
                    # and put it back in removed
                    if current_len > 10: # don't remove too many
                        r_idx = random.randrange(len(current_seq))
                        r_val = current_seq[r_idx]
                        current_seq.pop(r_idx)
                        current_len -= 1
                        
                        # Insert r_val into removed in sorted order
                        bisect.insort(removed, r_val)
                        
                        # Reset stuck counter
                        stuck_count = 0
                        
        else:
            # All elements inserted? (Impossible since max length is limited by collisions)
            # Or we ran out of candidates.
            # Try a restart?
            # For now, break.
            break
            
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final Polish: Try to insert any remaining elements greedily
    if is_valid(global_best):
        missing = [x for x in range(2, 129) if x not in global_best]
        for x in missing:
            if time.time() > deadline - 1.0: break
            pos = bisect.bisect_left(global_best, x)
            candidate = global_best[:pos] + [x] + global_best[pos:]
            if is_valid(candidate):
                global_best = candidate
    
    return global_best

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
