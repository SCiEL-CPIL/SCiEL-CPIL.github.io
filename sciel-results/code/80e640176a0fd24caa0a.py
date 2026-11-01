import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0  # Leave safety margin

    space = q ** n

    def w2i(w):
        i = 0
        for v in w:
            i = i * q + v
        return i

    def i2w(i):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = i % q
            i //= q
        return w

    # Precompute neighbor index lists for every word in the space (distance <= radius).
    # For radius 1: the word itself plus all single-position changes.
    if radius >= 1:
        nbrs = []
        for i in range(space):
            w = i2w(i)
            lst = [i]
            for p in range(n):
                ov = w[p]
                for nv in range(q):
                    if nv != ov:
                        w[p] = nv
                        lst.append(w2i(w))
                w[p] = ov
            nbrs.append(lst)
    else:
        nbrs = [[i] for i in range(space)]

    # Bitset representation for fast coverage operations (used for initial check and maybe validation)
    def make_bitset(indices):
        b = 0
        for i in indices:
            b |= (1 << i)
        return b

    cover_bitsets = [make_bitset(nbrs[i]) for i in range(space)]
    full_mask = (1 << space) - 1

    def check_valid_bitset(coverage_bits):
        return coverage_bits == full_mask

    def greedy_deletion_counts(cur, coverage_counts):
        """
        Greedy deletion using coverage counts.
        Returns new list of codewords and updated coverage_counts.
        """
        cur = list(cur)
        # Rebuild coverage_counts from scratch to be safe, or update? 
        # Let's rebuild to ensure consistency if called from different states.
        counts = [0] * space
        for c in cur:
            for w in nbrs[c]:
                counts[w] += 1
        
        improved = True
        while improved:
            improved = False
            # Find a removable codeword
            # A codeword c is removable if all w in nbrs[c] have counts[w] > 1
            removable_idx = -1
            for i, c in enumerate(cur):
                removable = True
                for w in nbrs[c]:
                    if counts[w] <= 1:
                        removable = False
                        break
                if removable:
                    removable_idx = i
                    break
            
            if removable_idx != -1:
                c = cur[removable_idx]
                # Remove c
                for w in nbrs[c]:
                    counts[w] -= 1
                cur.pop(removable_idx)
                improved = True
        
        return cur, counts

    def local_search(cur, counts, rng, max_iters=2000):
        """
        Local search: try to swap a removable codeword with a new one to enable further deletion.
        """
        cur = list(cur)
        counts = list(counts)
        cur_set = set(cur)
        all_idx = list(range(space))
        improved = True
        it = 0

        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 5.0:
                break

            # Try to find a removable codeword to swap
            # We prioritize codewords that are "least useful" (cover few unique words)
            # But for now, just pick a random removable one.
            
            # Identify removable codewords
            removable = []
            for c in cur:
                is_rem = True
                for w in nbrs[c]:
                    if counts[w] <= 1:
                        is_rem = False
                        break
                if is_rem:
                    removable.append(c)
            
            if not removable:
                break # No removable codewords, stuck (locally optimal)

            c = rng.choice(removable)
            
            # Tentatively remove c
            for w in nbrs[c]:
                counts[w] -= 1
            
            # Pick a random replacement x
            # To make it smarter, we could pick x that covers "scarce" words, but random is okay for start
            # We must ensure x is not in cur (except c which is removed)
            # Actually, x can be any word not in cur.
            # Let's pick a random x from all_idx not in cur_set (after removing c)
            
            # Optimization: Instead of shuffling all_idx every time, pick a random index
            # and check if it's in cur_set.
            # Since |cur| is small (~80) and space is 729, collision is rare.
            
            new_cur = [y for y in cur if y != c]
            new_cur_set = set(new_cur)
            
            # Try a few candidates
            found_improvement = False
            for _ in range(10): # Try 10 random candidates
                x = rng.choice(all_idx)
                if x in new_cur_set:
                    continue
                
                # Simulate adding x
                # Check if adding x makes the set valid? 
                # We already removed c. The set might be invalid if c was essential for some word 
                # that x doesn't cover. 
                # Wait, c was removable, so removing c leaves all words covered by >=1 other codeword.
                # So the set `new_cur` is VALID.
                # Adding x keeps it valid.
                # The goal is to see if adding x allows removing ANOTHER codeword.
                
                # Update counts for x
                for w in nbrs[x]:
                    counts[w] += 1
                
                # Run greedy deletion on new_cur + [x]
                # We have counts for new_cur + [x] now.
                # We need to check if any y in new_cur (or x) is removable.
                
                # Actually, we can just run the greedy deletion routine on the new set.
                # But we have the counts.
                
                # Let's check if any codeword in new_cur is now removable.
                # Or if x is removable (unlikely, it's new).
                
                candidate_new_cur = new_cur + [x]
                # We need to verify if we can delete anything.
                # Let's just run a quick check:
                # For each y in candidate_new_cur, is it removable?
                
                can_delete = False
                delete_idx = -1
                for i, y in enumerate(candidate_new_cur):
                    is_rem = True
                    for w in nbrs[y]:
                        if counts[w] <= 1:
                            is_rem = False
                            break
                    if is_rem:
                        can_delete = True
                        delete_idx = i
                        break
                
                if can_delete:
                    # Accept the swap and deletion
                    # Remove y
                    y = candidate_new_cur[delete_idx]
                    for w in nbrs[y]:
                        counts[w] -= 1
                    candidate_new_cur.pop(delete_idx)
                    
                    cur = candidate_new_cur
                    cur_set = set(cur)
                    improved = True
                    found_improvement = True
                    break
                else:
                    # Revert counts for x
                    for w in nbrs[x]:
                        counts[w] -= 1
            
            if not found_improvement:
                # Revert removal of c
                for w in nbrs[c]:
                    counts[w] += 1
                # If we couldn't find any improvement with this c, we might be stuck 
                # or need to try a different c. 
                # To avoid infinite loops on the same c, we could shuffle cur or just break?
                # If no removable c leads to improvement, we are locally optimal.
                # Let's break to avoid wasting time.
                break

        return cur, counts

    best = None
    best_len = 10 ** 9

    # Product construction start: tetracode (length 4, size 9) x full space (length n-4).
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    base = [h + list(t) for h in tetracode for t in tails]
    base_idx = [w2i(c) for c in base]

    # Initial Greedy Deletion
    # Use bitset for initial check? No, use counts for consistency with local search.
    # But let's use the fast bitset check to see if base is valid.
    base_coverage = 0
    for i in base_idx:
        base_coverage |= cover_bitsets[i]
    
    if check_valid_bitset(base_coverage):
        # Run greedy deletion with counts
        counts = [0] * space
        for c in base_idx:
            for w in nbrs[c]:
                counts[w] += 1
        
        # Greedy deletion
        cur = list(base_idx)
        improved = True
        while improved:
            improved = False
            for i, c in enumerate(cur):
                removable = True
                for w in nbrs[c]:
                    if counts[w] <= 1:
                        removable = False
                        break
                if removable:
                    for w in nbrs[c]:
                        counts[w] -= 1
                    cur.pop(i)
                    improved = True
                    break
        
        best = cur
        best_len = len(cur)

    # Multiple randomized restarts
    num_restarts = 20
    for s in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        rng = random.Random(s)
        
        # Start from base, do a randomized greedy deletion?
        # Or just start from best and do local search?
        # Let's try starting from base with different deletion orders.
        
        cur = list(base_idx)
        # Randomize order for greedy deletion
        rng.shuffle(cur)
        
        # Rebuild counts
        counts = [0] * space
        for c in cur:
            for w in nbrs[c]:
                counts[w] += 1
                
        # Greedy deletion in random order
        # We need to check removability in the shuffled order
        # This is tricky with the simple loop above. 
        # Let's stick to the standard greedy (first removable found) but maybe randomize the set?
        # Actually, the "first removable" depends on the order.
        
        # Let's just use the standard greedy on the shuffled list?
        # The `greedy_deletion_counts` function above uses a fixed order (0 to end).
        # If we shuffle `cur`, the order is random.
        
        res, res_counts = greedy_deletion_counts(cur, counts)
        
        if len(res) < best_len:
            best = res
            best_len = len(res)
            
        # Local Search
        res2, res2_counts = local_search(res, res_counts, rng, max_iters=500)
        
        # Verify validity of res2
        # Quick check: is coverage full?
        # We can trust the counts if the logic is right, but let's be safe.
        # Rebuild coverage bitset for res2
        res2_cov = 0
        for c in res2:
            res2_cov |= cover_bitsets[c]
            
        if check_valid_bitset(res2_cov) and len(res2) < best_len:
            best = res2
            best_len = len(res2)

    if best is None:
        return [i2w(i) for i in base_idx]
    return [i2w(i) for i in best]
