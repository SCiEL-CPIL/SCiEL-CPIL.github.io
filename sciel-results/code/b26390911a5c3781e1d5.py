"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    seed_elements = [1, 2, 5, 11]
    max_val = 284
    max_sum = 2 * max_val
    seed_set = set(seed_elements)
    
    def compute_sums_bitset(a):
        """Compute bitset of all pairwise sums a[i]+a[j] for i<=j."""
        bits = 0
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                bits |= 1 << (a[i] + a[j])
        return bits

    def can_add(x, a, sums_bits):
        """Check if x can be added to a without creating duplicate sums."""
        # Check if 2x conflicts
        if (sums_bits >> (2 * x)) & 1:
            return False
        # Check if x+y conflicts for any y in a
        for y in a:
            s = x + y
            if (sums_bits >> s) & 1:
                return False
        return True

    def greedy_extend_bitset(a, sums_bits, candidates_order=None):
        """Greedy extension using bitsets. Returns (new_a, new_sums_bits)."""
        a = list(a)
        sums_bits = sums_bits
        if candidates_order is None:
            candidates_order = list(range(1, max_val + 1))
        
        # Filter out elements already in a
        a_set = set(a)
        candidates = [x for x in candidates_order if x not in a_set]
        
        for x in candidates:
            if can_add(x, a, sums_bits):
                a.append(x)
                # Update sums_bits
                sums_bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        sums_bits |= 1 << (x + y)
                
        a.sort()
        return a, sums_bits

    def local_search_bitset(a, sums_bits, max_iters=1000, deadline=None):
        """Local search: remove one, add one, then greedy extend."""
        a = list(a)
        sums_bits = sums_bits
        best_a = list(a)
        best_bits = sums_bits
        best_size = len(a)
        
        # Pre-compute list of non-seed elements for removal
        # We will recompute removable list each iteration to be safe
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            if len(a) <= len(seed_elements):
                break
            
            # Pick a random non-seed element to remove
            removable = [x for x in a if x not in seed_set]
            if not removable:
                break
                
            x_remove = random.choice(removable)
            
            # Remove x_remove from a
            a.remove(x_remove)
            
            # Recompute sums (fast for small n)
            new_sums_bits = compute_sums_bitset(a)
            
            # Try to add a random candidate
            # Generate candidates not in a
            a_set = set(a)
            candidates = [x for x in range(1, max_val + 1) if x not in a_set]
            
            improved = False
            # Try a few random additions
            # We can also try a greedy addition here
            # Let's try 5 random candidates
            for _ in range(5):
                if not candidates:
                    break
                x_add = random.choice(candidates)
                if can_add(x_add, a, new_sums_bits):
                    a.append(x_add)
                    a.sort()
                    # Update sums
                    new_sums_bits |= 1 << (2 * x_add)
                    for y in a:
                        if y != x_add:
                            new_sums_bits |= 1 << (x_add + y)
                    
                    # Greedy extend
                    a, new_sums_bits = greedy_extend_bitset(a, new_sums_bits)
                    
                    if len(a) > best_size:
                        best_a = list(a)
                        best_bits = new_sums_bits
                        best_size = len(a)
                    improved = True
                    break
            
            if not improved:
                # Revert
                a.append(x_remove)
                a.sort()
                new_sums_bits = compute_sums_bitset(a)
                
        return best_a, best_bits

    # Initial greedy
    best_a = list(seed_elements)
    best_bits = compute_sums_bitset(best_a)
    best_a, best_bits = greedy_extend_bitset(best_a, best_bits)
    
    random.seed(42)
    
    # Multiple restarts with local search
    # Dynamically determine number of restarts based on time
    start_time = time.time()
    restarts_done = 0
    max_restarts_allowed = 500
    
    while time.time() < deadline - 10 and restarts_done < max_restarts_allowed:
        restarts_done += 1
        
        # Randomized greedy start
        a = list(seed_elements)
        bits = compute_sums_bitset(a)
        
        # Shuffle candidates for greedy
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        a, bits = greedy_extend_bitset(a, bits, candidates)
        
        # Local search
        # Increase iterations if time permits
        iters = 500 if time.time() < deadline - 30 else 100
        a_ls, bits_ls = local_search_bitset(a, bits, max_iters=iters, deadline=deadline - 5)
        
        if len(a_ls) > len(best_a):
            best_a = a_ls
            best_bits = bits_ls
            
        # If we found a very large set (e.g., 20), we can stop early?
        # But we want to be sure it's valid.
        if len(best_a) >= 20:
             # Try a few more restarts to see if we can get 21? 
             # Prompt says max is 20. So 20 is the target.
             # We can break if we are confident.
             # But let's run a few more to ensure stability.
             if restarts_done > 50:
                 break

    # Final verification
    final_a = sorted(best_a)
    assert all(x in final_a for x in seed_elements), "Missing seed elements"
    assert all(1 <= x <= 284 for x in final_a), "Out of range"
    assert len(set(final_a)) == len(final_a), "Duplicates"
    
    # Verify Sidon property
    sums = set()
    n = len(final_a)
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = final_a[i] + final_a[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    assert valid, "Invalid Sidon set"
    
    return final_a
# EVOLVE-BLOCK-END
