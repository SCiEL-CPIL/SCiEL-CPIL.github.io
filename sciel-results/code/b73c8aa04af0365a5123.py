import numpy as np
from itertools import combinations
import time

def construct():
    """
    Find the smallest n (5 <= n <= 90) for which a tournament exists where every
    set of 4 vertices has a common dominator.
    """
    start_time = time.time()
    time_limit = 110  # Safety margin before 120s hard limit

    # List of candidate n values to test, in ascending order (smaller is better)
    # We start from the lower bound (31) up to the known upper bound (83)
    # We prioritize primes p = 4k+3 as Paley tournaments are a strong candidate family.
    # Other n values can be filled with random or heuristic searches if time permits,
    # but Paley is the most likely to yield a small valid n quickly.
    
    # Known smallest Paley candidate is 67.
    # Let's check all n from 31 to 83.
    # However, brute forcing random tournaments for n < 67 is extremely hard.
    # We will focus on Paley tournaments for primes p = 4k+3.
    # Primes p = 4k+3 in range [31, 83]: 31, 43, 47, 59, 67, 71, 79, 83.
    
    # We'll also include non-prime n if we have time, but Paley is the primary strategy.
    # Let's start with the Paley candidates in ascending order.
    paley_primes = [31, 43, 47, 59, 67, 71, 79, 83]
    
    # Add other n values if we want to be thorough, but Paley is the best bet.
    # If no Paley works, we might try a simple random search for small n,
    # but the probability is low.
    
    # Actually, let's just test the Paley primes first.
    # If one works, return it.
    
    # Pre-calculate squares for each prime? Or calculate on the fly?
    # On the fly is fine.
    
    def is_paley_valid(p):
        """Check if Paley tournament on p vertices is valid."""
        if p < 5:
            return False
            
        # 1. Generate orientation bits
        # Bit (i, j) for i < j: 0 if i->j, 1 if j->i
        # i->j iff (j-i) is a square mod p
        
        # To speed up validation, let's build the adjacency matrix or bitsets.
        # n <= 83, so bitsets (Python integers) are very efficient.
        
        # out[i] is a bitmask where bit j is 1 if i -> j
        out = [0] * p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    # i -> j
                    out[i] |= (1 << j)
        
        # 2. Validate: Every 4-subset {a,b,c,d} must have a dominator.
        # Dominator v is such that v->a, v->b, v->c, v->d.
        # This means v is in out_inv[a] & out_inv[b] & out_inv[c] & out_inv[d]?
        # No, v->a means a is in out[v].
        # So we need v such that (out[v] & (1<<a)) and (out[v] & (1<<b)) ...
        # Equivalently, v is in the intersection of the "predecessors" of a, b, c, d.
        
        # Let's use the out bits.
        # For a subset S={a,b,c,d}, a vertex v dominates S if for all u in S, v->u.
        # This is equivalent to: (out[v] & mask_S) == mask_S.
        
        # Alternative: For each v, let D(v) be the set of vertices v dominates.
        # We need to check if the family of sets D(v) covers all 4-subsets.
        
        # Efficient check:
        # For each 4-subset, check if there exists v such that v dominates all 4.
        # Number of 4-subsets for p=83 is ~16M.
        # For each subset, checking all v is O(p). Total O(p^5) which is too slow for p=83 in Python loops.
        
        # Better approach:
        # For each v, let M_v be the bitmask of vertices dominated by v.
        # We need to know if any M_v covers the subset.
        
        # Even better:
        # Iterate through all 4-subsets.
        # For a subset {a,b,c,d}, the dominators are the intersection of:
        # {v | v->a}, {v | v->b}, {v | v->c}, {v | v->d}.
        
        # Let in_mask[u] be the bitmask of vertices that point TO u.
        # Then dominators of {a,b,c,d} = in_mask[a] & in_mask[b] & in_mask[c] & in_mask[d].
        # If this intersection is non-zero, a dominator exists.
        
        in_mask = [0] * p
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if (out[i] & (1 << j)):
                    # i -> j, so i is a dominator of j
                    in_mask[j] |= (1 << i)
        
        # Now check all 4-subsets
        # To speed this up, we can iterate using combinations
        # But 16M iterations in Python might take ~10-20s.
        # We can try to optimize by breaking early if invalid?
        # Or use numpy? Numpy might not be faster for bit operations on small ints.
        
        # Let's try a hybrid:
        # If p is small, brute force.
        # If p is large, we might need a smarter check or accept the time.
        
        # Optimization:
        # We can precompute in_masks.
        
        # Let's just run the loop. 16M iterations * simple bit ops.
        # Python loop overhead is ~100ns-1us. 16M * 1us = 16s.
        # This is close to the limit.
        
        # We can parallelize? No, standard library only.
        
        # Can we reduce the number of checks?
        # If a 4-subset has no dominator, return False.
        
        # Let's try to vectorize the check?
        # It's hard to vectorize "for each combination of 4 indices".
        
        # Alternative:
        # For each v, M_v is the set of dominated vertices.
        # The condition is that the union of all M_v (as sets of size >= 4) covers all 4-subsets?
        # No, that's not right. We need *a single* v to dominate the specific 4.
        
        # Let's stick to the in_mask intersection.
        # To speed up, we can use a list of combinations and check them.
        
        # One optimization:
        # If p is small, it's fast.
        # For p=83, it might be slow.
        
        # Let's try a different strategy for validation:
        # For each v, let C_v be the collection of 4-subsets of D(v).
        # We need the union of C_v to be the set of all 4-subsets.
        # This is a set cover problem, but we just need to check if the cover is complete.
        
        # Actually, the in_mask method is the most direct.
        
        # Let's try to make the Python loop as fast as possible.
        
        # We'll iterate through combinations of 4 vertices.
        
        # To avoid generator overhead, we can use a recursive generator or itertools.
        # itertools.combinations is C-implemented and fast.
        
        # Let's try it. If it's too slow for 83, we might have to skip 83 or optimize.
        # But 83 is the seed, so we MUST validate it or trust it?
        # The task says "The smallest known has 67 vertices".
        # So if we find 67, we are better than seed.
        # If we don't find anything smaller, we should return 83.
        # Do we need to validate 83?
        # The seed program returns 83. If we return 83, we need it to be valid.
        # It is known to be valid.
        # So we can just return 83 if we don't find a smaller one.
        
        # So we only need to VALIDATE candidates p < 83.
        # The largest candidate we validate is 79 (if 71, 67 etc fail).
        # \binom{79}{4} \approx 13.8M. Still a lot.
        # \binom{71}{4} \approx 9.2M.
        # \binom{67}{4} \approx 6.4M.
        
        # 6.4M iterations is manageable (~5-10s).
        # 13.8M might be ~15-20s.
        
        # We have 110s.
        # We can try 67 first. If valid, return it.
        # If invalid, try 59, 47, 43, 31.
        # If all invalid, return 83.
        
        # We don't need to validate 71, 79, 83 unless we want to check if they are valid too,
        # but since we want to MINIMIZE n, we only care if a smaller one works.
        
        # So the strategy is:
        # 1. Try p in [31, 43, 47, 59, 67].
        # 2. If any is valid, return it.
        # 3. If none valid, return 83 (known valid).
        
        # This saves time by not checking 71, 79, 83.
        
        # Wait, is 67 the smallest known?
        # "The smallest known has 67 vertices".
        # So 67 is valid.
        # If we test 67 and it is valid, we return 67.
        # If we test 59, 47, 43, 31 and they are invalid, we return 67.
        
        # So we MUST check 67.
        
        # Is it possible that 67 is invalid?
        # The prompt says "smallest known has 67". This implies 67 is valid.
        # But we should verify it to be sure, or just trust it?
        # If we trust it, we can just return 67 without checking?
        # No, we should check because the tournament definition might be slightly different or I might make a mistake in construction.
        # But Paley(67) is a standard construction.
        
        # Let's implement the check for p in [31, 43, 47, 59, 67].
        # If time runs out, return 83.
        
        # Implementation details for speed:
        # Use local variables for in_mask.
        # Avoid function calls in the loop.
        
        in_m = in_mask
        
        # Iterate over combinations
        # For each (a,b,c,d), check (in_m[a] & in_m[b] & in_m[c] & in_m[d]) != 0
        
        # We can break early if found invalid.
        
        # Let's write the check inside the loop.
        
        # One more optimization:
        # If p is small, we can unroll or use a different method?
        # No, combinations is fine.
        
        # Let's try to use a list of indices for the in_mask to avoid list lookup?
        # in_m is a list.
        
        # We'll try it.
        
        # Note: 1 << j for j up to 82 fits in a Python int.
        
        # Let's do it.
        
        # We'll wrap the check in a function to keep it clean, but use locals.
        
        # Actually, I'll just write the loop here.
        
        # To speed up the combination generation, we can use a custom iterator?
        # itertools.combinations is C-speed. It's the best.
        
        # Let's proceed.
        
        # But wait, if p=67, 6.4M iterations.
        # If p=59, \binom{59}{4} \approx 450k. Very fast.
        # If p=47, \binom{47}{4} \approx 178k. Very fast.
        # If p=43, \binom{43}{4} \approx 123k. Very fast.
        # If p=31, \binom{31}{4} \approx 31k. Very fast.
        
        # So the bottleneck is only if we have to check 67 and it is valid (we finish) or invalid (we finish).
        # If 67 is valid, we return.
        # If 67 is invalid, we return 83 (after checking 59, 47, 43, 31 which are fast).
        
        # So the worst case is checking 67.
        
        # Let's implement.
        
        # We'll return True if valid, False if invalid.
        
        # We'll do the check.
        
        # To avoid recreating the combination object, we just iterate.
        
        # One potential issue: Python's `combinations` yields tuples.
        # Unpacking is fast.
        
        # Let's go.
        
        for a, b, c, d in combinations(range(p), 4):
            if (in_m[a] & in_m[b] & in_m[c] & in_m[d]) == 0:
                return False
        return True

    # Strategy:
    # Check candidates in ascending order.
    # If valid, return.
    # If all fail, return 83.
    
    # Candidates: 31, 43, 47, 59, 67
    # We check 67 last among these? No, ascending order means 31 first.
    # If 31 is valid, we return 31.
    # If 31 invalid, check 43.
    # ...
    # If 67 valid, return 67.
    # If 67 invalid, return 83.
    
    # This is optimal because we stop as soon as we find a valid one.
    
    candidates = [31, 43, 47, 59, 67]
    
    for p in candidates:
        if time.time() - start_time > time_limit - 10: # Leave 10s buffer
            break
            
        if is_paley_valid(p):
            # Construct the orientation bits for this p
            squares = set()
            for x in range(1, p):
                squares.add((x * x) % p)
            
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (j - i) % p
                    if diff in squares:
                        orientation.append(0) # i -> j
                    else:
                        orientation.append(1) # j -> i
                        
            return {'n': p, 'orientation': orientation}
    
    # Fallback: Return the seed solution (83)
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-START
# (The code above is the implementation of construct())
# EVOLVE-BLOCK-END
