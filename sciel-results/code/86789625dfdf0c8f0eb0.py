import time
import random
import sys

DEADLINE = 85.0  # Safety margin before 90s limit

def construct():
    start_time = time.time()
    n_max = 4096
    target_max_size = 80  # Upper bound from problem statement, though likely much lower
    
    # We will use Simulated Annealing.
    # State: A set S of integers.
    # Move: Remove x from S, Add y to S (where y not in S).
    # Acceptance:
    #   - If new size > old size: Always accept.
    #   - If new size == old size: Accept with probability exp(-delta_energy / T).
    #     Here, "energy" can be a measure of how "tight" the packing is, 
    #     or simply 0 if valid. Since we must maintain validity, 
    #     all moves in the search space are valid sets.
    #     To escape local optima (same size, different composition), 
    #     we need a heuristic. 
    #     Let's use "Minimum Gap" or "Sum of Squares" as a secondary objective 
    #     to encourage exploration? 
    #     Actually, standard SA for maximization problems usually just tries 
    #     to increase size. If stuck at size K, we need to "shake" the set.
    #     
    #     Alternative: Random Restart with Local Search.
    #     But SA can handle the "temperature" to allow temporary "worse" 
    #     states if we define "worse" loosely.
    #     However, the constraint is hard: ALL triples must be unique.
    #     So we can only move to valid states.
    #     If we can only move to valid states, and we want to maximize size,
    #     we are looking for a path from size K to K+1.
    #     If no such path exists from current S, we are stuck.
    #     
    #     Strategy: 
    #     1. Start with a greedy set S.
    #     2. Try to increase size by 1. 
    #        - Iterate through all possible swaps (remove x, add y) that keep size constant.
    #        - Among those, check if any allows a subsequent addition.
    #        - This is expensive.
    #        
    #     Better Strategy: **Randomized Local Search with Random Restarts**
    #     1. Generate a random valid set S (using greedy with random order).
    #     2. Try to extend S by adding one element.
    #     3. If failed, try a swap (remove x, add y) to create a "different" 
    #        valid set of the same size, then try to extend again.
    #     4. Repeat until time limit.
    
    def _check_valid_addition(S, used_sums, pair_sums, x):
        """Check if x can be added to S."""
        if 3*x in used_sums:
            return False
        for y in S:
            if (2*x + y) in used_sums or (x + 2*y) in used_sums:
                return False
        for p in pair_sums:
            if (x + p) in used_sums:
                return False
        return True

    def _add_to_state(S, used_sums, pair_sums, x):
        """Update state after adding x. Assumes valid."""
        new_used = set(used_sums)
        new_pairs = set(pair_sums)
        
        new_used.add(3*x)
        for y in S:
            new_used.add(2*x + y)
            new_used.add(x + 2*y)
            new_pairs.add(x + y)
        for p in pair_sums:
            new_used.add(x + p)
        new_pairs.add(2*x)
        
        return S + [x], new_used, new_pairs

    def _remove_from_state(S, used_sums, pair_sums, x):
        """Update state after removing x. Assumes x in S."""
        # This is tricky because used_sums is a set of sums.
        # We need to know which sums were *only* formed by x.
        # This requires counting or tracking origins.
        # For simplicity in local search, it's often faster to just 
        # reconstruct the state from the new set S' if the set is small (< 50).
        # Given n=4096 and target size ~30-50, reconstruction is O(K^2) which is fast.
        
        S_new = [y for y in S if y != x]
        used_new = set()
        pairs_new = set()
        
        # Rebuild from scratch
        # Optimization: We can do this incrementally if we track counts, 
        # but for K < 100, rebuilding is simpler and less error-prone.
        for i in range(len(S_new)):
            a = S_new[i]
            # 3a
            used_new.add(3*a)
            # 2a+b, a+2b, a+b+c
            for j in range(i+1, len(S_new)):
                b = S_new[j]
                used_new.add(2*a + b)
                used_new.add(a + 2*b)
                used_new.add(a + b + b) # a+2b already added? No, a+2b is distinct from a+b+b? 
                # Wait, triple (a,b,b) sum is a+2b.
                # Triple (a,a,b) sum is 2a+b.
                # Triple (a,b,c) sum is a+b+c.
                
                # Let's just iterate all triples to be safe and simple for reconstruction.
                # O(K^3) might be too slow for K=50 (125k ops). 
                # Let's use the pair sum method for reconstruction too.
                
        # Let's use the efficient rebuild:
        used_new = set()
        pairs_new = set()
        for i, a in enumerate(S_new):
            # Add a to the set, updating used/pairs
            # This is O(K^2) total if done sequentially
            pass
        
        # Actually, let's just write a clean rebuild function.
        used_new = set()
        pairs_new = set()
        for a in S_new:
            # 3a
            used_new.add(3*a)
            # Pairs with existing elements in S_new (those processed before a)
            # Wait, S_new is a list. We need to process in order to maintain pairs.
            # Let's just sort S_new and process.
            pass
        
        # Simpler: Just return the new set S_new. 
        # The caller will rebuild used/pairs if needed, or we keep them in sync.
        # For the local search, we will maintain S as a list.
        # We will rebuild used/pairs only when we need to check a new candidate y.
        # Since checking y is O(K^2), and we do this for many y, it's expensive.
        
        # Let's stick to the "Rebuild on Swap" approach.
        # Swap: Remove x, Add y.
        # 1. S_temp = S \ {x}
        # 2. Rebuild used_temp, pairs_temp from S_temp. O(K^2)
        # 3. Check if y can be added to S_temp. O(K^2)
        # 4. If yes, new S = S_temp U {y}.
        
        return S_new

    def _rebuild_state(S):
        """Rebuild used_sums and pair_sums from scratch for set S."""
        used = set()
        pairs = set()
        # Sort to ensure deterministic pair generation if needed, though set handles it.
        for a in S:
            used.add(3*a)
            for b in pairs:
                used.add(a + b)
            for b in S:
                if a != b: # We only add pairs a+b where a<=b? 
                           # Actually, for checking a+c+d, we need all pairs c+d.
                           # If we maintain pairs as {c+d | c,d in S, c<=d}, 
                           # then when adding a, we check a+p for p in pairs.
                           # And we add new pairs: a+a, a+b for b in S (b<a to avoid dup? or all?)
                           # If we add a+b for all b in S, we get duplicates in set, which is fine.
                           # But we must be consistent.
                           # Let's use: pairs = {c+d | c,d in S}.
                           # Then a+p covers a+c+d.
                           # And 2a+b, a+2b are handled separately?
                           # No, 2a+b is (a,a)+b. If pairs contains a+a, then a+(a+a) = 3a.
                           # Wait.
                           # Triple sums:
                           # 1. 3a -> a + (2a). If pairs has 2a, then a+2a=3a.
                           # 2. 2a+b -> (a+a)+b. If pairs has a+a, then (a+a)+b = 2a+b.
                           # 3. a+2b -> a+(b+b). If pairs has b+b, then a+(b+b) = a+2b.
                           # 4. a+b+c -> a+(b+c). If pairs has b+c, then a+(b+c) = a+b+c.
                           
                           # So, if we maintain `pairs` as the set of ALL sums c+d for c,d in S,
                           # then `used` can be computed as {x + p | x in S, p in pairs}.
                           # Is this correct?
                           # Sum s is in used iff there exist x,y,z in S such that x+y+z=s.
                           # Let x be one of the elements. Then y+z must be in pairs.
                           # So s = x + p where p in pairs.
                           # Yes! This is a huge simplification.
                           # We don't need separate checks for 3x, 2x+y, etc.
                           # We just need:
                           # 1. `pairs`: set of all c+d for c,d in S.
                           # 2. `used`: set of all x+p for x in S, p in pairs.
                           
                           # Check collision for new element x:
                           # New sums are x + p for p in pairs_old.
                           # New pairs are x + c for c in S_old (and x+x).
                           # Collision iff (x + pairs_old) intersects used_old.
                           
                           # Update:
                           # new_pairs = pairs_old U {x+c | c in S_old} U {2x}
                           # new_used = used_old U {x+p | p in pairs_old}
                           # Wait, is new_used just that?
                           # new_used should contain sums from triples in S_new.
                           # A triple in S_new is either:
                           # - All in S_old: sum in used_old.
                           # - Involves x: sum is x + y + z where y,z in S_old (y,z can be x? No, x is new).
                           #   If y,z in S_old, sum is x + (y+z). y+z is in pairs_old.
                           #   So new sums are {x+p | p in pairs_old}.
                           #   What about triples with two x's? (x,x,y). Sum 2x+y.
                           #   Is 2x+y in {x+p | p in pairs_old}?
                           #   We need p = x+y. Is x+y in pairs_old?
                           #   pairs_old only contains sums of elements in S_old. x is not in S_old.
                           #   So x+y is NOT in pairs_old.
                           #   Therefore, 2x+y is NOT covered by {x+p | p in pairs_old}.
                           
                           # So the simple "x+p" logic misses triples with repetition of the new element.
                           # We must handle:
                           # 1. 3x
                           # 2. 2x + y (y in S_old)
                           # 3. x + 2y (y in S_old)
                           # 4. x + y + z (y,z in S_old) -> x + (y+z) where y+z in pairs_old.
                           
                           # So the original logic was correct. The "pairs only" trick fails for repetitions.
                           pass
        
        # Let's implement the rebuild properly.
        used = set()
        pairs = set()
        for a in S:
            # Add a to the set
            used.add(3*a)
            for b in S:
                if a <= b: # To avoid duplicate pair sums in set, though set handles it
                    pairs.add(a+b)
            # Now update used with new sums involving a and existing pairs?
            # No, we are building from scratch.
            pass
        
        # Efficient Rebuild:
        used = set()
        pairs = set()
        # We need to add elements one by one to maintain the invariant.
        # Let's just iterate all triples if K is small, or use the incremental method.
        # Incremental method for rebuild:
        S_sorted = sorted(S)
        used = set()
        pairs = set()
        for a in S_sorted:
            # a is the largest element so far in this iteration?
            # No, S_sorted is just a list.
            # Let's use the standard incremental build:
            # Start with empty.
            # Add a.
            # New used: 3a, 2a+b, a+2b, a+b+c for b,c in current S.
            # This is O(K^3) total.
            
            # For K=50, 50^3 = 125,000 operations. Very fast.
            # For K=100, 1,000,000 operations. Still fast in Python? Maybe 0.1s.
            # Since we do this for every swap, and we might do 1000 swaps, that's 1e9 ops. Too slow.
            
            # We need incremental update for swaps.
            # Remove x: 
            #   We need to remove sums that ONLY x formed.
            #   This requires counting.
            
            # Alternative: **Don't remove. Just try to ADD.**
            # If we can't add, we are stuck.
            # To get unstuck, we need to change the set.
            # 
            # Let's use a different strategy: **Randomized Greedy with Restarts**.
            # 1. Pick a random subset of candidates C (size M).
            # 2. Run Greedy on C to get S.
            # 3. If len(S) > best, update best.
            # 4. Repeat.
            # This is what the current program does (sort of).
            # 
            # To make it structurally different and potentially better:
            # **Use a Bitset for the 'Used Sums' to speed up the collision check.**
            # Current check: `if (x+p) in used_sums` is O(1) for set.
            # But iterating `for p in pair_sums` is O(K^2).
            # 
            # Can we make the check O(1) or O(K)?
            # We want to know if `(x + pair_sums) & used_sums` is non-empty.
            # If we represent `used_sums` and `pair_sums` as bitsets (numpy arrays or integers),
            # we can shift the pair_sums bitset by x and AND with used_sums.
            # 
            # Max sum is 3*4096 = 12288.
            # A bitset of size 12289 bits is ~1536 bytes.
            # We can use Python's `int` as a bitset!
            # 
            # Let `used_mask` be an integer where bit i is 1 if sum i is used.
            # Let `pair_mask` be an integer where bit i is 1 if pair sum i exists.
            # 
            # Check for x:
            # 1. 3x: `if (used_mask >> (3*x)) & 1: fail`
            # 2. 2x+y: We need to check if any y in S has 2x+y in used.
            #    This is equivalent to: Is there a y in S such that (used_mask >> (2x)) has bit y set?
            #    Let `S_mask` be an integer where bit i is 1 if i is in S.
            #    Check: `(used_mask >> (2*x)) & S_mask` != 0.
            # 3. x+2y: Is there a y in S such that x+2y in used?
            #    This is: Is there a y in S such that (used_mask >> x) has bit (2y) set?
            #    Let `double_S_mask` be an integer where bit i is 1 if i/2 is in S (and i is even).
            #    Actually, 2y is just 2*y.
            #    We can precompute `S_doubled_mask`? 
            #    Or just: `((used_mask >> x) & S_doubled_mask) != 0`.
            #    How to maintain `S_doubled_mask`? 
            #    When adding y, set bit `2*y`.
            # 4. x+y+z: Is there a pair (y,z) in S such that x+y+z in used?
            #    This is: Is there a pair sum p in pairs such that x+p in used?
            #    Check: `((used_mask >> x) & pair_mask) != 0`.
            
            # This reduces the collision check to O(1) bit operations (on big integers)!
            # Python big integer bitwise operations are very fast (C implementation).
            
            # State:
            # - S: list of ints (for iteration if needed, or just mask)
            # - S_mask: int
            # - S_doubled_mask: int (bits at 2*y for y in S)
            # - pair_mask: int (bits at y+z for y,z in S)
            # - used_mask: int (bits at all triple sums)
            
            # Update on adding x:
            # - S_mask |= (1 << x)
            # - S_doubled_mask |= (1 << (2*x))
            # - new_pairs = (S_mask << x) | (1 << (2*x))  # x+y for y in S, and 2x
            #   Wait, S_mask has bits at y. (S_mask << x) has bits at x+y.
            #   So `pair_mask |= new_pairs`
            # - new_used = (used_mask) | (1 << (3*x)) | ((used_mask >> ???) ... No
            #   New used sums:
            #   1. 3x -> bit 3x
            #   2. 2x+y -> bits at 2x+y for y in S. 
            #      This is `(S_mask << (2*x))`? No, S_mask has bits at y. 
            #      We want bits at 2x+y. So `(S_mask << (2*x))`.
            #   3. x+2y -> bits at x+2y for y in S.
            #      S_doubled_mask has bits at 2y. So `(S_doubled_mask << x)`.
            #   4. x+y+z -> bits at x+p for p in pairs.
            #      `pair_mask << x`.
            
            #   So `new_used_mask = used_mask | (1 << (3*x)) | (S_mask << (2*x)) | (S_doubled_mask << x) | (pair_mask << x)`
            
            # This is extremely fast!
            pass

    # Implementation of Bitset Strategy
    
    def _greedy_bitset(candidates):
        S = []
        S_mask = 0
        S_doubled_mask = 0
        pair_mask = 0
        used_mask = 0
        
        for x in candidates:
            # Check 1: 3x
            if (used_mask >> (3*x)) & 1:
                continue
            
            # Check 2: 2x+y
            # Is there y in S such that 2x+y is used?
            # S_mask has bits at y.
            # We want to know if any bit in S_mask corresponds to a y where (2x+y) is in used_mask.
            # Shift used_mask right by 2x. Bit y is set if 2x+y was in used.
            # Intersect with S_mask.
            if ((used_mask >> (2*x)) & S_mask) != 0:
                continue
                
            # Check 3: x+2y
            # S_doubled_mask has bits at 2y.
            # Shift used_mask right by x. Bit 2y is set if x+2y was in used.
            # Intersect with S_doubled_mask.
            if ((used_mask >> x) & S_doubled_mask) != 0:
                continue
                
            # Check 4: x+y+z
            # pair_mask has bits at y+z.
            # Shift used_mask right by x. Bit p is set if x+p was in used.
            # Intersect with pair_mask.
            if ((used_mask >> x) & pair_mask) != 0:
                continue
            
            # Valid. Add x.
            S.append(x)
            S_mask |= (1 << x)
            S_doubled_mask |= (1 << (2*x))
            
            # Update pair_mask
            # New pairs: 2x, and x+y for all y in S (old S, so x not included in S_mask yet? 
            # Wait, S_mask was updated. We need old S_mask for x+y?
            # x+y where y in old S.
            # old_S_mask = S_mask ^ (1 << x)
            old_S_mask = S_mask ^ (1 << x)
            new_pairs = (1 << (2*x)) | (old_S_mask << x)
            pair_mask |= new_pairs
            
            # Update used_mask
            # New sums: 3x, 2x+y, x+2y, x+y+z
            new_used = (1 << (3*x))
            # 2x+y: y in old S. (old_S_mask << (2*x))
            new_used |= (old_S_mask << (2*x))
            # x+2y: y in old S. (old_S_doubled_mask << x)
            old_S_doubled_mask = S_doubled_mask ^ (1 << (2*x))
            new_used |= (old_S_doubled_mask << x)
            # x+y+z: p in old pairs. (old_pair_mask << x)
            old_pair_mask = pair_mask ^ new_pairs
            new_used |= (old_pair_mask << x)
            
            used_mask |= new_used
            
        return S

    # Main Loop: Randomized Greedy with Bitsets
    best_set = []
    rng = random.Random(42)
    
    # Pre-sort candidates? Or randomize?
    # Randomizing the order of candidates in greedy often helps find larger sets 
    # because it avoids getting stuck in a specific "local" packing.
    
    while time.time() - start_time < DEADLINE:
        # Generate a random permutation of a subset of candidates
        # Using all 4096 is O(4096 * Cost) per run.
        # Cost is O(1) bit ops. So 4096 runs is very fast.
        # Let's try full range.
        
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        
        S = _greedy_bitset(candidates)
        
        if len(S) > len(best_set):
            best_set = S
            # If we hit the theoretical max or a high bar, we could stop, 
            # but let's keep searching for a bit.
            
    return best_set

# Final check on logic:
# The bitset shift logic:
# `used_mask >> k` shifts the bits to the right. Bit i in the result corresponds to bit i+k in the original.
# So if `used_mask` has bit s, then `used_mask >> k` has bit s-k.
# We want to check if `x+y` is in `used_mask`.
# This is equivalent to checking if bit `x+y` is set in `used_mask`.
# In `used_mask >> x`, bit `y` is set if bit `x+y` was set in `used_mask`.
# So `(used_mask >> x) & S_mask` checks if there is any y in S such that x+y is in used_mask.
# This is correct for Check 4 (x+y+z where y+z=p, so we check x+p in used).
# Wait, for Check 4, we have `pair_mask` which has bits at p=y+z.
# We want to check if x+p is in used_mask.
# So we check `(used_mask >> x) & pair_mask`.
# Bit p in `used_mask >> x` is set if bit x+p is in used_mask.
# Intersect with pair_mask (bit p set if p is a pair sum).
# If non-zero, there is a pair sum p such that x+p is in used. Collision.
# Correct.

# Check 2: 2x+y.
# We want to check if 2x+y is in used_mask.
# `used_mask >> (2*x)` has bit y set if 2x+y is in used_mask.
# Intersect with S_mask (bit y set if y in S).
# Correct.

# Check 3: x+2y.
# We want to check if x+2y is in used_mask.
# `used_mask >> x` has bit 2y set if x+2y is in used_mask.
# Intersect with S_doubled_mask (bit 2y set if y in S).
# Correct.

# Update Logic:
# new_pairs = {2x} U {x+y | y in old_S}
# old_S_mask has bits at y.
# `old_S_mask << x` has bits at x+y.
# `1 << (2*x)` is bit 2x.
# Union is correct.

# new_used:
# 3x: `1 << (3*x)`
# 2x+y: y in old_S. `old_S_mask << (2*x)` has bits at 2x+y.
# x+2y: y in old_S. `old_S_doubled_mask << x` has bits at x+2y.
# x+y+z: p in old_pairs. `old_pair_mask << x` has bits at x+p.
# Union is correct.

# One detail: `old_S_mask` calculation.
# `S_mask` was updated BEFORE `old_S_mask` calculation in my draft?
# In the code:
# S_mask |= (1 << x)
# old_S_mask = S_mask ^ (1 << x)
# This correctly recovers the mask before adding x.

# Performance:
# Python big integers are efficient for bitwise ops.
# 4096 candidates * O(1) ops = Very fast.
# We can run many iterations.

