import time
from math import gcd
import sys

N = 512
DEADLINE = 88.0  # Safety margin from 90s

def construct():
    start_time = time.time()
    
    # Precompute LCM conflicts
    # conflict[i][j] is True if lcm(i, j) > N
    # We can store this as a list of bitmasks for fast operations
    # But for N=512, a list of sets or a 2D boolean array is also fine.
    # Let's use a list of sets for clarity and moderate speed, 
    # but for the greedy step, we need to count conflicts quickly.
    
    # Actually, let's stick to bitmasks for the "conflict with current set" check
    # and a list of conflict sets for the "conflict with remaining" heuristic.
    
    conflict_sets = [set() for _ in range(N + 1)]
    conflict_masks = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i // gcd(i, j)) * j > N:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
    
    # Greedy Construction Strategy:
    # 1. Start with an empty set.
    # 2. At each step, from the remaining candidates, pick the one that has the 
    #    fewest conflicts with the *other remaining candidates*.
    #    Why? We want to add "compatible" numbers. If a number conflicts with many 
    #    others, adding it removes many options. If it conflicts with few, it's "safe".
    #    However, we must also ensure it doesn't conflict with the *already chosen* set.
    
    # A more effective heuristic for "Maximize Cardinality" under "Pairwise Constraint":
    # This is equivalent to finding a large clique in the compatibility graph (complement of conflict graph).
    # Greedy clique finding:
    # 1. Sort vertices by degree in compatibility graph (descending).
    # 2. Iterate through sorted list. If vertex is compatible with all already chosen, add it.
    
    # Let's try a dynamic greedy:
    # Candidates = set(1..N)
    # Chosen = set()
    # While Candidates:
    #    Pick v in Candidates that maximizes:
    #       - Compatibility with Chosen (must be 100%)
    #       - "Potential": Number of other candidates compatible with v.
    #       - Or simply: Pick v with minimum degree in the *conflict* subgraph induced by Candidates.
    
    candidates = set(range(1, N + 1))
    chosen = set()
    
    # Pre-compute degrees in the conflict graph? 
    # The degree changes as we remove vertices.
    # For N=512, calculating the "effective degree" (conflicts within remaining candidates)
    # for all candidates at each step is O(N^2) per step, O(N^3) total. 512^3 ~ 134M ops.
    # In Python, this might be too slow if done naively in a loop 512 times.
    
    # Optimization: 
    # We can approximate. Or we can use a simpler greedy:
    # Sort numbers 1..N.
    # Smaller numbers generally have fewer LCM conflicts.
    # Let's test natural order first. It's O(N^2) total.
    
    # Strategy A: Natural Order Greedy
    # This is very fast.
    sol_a = set()
    for i in range(1, N + 1):
        # Check if i is compatible with all in sol_a
        compatible = True
        # Use bitmask check for speed? 
        # conflict_masks[i] & mask_of_sol_a == 0
        # Let's maintain a bitmask of chosen
        pass # We'll implement a faster version below.
        
    # Let's use a bitmask for the chosen set to speed up compatibility checks.
    chosen_mask = 0
    chosen_set = set()
    
    # Heuristic: Process numbers in increasing order.
    # This works well because 1, 2, 3... are "divisors" of many numbers, 
    # but wait, LCM(1, x) = x <= 512 always.
    # LCM(2, x) = x if x even, 2x if x odd.
    # So 1 is compatible with all.
    # 2 is compatible with all even numbers and odd numbers x where 2x <= 512 (x <= 256).
    # So 2 conflicts with odd numbers > 256.
    
    # Let's try a "Least Conflicts Remaining" greedy.
    # To make it fast, we'll iterate a few times or use a simplified metric.
    
    # Actually, the "Natural Order" greedy is extremely strong for this specific problem 
    # because small numbers are "universal" connectors.
    
    # Let's run Natural Order Greedy first.
    for i in range(1, N + 1):
        # Check compatibility with chosen_set
        # (i // gcd(i, j)) * j <= 512 for all j in chosen_set
        # Using bitmask:
        # if conflict_masks[i] & chosen_mask == 0:
        if conflict_masks[i] & chosen_mask == 0:
            chosen_set.add(i)
            chosen_mask |= (1 << i)
            
    best_set = set(chosen_set)
    best_size = len(best_set)
    
    # Now, Local Search to improve.
    # Try to swap a chosen element with an unchosen element to increase size?
    # Or just try to add more?
    # The greedy might have missed some numbers because of the order.
    # Example: Greedy picks 3. 3 conflicts with 257 (lcm=771). 
    # If we didn't pick 3, we could pick 257? 
    # But 257 is large, it conflicts with almost everything else.
    # Generally, smaller numbers are better "hubs".
    
    # Let's try a different greedy order: 
    # Sort by "number of conflicts" in the full graph (ascending).
    # Numbers with fewest total conflicts are "safe" everywhere.
    
    degrees = [len(conflict_sets[i]) for i in range(N + 1)]
    sorted_by_degree = sorted(range(1, N + 1), key=lambda x: degrees[x])
    
    chosen_mask_b = 0
    chosen_set_b = set()
    for i in sorted_by_degree:
        if conflict_masks[i] & chosen_mask_b == 0:
            chosen_set_b.add(i)
            chosen_mask_b |= (1 << i)
            
    if len(chosen_set_b) > best_size:
        best_set = chosen_set_b
        best_size = len(best_set)
        
    # Let's try a randomized greedy with multiple restarts if time permits.
    # And a local search (1-move) on the best solution.
    
    def try_local_search(current_set, current_mask, time_limit):
        # Try to find a number 'in' in current_set and 'out' not in current_set
        # such that swapping them allows us to add another number? 
        # Actually, simpler: 
        # Can we add any number from the complement? 
        # The greedy already added all compatible numbers. 
        # So to improve, we must *remove* some to allow *more* to be added.
        # This is a 2-swap or higher order search.
        
        # Let's stick to a simple "re-greedy" with a different start or randomization.
        pass
        
    # Given the structure, the "Natural Order" or "Low Degree" greedy is likely near-optimal.
    # The maximum independent set of the conflict graph is likely very close to the greedy result.
    # Why? Because the conflict graph is "interval-like" or "threshold-like" in terms of magnitude.
    
    # Let's do a quick check: 
    # Is there a number > 256 that can be included?
    # If x > 256, then for any odd y > 1, lcm(x, y) >= x*y.
    # If x is odd and > 256, it conflicts with all odd y > 1.
    # It is compatible with 1.
    # It is compatible with even y only if x divides y or y divides x? 
    # lcm(x, y) <= 512.
    # If x=257 (prime), lcm(257, y) <= 512 implies y=1 or y=257.
    # So 257 is only compatible with 1 and 257.
    # In a large set, including 257 forces us to exclude almost all other odds.
    # Since we have many small odds (3, 5, 7...), it's better to exclude 257.
    
    # Therefore, the optimal set likely consists mostly of numbers <= 256, plus 1, and maybe some specific even numbers > 256 that are powers of 2?
    # 512 is a power of 2. lcm(512, 2^k) = 512.
    # So 512 is compatible with 1, 2, 4, ..., 256.
    # If we include 512, we can include all powers of 2.
    # Does including 512 help? 
    # Greedy natural order:
    # 1 (ok)
    # 2 (ok, lcm(1,2)=2, lcm(2,2)=2)
    # 3 (ok, lcm(1,3)=3, lcm(2,3)=6)
    # ...
    # 256 (ok)
    # 257 (conflict with 3? lcm(3,257)=771 > 512. Yes. So 257 skipped if 3 is in set).
    # ...
    # 512 (conflict with 3? lcm(3,512)=1536 > 512. Yes. So 512 skipped if 3 is in set).
    
    # So the natural order greedy will pick 1..256 (all of them? let's check 3 and 256. lcm=768 > 512. So 3 and 256 conflict!)
    # Wait. lcm(3, 256) = 768. So 3 and 256 CANNOT be in the set together.
    # My previous assumption "small numbers are universal" is flawed for large powers of 2.
    
    # Let's re-evaluate the Greedy.
    # Natural Order:
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10...
    # 3 is picked.
    # 4 is picked (lcm(3,4)=12).
    # ...
    # 16 is picked (lcm(3,16)=48).
    # 32 is picked (lcm(3,32)=96).
    # 64 is picked (lcm(3,64)=192).
    # 128 is picked (lcm(3,128)=384).
    # 256: lcm(3, 256) = 768 > 512. So 256 is REJECTED if 3 is in the set.
    # 512: lcm(3, 512) > 512. REJECTED.
    
    # So Natural Order Greedy picks 1, 2, 3, ..., 128, skips 256, 512.
    # What if we excluded 3? 
    # Then we could include 256 and 512.
    # But excluding 3 means we might also have to exclude 9, 15, etc? No, just 3.
    # If we remove 3, we gain 256 and 512. Net change: -1 + 2 = +1.
    # So a local search that swaps 3 for {256, 512} would improve the solution!
    
    # The current program (score 1.0455) uses a B&B which *should* find this, but maybe it times out or the heuristic is weak.
    # My new strategy: 
    # 1. Run Natural Order Greedy.
    # 2. Run "Swap Search": 
    #    For each element x in the set, try removing it.
    #    Then, run a greedy pass on the remaining candidates to see if we can add more.
    #    Specifically, after removing x, check if any previously rejected number y can now be added.
    #    If yes, and the net gain is positive, accept the swap.
    
    # Implementation of Local Search:
    
    def greedy_fill(current_set, current_mask, candidates_mask):
        """
        Given a current independent set, try to add as many more as possible 
        from candidates_mask using natural order.
        Returns the new set and mask.
        """
        # Iterate 1..N
        for i in range(1, N + 1):
            if current_mask & (1 << i):
                continue
            if candidates_mask & (1 << i):
                if conflict_masks[i] & current_mask == 0:
                    current_set.add(i)
                    current_mask |= (1 << i)
        return current_set, current_mask

    # Initial Solution: Natural Order Greedy
    sol_set = set()
    sol_mask = 0
    for i in range(1, N + 1):
        if conflict_masks[i] & sol_mask == 0:
            sol_set.add(i)
            sol_mask |= (1 << i)
            
    best_size = len(sol_set)
    best_set = set(sol_set)
    
    # Local Search Loop
    # We will iterate through elements in the set and try to remove them to see if it unlocks better additions.
    # To be efficient, we only consider removing elements that "block" many others.
    # Or just iterate all.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Copy current state
        cur_set = set(sol_set)
        cur_mask = sol_mask
        
        # Try removing each element
        # Sort elements by "impact" (number of conflicts they cause in the *rejected* set)?
        # Just iterate.
        for x in list(cur_set):
            if time.time() > DEADLINE:
                break
            
            # Tentatively remove x
            new_set = cur_set - {x}
            new_mask = cur_mask & ~(1 << x)
            
            # Try to fill with greedy
            # We can optimize: only check numbers that were previously rejected due to x?
            # For simplicity, run full greedy fill. N=512 is small.
            new_set, new_mask = greedy_fill(new_set, new_mask, (1 << (N+1)) - 2) # Mask for 1..N
            
            if len(new_set) > best_size:
                sol_set = new_set
                sol_mask = new_mask
                best_size = len(sol_set)
                improved = True
                # Restart loop with new base? 
                # Or continue with this new base?
                # Restarting is safer to avoid local minima traps from bad swaps.
                break
        
    # One more strategy: Randomized Restart
    # If time remains, try random permutations or random initial seeds.
    if time.time() < DEADLINE - 5:
        import random
        random.seed(42)
        
        for _ in range(10): # 10 restarts
            if time.time() > DEADLINE - 2:
                break
            
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
            
            r_set = set()
            r_mask = 0
            for i in order:
                if conflict_masks[i] & r_mask == 0:
                    r_set.add(i)
                    r_mask |= (1 << i)
            
            if len(r_set) > best_size:
                best_set = r_set
                best_size = len(r_set)
                
                # Run local search on this new best
                # (Simplified: just accept it, maybe one pass of local search)
                # For brevity and time, we'll just take the greedy result.
                # The local search above is more powerful.
                
    return list(best_set)

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("Valid Solution")
