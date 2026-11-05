"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import heapq
from collections import defaultdict

N = 5000
DEADLINE = 315  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
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
    
    # 1. Get primes up to N+1 (max possible tuple size)
    primes = _primes(N + 1)
    
    # 2. Initialize state
    # present[v] is True if v is in the current set
    present = bytearray(N + 1)
    for i in range(N + 1):
        present[i] = 1
    
    # residue_counts[p][r] = number of elements v in set such that v % p == r
    # We only care about primes p <= current_size.
    # Initially current_size = N+1.
    # We can pre-compute residue counts for all primes p <= N+1.
    # But memory might be an issue if we store all at once? 
    # Sum of p for p <= 5000 is roughly 5000 * pi(5000) / 2? No, sum of primes is ~ 1.5M.
    # Storing counts for all primes is fine.
    
    # However, we only need to track primes that are currently "active" (p <= current_size) 
    # and "unsatisfied" (all residues covered).
    
    # Let's maintain:
    # - active_primes: set of primes p <= current_size
    # - unsatisfied_primes: set of primes p in active_primes where min(counts[p]) > 0
    
    # To make it efficient, we can just re-evaluate or update incrementally.
    
    # Given the complexity of incremental updates for all primes when removing one element,
    # and the fact that N=5000 is small, let's try a hybrid:
    # 1. Use the simple greedy to get a good starting point quickly.
    # 2. Then use local search (swap) to improve.
    
    # Wait, the prompt suggests a specific strategy: "greedy vertex cover approximation by iteratively selecting the integer that covers the most uncovered constraints".
    # This is equivalent to: At each step, find the element v that, if removed, fixes the most "bad" primes.
    # A prime is "bad" if all its residues are present.
    # Removing v fixes prime p if v is the only element in some residue class r mod p? No.
    # Removing v makes residue class r mod p empty if count[p][r] == 1 and v % p == r.
    # If a prime p is currently "bad" (all counts > 0), removing v fixes p IF AND ONLY IF
    # there exists a residue r such that count[p][r] == 1 and v % p == r.
    # Actually, if count[p][r] == 1, removing v (if v%p==r) makes count[p][r]=0, so p becomes "good".
    # If count[p][r] > 1, removing v doesn't make p good.
    # So, for a "bad" prime p, the elements that can fix it are those that are the *unique* representative of some residue class.
    # Let U_p be the set of residues r mod p where count[p][r] == 1.
    # The elements that fix p are { v in set | v % p in U_p }.
    # Note: If count[p][r]==1, there is exactly one such v.
    # So the set of elements that fix p is exactly the set of unique representatives of the "singleton" residues.
    
    # Strategy:
    # 1. Start with full set.
    # 2. Identify all "bad" primes (p <= size, all counts > 0).
    # 3. For each bad prime p, identify the set of "fixers" F_p (elements that are unique in some residue class of p).
    # 4. For each element v in the set, calculate score(v) = number of bad primes p such that v in F_p.
    # 5. Remove the element v with max score(v). If tie, pick randomly or by index.
    # 6. Update state:
    #    - Remove v from set.
    #    - Update counts for all p <= new_size.
    #    - Update bad primes set.
    #    - Update fixers sets? This is expensive.
    
    # Optimization:
    # Instead of maintaining fixers sets dynamically, we can recompute scores from scratch or incrementally.
    # Given N=5000, the number of removals is roughly N - OptimalSize.
    # OptimalSize is likely around 1000-2000?
    # If we remove ~3000 elements, and each step takes O(N * pi(N)) to recompute, it might be too slow.
    # 3000 * 3.3M = 10B ops. Too slow in Python.
    
    # We need a faster heuristic.
    # Let's stick to the simpler "Greedy Removal by Least Populated Class" but optimize the implementation.
    # The previous greedy was:
    # for p in primes:
    #   if p > len(points): break
    #   if min(counts) > 0: remove least populated class.
    
    # This is actually quite good because it processes primes in increasing order.
    # Small primes are the most restrictive.
    # By fixing p=2, we remove ~N/2 elements.
    # Then p=3, we remove ~N/6 elements? No, we remove the least populated class of the *remaining* set.
    
    # Let's try to improve the greedy by looking ahead or using a better tie-breaking.
    # Or, let's try the "Swap" local search on the result of the greedy.
    
    # Plan:
    # 1. Run the standard greedy to get `base_set`.
    # 2. Compute `removed_set` = [0..N] \ `base_set`.
    # 3. Try to swap: Remove x from `base_set`, Add y from `removed_set`.
    #    Check if the new set is admissible.
    #    If yes, accept.
    #    To make this fast, we need a fast admissibility check.
    
    def _is_admissible_fast(pts_set, primes_list, L):
        """Check if pts_set (set of ints) is admissible for size L."""
        # We only need to check primes p <= L.
        # Optimization: Early exit.
        for p in primes_list:
            if p > L:
                break
            # Check if all residues are covered.
            # We can do this by iterating through the set? No, that's O(L).
            # Better: Maintain counts? But we are doing swaps, so counts change.
            # For a single check, O(L * pi(L)) is too slow if done many times.
            
            # Alternative: Pre-compute residue masks?
            # For a given p, we can compute the set of residues present.
            # If len(residues_present) == p, it's bad.
            # Computing residues_present for a set of size L takes O(L).
            # Doing this for all p <= L takes O(L * pi(L)).
            # If L ~ 2000, pi(L) ~ 300. 2000 * 300 = 600k ops per check.
            # If we do 1000 checks, 600M ops. Feasible in 30s? Maybe.
            
            residues = set()
            for v in pts_set:
                residues.add(v % p)
            if len(residues) == p:
                return False
        return True

    # 1. Greedy Construct
    def _greedy_construct():
        points = list(range(N + 1))
        in_set = set(points)
        
        # We need to check primes in increasing order.
        # But as we remove points, len(points) decreases.
        # So the set of relevant primes shrinks.
        
        # To avoid re-computing counts for all primes from scratch every time we remove a class,
        # we can just do the removal and then re-check?
        # The previous implementation did:
        # for p in primes:
        #   if p > len(points): break
        #   compute counts
        #   if min(counts) > 0: remove class
        
        # This is O(pi(N) * N) for the first few primes, but as len(points) drops, it's faster.
        # Let's implement this carefully.
        
        current_primes = [p for p in primes if p <= len(points)]
        
        # We iterate through the list of primes.
        # If we remove points, some primes might become irrelevant (p > len).
        # We can just filter the list or break.
        
        # Note: Removing points for p does NOT make other primes q "more bad".
        # It can only make them "less bad" (more missed classes).
        # So if a prime q was safe, it remains safe.
        # If a prime q was unsafe, it might become safe.
        # So we only need to check if a prime is unsafe.
        # If it is unsafe, we fix it. It becomes safe. We never need to check it again.
        # Wait, is this true?
        # If we fix p by removing a class, we remove elements.
        # This removal might cause a *different* prime q (which was safe) to become unsafe?
        # No. Removing elements decreases counts.
        # If q was safe, there was some r with count[q][r] == 0.
        # Removing elements cannot increase count[q][r]. So it remains 0.
        # So q remains safe.
        # Therefore, the set of unsafe primes is monotonically decreasing (in terms of "badness").
        # Once a prime is safe, it stays safe.
        # So we can just iterate through primes in increasing order, and if one is unsafe, fix it.
        # But wait, if we fix p, we remove elements. This might make a *later* prime q (which we haven't checked yet) safe?
        # Yes. So when we get to q, we check it. If it's safe, we skip. If unsafe, we fix.
        
        # So the algorithm is:
        # for p in primes:
        #   if p > len(points): break
        #   check if p is unsafe (all counts > 0)
        #   if unsafe: remove least populated class
        
        # This is correct and efficient.
        
        for p in primes:
            if p > len(points):
                break
            
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            if min(counts) > 0:
                min_count = min(counts)
                min_idx = counts.index(min_count)
                # Remove all v where v % p == min_idx
                # To do this efficiently, we can filter the list.
                # points = [v for v in points if v % p != min_idx]
                # This is O(N).
                # We do this for each unsafe prime.
                # How many unsafe primes are there?
                # Initially all are unsafe.
                # But as we remove, many become safe.
                # In the worst case, we might remove for many primes.
                # But each removal reduces the size significantly?
                # Not necessarily.
                
                # Let's just do the filter.
                points = [v for v in points if v % p != min_idx]
        
        return points

    base_points = _greedy_construct()
    base_set = set(base_points)
    removed_set = set(range(N + 1)) - base_set
    
    best_size = len(base_points)
    best_points = base_points[:]
    
    # 2. Local Search (Swaps)
    # We try to swap x in base_points with y in removed_set.
    # If the new set is admissible, we accept.
    # To speed up, we can try multiple swaps or random sampling.
    
    # Fast Admissibility Check for a set
    def is_admissible_set(pts_set, L):
        # L is len(pts_set)
        for p in primes:
            if p > L:
                break
            # Check if all residues are present
            # Optimization: If p is large, checking all residues is expensive.
            # But p <= L <= 5000.
            # We can use a boolean array or set.
            residues = bytearray(p) # 0 or 1
            for v in pts_set:
                residues[v % p] = 1
            if all(residues):
                return False
        return True

    # We will perform a limited number of swap attempts.
    # Time budget: ~250s for search.
    
    # To make swaps effective, we should prioritize y's that "help" the most constrained primes.
    # But for simplicity, let's start with random swaps.
    
    # Optimization: 
    # Instead of checking the entire set from scratch for each swap,
    # we can check only the primes that might be affected.
    # Adding y can only make primes "worse" (more covered).
    # Removing x can only make primes "better" (less covered).
    # So a swap is valid if:
    # 1. The new set is admissible.
    # This is equivalent to: For all p <= L, not (all residues covered).
    
    # Let's just use the full check. It's O(L * pi(L)).
    # If L ~ 2000, pi(L) ~ 300. 600k ops.
    # In Python, 600k ops ~ 0.05s?
    # So we can do ~5000 checks in 250s.
    
    # Let's try 5000 random swaps.
    
    candidates_y = list(removed_set)
    random.shuffle(candidates_y)
    
    # We need to be careful with time.
    # Let's measure the time for one check.
    
    check_count = 0
    last_time = time.time()
    
    while time.time() - start_time < DEADLINE - 10:
        if not candidates_y:
            break
            
        y = candidates_y.pop()
        if y in base_set: # Should not happen
            continue
            
        # Try to swap with a random x from base_set
        # To be smarter, we can pick x that is "least useful".
        # But for now, random x.
        
        # Pick a random x from base_set
        # To avoid O(N) selection, we can maintain a list of base_points.
        # But base_points changes.
        # Let's just pick a random index.
        
        if not base_points:
            break
            
        idx = random.randint(0, len(base_points) - 1)
        x = base_points[idx]
        
        # Perform swap
        # Remove x, Add y
        # Check admissibility of (base_set - {x} + {y})
        
        # To avoid modifying base_set directly and reverting, we can create a new set?
        # That's O(N) memory and time.
        # Better: Modify in place, check, revert if fail.
        
        # But is_admissible_set takes a set.
        # We can pass a set view? No.
        # Let's just create a temporary set? No, too slow.
        
        # We can write a specialized check that takes base_set, x, y.
        
        def is_admissible_swap(base_set, x, y, L):
            # L is len(base_set) (size remains same)
            # We check primes p <= L.
            # For each p, we need to know if all residues are covered in (base_set - {x} + {y}).
            # This is equivalent to:
            # Let S' = base_set - {x} + {y}.
            # For each p, let R' be the set of residues of S'.
            # We need len(R') < p for all p.
            
            # We can compute R' efficiently?
            # R' = (R_base - {x%p}) union {y%p} ... roughly.
            # But we don't have R_base stored for all p.
            
            # So we are back to O(L * pi(L)).
            # Let's just do it.
            
            for p in primes:
                if p > L:
                    break
                residues = bytearray(p)
                for v in base_set:
                    if v == x:
                        continue
                    residues[v % p] = 1
                residues[y % p] = 1
                if all(residues):
                    return False
            return True

        if is_admissible_swap(base_set, x, y, len(base_set)):
            # Accept swap
            base_set.remove(x)
            base_set.add(y)
            base_points[idx] = y # Update list
            candidates_y.append(x) # x is now a candidate for removal
            random.shuffle(candidates_y) # Re-shuffle? Maybe expensive.
            # Actually, we can just leave x in candidates_y for later.
            # But we popped y.
            # So we add x to the pool.
            
            # We don't update best_size because size is constant.
            # But we update best_points if we want to keep the best found?
            # Since size is constant, any valid swap is an improvement in "quality" (density),
            # but the score is just the size.
            # Wait, the score is `tuple size`.
            # So swaps do NOT improve the score!
            # The score is the number of elements.
            # So why do swaps?
            # Swaps don't change the size.
            # So they don't improve the score directly.
            # However, a better configuration might allow *insertions* later.
            
            # So the strategy should be:
            # 1. Greedy removal to get a valid set of size S.
            # 2. Try to INSERT elements into the set to increase size.
            # 3. If insertion fails, try SWAPS to rearrange the set to make insertion possible.
            
            pass # Swap logic is only useful if it enables insertion.

    # 3. Insertion Search
    # Now we try to add elements from `removed_set` to `base_set`.
    
    # We need a fast check for insertion.
    # Adding y to base_set (size L) creates a set of size L+1.
    # We need to check primes p <= L+1.
    # This is O(L * pi(L)).
    
    # Let's try to insert elements one by one.
    # We can prioritize y's that are "safe" for many primes.
    
    # For each y in removed_set:
    #   Check if base_set U {y} is admissible.
    #   If yes, add y to base_set, remove from removed_set.
    #   Update L.
    
    # This is O(|removed| * L * pi(L)).
    # |removed| ~ 3000. L ~ 2000. pi(L) ~ 300.
    # 3000 * 600k = 1.8B ops. Too slow.
    
    # We need a faster insertion check.
    # Optimization:
    # When adding y, only primes p where y%p was a "missed" class are at risk.
    # If a prime p had >= 2 missed classes, adding y cannot make it unsafe.
    # If a prime p had exactly 1 missed class, and y%p is that class, it becomes unsafe.
    # If a prime p had 0 missed classes (unsafe), it stays unsafe.
    
    # So we only need to check primes p <= L+1 that have exactly 1 missed class in base_set.
    # Let's maintain a set of "critical" primes.
    
    # Re-initialize critical primes for base_set.
    
    def get_critical_primes(pts_set, L, primes_list):
        critical = set()
        for p in primes_list:
            if p > L:
                break
            counts = [0] * p
            for v in pts_set:
                counts[v % p] += 1
            missed = 0
            for c in counts:
                if c == 0:
                    missed += 1
            if missed == 1:
                critical.add(p)
        return critical

    critical_primes = get_critical_primes(base_set, len(base_set), primes)
    
    # Now, for each y in removed_set:
    #   Check if y causes any critical prime to become unsafe.
    #   i.e., for any p in critical_primes, if y % p is the unique missed class, then y is bad.
    #   If y is good for all critical primes, it can be added.
    #   After adding y, update critical_primes.
    #   - y might make some critical primes unsafe (if it fills the gap). We check this.
    #   - y might make non-critical primes critical?
    #     If a prime p had 2 missed classes, and y fills one, it now has 1 missed class.
    #     So p becomes critical.
    #     So we need to update critical_primes to include new criticals.
    
    # This update is expensive if done naively.
    # But we can just re-compute critical_primes after each insertion?
    # Re-computation is O(L * pi(L)).
    # If we do ~1000 insertions, 1000 * 600k = 600M ops. Feasible.
    
    # Let's try this.
    
    # Sort removed_set? Maybe by some heuristic.
    # Let's just iterate.
    
    current_set = base_set
    current_size = len(current_set)
    candidates = list(removed_set)
    random.shuffle(candidates)
    
    for y in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        if y in current_set:
            continue
            
        # Check if y can be added
        can_add = True
        for p in critical_primes:
            # Find the unique missed class for p in current_set
            # We need to know which class is missed.
            # We didn't store it.
            # So we have to re-check counts for p?
            # Or we can store the missed class for each critical prime.
            
            # Let's store missed_class[p] = r
            # We need to update this.
            
            pass # This approach is getting complex.
        
        # Simpler: Just do the full check for insertion, but only for primes <= current_size + 1.
        # And use the fact that we only need to check if the NEW set is admissible.
        # We can speed it up by early exiting.
        
        def is_admissible_insertion(current_set, y, L):
            # L is current_size
            # New size is L+1
            # Check primes p <= L+1
            for p in primes:
                if p > L + 1:
                    break
                residues = bytearray(p)
                for v in current_set:
                    residues[v % p] = 1
                residues[y % p] = 1
                if all(residues):
                    return False
            return True

        if is_admissible_insertion(current_set, y, current_size):
            current_set.add(y)
            current_size += 1
            # Update critical_primes? 
            # If we don't, we might miss opportunities or do unnecessary checks.
            # But if we just do full checks, we don't need critical_primes.
            # The full check is O(L * pi(L)).
            # If we do 1000 insertions, 600M ops.
            # Let's see how fast it is.
            
            if current_size > best_size:
                best_size = current_size
                best_points = list(current_set)
                
    return best_points

# EVOLVE-BLOCK-END
