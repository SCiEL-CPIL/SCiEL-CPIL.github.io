"""
Strategy: Greedy Construction + Simulated Annealing.

Instead of starting with [0, N] and deleting (repair), we start with an empty set
and add elements one by one. The order of addition is determined by a heuristic
score based on the "rarity" of the residue classes an element occupies for small
primes. Elements that are "rare" (i.e., their residue class is already heavily
populated by few other elements, or conversely, they contribute to filling a class
that is close to being full) are prioritized. 

Actually, a better heuristic for construction is:
An element x is "good" to add if it does not complete a residue class modulo any
prime p <= current_size + 1.
To maximize size, we should add elements that are "least likely" to cause a conflict
in the future. However, the constraint is dynamic: as size grows, more primes become
relevant.

Let's refine the construction heuristic:
1. Precompute for each x in [0, N] the list of primes p <= N such that x mod p is
   a "critical" residue. 
2. Simpler: Just use a randomized greedy construction with multiple restarts, 
   guided by a score that penalizes elements which would immediately create a 
   "full" class for a prime p that is currently near the threshold (len+1).
   
   Actually, the most robust approach for this specific problem (admissible tuples)
   is often to notice that the "bad" primes are the small ones. 
   The repair method works well because small primes are the bottleneck.
   
   Let's try a different structural change: **Iterative Deepening A* style or 
   Beam Search on the construction order?** 
   
   No, let's stick to **Simulated Annealing on the set of elements**.
   State: A subset S of [0, N].
   Objective: Maximize |S| subject to Admissible(S).
   Moves: 
     1. Add a random x not in S (if it keeps S admissible).
     2. Remove a random x in S.
     3. Swap x in S with y not in S (if it keeps S admissible).
   
   The key is the acceptance criterion. If we just do greedy add, we get stuck.
   SA allows removing elements temporarily to allow adding many more later.
   
   To make SA fast, we need an efficient admissibility check and delta evaluation.
   Checking admissibility from scratch is O(P * |S|) which is slow.
   We can maintain counts of residues for all primes p <= N.
   
   When adding x:
     For each p <= |S|+1:
       count[x % p] += 1
       If count[x % p] == p: Conflict!
       
   When removing x:
     For each p <= |S|:
       count[x % p] -= 1
       
   This is O(pi(|S|)) per move, which is ~ |S|/log|S|. For |S|~4000, this is ~500 ops.
   300s / 500 ops = 600,000 moves. This is feasible.
   
   Initialization: Use the greedy repair result as the starting point for SA.
   This ensures we start with a valid, large set.
"""

import time
import random
import math

N = 5000
T_START = time.time()
DEADLINE_MARGIN = 10.0

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _primes(limit):
    """Sieve of Eratosthenes: list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]

# Precompute primes up to N
PRIMES = _primes(N)

class AdmissibleState:
    def __init__(self, points):
        self.points = set(points)
        self.size = len(self.points)
        # counts[p_index][residue] = count
        # We only need to track primes p <= size.
        # But as size changes, the set of relevant primes changes.
        # To be safe and simple, we track counts for ALL primes in PRIMES.
        # Memory: ~669 primes * avg p residues. Sum of p for p<=5000 is ~ 5000^2/2 / log 5000 ~ 2M.
        # This is fine.
        self.counts = {}
        for p in PRIMES:
            self.counts[p] = [0] * p
        
        for x in self.points:
            for p in PRIMES:
                if p > self.size: 
                    # Optimization: We only care about p <= size.
                    # But if we add elements, size grows. 
                    # If we maintain counts for all primes, we can check any p <= new_size instantly.
                    pass
                self.counts[p][x % p] += 1
        
        # Verify initial state is admissible
        self._verify()

    def _verify(self):
        # For every prime p <= size, no residue class should be full (count == p)
        for p in PRIMES:
            if p > self.size:
                break
            c = self.counts[p]
            if any(x == p for x in c):
                raise ValueError(f"Initial state not admissible for p={p}")

    def is_admissible(self):
        for p in PRIMES:
            if p > self.size:
                break
            c = self.counts[p]
            if any(x == p for x in c):
                return False
        return True

    def try_add(self, x):
        """Attempt to add x. Returns True if successful (and state updated)."""
        if x in self.points:
            return False
        
        # Check if adding x would create a conflict
        # Conflict if for some p <= new_size, count[x%p] == p-1 (so it becomes p)
        # Note: new_size = self.size + 1
        new_size = self.size + 1
        
        for p in PRIMES:
            if p > new_size:
                break
            res = x % p
            if self.counts[p][res] == p - 1:
                return False
        
        # Safe to add
        self.points.add(x)
        for p in PRIMES:
            if p > new_size:
                break
            self.counts[p][x % p] += 1
        self.size = new_size
        return True

    def try_remove(self, x):
        """Attempt to remove x. Returns True if successful."""
        if x not in self.points:
            return False
        
        self.points.discard(x)
        old_size = self.size
        for p in PRIMES:
            if p > old_size:
                break
            self.counts[p][x % p] -= 1
        self.size = old_size - 1
        return True

    def try_swap(self, x_out, x_in):
        """Remove x_out, add x_in. Returns True if successful."""
        if x_out not in self.points or x_in in self.points:
            return False
        
        # 1. Check if x_in can be added to (S \ {x_out})
        # The set of relevant primes for the new size (same size) is p <= self.size
        # However, the counts are currently for S.
        # If we remove x_out, counts decrease. If we add x_in, counts increase.
        # Conflict condition: 
        # For any p <= self.size:
        #   new_count[res] = count[res] - (1 if x_out%p==res else 0) + (1 if x_in%p==res else 0)
        #   We need new_count[res] < p for all res.
        
        for p in PRIMES:
            if p > self.size:
                break
            res_out = x_out % p
            res_in = x_in % p
            c = self.counts[p]
            
            # Calculate potential new count for the specific residues affected
            # Only res_out and res_in change.
            
            if res_out == res_in:
                # Net change 0. No conflict possible from this p.
                continue
            
            # Check res_out: count decreases by 1. Never causes a full class (was <= p-1, now <= p-2).
            # Check res_in: count increases by 1.
            # We need c[res_in] - (1 if res_out==res_in else 0) + 1 < p
            # Since res_out != res_in, we need c[res_in] + 1 < p => c[res_in] < p - 1
            if c[res_in] >= p - 1:
                return False
        
        # Safe to swap
        self.points.discard(x_out)
        self.points.add(x_in)
        for p in PRIMES:
            if p > self.size:
                break
            self.counts[p][x_out % p] -= 1
            self.counts[p][x_in % p] += 1
        return True

def _greedy_repair_baseline():
    """Fast baseline to get a good starting point for SA."""
    points = list(range(N + 1))
    # Simple greedy deletion
    changed = True
    while changed:
        changed = False
        m = len(points)
        for p in _primes(m):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                # Remove all elements in smallest class
                remove_set = set(smallest)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    return points

def construct():
    # 1. Get a strong baseline
    baseline_points = _greedy_repair_baseline()
    
    # 2. Initialize SA state
    try:
        state = AdmissibleState(baseline_points)
    except ValueError:
        # Fallback if baseline repair failed (shouldn't happen)
        state = AdmissibleState([])
        # Re-add greedily? Just return empty or try simple add
        for x in range(N+1):
            state.try_add(x)
    
    best_size = state.size
    best_points = set(state.points)
    
    # 3. Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.99999  # Cooling rate
    
    # We need a list of available elements for random selection
    # To avoid O(N) checks for "not in points", we can maintain a list of removed items.
    # But swaps are more effective.
    
    # Let's use a hybrid: 
    # - Try random swaps (x in S, y not in S)
    # - If stuck, try random additions (if possible) or removals (to enable additions)
    
    # For efficiency, we pick x_in from a random sample of [0, N] not in state.points.
    # To make this fast, we can keep a list of "removed" elements from the baseline.
    removed_candidates = [x for x in range(N + 1) if x not in state.points]
    random.shuffle(removed_candidates)
    
    T = T_start
    last_improvement = time.time()
    
    while _time_left() > 5.0:
        if T < T_end:
            T = T_end
            
        # Pick a random move
        # 1. Swap: pick random x_out from state.points, random x_in from removed_candidates
        #    But removed_candidates might be stale if we add/remove.
        #    Let's just pick random x_in from [0, N] and check if it's in state.
        #    To avoid bias, we can maintain a dynamic list.
        
        # Efficient move generation:
        # 1. Try to add a random element from removed_candidates
        # 2. If that fails, try a swap
        
        # Let's alternate or pick randomly.
        
        # Update removed_candidates if it gets too small or stale?
        # For now, just pick random indices.
        
        if not removed_candidates:
            break
            
        # Pick random x_in
        idx_in = random.randint(0, len(removed_candidates) - 1)
        x_in = removed_candidates[idx_in]
        
        # Pick random x_out from state.points
        # Converting set to list every time is O(S). S~4000. 4000 ops per move.
        # 300s / 4000 = 75,000 moves. This is low.
        # Optimization: Keep a list of current points for random access.
        # We can update the list on add/remove/swap.
        
        # Let's refactor state to keep a list for O(1) random access to elements in S.
        # But state.points is a set for O(1) membership.
        # We can maintain a separate list `active_list` in the loop.
        
        # For this implementation, we will just accept the O(S) cost of list(state.points) 
        # if the number of moves is high, or optimize by keeping a list in the state object.
        
        # Let's add a list to AdmissibleState for this purpose.
        # (Self-correction: I didn't add it in the class above. I will use a local list in construct and update it, or just use list(state.points) which is fast enough for 50k moves in 300s? 
        # 50k * 4000 = 200M ops. Might be too slow in Python.
        # Let's use a list in the state class.)
        
        # I will modify the logic to use a list for points in the state to allow O(1) random selection.
        # But I already defined the class. I can just use `random.choice(list(state.points))` and hope for the best, or use a fixed list of candidates.
        
        # Alternative: Use the `removed_candidates` list which is already a list.
        # For x_out, we can pick from a pre-generated list of all elements and check membership? No, that's O(1) but we need to pick from S.
        
        # Let's just use `list(state.points)` and see. If it's too slow, the deadline guard will catch it.
        # To speed up, we can pick x_out from a small random sample of state.points? No, biased.
        
        # Let's try to optimize the swap check itself first.
        
        # Actually, let's just do a simpler SA:
        # 1. Pick random x_in from removed_candidates.
        # 2. Try to add it. If success, update removed_candidates.
        # 3. If fail, pick random x_out from state.points (using list conversion).
        # 4. Try swap.
        
        # To avoid list conversion cost, we can maintain a list `active_elements` in the construct function.
        
        active_elements = list(state.points)
        
        # We need to keep active_elements in sync with state.
        
        # Let's restructure the loop to manage this list.
        
        # Reset loop logic to handle list sync
        
        # I will rewrite the loop section in the final code to be more efficient.
        
        # For now, this draft is sufficient to show the logic. The final code will include the list optimization.
        
        # Break out of this draft thinking and write the final code.
        
        pass

    # Fallback if SA didn't run or failed
    if state.is_admissible():
        return list(state.points)
    else:
        return list(baseline_points)

# REWRITING construct to be efficient and correct

def construct():
    # 1. Baseline
    baseline_points = _greedy_repair_baseline()
    
    # 2. Initialize State
    try:
        state = AdmissibleState(baseline_points)
    except ValueError:
        # If baseline is invalid (unlikely), start from empty and greedily add
        state = AdmissibleState([])
        for x in range(N+1):
            if state.try_add(x):
                pass
    
    best_points = list(state.points)
    best_size = state.size
    
    # 3. SA Setup
    T = 1.0
    T_end = 0.001
    alpha = 0.99999
    
    # List of elements NOT in state, for random x_in selection
    # We start with all elements not in baseline
    removed_candidates = [x for x in range(N + 1) if x not in state.points]
    random.shuffle(removed_candidates)
    
    # List of elements IN state, for random x_out selection
    active_elements = list(state.points)
    
    # We will track the current best size and points
    curr_size = state.size
    curr_points_set = state.points  # Reference to state's set
    
    # To allow fast random access to active elements, we maintain active_elements list.
    # We need a way to remove an element from active_elements efficiently when it's swapped out.
    # Swapping x_out with x_in:
    # 1. Find index of x_out in active_elements.
    # 2. Replace with x_in.
    # 3. Remove x_in from removed_candidates (or mark it).
    
    # For removed_candidates, we can just pop from the end if we pick randomly? 
    # No, we pick randomly. 
    # To remove x_in from removed_candidates efficiently, we can use a "lazy deletion" approach or just filter it occasionally.
    # Or, we can keep removed_candidates as a set and a list for random access?
    # Let's use a list and a set for removed_candidates.
    removed_set = set(removed_candidates)
    
    # Similarly for active_elements, we have state.points (set).
    # We need active_elements to be a list for O(1) random choice.
    # We need to update active_elements when a swap occurs.
    
    # Helper to get index of x in list. O(N). 
    # To make this O(1), we can maintain a dict {value: index} for active_elements.
    # But indices change on swap? No, if we swap values, indices stay same.
    # If we swap x_out (at idx) with x_in, we just do active_elements[idx] = x_in.
    # And update the index map: del index_map[x_out], index_map[x_in] = idx.
    
    index_map = {x: i for i, x in enumerate(active_elements)}
    
    # Let's refine the loop
    
    while _time_left() > 5.0:
        if T < T_end:
            T = T_end
            
        # 1. Try a move
        # Pick random x_in from removed_candidates
        if not removed_candidates:
            break
            
        idx_in = random.randint(0, len(removed_candidates) - 1)
        x_in = removed_candidates[idx_in]
        
        # Check if x_in is still in removed_set (lazy deletion check)
        if x_in not in removed_set:
            # Clean up
            removed_candidates.pop(idx_in)
            continue
            
        # Try to add x_in directly? 
        # If we can add it, size increases. This is always an improvement.
        # But SA allows worsening moves. 
        # However, adding is strictly better than not adding.
        # So, if try_add succeeds, we accept it immediately (no T needed).
        
        # But wait, if we add x_in, we must remove it from removed_candidates and add to active_elements.
        
        # Let's try a swap first, as it keeps size constant and allows exploration.
        # Or try add first? 
        # If we can add, we should. 
        # If we can't add, we try swap.
        
        # Try Add
        if state.try_add(x_in):
            # Success!
            curr_size += 1
            if curr_size > best_size:
                best_size = curr_size
                best_points = list(state.points)
            
            # Update structures
            removed_candidates.pop(idx_in)
            removed_set.discard(x_in)
            
            active_elements.append(x_in)
            index_map[x_in] = len(active_elements) - 1
            
            # Cool down? Or keep T? 
            # Successful add is a "good" move. 
            pass
            
        else:
            # Add failed. Try Swap.
            # Pick random x_out from active_elements
            if not active_elements:
                break
                
            idx_out = random.randint(0, len(active_elements) - 1)
            x_out = active_elements[idx_out]
            
            # Try Swap
            if state.try_swap(x_out, x_in):
                # Success!
                # Size unchanged. 
                # Accept with probability 1 (since size is same, and we are just exploring)
                # Or use SA acceptance? 
                # Since size is same, it's a neutral move. We accept it to explore.
                
                # Update structures
                # x_out is now removed, x_in is now active.
                
                # Update active_elements
                active_elements[idx_out] = x_in
                del index_map[x_out]
                index_map[x_in] = idx_out
                
                # Update removed_candidates
                # Remove x_in from removed_candidates (it's at idx_in)
                removed_candidates.pop(idx_in)
                removed_set.discard(x_in)
                
                # Add x_out to removed_candidates
                removed_candidates.append(x_out)
                removed_set.add(x_out)
                
                # No size change, so no best update.
                
            else:
                # Swap failed. 
                # Maybe try a different x_out? 
                # For now, just cool down.
                pass
        
        # Cooling
        T *= alpha
        
    # Final check
    if state.is_admissible() and len(state.points) > len(best_points):
        return list(state.points)
    else:
        return best_points

# One more optimization: 
# The _greedy_repair_baseline is O(N^2) potentially if it loops many times.
# But it's fast enough.

# Let's ensure the AdmissibleState class is efficient.
# The try_add and try_swap loops over primes.
# pi(5000) ~ 669.
# 669 checks per move.
# 300s / 669 ~ 44,000 moves.
# This is reasonable.

# Final Code Structure:
# 1. Imports
# 2. Constants
# 3. _primes
# 4. AdmissibleState class
# 5. _greedy_repair_baseline
# 6. construct

