"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000
_DEADLINE = time.time() + 310.0  # Safety margin before 330s kill

# EVOLVE-BLOCK-START

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

# Pre-compute primes up to N
PRIMES_UP_TO_N = _primes(N)
MAX_P = len(PRIMES_UP_TO_N)

def construct():
    # 1. Generate initial greedy solution
    points = [1] + [p for p in PRIMES_UP_TO_N if p * p > N]
    
    # Helper class to manage admissibility state efficiently
    class AdmissibilityTracker:
        def __init__(self, points):
            self.points = list(points)
            self.size = len(self.points)
            # residue_counts[p_idx][r] = count of points with residue r mod p
            # We only need primes p <= size.
            # But size changes.
            # Let's store counts for ALL primes in PRIMES_UP_TO_N.
            # p_idx is index in PRIMES_UP_TO_N.
            self.residue_counts = [ [0] * p for p in PRIMES_UP_TO_N ]
            self.num_present = [0] * MAX_P # Number of distinct residues present for each prime
            self.violations = 0
            
            # Initialize
            for v in self.points:
                self._add_point(v)
                
        def _add_point(self, v):
            # Update counts for all primes
            for i, p in enumerate(PRIMES_UP_TO_N):
                # If p > current max possible size? 
                # Actually, the condition is "for every prime p <= len(tuple)".
                # So if p > self.size, it doesn't matter if it's violated or not.
                # But we update counts anyway to be ready if size increases.
                r = v % p
                if self.residue_counts[i][r] == 0:
                    self.num_present[i] += 1
                    if self.num_present[i] == p and p <= self.size:
                        # Becomes violated
                        self.violations += 1
                self.residue_counts[i][r] += 1
                
        def _remove_point(self, v):
            for i, p in enumerate(PRIMES_UP_TO_N):
                r = v % p
                self.residue_counts[i][r] -= 1
                if self.residue_counts[i][r] == 0:
                    self.num_present[i] -= 1
                    if self.num_present[i] == p - 1 and p <= self.size:
                        # Was violated, now not (if size still >= p)
                        # Wait, if p <= self.size, and num_present drops to p-1, it's no longer violated.
                        self.violations -= 1

        def add_point(self, v):
            # Must check if v is already in points? 
            # The caller ensures v is not in points.
            self._add_point(v)
            self.points.append(v)
            self.size += 1
            
            # After adding, new primes p where p == self.size might become relevant.
            # Also, primes p <= self.size that were not violated might become violated?
            # No, adding a point can only increase num_present.
            # So a prime that was NOT violated (num_present < p) might become violated (num_present == p).
            # A prime that WAS violated stays violated.
            # So we only need to check primes p where p <= new_size and p > old_size?
            # No, if p <= old_size, it was already checked.
            # If p <= new_size and p > old_size, it's a new prime to consider.
            # Also, for p <= old_size, num_present increased, so it might have crossed the threshold p.
            # My _add_point already handles the violation count update based on p <= self.size.
            # BUT, self.size is updated AFTER _add_point in my current structure?
            # Let's fix the order.
            
        def remove_point(self, v):
            self._remove_point(v)
            self.points.remove(v) # O(N)
            self.size -= 1
            
            # Removing a point decreases num_present.
            # A prime that was violated (num_present == p) might become not violated (num_present == p-1).
            # This is handled in _remove_point.
            # Primes p > new_size are no longer checked.
            # If a prime p was violated, but now p > self.size, it shouldn't count as violation.
            # My _remove_point checks `p <= self.size`. Since self.size is updated AFTER _remove_point, 
            # this logic is flawed.
            
    # The class above is getting messy with size updates. 
    # Let's simplify: 
    # We maintain `points` list.
    # We maintain `counts` and `present` arrays.
    # We calculate `violations` from scratch? No, too slow.
    # Let's just recalculate violations for the specific primes that might have changed status.
    # Status changes only for primes p where num_present crosses p.
    
    # Let's try a simpler, slightly less optimized but robust approach first, 
    # relying on the fact that N=5000 is small.
    
    # Optimized Check:
    # We store `counts[p_idx][r]` and `present[p_idx]`.
    # `violations` is the count of i where present[i] == PRIMES_UP_TO_N[i] AND PRIMES_UP_TO_N[i] <= current_size.
    
    # Let's implement the swap logic with careful state management.
    
    # Initial state
    points_set = set(points)
    # Initialize counts
    counts = [ [0] * p for p in PRIMES_UP_TO_N ]
    present = [0] * MAX_P
    
    for v in points:
        for i, p in enumerate(PRIMES_UP_TO_N):
            r = v % p
            if counts[i][r] == 0:
                present[i] += 1
            counts[i][r] += 1
            
    def calc_violations(size):
        v = 0
        for i, p in enumerate(PRIMES_UP_TO_N):
            if p <= size and present[i] == p:
                v += 1
        return v
        
    current_size = len(points)
    current_violations = calc_violations(current_size)
    
    # Greedy repair if needed
    if current_violations > 0:
        changed = True
        while changed and time.time() < _DEADLINE - 50:
            changed = False
            # Find a violated prime
            # We need to find p where present[i] == p and p <= current_size
            # And remove a point from the least populated residue class for that p.
            
            # To do this efficiently, we need to know which residue class is least populated.
            # We have counts[i].
            
            for i, p in enumerate(PRIMES_UP_TO_N):
                if p <= current_size and present[i] == p:
                    # Violation found
                    # Find min count in counts[i]
                    min_count = min(counts[i])
                    # Find residue r with min count
                    # There might be multiple. Pick one.
                    target_r = counts[i].index(min_count)
                    
                    # Find a point v in points such that v % p == target_r
                    # Remove it.
                    v_to_remove = None
                    for v in points:
                        if v % p == target_r:
                            v_to_remove = v
                            break
                    
                    if v_to_remove is not None:
                        # Remove v_to_remove
                        points.remove(v_to_remove)
                        points_set.remove(v_to_remove)
                        current_size -= 1
                        
                        # Update counts
                        for j, pj in enumerate(PRIMES_UP_TO_N):
                            rj = v_to_remove % pj
                            counts[j][rj] -= 1
                            if counts[j][rj] == 0:
                                present[j] -= 1
                                
                        current_violations = calc_violations(current_size)
                        changed = True
                        break
                        
    best_points = list(points)
    best_size = current_size
    best_violations = current_violations
    
    # Simulated Annealing
    # Fitness: -violations * 10000 + size
    current_fitness = -current_violations * 10000 + current_size
    best_fitness = current_fitness
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    pool = list(range(N + 1))
    
    # To speed up "find a point with residue r", we can maintain lists of points by residue?
    # That's complex. 
    # Given N=5000, `points.remove` and `v % p` checks are O(N).
    # SA step: 
    # 1. Pick random v_out from points (O(1) if index, O(N) if value removal)
    # 2. Pick random v_in from pool
    # 3. Update counts (O(pi(N)))
    # 4. Check violations (O(pi(N)))
    # 5. Update points list (O(N))
    
    # Total per step: O(N + pi(N)) ~ 5000 + 670 ~ 6000 ops.
    # 10,000 steps -> 60M ops. Feasible.
    
    while time.time() < _DEADLINE - 5:
        if T < T_min:
            T = T_min
            
        # Generate move
        if current_size == 0:
            v_in = random.choice(pool)
            v_out = None
            new_size = 1
        else:
            idx_out = random.randint(0, current_size - 1)
            v_out = points[idx_out]
            
            # Pick v_in
            v_in = random.choice(pool)
            
            # If v_in == v_out, skip or pick another
            if v_in == v_out:
                v_in = random.choice(pool)
                if v_in == v_out:
                    v_in = v_out # No change
            
            # If v_in is already in points (and v_in != v_out), it's invalid (duplicate)
            if v_in in points_set and v_in != v_out:
                # Try again? Or just pick v_out (no change)
                # Let's just pick a random one until valid? 
                # Or accept the duplicate? No, must be distinct.
                # Let's just skip this move.
                T *= cooling
                continue
            
            new_size = current_size # If v_in != v_out and v_in not in set, size stays same?
            # Wait. We remove v_out, add v_in.
            # If v_in was not in set, size stays same.
            # If v_in was v_out, size stays same.
            # So size is constant in a swap?
            # Yes, swap keeps size constant.
            # To GROW, we need a move that adds a point without removing one?
            # But if we add a point, size increases.
            # If the new state is valid, we accept.
            # If not, we might reject.
            # So we should allow "Add" moves and "Remove" moves?
            # Or just "Swap" and rely on the fact that we start with a valid set?
            # If we start valid, and only swap, we stay valid?
            # No, swapping can break validity.
            # And swapping cannot increase size.
            # So we will never grow beyond the initial greedy size!
            
            # CRITICAL INSIGHT: 
            # The greedy start is [1] + primes > sqrt(N).
            # Size is approx 1 + (669 - 30) = 640.
            # Can we get larger?
            # Yes, if we can find a set of size > 640 that is admissible.
            # To get larger, we MUST add points.
            # So we need moves that change size.
            
            # Move types:
            # 1. Add: Pick v_in not in set. Add it. Size +1.
            # 2. Remove: Pick v_out in set. Remove it. Size -1.
            # 3. Swap: Pick v_out, v_in. Replace. Size 0.
            
            # Let's use a mix.
            # If we are in a valid state (violations=0), we prefer Add.
            # If we are in an invalid state, we prefer Remove or Swap to fix violations.
            
            pass # Placeholder for logic below

    # Re-implementing the loop with proper move types
    
    # Reset for SA
    current_size = len(points)
    current_violations = calc_violations(current_size)
    current_fitness = -current_violations * 10000 + current_size
    best_fitness = current_fitness
    best_points = list(points)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < _DEADLINE - 5:
        if T < T_min:
            T = T_min
            
        # Decide move type
        # If valid, try to grow (Add) or explore (Swap)
        # If invalid, try to shrink (Remove) or explore (Swap)
        
        r = random.random()
        if current_violations == 0:
            if r < 0.5:
                move_type = 'add'
            else:
                move_type = 'swap'
        else:
            if r < 0.5:
                move_type = 'remove'
            else:
                move_type = 'swap'
                
        # Execute move
        old_points = list(points)
        old_size = current_size
        old_violations = current_violations
        
        # Temporarily apply move to counts and points
        # To avoid deep copying counts, we will just calculate the new violations 
        # based on the change.
        
        # This is getting complicated to do incrementally without errors.
        # Given the time constraint, let's stick to a simpler but effective heuristic:
        # **Local Search with Restart**
        # 1. Greedy start.
        # 2. Try to add random points. If valid, keep.
        # 3. If stuck, try to swap.
        # 4. If stuck, restart with a different seed?
        
        # Actually, the "Add" move is the key to growth.
        # Let's just do a simple hill climbing on "Add" moves.
        # Start with valid set.
        # Try adding random points. If valid, accept.
        # If no more points can be added, try swapping to unblock?
        
        # Let's implement "Greedy Add" first.
        # It might be faster and sufficient.
        
        # We'll break out of the SA loop idea and do:
        # 1. Greedy Repair (done)
        # 2. Random Add Loop:
        #    - Pick random v not in set.
        #    - If adding v keeps it admissible, add it.
        #    - Repeat until time runs out or no more adds possible.
        # 3. Random Swap Loop:
        #    - Pick random v_in, v_out.
        #    - If swap keeps it admissible, do it.
        #    - This explores the space of same-size sets.
        
        # This is simpler and less error-prone than full SA with incremental updates.
        
        # Let's implement this simpler strategy.
        
        # Current state: points, points_set, counts, present
        # We need a function `try_add(v)` that checks if adding v is valid.
        # And `try_swap(v_in, v_out)`.
        
        # To check validity of adding v:
        # For each prime p <= new_size (old_size + 1):
        #   If p <= old_size: 
        #      Check if adding v makes present[p] == p.
        #      It does if present[p] == p-1 and v % p is the missing residue.
        #      Or if present[p] == p (already violated, stays violated).
        #   If p == new_size:
        #      Check if all residues are present.
        
        # This is doable.
        
        # Let's just use the full check for simplicity, as N is small.
        # O(N * pi(N)) per check.
        # If we do 1000 checks, it's 3M * 1000 = 3B. Too slow.
        
        # We MUST use incremental or fast check.
        
        # Let's go back to the class-based incremental approach but fix the logic.
        
        # I will implement a robust `State` object.
        
        class State:
            def __init__(self, points):
                self.points = list(points)
                self.size = len(self.points)
                self.set = set(self.points)
                self.counts = [ [0] * p for p in PRIMES_UP_TO_N ]
                self.present = [0] * MAX_P
                for v in self.points:
                    self._update_counts(v, 1)
                self.violations = self._calc_violations()
                
            def _update_counts(self, v, delta):
                for i, p in enumerate(PRIMES_UP_TO_N):
                    r = v % p
                    old_c = self.counts[i][r]
                    new_c = old_c + delta
                    self.counts[i][r] = new_c
                    if old_c == 0 and new_c > 0:
                        self.present[i] += 1
                    elif old_c > 0 and new_c == 0:
                        self.present[i] -= 1
                        
            def _calc_violations(self):
                v = 0
                for i, p in enumerate(PRIMES_UP_TO_N):
                    if p <= self.size and self.present[i] == p:
                        v += 1
                return v
                
            def add(self, v):
                if v in self.set:
                    return False
                self._update_counts(v, 1)
                self.points.append(v)
                self.set.add(v)
                self.size += 1
                self.violations = self._calc_violations()
                return True
                
            def remove(self, v):
                if v not in self.set:
                    return False
                self._update_counts(v, -1)
                self.points.remove(v)
                self.set.remove(v)
                self.size -= 1
                self.violations = self._calc_violations()
                return True
                
            def swap(self, v_out, v_in):
                if v_in == v_out:
                    return True
                if v_in in self.set and v_in != v_out:
                    return False
                self.remove(v_out)
                self.add(v_in)
                return True
                
            def fitness(self):
                return -self.violations * 10000 + self.size

        state = State(points)
        best_state = State(list(points))
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        pool = list(range(N + 1))
        
        while time.time() < _DEADLINE - 5:
            if T < T_min:
                T = T_min
                
            r = random.random()
            if state.violations == 0:
                if r < 0.7:
                    # Try add
                    v = random.choice(pool)
                    # Make a copy? No, just try and revert if bad?
                    # Or check validity first?
                    # Checking validity is O(pi(N)).
                    # Let's just do the move, check fitness, and revert if worse.
                    
                    # To revert, we need to undo the move.
                    # If we add v, we can remove v.
                    
                    old_fitness = state.fitness()
                    state.add(v)
                    new_fitness = state.fitness()
                    
                    delta = new_fitness - old_fitness
                    if delta > 0 or random.random() < math.exp(delta / T):
                        if new_fitness > best_state.fitness():
                            best_state = State(list(state.points))
                    else:
                        state.remove(v) # Revert
                        
                else:
                    # Try swap
                    v_out = random.choice(state.points)
                    v_in = random.choice(pool)
                    
                    old_fitness = state.fitness()
                    # Check if swap is possible
                    if v_in not in state.set or v_in == v_out:
                        state.swap(v_out, v_in)
                        new_fitness = state.fitness()
                        
                        delta = new_fitness - old_fitness
                        if delta > 0 or random.random() < math.exp(delta / T):
                            if new_fitness > best_state.fitness():
                                best_state = State(list(state.points))
                        else:
                            # Revert swap: swap back
                            state.swap(v_in, v_out)
            else:
                # Invalid state, try to fix
                if r < 0.5:
                    # Remove
                    v = random.choice(state.points)
                    old_fitness = state.fitness()
                    state.remove(v)
                    new_fitness = state.fitness()
                    
                    delta = new_fitness - old_fitness
                    if delta > 0 or random.random() < math.exp(delta / T):
                        if new_fitness > best_state.fitness():
                            best_state = State(list(state.points))
                    else:
                        state.add(v) # Revert
                else:
                    # Swap
                    v_out = random.choice(state.points)
                    v_in = random.choice(pool)
                    
                    old_fitness = state.fitness()
                    if v_in not in state.set or v_in == v_out:
                        state.swap(v_out, v_in)
                        new_fitness = state.fitness()
                        
                        delta = new_fitness - old_fitness
                        if delta > 0 or random.random() < math.exp(delta / T):
                            if new_fitness > best_state.fitness():
                                best_state = State(list(state.points))
                        else:
                            state.swap(v_in, v_out)
                            
            T *= cooling
            
        return best_state.points

# EVOLVE-BLOCK-END
