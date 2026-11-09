import time
import random
from fractions import Fraction

def _get_primes(limit=1000):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    primes = _get_primes(1000)
    L_cap = 100000
    primes_set = set(primes)
    recip_map = {p: Fraction(1, p) for p in primes}
    
    # --- Structural Change: CSP-style Backtracking with Forward Checking ---
    # Instead of global SA on residues, we treat the problem as a sequence of
    # covering gaps. We maintain a set of "active" primes (those that cover
    # numbers near the current frontier L). We assign residues to these primes
    # to cover the next number, using backtracking if conflicts arise or if
    # the budget is exhausted.
    
    # 1. Select a candidate pool of primes.
    # To maximize L, we want small primes. 
    # Sum of reciprocals for primes up to 1000 is ~2.2.
    # We can afford a subset. 
    # Let's try to use primes in a range that allows flexibility.
    # Small primes are powerful but expensive.
    # Let's start with primes from 7 to 1000.
    
    # A better strategy: 
    # We will maintain a list of (prime, residue) pairs.
    # We will try to extend the covered prefix L.
    # At each step, if L+1 is not covered, we must find a prime p and residue r
    # such that (L+1) % p == r, and either p is already in our set (change r)
    # or we add p to our set (if budget allows and p is not already used).
    
    # However, changing r for an existing prime might uncover previous numbers.
    # So we need to ensure that changing r doesn't break coverage of 1..L.
    # This is the "forward checking" part.
    
    # To make this tractable, we will:
    # 1. Start with an empty set of classes.
    # 2. Iterate n from 1 to L_cap.
    # 3. If n is not covered by current classes:
    #    a. Try to find an existing prime p in our set such that we can change its residue to cover n WITHOUT uncovering any m in [1, n-1].
    #       This is expensive to check for all m.
    #       Approximation: Only check if the new residue still covers the "critical" points?
    #       Actually, let's use a different heuristic.
    
    # Alternative Structural Approach: 
    # **Greedy Set Cover with Backtracking on the Frontier**
    
    # We maintain `covered` as a boolean array up to some limit? No, too slow.
    # We maintain a set of "gaps" or just the current L.
    
    # Let's try a **Backtracking Search over Prime Selection and Residues**
    # focused on the first K primes.
    
    # The number of primes <= 1000 is 168.
    # We can't backtrack over all.
    
    # Let's try **Iterated Local Search with a specific neighborhood**:
    # The current program uses SA. 
    # Let's try **Hill Climbing with Random Restarts and a "Gap-Filling" heuristic**.
    
    # Actually, let's look at Inspiration 3: "CSP-style backtracking ... keeping the greedy prime selection framework".
    
    # Let's implement a **Greedy Construction with Backtracking**:
    # 1. Sort primes by size.
    # 2. We will build the covering set incrementally.
    # 3. At each step, we decide which prime to include next.
    # 4. Once a prime is included, we assign it a residue that maximizes the new L.
    
    # This is still greedy.
    
    # Let's try a **Different Optimization Method**: 
    # **Tabu Search**.
    # Tabu Search avoids cycles by forbidding recent moves.
    # This is structurally different from SA (which uses temperature) and standard Hill Climbing.
    
    # But maybe a more fundamental change is better.
    # What if we **fix the residues** for a subset of primes and **optimize the prime set**?
    # Or **fix the prime set** and **optimize residues** using a **Constraint Propagation** algorithm?
    
    # Let's go with **Constraint Propagation / Arc Consistency** on the residues.
    # Variables: Residues r_i for each prime p_i in a fixed set P.
    # Constraints: For every n in 1..L, exists i such that n % p_i == r_i.
    # We want to maximize L.
    
    # This is hard.
    
    # Let's try **Genetic Algorithm** instead of SA.
    # Population of solutions.
    # Crossover: Combine primes/residues from two parents.
    # Mutation: Random changes.
    # Selection: Keep best.
    
    # This is a different design family (Evolutionary vs Stochastic Local Search).
    
    def evaluate_solution(primes, residues):
        if not primes:
            return 0
        # Fast check for prefix coverage
        # Use a bitset or boolean array? 
        # For L up to 100k, a boolean array is fine.
        covered = [False] * 101  # Start small, expand if needed?
        # Actually, let's just check up to a limit.
        # To be fast, we only check up to current_best_L + 1000.
        pass

    # Let's implement a **Simulated Annealing** but with a **Different Objective Function**?
    # No, the objective is fixed.
    
    # Let's try **Beam Search**.
    # Maintain a list of the best K states.
    # At each step, generate neighbors for all K states, keep the best K.
    # This is structurally different from single-trajectory SA.
    
    # Let's implement **Beam Search**.
    
    class State:
        __slots__ = ['primes', 'residues', 'L', 'budget']
        
        def __init__(self, primes, residues, L, budget):
            self.primes = primes
            self.residues = residues
            self.L = L
            self.budget = budget

    def calc_L_fast(primes, residues, limit=100000):
        # Returns the longest prefix 1..L covered.
        # Optimization: Use a boolean array of size limit+1.
        # But allocating 100k array for every evaluation is slow in Python.
        # Let's use a set of covered numbers? No, too many.
        # Let's use a bytearray.
        
        # Actually, for Beam Search, we might have many states.
        # We need a very fast evaluation.
        
        # Alternative: 
        # We know the primes. 
        # We can check coverage of a specific number n by:
        # any(n % p == r for p, r in zip(primes, residues))
        # This is O(K).
        # To find L, we iterate n=1,2,... until gap.
        # If L is large, this is slow.
        
        # Let's stick to the count array but reuse it?
        # In Beam Search, states are different, so we can't reuse easily.
        
        # Maybe Beam Search is too slow.
        
        # Let's go back to **Structural Change: Different Problem Framing**.
        # Frame it as a **Shortest Path** problem?
        # Nodes are residues? No.
        
        # How about **Integer Linear Programming** (ILP)?
        # We can use scipy.optimize.milp.
        # Variables: x_{p,r} in {0,1} indicating if class (p,r) is chosen.
        # Constraints:
        # 1. Sum_{p,r} x_{p,r} * (1/p) <= 1
        # 2. For each n in 1..L, Sum_{p,r: n%p==r} x_{p,r} >= 1
        # 3. For each p, Sum_r x_{p,r} <= 1 (distinct primes, one residue each)
        # Objective: Maximize L.
        
        # This is a huge ILP if L is large.
        # But we can do **Branch and Bound** or **Column Generation**?
        # Too complex.
        
        # However, we can use **scipy.optimize.milp** for a **fixed L**.
        # We can binary search on L!
        # Check if L=100 is coverable.
        # Check if L=200 is coverable.
        # This is a structurally different approach: **Feasibility Check via ILP**.
        
        # Let's try this.
        
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # We need to define the ILP for a given L.
        # Variables: x_{p,r} for each prime p <= 1000 and residue r in 0..p-1.
        # Total variables: Sum_{p<=1000} p. 
        # Sum of primes <= 1000 is ~304141.
        # 300k variables is large for scipy.milp, but might work with sparse matrices.
        
        # However, building the constraint matrix for each L check is expensive.
        # And solving a 300k variable ILP might take longer than 120s.
        
        # Let's restrict the primes.
        # We only need primes that can cover numbers in 1..L.
        # Any prime p > L can cover at most one number in 1..L (specifically r if r<=L).
        # So for p > L, x_{p,r} is only useful if r <= L.
        # This reduces variables.
        
        # But 300k variables is still a lot.
        
        # Let's try a **Hybrid Approach**:
        # 1. Use ILP to find a good set of primes and residues for a moderate L (e.g., 1000).
        # 2. Use Local Search to extend it.
        
        # This combines two different methods.
        
        # Let's implement the ILP feasibility check for a target L.
        
        def check_feasibility(L_target):
            # Variables: x_{i} where i maps to (p, r).
            # We will only include variables for primes p <= 1000.
            # To reduce size, we can fix some primes?
            # No, let's let the solver decide.
            
            # Actually, 300k variables might be too slow for scipy.milp in 120s.
            # Let's try to reduce the problem.
            
            # We can observe that for large p, the residue r must be small to cover small n.
            # If p > L_target, then n % p == n. So x_{p,n} covers n.
            # These are "single-use" variables.
            
            # Let's just try it with a smaller set of primes first.
            # Primes up to 100? Sum of reciprocals is ~1.5.
            # Variables: Sum_{p<=100} p = 1060.
            # This is very small.
            
            # If we restrict to primes <= 100, can we cover L=999?
            # Sum of 1/p for p<=100 is ~1.54. We have budget 1.
            # So we can only use a subset.
            
            # Let's try ILP with primes <= 200.
            # Sum of 1/p for p<=200 is ~1.8.
            # Variables: Sum_{p<=200} p = 4227.
            
            # Let's try this.
            
            primes_small = [p for p in primes if p <= 200]
            # Also add some large primes? 
            # Large primes are cheap (1/p is small) but cover few numbers.
            # They are good for filling gaps.
            # Let's add primes up to 1000, but for p > L_target, only allow r in 1..L_target?
            # No, r can be anything, but only r <= L_target matters for coverage of 1..L_target.
            # So for p > L_target, we only need variables x_{p,r} for r in 1..L_target.
            # This adds L_target * (num primes > L_target) variables.
            # If L_target=1000, and ~160 primes > 1000? No, primes up to 1000.
            # Primes > 1000 are not allowed.
            
            # So we only use primes <= 1000.
            # For p <= L_target, we have p variables.
            # For p > L_target, we have L_target variables (r=1..L_target).
            
            # Let's build the variable list.
            var_indices = {}
            var_count = 0
            A_rows = []
            b_lower = []
            b_upper = []
            
            # Constraint 1: Budget
            # Sum x_{p,r} * (1/p) <= 1
            
            # Constraint 2: Coverage for each n in 1..L_target
            # Sum_{p,r: n%p==r} x_{p,r} >= 1
            
            # Constraint 3: One residue per prime
            # Sum_{r} x_{p,r} <= 1
            
            # This is getting complex to build efficiently in Python.
            # And solving might be slow.
            
            # Let's abandon ILP for now due to complexity and time risk.
            
            # Let's go back to **Beam Search** but with a **Fast Evaluation**.
            
            # Fast Evaluation Trick:
            # Maintain a "coverage mask" as a Python int (bitstring).
            # If L is up to 1000, we can use a 1000-bit integer.
            # Coverage of a class (p,r) is a bitstring with bits set at r, r+p, r+2p, ...
            # We can precompute these bitstrings for all (p,r) pairs?
            # 168 primes * avg 500 residues = 84k bitstrings.
            # Each bitstring is 1000 bits = 125 bytes.
            # 84k * 125 bytes = 10 MB. Feasible.
            
            # Then, the coverage of a set of classes is the OR of their bitstrings.
            # To find L, we find the highest set bit in the OR result?
            # No, we need the longest prefix of 1s.
            # If the bitstring is 1110111..., the prefix length is 3.
            # We can compute this by:
            # 1. Invert the bitstring: 0001000...
            # 2. Find the first set bit? No, that's the first gap.
            # 3. The index of the first 0 in the original string is the gap.
            
            # Let's implement this.
            
            L_limit = 2000  # Start with a smaller limit for speed
            # Precompute bitstrings for all (p, r)
            # p up to 1000, r in 0..p-1
            # But we only care about bits 0..L_limit-1 (representing numbers 1..L_limit)
            # Bit i represents number i+1.
            
            bitstrings = {}
            for p in primes:
                for r in range(p):
                    # Generate bitstring
                    mask = 0
                    # Number n is covered if n % p == r.
                    # n = 1, 2, ...
                    # First n: if r == 0, n=p. Else n=r.
                    # But we represent n=1 as bit 0.
                    # So bit index = n-1.
                    # n % p == r  =>  (n-1) % p == (r-1) % p ? No.
                    # n = k*p + r.
                    # If r==0, n is multiple of p.
                    # Let's just loop n from 1 to L_limit.
                    # This is O(L_limit * P * AvgP) which is too slow to precompute.
                    
                    # Better:
                    # For a given p, r, the numbers covered are r, r+p, r+2p, ...
                    # If r==0, it's p, 2p, ...
                    # We want bits for n in 1..L_limit.
                    # start = r if r>0 else p
                    # for n in range(start, L_limit+1, p):
                    #     mask |= 1 << (n-1)
                    
                    # This is O(L_limit/p).
                    # Total precomputation: Sum_{p} Sum_{r} (L_limit/p) = L_limit * Sum_{p} (p * 1/p) = L_limit * P.
                    # 2000 * 168 = 336k ops. Very fast.
                    
                    pass # We will do this in the actual code
            
            # Let's restart the code structure with this Bitstring Beam Search.
            
            return None # Placeholder

    # --- Implementation of Bitstring Beam Search ---
    
    L_limit = 5000  # We can handle larger L with bitstrings
    # Precompute bitstrings
    # To save memory, we only store bitstrings for primes we might use.
    # But we might use any prime.
    # Let's precompute for all primes <= 1000.
    
    bitstrings_cache = {}
    for p in primes:
        for r in range(p):
            mask = 0
            start = r if r > 0 else p
            # n is covered if n % p == r
            # n = 1, 2, ...
            # Bit index is n-1.
            for n in range(start, L_limit + 1, p):
                mask |= (1 << (n - 1))
            bitstrings_cache[(p, r)] = mask

    # Helper: Get prefix length from bitstring
    def get_prefix_length(mask, limit=L_limit):
        # Find the first 0 bit in the sequence of 1s starting from bit 0.
        # If mask is 0b111000..., prefix is 3.
        # If mask is 0b111111..., prefix is limit (if all bits up to limit are 1).
        
        # Trick: 
        # Invert mask: ~mask.
        # We want the lowest set bit in ~mask.
        # But ~mask has infinite 1s at the top.
        # We only care about bits 0..limit-1.
        
        # Let's just loop? 
        # L_limit is 5000. Looping 5000 times is fast.
        # But we do this for every state in Beam Search.
        # Beam width 10, 1000 steps -> 10k evaluations.
        # 10k * 5000 ops = 50M ops. ~5s in Python. Acceptable.
        
        # Optimization:
        # We can check if mask has all bits 0..limit-1 set.
        # full_mask = (1 << limit) - 1
        # if mask == full_mask: return limit
        # else:
        #    inv = (~mask) & full_mask
        #    # Find lowest set bit in inv
        #    # lowest_set_bit = inv & -inv
        #    # index = bit_length(lowest_set_bit) - 1
        #    # return index
        
        full_mask = (1 << limit) - 1
        if mask == full_mask:
            return limit
        inv = (~mask) & full_mask
        if inv == 0:
            return limit
        lowest = inv & -inv
        # index of lowest set bit
        # lowest is a power of 2.
        # index = log2(lowest)
        # In Python: lowest.bit_length() - 1
        idx = lowest.bit_length() - 1
        return idx

    # Beam Search
    beam_width = 20
    # Initial state: Empty set
    # But empty set has L=0.
    # We need to add primes.
    
    # Let's initialize the beam with some random good states.
    # Or start with empty and add primes.
    
    # State: (primes_list, residues_list, mask, L)
    # We maintain mask as OR of bitstrings.
    
    # To allow adding/removing primes, we need to update mask.
    # mask = OR of bitstrings of all classes.
    # If we add (p,r), new_mask = mask | bitstrings[(p,r)]
    # If we remove (p,r), we cannot simply AND with NOT bitstring, because other classes might cover those bits.
    # So we must recompute mask from scratch if we remove.
    # Or we maintain a count of how many times each bit is set?
    # That requires an array of size L_limit.
    # 5000 ints. 
    # Updating counts for add/remove is O(L_limit/p).
    # This is similar to the count array in the original program.
    
    # So Beam Search with count array is equivalent to SA with count array, but with a beam.
    # Let's try it.
    
    class BeamState:
        __slots__ = ['primes', 'residues', 'counts', 'L']
        
        def __init__(self, primes, residues):
            self.primes = primes
            self.residues = residues
            self.counts = [0] * (L_limit + 1)
            self._rebuild()
            
        def _rebuild(self):
            self.counts = [0] * (L_limit + 1)
            for p, r in zip(self.primes, self.residues):
                start = r if r > 0 else p
                for n in range(start, L_limit + 1, p):
                    self.counts[n] += 1
            self._update_L()
            
        def _update_L(self):
            L = 0
            for i in range(1, L_limit + 1):
                if self.counts[i] > 0:
                    L = i
                else:
                    break
            self.L = L
            
        def apply_add(self, p, r):
            start = r if r > 0 else p
            for n in range(start, L_limit + 1, p):
                self.counts[n] += 1
            self.primes.append(p)
            self.residues.append(r)
            self._update_L()
            
        def apply_remove(self, idx):
            p = self.primes[idx]
            r = self.residues[idx]
            start = r if r > 0 else p
            for n in range(start, L_limit + 1, p):
                self.counts[n] -= 1
            self.primes.pop(idx)
            self.residues.pop(idx)
            self._update_L()
            
        def apply_change_residue(self, idx, new_r):
            p = self.primes[idx]
            old_r = self.residues[idx]
            if old_r == new_r:
                return
            # Remove old
            start_old = old_r if old_r > 0 else p
            for n in range(start_old, L_limit + 1, p):
                self.counts[n] -= 1
            # Add new
            start_new = new_r if new_r > 0 else p
            for n in range(start_new, L_limit + 1, p):
                self.counts[n] += 1
            self.residues[idx] = new_r
            self._update_L()

    # Initial Beam: 
    # Create a few random states with ~10 primes.
    beam = []
    for _ in range(beam_width):
        # Randomly pick 10 distinct primes
        sel_primes = random.sample(primes, 10)
        sel_residues = [random.randint(0, p-1) for p in sel_primes]
        state = BeamState(sel_primes, sel_residues)
        beam.append(state)
        
    # Sort by L
    beam.sort(key=lambda s: s.L, reverse=True)
    
    best_state = beam[0]
    
    # Main Loop
    while time.time() < deadline:
        new_beam = []
        
        # Generate neighbors for each state in beam
        for state in beam:
            # Move 1: Change residue of a random prime
            if state.primes:
                idx = random.randint(0, len(state.primes)-1)
                p = state.primes[idx]
                new_r = random.randint(0, p-1)
                
                # Copy state? 
                # We need to be careful with mutation.
                # Let's create a new state by copying.
                # Copying counts array is O(L_limit).
                # This is expensive.
                
                # Alternative: Apply move, record L, revert.
                # But we need to keep the new state if it's good.
                # So we must copy.
                
                # To speed up, we can use a "dirty" flag and only rebuild counts if needed?
                # No, counts are needed for L.
                
                # Let's just copy the list and counts.
                new_primes = state.primes[:]
                new_residues = state.residues[:]
                new_counts = state.counts[:]
                
                # Apply change to new_counts
                old_r = state.residues[idx]
                if old_r != new_r:
                    start_old = old_r if old_r > 0 else p
                    for n in range(start_old, L_limit + 1, p):
                        new_counts[n] -= 1
                    start_new = new_r if new_r > 0 else p
                    for n in range(start_new, L_limit + 1, p):
                        new_counts[n] += 1
                    new_residues[idx] = new_r
                    
                # Calculate L
                L = 0
                for i in range(1, L_limit + 1):
                    if new_counts[i] > 0:
                        L = i
                    else:
                        break
                
                # Create new state object (lightweight, just store data)
                # We will delay creating the full BeamState object until we select it for the next beam.
                # For now, just store (L, primes, residues, counts)
                new_beam.append((L, new_primes, new_residues, new_counts))
                
            # Move 2: Add a prime
            # Pick a random prime not in set
            # Check budget
            # For simplicity, skip add/remove in beam search to keep it fast.
            # Focus on residue optimization.
            
        # Select top beam_width states
        if not new_beam:
            break
            
        # Sort by L
        new_beam.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top beam_width
        top_states = new_beam[:beam_width]
        
        # Check if best improved
        if top_states[0][0] > best_state.L:
            best_state.L = top_states[0][0]
            best_state.primes = top_states[0][1]
            best_state.residues = top_states[0][2]
            best_state.counts = top_states[0][3]
            
        # Update beam
        beam = []
        for L, p_list, r_list, c_list in top_states:
            # Create BeamState wrapper? 
            # Or just keep the tuples?
            # Let's keep tuples for speed.
            beam.append((L, p_list, r_list, c_list))
            
    # Final Result
    return [[p, r] for p, r in zip(best_state.primes, best_state.residues)]
