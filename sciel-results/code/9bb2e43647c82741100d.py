import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    N = 256

    # Precompute APs
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(np.array(indices, dtype=np.int32))
                
    num_aps = len(ap_indices)

    # Try to use pysat for Max-SAT formulation
    try:
        from pysat.solvers import Solver
        from pysat.formula import CNF
        use_pysat = True
    except ImportError:
        use_pysat = False

    if use_pysat:
        # Max-SAT formulation
        # Variables: 1..N represent signs (1 = +1, 0 = -1)
        # For each AP r, we want |sum| <= D
        # sum = sum_{i in AP_r} x_i - (len(AP_r) - sum_{i in AP_r} x_i)
        #     = 2*sum_{i in AP_r} x_i - len(AP_r)
        # So |2*S_r - L_r| <= D where S_r = sum of x_i in AP_r, L_r = len(AP_r)
        
        # This is equivalent to:
        # L_r - D <= 2*S_r <= L_r + D
        # ceil((L_r - D)/2) <= S_r <= floor((L_r + D)/2)
        
        # We'll use a binary counter for each AP sum
        # For an AP of length L, S_r ranges from 0 to L
        # We need log2(L+1) bits to represent S_r
        
        # Actually, let's use a simpler approach:
        # For each AP, introduce auxiliary variables to compute the sum
        # Then enforce the constraint
        
        def try_discrepancy(D, time_budget):
            """Try to find assignment with max discrepancy <= D"""
            cnf = CNF()
            
            # Variables 1..N: sign of position i (1 = +1, 0 = -1)
            # For each AP, we'll use a binary adder circuit
            
            var_counter = N + 1
            
            # For each AP, build a sum constraint
            # We'll use a ripple-carry adder approach
            
            for r, indices in enumerate(ap_indices):
                L = len(indices)
                
                # We need to compute S = sum of x_i for i in indices
                # S ranges from 0 to L
                # We need: ceil((L-D)/2) <= S <= floor((L+D)/2)
                
                lo = max(0, (L - D + 1) // 2)  # ceil((L-D)/2)
                hi = min(L, (L + D) // 2)       # floor((L+D)/2)
                
                if lo > hi:
                    return None  # Infeasible
                
                # If lo == 0 and hi == L, no constraint needed
                if lo == 0 and hi == L:
                    continue
                
                # Build binary representation of S
                # We need log2(L+1) bits
                num_bits = max(1, L.bit_length())
                
                # Create bits for this sum: b_0, b_1, ..., b_{num_bits-1}
                bits = []
                for b in range(num_bits):
                    bits.append(var_counter)
                    var_counter += 1
                
                # Build the adder: S = x_{i1} + x_{i2} + ... + x_{iL}
                # Use sequential addition with carry bits
                
                # Start with sum = 0
                # Add each x_i one by one
                
                # Current sum bits: start with all 0
                current_bits = [0] * num_bits  # 0 means "not yet assigned"
                
                for idx in indices:
                    # Add x_idx to current sum
                    # This requires a full adder for each bit position
                    
                    # New sum bits
                    new_bits = []
                    carry_in = 0
                    
                    for b in range(num_bits):
                        if b == 0:
                            # LSB: sum = (current_bit_0 + x_idx + carry_in) mod 2
                            # carry_out = (current_bit_0 + x_idx + carry_in) // 2
                            pass
                        
                        # This is getting complex. Let me use a simpler approach.
                        break
                    
                    # Alternative: use pysat's built-in cardinality constraints
                    # or just encode the sum directly using auxiliary variables
                    
                    break
                
                # This is getting too complex. Let me use a different approach.
                # Instead of building adders, let's use the fact that we can
                # encode "sum >= K" and "sum <= K" using cardinality constraints.
                
                # pysat has support for at-least and at-most constraints
                # For "sum of variables in set >= K", we can use:
                #   not (sum < K) which is not (at most K-1)
                
                # Actually, pysat.solvers.Solver has methods for this.
                # Let me use a simpler encoding.
                
                # For each AP, we'll just add soft constraints and let the solver
                # minimize the maximum violation. But that's not quite right.
                
                # Let me try a different approach: use the solver's built-in
                # support for linear constraints if available, or use a
                # simpler heuristic.
                
                return None  # Skip this complex encoding for now
            
            # This approach is getting too complex. Let me fall back to a
            # simpler method: use the SAT solver to find a valid assignment
            # for a given D by encoding the constraints differently.
            
            return None

        # The above is too complex. Let me use a simpler approach:
        # Binary search on D, and for each D, use a SAT solver with
        # carefully encoded constraints.
        
        # Actually, let me just use a hybrid: use pysat for the core SAT
        # part but with a simpler constraint encoding.
        
        def solve_with_pysat(D, time_budget):
            """Solve for assignment with max discrepancy <= D"""
            cnf = CNF()
            
            # Variables 1..N: sign of position i (1 = +1, 0 = -1)
            
            # For each AP, we need |sum| <= D
            # sum = 2*S - L where S = number of +1s in the AP
            # So |2*S - L| <= D
            # Which means: L - D <= 2*S <= L + D
            # i.e., ceil((L-D)/2) <= S <= floor((L+D)/2)
            
            # We'll encode "S >= K" and "S <= K" using cardinality constraints
            # pysat supports these via the solver's built-in methods
            
            solver = Solver(bootstrap_with=cnf)
            
            # Add constraints for each AP
            for r, indices in enumerate(ap_indices):
                L = len(indices)
                lo = max(0, (L - D + 1) // 2)
                hi = min(L, (L + D) // 2)
                
                if lo > hi:
                    solver.close()
                    return None
                
                if lo == 0 and hi == L:
                    continue
                
                # Encode S >= lo: at least lo of the variables are true
                # Encode S <= hi: at most hi of the variables are true
                
                # pysat's Solver has at_least and at_most methods
                # But we need to use them correctly
                
                # For at_least K: not (at most K-1)
                # For at_most K: use the built-in
                
                if lo > 0:
                    # At least lo variables must be true
                    # This is equivalent to: not (at most lo-1)
                    # We can use the dual: at least lo is true
                    clauses = []
                    # Simple encoding: for each subset of size lo, at least one must be true
                    # But this is exponential. Use a better encoding.
                    
                    # Use a sequential counter or just use the solver's built-in
                    # if available
                    
                    # For now, skip complex encoding and use a simpler heuristic
                    pass
                
                if hi < L:
                    # At most hi variables can be true
                    # This is equivalent to: not (at least hi+1)
                    pass
            
            solver.close()
            return None

        # The SAT encoding is too complex to do correctly in this time.
        # Let me use a different strategy: use pysat for a simpler problem
        # or fall back to improved local search.
        
        # Actually, let me try a much simpler approach:
        # Use the SAT solver to find ANY valid assignment for small D,
        # starting from D=1 and increasing.
        
        # But even that requires proper encoding. Let me just use an
        # improved local search with better initialization.
        
        pass

    # Fall back to improved local search
    # Use Thue-Morse as initial guess, then do hill climbing
    
    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            seq.append(1 if count % 2 == 0 else -1)
        return np.array(seq, dtype=np.int8)

    def compute_sums(s):
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
        return sums

    def max_abs_sum(s):
        sums = compute_sums(s)
        return int(np.max(np.abs(sums)))

    # Start with Thue-Morse
    s = thue_morse(N)
    best_s = s.copy()
    best_score = max_abs_sum(s)
    
    # Precompute aps_containing for fast updates
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)
    
    aps_containing_arr = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Hill climbing with multiple restarts
    rng = random.Random(42)
    t_start = time.time()
    
    for restart in range(20):
        if time.time() - t_start > time_limit:
            break
        
        # Initialize: Thue-Morse for first restart, random for others
        if restart == 0:
            s = thue_morse(N)
        else:
            s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        
        if current_max < best_score:
            best_score = current_max
            best_s = s.copy()
        
        # Local search
        improved = True
        while improved and (time.time() - t_start < time_limit):
            improved = False
            best_delta = 0
            best_idx = -1
            best_new_max = current_max
            
            # Try flipping each index
            for i in range(N):
                delta = -2 * s[i]
                affected = aps_containing_arr[i]
                
                old_vals = sums[affected].copy()
                sums[affected] += delta
                
                new_max = int(np.max(np.abs(sums)))
                
                sums[affected] = old_vals
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_idx = i
                    best_delta = delta
                    improved = True
            
            if improved:
                s[best_idx] = -s[best_idx]
                affected = aps_containing_arr[best_idx]
                sums[affected] += best_delta
                current_max = best_new_max
                
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
        
        # Perturb for next restart
        if restart < 19:
            num_flips = rng.randint(1, 5)
            for _ in range(num_flips):
                idx = rng.randint(0, N - 1)
                s[idx] = -s[idx]
    
    return [int(x) for x in best_s]
