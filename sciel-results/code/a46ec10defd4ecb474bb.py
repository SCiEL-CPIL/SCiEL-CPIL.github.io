"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure
    # lit_bits[v][p] is a bitset of clauses containing literal (v, p)
    # v is 0-indexed, p is 0 (False) or 1 (True)
    lit_bits = [[0, 0] for _ in range(n)]
    clause_vars = []  # list of lists of (v, p)
    
    for ci, clause in enumerate(clauses):
        bit = 1 << ci
        cv = []
        for l in clause:
            v = abs(l) - 1
            p = 1 if l > 0 else 0
            lit_bits[v][p] |= bit
            cv.append((v, p))
        clause_vars.append(cv)

    def popcount(x):
        return bin(x).count('1')

    def initialize_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_bits = 0
        satisfied_by = [[0, 0] for _ in range(n)]
        
        for ci, cv in enumerate(clause_vars):
            bit = 1 << ci
            sat_lits = []
            is_sat = False
            for v, p in cv:
                if assignment[v] == (p == 1):
                    is_sat = True
                    sat_lits.append((v, p))
            
            if not is_sat:
                unsat_bits |= bit
            else:
                if len(sat_lits) == 1:
                    v, p = sat_lits[0]
                    satisfied_by[v][p] |= bit
                # If len > 1, no single variable is exclusively satisfying
        
        return assignment, unsat_bits, satisfied_by

    def flip_variable(v, new_p, assignment, unsat_bits, satisfied_by):
        old_p = 1 - new_p
        
        # Clauses that become unsatisfied: those exclusively satisfied by (v, old_p)
        newly_unsat = satisfied_by[v][old_p]
        
        # Clauses that become satisfied: those currently unsat and containing (v, new_p)
        newly_sat = unsat_bits & lit_bits[v][new_p]
        
        # Update unsat_bits
        # Add newly_unsat, remove newly_sat
        unsat_bits = (unsat_bits | newly_unsat) & ~newly_sat
        
        # Update satisfied_by
        # 1. Clear the old exclusive satisfaction for (v, old_p)
        #    These clauses are now unsat, so they are no longer in satisfied_by
        satisfied_by[v][old_p] &= ~newly_unsat
        
        # 2. Determine new exclusive satisfaction for (v, new_p)
        #    We need to check clauses in newly_sat.
        #    A clause c in newly_sat is exclusively satisfied by (v, new_p) if no other literal in c is true.
        new_exclusive = 0
        # Iterate through set bits in newly_sat
        temp_ns = newly_sat
        while temp_ns:
            # Get index of least significant set bit
            lsb = temp_ns & -temp_ns
            ci = lsb.bit_length() - 1
            temp_ns ^= lsb
            
            # Check if any other literal in clause ci is true
            is_exclusive = True
            for u, p in clause_vars[ci]:
                if u == v and p == new_p:
                    continue
                # Check if literal (u, p) is true in current assignment
                # Note: assignment[v] is not yet updated to new_p, but for u != v it is current.
                # For u == v and p != new_p, it is false (since we are flipping to new_p).
                # So we just check assignment[u] == (p == 1)
                if assignment[u] == (p == 1):
                    is_exclusive = False
                    break
            
            if is_exclusive:
                new_exclusive |= lsb
        
        satisfied_by[v][new_p] |= new_exclusive
        
        # Update assignment
        assignment[v] = (new_p == 1)
        
        return unsat_bits, satisfied_by

    rng = random.Random(42)
    
    best_unsat_count = float('inf')
    best_assignment = None
    
    # Restart strategy
    max_restarts = 50
    steps_per_restart = 50000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment, unsat_bits, satisfied_by = initialize_state(rng)
        current_unsat = popcount(unsat_bits)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_bits == 0:
                return assignment
            
            # If we found a better state, save it
            # (Optional: frequent checking is expensive, check every 1000 steps?)
            # For now, just check if we hit 0.
            
            # Get list of unsatisfied clauses
            # Instead of converting to list every time, we can pick a random bit from unsat_bits
            # But we need the clause index to get the variables.
            # Let's pick a random unsatisfied clause index.
            # We can iterate through bits or just pick a random index and check if it's unsat?
            # Better: convert unsat_bits to a list of indices? 
            # popcount is small usually. 
            # To be fast, let's just pick a random variable from the unsat clauses.
            
            # Strategy: Pick a random unsatisfied clause.
            # To pick a random set bit:
            # x = unsat_bits
            # count = popcount(x)
            # k = rng.randint(0, count-1)
            # Then find the k-th set bit. This is O(n) in worst case if done naively.
            # Alternative: Just iterate through all clauses and pick a random unsat one? O(m).
            # With m=450, O(m) is fine if we do it once per step? 
            # 50k steps * 450 checks = 22.5M ops. ~2-3 seconds. Acceptable.
            
            # Let's try to be smarter: maintain a list of unsat clauses? 
            # Updating the list is hard.
            # Let's stick to picking a random clause from the unsat_bits.
            
            # Efficient random set bit selection:
            # We can use a precomputed table or just a loop.
            # Given m is small, let's just do:
            unsat_clauses = []
            temp_ubit = unsat_bits
            while temp_ubit:
                lsb = temp_ubit & -temp_ubit
                ci = lsb.bit_length() - 1
                unsat_clauses.append(ci)
                temp_ubit ^= lsb
            
            if not unsat_clauses:
                return assignment
                
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Pick a literal to flip
            # Min-conflicts heuristic:
            best_delta = float('inf')
            best_v = -1
            best_p = -1
            
            for v, p in cv:
                new_p = p
                old_p = 1 - p
                # Calculate delta
                # Clauses becoming unsat: satisfied_by[v][old_p]
                # Clauses becoming sat: unsat_bits & lit_bits[v][new_p]
                delta = popcount(satisfied_by[v][old_p]) - popcount(unsat_bits & lit_bits[v][new_p])
                
                # Add some noise to escape local minima
                if rng.random() < 0.1:
                    delta += rng.gauss(0, 1)
                
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
                    best_p = new_p
            
            # Flip
            unsat_bits, satisfied_by = flip_variable(best_v, best_p, assignment, unsat_bits, satisfied_by)
            
            if unsat_bits == 0:
                return assignment

    # Return best found
    if best_assignment is None:
        return [False] * n
    return best_assignment
# EVOLVE-BLOCK-END
