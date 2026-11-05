"""Planted 3-SAT solver (50 vars) using Simulated Annealing and Greedy Polish.

This solver models the 3-SAT problem as a Max-SAT optimization problem, 
minimizing the number of unsatisfied clauses (energy). 
It employs Simulated Annealing to escape local minima, followed by a 
greedy Min-Conflict polish phase to refine near-solutions.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses_in = instance["clauses"]

    # Deadline guard: return best found with a safety margin before the kill.
    # The evaluator kills after 330s. We use 290s to be safe.
    deadline = time.monotonic() + 290.0

    # Precompute per-literal variable index and clause polarity.
    # clauses: list of (var, sign) with var in 0..n-1, sign in {+1,-1}
    clauses = []
    for c in clauses_in:
        clauses.append(tuple((abs(l) - 1, 1 if l > 0 else -1) for l in c))
    m = len(clauses)

    # Build variable -> parallel lists of (clause index, position in clause).
    # This allows O(1) access to clauses containing a variable for energy calculation.
    var_clauses = [[] for _ in range(n)]
    var_positions = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for idx, (v, s) in enumerate(cl):
            var_clauses[v].append(ci)
            var_positions[v].append(idx)

    def build_state(assign):
        """
        Calculate initial state: 
        clause_sat: count of satisfied literals per clause
        sat_lit: boolean list of lists, sat_lit[ci][idx] is True if literal is satisfied
        unsat_count: total unsatisfied clauses
        """
        clause_sat = [0] * m
        sat_lit = []
        unsat_count = 0
        for ci, cl in enumerate(clauses):
            cnt = 0
            sl = []
            for v, s in cl:
                ok = assign[v] == (s > 0)
                sl.append(ok)
                if ok:
                    cnt += 1
            sat_lit.append(sl)
            clause_sat[ci] = cnt
            if cnt == 0:
                unsat_count += 1
        return clause_sat, sat_lit, unsat_count

    def delta_energy(v, assign, clause_sat, sat_lit):
        """
        Calculate the change in energy (number of unsatisfied clauses) if variable v is flipped.
        Energy = number of unsatisfied clauses.
        
        For each clause cj containing v:
        - If v currently satisfies cj:
            - If clause_sat[cj] == 1: Flipping v makes cj unsat. Energy increases by 1.
            - If clause_sat[cj] > 1: Flipping v leaves cj satisfied (by other literals). Energy unchanged.
        - If v currently does NOT satisfy cj:
            - Flipping v makes v satisfy cj.
            - If clause_sat[cj] == 0: cj becomes satisfied. Energy decreases by 1.
            - If clause_sat[cj] > 0: cj was already satisfied. Energy unchanged.
        """
        c_list = var_clauses[v]
        p_list = var_positions[v]
        d = 0
        for k in range(len(c_list)):
            cj = c_list[k]
            idx = p_list[k]
            was_sat = sat_lit[cj][idx]
            if was_sat:
                # v is currently satisfying this clause
                if clause_sat[cj] == 1:
                    d += 1  # Clause becomes unsat
            else:
                # v is currently NOT satisfying this clause
                if clause_sat[cj] == 0:
                    d -= 1  # Clause becomes sat
        return d

    def flip(v, assign, clause_sat, sat_lit):
        """Perform the flip and update state."""
        c_list = var_clauses[v]
        p_list = var_positions[v]
        for k in range(len(c_list)):
            cj = c_list[k]
            idx = p_list[k]
            was_sat = sat_lit[cj][idx]
            if was_sat:
                clause_sat[cj] -= 1
            else:
                clause_sat[cj] += 1
            sat_lit[cj][idx] = not was_sat
        assign[v] = not assign[v]

    rng = random.Random(42)
    
    best_assign = None
    best_unsat = m + 1

    # Simulated Annealing Parameters
    # We use multiple restarts. Each run has a specific temperature schedule.
    # T_start is chosen such that initial moves are mostly accepted.
    # For 3-SAT with 50 vars, typical unsat counts are around 10-30.
    # Delta is usually -1, 0, or 1.
    # P(accept) = exp(-delta / T). If delta=1, T=1 -> P=0.37. T=0.1 -> P=0.004.
    
    num_restarts = 50
    t0 = time.monotonic()
    
    for r in range(num_restarts):
        if time.monotonic() > deadline:
            break
            
        # Initialize random assignment
        assign = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, sat_lit, unsat_count = build_state(assign)
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign[:]
            
        if unsat_count == 0:
            break

        # Simulated Annealing Loop
        # We use a geometric cooling schedule.
        # T_start: high enough to accept many worsening moves.
        # T_end: low enough to freeze.
        # Steps per restart: Adaptive based on time remaining.
        
        # Estimate time per step. Let's aim for ~10,000 steps per restart if time permits.
        # 50 restarts * 10,000 steps = 500,000 steps.
        # Each step is O(degree) ~ O(10-20) operations. Very fast in Python if optimized, 
        # but let's be conservative. 
        # 300s / 50 restarts = 6s per restart.
        # 6s / 10,000 steps = 0.6ms per step. This is feasible.
        
        steps_per_restart = 10000
        T_start = 5.0
        T_end = 0.001
        # Cooling factor: T_end = T_start * alpha^steps
        # alpha = (T_end/T_start)^(1/steps)
        if steps_per_restart > 0:
            alpha = (T_end / T_start) ** (1.0 / steps_per_restart)
        else:
            alpha = 0.99

        T = T_start
        
        for step in range(steps_per_restart):
            if time.monotonic() > deadline:
                break
            
            # Pick a random variable to flip
            v = rng.randrange(n)
            
            # Calculate delta energy
            dE = delta_energy(v, assign, clause_sat, sat_lit)
            
            # Metropolis criterion
            if dE <= 0:
                # Always accept improving or neutral moves
                accept = True
            else:
                # Accept worsening moves with probability exp(-dE / T)
                # Use log to avoid underflow for small T, but T is not too small initially
                if T > 0.001:
                    # dE is usually 1 or 2. 
                    # If T is small, exp(-dE/T) is very small.
                    accept = rng.random() < math.exp(-dE / T)
                else:
                    accept = False
            
            if accept:
                flip(v, assign, clause_sat, sat_lit)
                unsat_count += dE
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assign[:]
                
                if unsat_count == 0:
                    break
            
            # Cool down
            T *= alpha
            
            # Periodic time check
            if step % 1024 == 0 and time.monotonic() > deadline:
                break

        if unsat_count == 0:
            break

    # Final Polish Phase: Greedy Min-Conflict Descent
    # If we haven't found a solution, try to polish the best assignment found.
    if best_assign is not None and best_unsat > 0:
        assign = best_assign[:]
        clause_sat, sat_lit, unsat_count = build_state(assign)
        
        # Greedy descent: repeatedly flip the variable that reduces unsat count the most.
        # If no single flip reduces unsat count, stop (local minimum).
        max_polish_steps = 10000
        polished = False
        
        for _ in range(max_polish_steps):
            if time.monotonic() > deadline:
                break
            
            if unsat_count == 0:
                polished = True
                break
                
            best_v = -1
            best_dE = 0  # We only want to accept if dE < 0
            
            # To speed up, we can just check a few random variables or all variables?
            # With n=50, checking all 50 variables is cheap (50 * O(degree) ~ 500 ops).
            # This is a full scan for the best move.
            
            for v in range(n):
                dE = delta_energy(v, assign, clause_sat, sat_lit)
                if dE < best_dE:
                    best_dE = dE
                    best_v = v
            
            if best_v == -1:
                # No improving move found. Stuck in local minimum.
                break
            
            # Apply the best move
            flip(best_v, assign, clause_sat, sat_lit)
            unsat_count += best_dE
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign[:]
                
        if unsat_count == 0:
            polished = True

    if best_assign is None:
        best_assign = [False] * n
        
    return best_assign
# EVOLVE-BLOCK-END
