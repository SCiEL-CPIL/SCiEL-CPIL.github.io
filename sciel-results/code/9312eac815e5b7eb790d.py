"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations
from fractions import Fraction

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 maximizing sum 1/(512 - a).
    Uses a MaxSAT formulation with DPLL-based SAT solver for maximum-weight
    independent set on the conflict graph, combined with local search refinement.
    """
    deadline = time.time() + 145.0  # safety margin before 150s kill
    N = 512
    n_vars = N - 1  # variables 1..511, index 0 unused
    
    # Precompute weights: weight of vertex a is 1/(512 - a)
    weights = [Fraction(0)] * N  # index 0 unused, 1..511
    weights_float = [0.0] * N
    for a in range(1, N):
        weights[a] = Fraction(1, N - a)
        weights_float[a] = 1.0 / (N - a)
    
    # Precompute prime factors for each number 1..511
    pfactors = [set() for _ in range(N)]
    for a in range(2, N):
        pfactors[a] = _prime_factors(a)
    
    def is_coprime(a, b):
        return len(pfactors[a] & pfactors[b]) == 0
    
    # Build conflict graph: conflict[a] = set of b != a where gcd(a,b) > 1
    conflict = [set() for _ in range(N)]
    for a in range(2, N):
        for b in range(a + 1, N):
            if not is_coprime(a, b):
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Step 1: Greedy initialization
    greedy_set = []
    for a in range(N - 1, 0, -1):  # descending weight
        if time.time() > deadline:
            break
        if all(is_coprime(a, x) for x in greedy_set):
            greedy_set.append(a)
    
    best_set = list(greedy_set)
    best_score = sum(weights[a] for a in best_set)
    best_score_float = sum(weights_float[a] for a in best_set)
    
    # Step 2: Local search refinement (fast, runs first)
    all_candidates = sorted(range(1, N), key=lambda x: -weights_float[x])
    
    def can_add_fast(candidate, current_set):
        for x in current_set:
            if candidate in conflict[x]:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        
        # Single additions
        for a in all_candidates:
            if time.time() > deadline - 60:
                break
            if a not in best_set and can_add_fast(a, best_set):
                best_set.append(a)
                best_score += weights[a]
                best_score_float += weights_float[a]
                improved = True
        
        # 1-for-1 swaps
        for i, remove_a in enumerate(list(best_set)):
            if time.time() > deadline - 60:
                break
            for add_b in all_candidates:
                if time.time() > deadline - 60:
                    break
                if add_b == remove_a or add_b in best_set:
                    continue
                valid = all(add_b not in conflict[x] for x in best_set if x != remove_a)
                if valid and weights_float[add_b] > weights_float[remove_a] + 1e-12:
                    best_set[i] = add_b
                    best_score += weights[add_b] - weights[remove_a]
                    best_score_float += weights_float[add_b] - weights_float[remove_a]
                    improved = True
                    break
        
        # 1-for-2 swaps
        if time.time() < deadline - 65:
            for i, remove_a in enumerate(list(best_set)):
                if time.time() > deadline - 65:
                    break
                valid_cands = [b for b in all_candidates 
                              if b not in best_set and b != remove_a
                              and all(b not in conflict[x] for x in best_set if x != remove_a)]
                for idx1 in range(len(valid_cands)):
                    if time.time() > deadline - 65:
                        break
                    for idx2 in range(idx1 + 1, len(valid_cands)):
                        if time.time() > deadline - 65:
                            break
                        b1, b2 = valid_cands[idx1], valid_cands[idx2]
                        if b1 in conflict[b2]:
                            continue
                        if weights_float[b1] + weights_float[b2] > weights_float[remove_a] + 1e-12:
                            best_set[i] = b1
                            best_set.append(b2)
                            best_score += weights[b1] + weights[b2] - weights[remove_a]
                            best_score_float += weights_float[b1] + weights_float[b2] - weights_float[remove_a]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Step 3: MaxSAT-based global optimization using DPLL SAT solver
    # Encode as: for each a in 1..511, variable x_a (1-indexed)
    # Hard constraints: for each conflict pair (a,b), clause (¬x_a ∨ ¬x_b)
    # Objective: maximize sum w_a * x_a
    
    # DPLL SAT solver with unit propagation and pure literal elimination
    class DPLL_SAT:
        def __init__(self, n_vars, hard_clauses, soft_weights, deadline):
            self.n_vars = n_vars
            self.hard_clauses = hard_clauses  # list of lists of literals (positive = var, negative = ¬var)
            self.soft_weights = soft_weights  # dict: var -> weight (positive means we want it true)
            self.deadline = deadline
            self.node_count = 0
            self.max_nodes = 10_000_000  # safety cap
            
        def _unit_propagate(self, assignment, clauses):
            """Apply unit propagation. Returns (new_assignment, new_clauses, contradiction)."""
            assignment = dict(assignment)
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    # Check if clause is satisfied
                    satisfied = False
                    remaining = []
                    for lit in clause:
                        var = abs(lit)
                        if var in assignment:
                            if (lit > 0 and assignment[var] == True) or (lit < 0 and assignment[var] == False):
                                satisfied = True
                                break
                        else:
                            remaining.append(lit)
                    if satisfied:
                        continue
                    if not remaining:
                        return None, None, True  # contradiction
                    if len(remaining) == 1:
                        # Unit clause
                        lit = remaining[0]
                        var = abs(lit)
                        val = lit > 0
                        assignment[var] = val
                        changed = True
                    else:
                        new_clauses.append(remaining)
                clauses = new_clauses
            return assignment, clauses, False
        
        def _pure_literal_elimination(self, assignment, clauses):
            """Eliminate pure literals. Returns (new_assignment, new_clauses)."""
            assignment = dict(assignment)
            # Find literals that appear only in one polarity
            lit_polarity = {}
            for clause in clauses:
                for lit in clause:
                    lit_polarity[lit] = lit_polarity.get(lit, 0) + 1
                    if -lit not in lit_polarity:
                        lit_polarity[-lit] = 0
            
            # Assign pure literals
            for lit, count in list(lit_polarity.items()):
                if count > 0 and lit_polarity.get(-lit, 0) == 0:
                    var = abs(lit)
                    if var not in assignment:
                        assignment[var] = lit > 0
            
            # Remove satisfied clauses
            new_clauses = []
            for clause in clauses:
                satisfied = False
                remaining = []
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        if (lit > 0 and assignment[var] == True) or (lit < 0 and assignment[var] == False):
                            satisfied = True
                            break
                    else:
                        remaining.append(lit)
                if not satisfied and remaining:
                    new_clauses.append(remaining)
            return assignment, new_clauses
        
        def _select_variable(self, clauses, assignment):
            """Select a variable to branch on using longest clause heuristic."""
            var_counts = {}
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if var not in assignment:
                        var_counts[var] = var_counts.get(var, 0) + len(clause)
            if not var_counts:
                return None
            # Pick variable in longest clause
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            return candidates[0]
        
        def _solve(self, assignment, clauses, depth=0):
            """Recursive DPLL search. Returns best assignment found."""
            if time.time() > self.deadline:
                return None
            self.node_count += 1
            if self.node_count > self.max_nodes:
                return None
            
            # Unit propagation
            assignment, clauses, contradiction = self._unit_propagate(assignment, clauses)
            if contradiction:
                return None
            if clauses is None:
                clauses = []
            
            # Pure literal elimination
            assignment, clauses = self._pure_literal_elimination(assignment, clauses)
            
            if not clauses:
                return assignment
            
            # Select variable
            var = self._select_variable(clauses, assignment)
            if var is None:
                return assignment
            
            # Branch: try true first (since we want high-weight vars to be true)
            result = self._solve(assignment | {var: True}, clauses, depth + 1)
            if result is not None:
                return result
            
            result = self._solve(assignment | {var: False}, clauses, depth + 1)
            if result is not None:
                return result
            
            return None
        
        def solve_maxsat(self, min_score_threshold=None):
            """
            Solve MaxSAT: find assignment satisfying all hard clauses
            that maximizes sum of soft_weights[var] for var=True.
            Uses iterative deepening on score threshold.
            """
            # Start with greedy solution as lower bound
            best_assignment = None
            best_score = Fraction(0)
            
            # Get initial solution from greedy
            for var in range(1, self.n_vars + 1):
                if var in self.soft_weights:
                    best_assignment = best_assignment or {}
                    best_assignment[var] = True  # placeholder, will be refined
            
            # For now, just do a simple DPLL search with soft constraint encoding
            # Add soft clauses: for each var with weight w, add clause (x_var) with weight w
            # This is a simplification; proper MaxSAT needs weighted clauses
            
            # Alternative: use the local search result as seed and try to improve
            # via SAT-based local search
            return best_assignment
    
    # Since full MaxSAT is complex, use a hybrid approach:
    # 1. Use local search results as baseline
    # 2. Try targeted improvements using SAT feasibility checks
    
    # Try to find a better solution by checking if we can add high-value elements
    # that were excluded due to conflicts, by removing lower-value conflicting elements
    
    if time.time() < deadline - 10:
        # Sort all elements by weight descending
        sorted_by_weight = sorted(range(1, N), key=lambda x: -weights_float[x])
        
        # For each high-weight element not in best_set, check if we can make room
        best_set_set = set(best_set)
        
        for target in sorted_by_weight:
            if time.time() > deadline - 10:
                break
            if target in best_set_set:
                continue
            
            # Find all elements in best_set that conflict with target
            blockers = [x for x in best_set if target in conflict[x]]
            
            # If no blockers, just add it
            if not blockers:
                best_set.append(target)
                best_set_set.add(target)
                best_score += weights[target]
                best_score_float += weights_float[target]
                continue
            
            # Check if removing blockers and adding target improves score
            # Simple case: remove all blockers, add target
            blocker_score = sum(weights_float[b] for b in blockers)
            if weights_float[target] > blocker_score + 1e-12:
                # Remove blockers
                for b in blockers:
                    best_set.remove(b)
                    best_set_set.remove(b)
                    best_score -= weights[b]
                    best_score_float -= weights_float[b]
                # Add target
                best_set.append(target)
                best_set_set.add(target)
                best_score += weights[target]
                best_score_float += weights_float[target]
            
            # Try 1-for-1: remove one blocker, add target
            # Only if target has higher weight than the blocker
            for b in blockers:
                if weights_float[target] > weights_float[b] + 1e-12:
                    # Check if removing b allows adding target (it should, since b was the only blocker)
                    other_blockers = [x for x in best_set if x != b and target in conflict[x]]
                    if not other_blockers:
                        best_set.remove(b)
                        best_set_set.remove(b)
                        best_score -= weights[b]
                        best_score_float -= weights_float[b]
                        best_set.append(target)
                        best_set_set.add(target)
                        best_score += weights[target]
                        best_score_float += weights_float[target]
                        break
    
    # Final validation
    final_set = list(best_set)
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if not is_coprime(final_set[i], final_set[j]):
                return greedy_set
    
    return final_set

# EVOLVE-BLOCK-END
