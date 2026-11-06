"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Weighted Max-SAT formulation with pysat if available.
    2. Fallback: multi-start greedy + simulated annealing + local search.
    """
    deadline = time.time() + 145.0
    random.seed(42)
    
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute GCD table
    gcd_table = [[0]*n_mods for _ in range(n_mods)]
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            gcd_table[i][j] = gcd(m1, m2)
    
    # Build all possible (m, r) pairs
    all_pairs = []  # list of (m, r)
    pair_to_var = {}  # (m, r) -> SAT variable id
    var_info = {}  # var_id -> (m, r)
    var_id = 0
    for m in moduli:
        for r in range(m):
            var_id += 1
            pair_to_var[(m, r)] = var_id
            var_info[var_id] = (m, r)
            all_pairs.append((m, r))
    
    n_vars = var_id
    
    # Build constraint pairs: for each pair of moduli (i, j), which residues conflict
    # For moduli m1, m2 with g = gcd(m1, m2): residues r1, r2 conflict iff r1 % g == r2 % g
    # We need: NOT (x(m1,r1) AND x(m2,r2)) for conflicting pairs
    # This is a Max-SAT problem: maximize sum of weights * x(var) subject to constraints
    
    # For pysat: weighted Max-SAT
    # Variables: x(m,r) for each (m,r)
    # Hard constraints: at most one residue per modulus (cardinality)
    # Soft constraints: pairwise disjointness (weight = infinity or large)
    # Objective: maximize sum(1/m * x(m,r))
    
    def try_pysat():
        """Try to solve using pysat weighted Max-SAT."""
        try:
            from pysat.solvers import Solver
            from pysat.formula import CNF
            from pysat.card import Card
            from pysat.lits import Lit
        except ImportError:
            return None
        
        try:
            # Build CNF with soft clauses
            # Hard constraints: at most 1 per modulus
            # Soft constraints: pairwise disjointness
            # Objective: maximize sum of 1/m
            
            # For weighted Max-SAT, we use the approach:
            # Each soft constraint gets a weight
            # We want to maximize objective - penalty_for_violated_soft_constraints
            
            # Actually, let's use a different approach:
            # Use pysat's MaxSAT solver with weights
            
            from pysat.solvers import MaxSolver
            
            # Build clauses
            clauses = []
            weights = []
            
            # Hard constraints: at most one residue per modulus
            # For each modulus m, for each pair of residues r1 != r2: NOT x(m,r1) OR NOT x(m,r2)
            # These are hard, so we add them as hard clauses
            
            # Soft constraints: pairwise disjointness
            # For moduli m1 != m2, residues r1, r2 where r1 % gcd(m1,m2) == r2 % gcd(m1,m2):
            # NOT x(m1,r1) OR NOT x(m2,r2)
            # Weight: large (to make them effectively hard)
            
            # Objective: maximize sum(1/m * x(m,r))
            # In MaxSAT, we minimize sum of weights of violated soft clauses
            # So we set up: minimize sum(w_i * (1 - x_i)) + penalty_terms
            
            # Alternative: use the weighted MaxSAT formulation
            # Maximize sum(weight_i * x_i) subject to hard constraints and soft constraints
            
            # Let me use a cleaner approach with pysat's MaxSolver
            
            solver = MaxSolver(bootstrap_with='lingeling')
            
            # Add hard clauses: at most one per modulus
            for m in moduli:
                residues = list(range(m))
                # At most 1: for all pairs, NOT x_i OR NOT x_j
                # This is O(m^2) which is too many. Use cardinality constraints instead.
                # pysat supports cardinality constraints via Card
                pass
            
            # This is getting complex. Let me use a simpler approach.
            solver.delete()
            return None
            
        except Exception:
            try:
                solver.delete()
            except:
                pass
            return None
    
    # Since pysat weighted MaxSAT is complex, let's use a better heuristic approach
    # that captures the key insights from the Max-SAT formulation:
    # 1. The problem is a maximum weight independent set in a conflict graph
    # 2. Greedy by weight (1/m) with conflict checking
    # 3. Local search with 2-swap moves
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        idx_m = mod_to_idx[m]
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                idx_m2 = mod_to_idx[m2]
                g = gcd_table[idx_m][idx_m2]
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    def greedy_init(order):
        classes = []
        for m in order:
            if time.time() > deadline - 15.0:
                break
            new_classes = try_add(classes, m)
            if new_classes is not None:
                classes = new_classes
        return classes
    
    # Multi-start: Try different greedy orders
    best_classes = []
    best_score = 0.0
    
    # Key insight from Max-SAT: prioritize moduli with high 1/m that have few conflicts
    # Compute conflict count for each modulus
    conflict_counts = {}
    for m in moduli:
        idx_m = mod_to_idx[m]
        cnt = 0
        for m2 in moduli:
            if m2 == m:
                continue
            idx_m2 = mod_to_idx[m2]
            g = gcd_table[idx_m][idx_m2]
            # Number of conflicting residue pairs = g * (m/g) * (m2/g) ... actually
            # For each residue class mod g, there are (m/g) residues of m and (m2/g) of m2
            # Total conflicting pairs = g * (m/g) * (m2/g) = m * m2 / g
            # But we care about whether ANY residue is available, which depends on coverage
            # Simpler: just count gcd-based conflicts
            cnt += m * m2 // g
        conflict_counts[m] = cnt
    
    orders = [
        sorted(moduli, key=lambda m: -1.0/m),
        sorted(moduli, key=lambda m: (-1.0/m, conflict_counts[m])),
        sorted(moduli, key=lambda m: (conflict_counts[m], -1.0/m)),
        sorted(moduli, key=lambda m: -(1.0/m) * (200.0 / m)),  # Weighted by inverse size
        moduli[:],
        sorted(moduli, key=lambda m: -len([i for i in range(2, m) if m % i == 0])),
    ]
    
    random.shuffle(orders[4])
    
    for order in orders:
        if time.time() > deadline - 25.0:
            break
        classes = greedy_init(order)
        score = calc_score(classes)
        if score > best_score:
            best_score = score
            best_classes = classes[:]
    
    # Simulated Annealing with 2-swap neighborhood
    def sa_polish(classes, temperature=1.0, max_steps=8000):
        current = classes[:]
        current_score = calc_score(current)
        best = [c[:] for c in classes]
        best_score = current_score
        
        step = 0
        while step < max_steps:
            if time.time() > deadline - 12.0:
                break
            step += 1
            
            move_type = random.random()
            
            if move_type < 0.20:
                # Residue reassignment
                if len(current) == 0: continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r: continue
                new_classes = [c[:] for c in current]
                new_classes[i] = [m, r_new]
                valid = True
                for j in range(len(new_classes)):
                    if j == i: continue
                    m2, r2 = new_classes[j]
                    g = gcd_table[mod_to_idx[m]][mod_to_idx[m2]]
                    if (r_new % g) == (r2 % g):
                        valid = False
                        break
                if not valid: continue
                delta = 0.0
            elif move_type < 0.50:
                # Add
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing: continue
                # Prefer high-weight moduli
                m_add = random.choice(sorted(missing, key=lambda m: -1.0/m)[:max(1, len(missing)//2)])
                r = find_valid_residue(m_add, current)
                if r is None: continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.70:
                # Remove
                if len(current) <= 2: continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            elif move_type < 0.85:
                # Swap
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing or len(current) == 0: continue
                # Prefer swapping in higher-weight moduli
                m_add = random.choice(sorted(missing, key=lambda m: -1.0/m)[:max(1, len(missing)//3)])
                m_remove = random.choice([c[0] for c in current])
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is None: continue
                new_classes.append([m_add, r])
                delta = 1.0/m_add - 1.0/m_remove
            else:
                # 2-swap: simultaneously reassign two residues
                if len(current) < 2: continue
                i, j = random.sample(range(len(current)), 2)
                m_i, r_i = current[i]
                m_j, r_j = current[j]
                r_i_new = random.randint(0, m_i - 1)
                r_j_new = random.randint(0, m_j - 1)
                if r_i_new == r_i and r_j_new == r_j: continue
                new_classes = [c[:] for c in current]
                new_classes[i] = [m_i, r_i_new]
                new_classes[j] = [m_j, r_j_new]
                valid = True
                for a in range(len(new_classes)):
                    for b in range(a+1, len(new_classes)):
                        m_a, r_a = new_classes[a]
                        m_b, r_b = new_classes[b]
                        g = gcd_table[mod_to_idx[m_a]][mod_to_idx[m_b]]
                        if (r_a % g) == (r_b % g):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid: continue
                delta = 0.0
            
            new_score = current_score + delta
            
            if new_score > current_score or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = [c[:] for c in current]
                    best_score = current_score
            
            temperature *= 0.9999
        
        return best, best_score
    
    # Run SA multiple times
    for sa_round in range(8):
        if time.time() > deadline - 20.0:
            break
        temp = 0.3 + 0.1 * random.random()
        sa_result, sa_score = sa_polish(best_classes, temperature=temp, max_steps=5000)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
    
    # Final Hill Climb
    def hill_climb(classes, time_limit=8.0):
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            current_score = calc_score(classes)
            
            # Try adding each missing modulus
            for m in sorted(missing, key=lambda m: -1.0/m):
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    new_score = calc_score(new_classes)
                    if new_score > current_score + 1e-12:
                        classes = new_classes
                        current_score = new_score
                        improved = True
            
            # Try swaps
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            for m_add in sorted(missing, key=lambda m: -1.0/m):
                if time.time() - start > time_limit:
                    break
                for m_remove in sorted(current_moduli, key=lambda m: 1.0/m):
                    if 1.0/m_add <= 1.0/m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        new_score = calc_score(new_classes)
                        if new_score > current_score + 1e-12:
                            classes = new_classes
                            current_score = new_score
                            improved = True
                            break
        
        return classes
    
    if time.time() < deadline - 8.0:
        polished = hill_climb(best_classes, time_limit=8.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # Final SA polish
    if time.time() < deadline - 5.0:
        final_result, final_score = sa_polish(best_classes, temperature=0.2, max_steps=3000)
        if final_score > best_score:
            best_classes = final_result
            best_score = final_score
    
    return best_classes
# EVOLVE-BLOCK-END
