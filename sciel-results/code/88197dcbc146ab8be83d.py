"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START_TIME = None


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect (coprime moduli)
    return (r2 - r1) % g != 0


def _can_add(r_new, m_new, existing):
    """Check if new class is disjoint from all existing."""
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True


def _can_add_fast(r_new, m_new, existing, gcd_cache):
    """Faster check using precomputed gcd values."""
    for i, (r, m) in enumerate(existing):
        g = gcd_cache.get((m_new, m), None)
        if g is None:
            g = _gcd(m_new, m)
        if g > 1 and (r_new - r) % g == 0:
            return False
    return True


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    deadline = _START_TIME + 80.0  # safety margin before 90s kill

    # Candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    n = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # constraint_score: how many other moduli share a factor (higher = more constrained)
    constraint = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd_table[i][j] > 1:
                constraint[i] += 1
                constraint[j] += 1

    # Build several deterministic orderings
    order_most_constrained = sorted(range(n), key=lambda i: (-constraint[i], -moduli[i]))
    order_least_constrained = sorted(range(n), key=lambda i: (constraint[i], moduli[i]))
    order_desc = list(range(n - 1, -1, -1))  # moduli 128..2
    order_asc = list(range(n))               # moduli 2..128

    def greedy_pass(order, residue_mode, rng):
        """
        Greedy: iterate moduli in `order`, pick a valid residue.
        residue_mode: 0 = first valid (deterministic), 1 = random valid,
                      2 = pick residue minimizing future blocking,
                      3 = pick residue maximizing flexibility for remaining moduli.
        Returns list of [r, m].
        """
        current = []
        used = set()
        # Precompute gcd for all pairs of moduli we might encounter
        gcd_cache = {}
        
        def get_gcd(m1, m2):
            key = (min(m1, m2), max(m1, m2))
            if key not in gcd_cache:
                gcd_cache[key] = _gcd(m1, m2)
            return gcd_cache[key]
        
        for idx in order:
            m = moduli[idx]
            if m in used:
                continue
            if not current:
                # First class: any residue works; pick 0 for determinism or random
                r = rng.randrange(m) if residue_mode == 1 else 0
                current.append([r, m])
                used.add(m)
                continue

            if residue_mode == 0:
                # First valid residue
                for r in range(m):
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used.add(m)
                        break
            elif residue_mode == 1:
                # Random valid residue
                residues = list(range(m))
                rng.shuffle(residues)
                for r in residues:
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used.add(m)
                        break
            else:
                # residue_mode == 2 or 3: smart residue selection
                best_r = None
                best_score = None
                # For mode 3, also consider which residues leave more options
                # for the remaining (unused) moduli
                remaining_moduli = [mm for mm in moduli if mm not in used]
                
                for r in range(m):
                    if not _can_add(r, m, current):
                        continue
                    
                    # Score 1: count how many existing classes share a factor with m
                    # (more shared factors = more constrained choice)
                    shared_factor_count = 0
                    for (r_e, m_e) in current:
                        g = get_gcd(m, m_e)
                        if g > 1:
                            shared_factor_count += 1
                    
                    # Score 2 (mode 3): estimate how many remaining moduli would be
                    # blocked by this residue choice. For each remaining modulus m_f,
                    # check if choosing r for m would make it harder to add m_f.
                    future_block_count = 0
                    if residue_mode == 3:
                        for m_f in remaining_moduli:
                            g_f = get_gcd(m, m_f)
                            if g_f > 1:
                                # Our residue r mod g_f is fixed. For m_f to be addable,
                                # we need a residue r_f such that for all existing (r_e, m_e)
                                # with gcd(m_f, m_e) > 1, (r_f - r_e) % gcd(m_f, m_e) != 0
                                # AND (r_f - r) % g_f != 0.
                                # The constraint from us blocks residue class r % g_f mod g_f.
                                # Count how many residue classes mod g_f are already blocked
                                # by existing classes.
                                blocked_classes = set()
                                for (r_e, m_e) in current:
                                    g_e = get_gcd(m_f, m_e)
                                    if g_e > 1:
                                        blocked_classes.add(r_e % g_e)
                                # Add our blocked class
                                blocked_classes.add(r % g_f)
                                # If all residue classes mod g_f are blocked, m_f can't be added
                                # (but this is an overestimate since different g_e values)
                                # Better: just count if we're adding to a small set
                                if len(blocked_classes) >= g_f:
                                    future_block_count += 1
                                else:
                                    # Partial block - add fractionally
                                    future_block_count += len(blocked_classes) / g_f
                    
                    if residue_mode == 2:
                        score = shared_factor_count
                    else:  # mode 3
                        score = shared_factor_count + 0.1 * future_block_count
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_r = r
                
                if best_r is None:
                    # Fallback: try each residue
                    for r in range(m):
                        if _can_add(r, m, current):
                            best_r = r
                            break
                if best_r is not None:
                    current.append([best_r, m])
                    used.add(m)
        return current

    def local_search(sol, max_time=None):
        """Try to improve sol by adding missing moduli and optimizing residues."""
        if not sol:
            return sol
        sol = [x[:] for x in sol]
        used = set(m for _, m in sol)
        
        if max_time is None:
            max_time = 10.0
        start = time.time()
        
        # Precompute gcd for all pairs
        gcd_cache = {}
        def get_gcd(m1, m2):
            key = (min(m1, m2), max(m1, m2))
            if key not in gcd_cache:
                gcd_cache[key] = _gcd(m1, m2)
            return gcd_cache[key]
        
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            
            # Try adding a missing modulus
            for m in moduli:
                if m in used:
                    continue
                for r in range(m):
                    if _can_add(r, m, sol):
                        sol.append([r, m])
                        used.add(m)
                        improved = True
                        break
                if improved:
                    break
            
            # Try optimizing residues to enable adding more moduli
            if not improved:
                # For each class, try all residues and keep the one that
                # allows the most additional moduli to be added
                best_improvement = 0
                best_change = None
                
                for i, (r1, m1) in enumerate(sol):
                    for r_new in range(m1):
                        if r_new == r1:
                            continue
                        # Check if this residue is valid with all others
                        ok = all(_disjoint(r_new, m1, r2, m2)
                                 for j, (r2, m2) in enumerate(sol) if j != i)
                        if not ok:
                            continue
                        # Try this residue and see how many moduli we can add
                        sol[i][0] = r_new
                        count_added = 0
                        added = []
                        for m_add in moduli:
                            if m_add in used:
                                continue
                            for r_add in range(m_add):
                                if _can_add(r_add, m_add, sol):
                                    sol.append([r_add, m_add])
                                    used.add(m_add)
                                    added.append([r_add, m_add])
                                    count_added += 1
                                    break
                        # Undo additions
                        for a in added:
                            sol.pop()
                            used.discard(a[1])
                        if count_added > best_improvement:
                            best_improvement = count_added
                            best_change = (i, r_new)
                        # Restore original residue
                        sol[i][0] = r1
                
                if best_change is not None:
                    i, r_new = best_change
                    sol[i][0] = r_new
                    # Now add all possible moduli
                    for m_add in moduli:
                        if m_add in used:
                            continue
                        for r_add in range(m_add):
                            if _can_add(r_add, m_add, sol):
                                sol.append([r_add, m_add])
                                used.add(m_add)
                                improved = True
                                break
            else:
                # We added something, continue
                continue
        
        return sol

    best_solution = []
    best_size = 0

    def update(sol):
        nonlocal best_solution, best_size
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = [x[:] for x in sol]

    # Phase 1: deterministic greedy passes with different orderings and residue modes
    deterministic_orders = [
        (order_most_constrained, 0),
        (order_most_constrained, 2),
        (order_most_constrained, 3),
        (order_least_constrained, 0),
        (order_least_constrained, 2),
        (order_least_constrained, 3),
        (order_desc, 0),
        (order_desc, 2),
        (order_desc, 3),
        (order_asc, 0),
        (order_asc, 2),
        (order_asc, 3),
    ]
    for order, mode in deterministic_orders:
        if time.time() > deadline:
            break
        sol = greedy_pass(order, mode, random)
        update(sol)
        if time.time() < deadline - 15:
            sol2 = local_search(sol, max_time=5.0)
            update(sol2)

    # Phase 2: randomized restarts
    rng = random
    for restart in range(500):
        if time.time() > deadline:
            break
        # Random order, sometimes biased toward constrained
        if restart % 3 == 0:
            order = order_most_constrained[:]
            rng.shuffle(order)
        elif restart % 3 == 1:
            order = order_least_constrained[:]
            rng.shuffle(order)
        else:
            order = list(range(n))
            rng.shuffle(order)
        mode = rng.choice([0, 1, 2, 3])
        sol = greedy_pass(order, mode, rng)
        update(sol)
        # Local search on the best found so far occasionally
        if len(sol) >= best_size and time.time() < deadline - 10:
            sol2 = local_search(sol, max_time=3.0)
            update(sol2)

    # Final local search on the best solution
    if best_solution and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_solution = local_search(best_solution, max_time=remaining_time * 0.7)
        update(best_solution)
        # Additional polish: try to add any remaining moduli with optimized residues
        if best_solution and time.time() < deadline - 2:
            used = set(m for _, m in best_solution)
            for m in moduli:
                if m in used:
                    continue
                for r in range(m):
                    if _can_add(r, m, best_solution):
                        best_solution.append([r, m])
                        used.add(m)
                        break
            update(best_solution)
            # One more local search pass if time allows
            if time.time() < deadline - 1:
                best_solution = local_search(best_solution, max_time=1.0)
                update(best_solution)

    if not best_solution:
        return [[1, 2]]

    return best_solution
# EVOLVE-BLOCK-END
