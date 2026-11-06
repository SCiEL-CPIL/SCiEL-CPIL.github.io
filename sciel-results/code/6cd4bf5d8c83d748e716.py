"""CSP-based approach with AC-3 arc consistency for pairwise disjoint residue classes."""
import time
import math
import random
from collections import deque

MAX_MOD = 128
DEADLINE = 82.0  # safety margin before 90s kill

def _gcd(m1, m2):
    return math.gcd(m1, m2)

def _incompatible(r1, m1, r2, m2):
    """Two classes are NOT disjoint iff gcd(m1,m2) divides (r2-r1)."""
    g = _gcd(m1, m2)
    return (r2 - r1) % g == 0

def _ac3_prune(moduli, domains, start_pair=None):
    """Apply AC-3 arc consistency. domains: dict m -> set of valid residues.
    Returns True if no domain wiped out."""
    # Build constraint graph: edge between m1, m2 if gcd(m1,m2) > 1
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(moduli[i], moduli[j]) > 1:
                adj[i].append(j)
                adj[j].append(i)
    
    queue = deque()
    if start_pair is not None:
        queue.append(start_pair)
    else:
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    queue.append((i, j))
    
    while queue:
        i, j = queue.popleft()
        if not domains[moduli[i]] or not domains[moduli[j]]:
            continue
        gi = _gcd(moduli[i], moduli[j])
        mi = moduli[i]
        mj = moduli[j]
        # For each value in domain[i], check if there's a compatible value in domain[j]
        new_domain_i = set()
        di = domains[moduli[i]]
        dj = domains[moduli[j]]
        # Precompute: for residue r in di, it's compatible with some s in dj iff
        # NOT (r - s) % gi == 0 for all s in dj
        # i.e., s % gi != r % gi for some s in dj
        # Equivalently: the set of residues mod gi present in dj must not be a subset of {r % gi}
        # i.e., there exists s in dj with s % gi != r % gi
        residues_j_mod_gi = set(s % gi for s in dj)
        if len(residues_j_mod_gi) > 1:
            # All values in di are compatible (there's always some s with different mod)
            new_domain_i = di
        else:
            # Only one residue class mod gi in dj; r is compatible iff r % gi != that value
            bad_mod = list(residues_j_mod_gi)[0]
            new_domain_i = {r for r in di if r % gi != bad_mod}
        
        removed = di - new_domain_i
        if removed:
            domains[moduli[i]] = new_domain_i
            if not new_domain_i:
                return False
            # Re-check neighbors of i (except j)
            for k in adj[i]:
                if k != j:
                    queue.append((k, i))
        
        # Symmetric: prune j based on i
        new_domain_j = set()
        di = domains[moduli[i]]
        dj = domains[moduli[j]]
        residues_i_mod_gi = set(r % gi for r in di)
        if len(residues_i_mod_gi) > 1:
            new_domain_j = dj
        else:
            bad_mod = list(residues_i_mod_gi)[0]
            new_domain_j = {s for s in dj if s % gi != bad_mod}
        
        removed_j = dj - new_domain_j
        if removed_j:
            domains[moduli[j]] = new_domain_j
            if not new_domain_j:
                return False
            for k in adj[j]:
                if k != i:
                    queue.append((k, j))
    
    return True

def _backtrack(moduli, domains, idx, assignment, time_limit, start_time):
    """Backtracking search with forward checking. assignment: dict m -> r."""
    if time.time() - start_time > time_limit:
        return None
    if idx == len(moduli):
        return dict(assignment)
    
    m = moduli[idx]
    # Try residues in order of least constraining (heuristic: try values that keep domains open)
    residues = sorted(domains[m], key=lambda r: r)
    
    for r in residues:
        # Check compatibility with already assigned
        ok = True
        for m2, r2 in assignment.items():
            if _incompatible(r, m, r2, m2):
                ok = False
                break
        if not ok:
            continue
        
        assignment[m] = r
        # Forward check: prune next variable's domain
        next_idx = idx + 1
        if next_idx < len(moduli):
            m_next = moduli[next_idx]
            g_next = _gcd(m, m_next)
            if g_next > 1:
                # Filter next domain
                new_next = {r2 for r2 in domains[m_next] if (r2 - r) % g_next != 0}
                if not new_next:
                    del assignment[m]
                    continue
                saved_next = domains[m_next]
                domains[m_next] = new_next
                result = _backtrack(moduli, domains, next_idx, assignment, time_limit, start_time)
                domains[m_next] = saved_next
            else:
                result = _backtrack(moduli, domains, next_idx, assignment, time_limit, start_time)
            del assignment[m]
            if result is not None:
                return result
        else:
            del assignment[m]
            return dict(assignment)
    
    return None

def _try_order(moduli_order, time_limit, start_time):
    """Try to find a full assignment for the given moduli ordering."""
    domains = {m: set(range(m)) for m in moduli_order}
    if not _ac3_prune(moduli_order, domains):
        return []
    
    # Order variables by most-constrained (smallest domain) first for backtracking
    # But we need to respect the constraint structure; use the given order but sort by domain size
    sorted_moduli = sorted(moduli_order, key=lambda m: (len(domains[m]), m))
    
    assignment = {}
    result = _backtrack(sorted_moduli, domains, 0, assignment, time_limit, start_time)
    if result is not None:
        return [[result[m], m] for m in moduli_order]
    return []

def construct():
    start_time = time.time()
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Strategy: Use CSP with AC-3 + backtracking
    # Key insight: we don't need ALL moduli; we want to maximize count.
    # So we try to include as many as possible.
    
    # First, try to include all moduli 2..128
    # Use AC-3 to prune, then backtracking
    
    # Variable ordering: process moduli in an order that helps
    # Moduli that are coprime to everything else are easiest (no constraints)
    # Moduli with many shared factors are hardest
    
    # Compute constraint degree for each modulus
    def constraint_degree(m):
        deg = 0
        for m2 in all_moduli:
            if m2 != m and _gcd(m, m2) > 1:
                deg += 1
        return deg
    
    # Order: most constrained first (hardest first)
    order_hard = sorted(all_moduli, key=lambda m: (-constraint_degree(m), m))
    # Order: least constrained first
    order_easy = sorted(all_moduli, key=lambda m: (constraint_degree(m), m))
    # Order: descending
    order_desc = sorted(all_moduli, reverse=True)
    # Order: ascending
    order_asc = sorted(all_moduli)
    
    best_pairs = []
    
    # Try different orderings
    for order in [order_hard, order_easy, order_desc, order_asc]:
        if time.time() - start_time > DEADLINE - 2:
            break
        pairs = _try_order(order, DEADLINE - 2, start_time)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # If we got all 127, great. Otherwise try to find a good subset.
    if len(best_pairs) == MAX_MOD - 1:
        return best_pairs
    
    # If we couldn't get all, try greedy as fallback but with AC-3 informed
    # Greedy with AC-3: at each step, pick the modulus-residue that allows the most
    # remaining moduli to still be assignable
    
    # Simpler: try random subsets of moduli, largest first
    rng = random.Random(123)
    
    # Try to build up: start with all, if AC-3 fails, remove the most constrained
    # and retry
    
    # Alternative: greedy with lookahead
    def greedy_with_ac3(moduli_pool, time_limit, start_time):
        """Greedily add moduli, using AC-3 to check feasibility."""
        selected = []
        domains = {}
        
        for m in moduli_pool:
            if time.time() - start_time > time_limit:
                break
            # Try to add m with some residue
            # First, check if any residue works with AC-3 on current + m
            test_domains = dict(domains)
            test_domains[m] = set(range(m))
            test_moduli = [x[1] for x in selected] + [m]
            
            if _ac3_prune(test_moduli, test_domains):
                # Find a residue that works
                for r in range(m):
                    # Check direct compatibility
                    ok = True
                    for (r2, m2) in selected:
                        if _incompatible(r, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        selected.append([r, m])
                        domains[m] = set(range(m))
                        # Update domains of existing moduli via AC-3
                        all_m = [x[1] for x in selected]
                        all_domains = {x[1]: set(range(x[1])) for x in selected}
                        # Fix assigned residues
                        for (r2, m2) in selected:
                            all_domains[m2] = {r2}
                        _ac3_prune(all_m, all_domains)
                        break
        
        return selected
    
    # Greedy with AC-3, trying different orderings
    for order in [order_hard, order_easy, order_desc, order_asc]:
        if time.time() - start_time > DEADLINE - 2:
            break
        pairs = greedy_with_ac3(order, DEADLINE - 2, start_time)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # Try random orderings with greedy+AC3
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 2:
            break
        order = all_moduli[:]
        rng.shuffle(order)
        pairs = greedy_with_ac3(order, DEADLINE - 2, start_time)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # If still not great, try local improvement
    # Remove a modulus and try to add a different one
    
    if len(best_pairs) < MAX_MOD - 1:
        # Try to improve by swapping
        current_mods = set(m for _, m in best_pairs)
        missing = [m for m in all_moduli if m not in current_mods]
        
        for m_remove in [m for _, m in best_pairs]:
            if time.time() - start_time > DEADLINE - 2:
                break
            for m_add in missing:
                if time.time() - start_time > DEADLINE - 2:
                    break
                # Try replacing m_remove with m_add
                new_pairs = [[r, m] for (r, m) in best_pairs if m != m_remove]
                # Find residue for m_add
                for r in range(m_add):
                    ok = True
                    for (r2, m2) in new_pairs:
                        if _incompatible(r, m_add, r2, m2):
                            ok = False
                            break
                    if ok:
                        new_pairs.append([r, m_add])
                        # Verify all pairwise
                        valid = True
                        for i in range(len(new_pairs)):
                            for j in range(i + 1, len(new_pairs)):
                                if _incompatible(new_pairs[i][0], new_pairs[i][1], 
                                                  new_pairs[j][0], new_pairs[j][1]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(new_pairs) > len(best_pairs):
                            best_pairs = new_pairs
                        break
    
    return best_pairs
