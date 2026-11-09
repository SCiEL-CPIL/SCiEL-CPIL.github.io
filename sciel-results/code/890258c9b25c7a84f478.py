"""Combined: greedy + branch-and-bound MWIS on conflict graph + local search."""
import time
import random
import math
from fractions import Fraction
from collections import defaultdict

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    """Check pairwise disjointness for a list of [m, r] pairs with distinct moduli."""
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _find_valid_residue(m, selected):
    """Find a residue r in [0, m) such that (m, r) is disjoint from all in selected.
    Returns r or None."""
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            if not _disjoint(m, r, m2, r2):
                ok = False
                break
        if ok:
            return r
    return None

def _greedy_construction():
    """Greedy construction: process moduli in ascending order, pick first valid residue."""
    selected = []
    for m in range(20, 201):
        r = _find_valid_residue(m, selected)
        if r is not None:
            selected.append([m, r])
    return selected

def _branch_and_bound_mwis(moduli, deadline, rng):
    """
    Branch-and-bound Maximum Weight Independent Set.
    
    Nodes: (m, r) pairs for m in moduli, r in [0, m).
    Edges: conflict between (m1,r1) and (m2,r2) if not _disjoint(m1,r1,m2,r2).
    Also edges between all (m,r) pairs for same m (since we can only pick one per modulus).
    
    We solve this exactly with branch and bound using bitsets.
    """
    n_moduli = len(moduli)
    m_index = {m: i for i, m in enumerate(moduli)}
    
    # Build node list: each node is (modulus_idx, residue)
    nodes = []
    node_to_modres = []  # (m, r) for each node
    node_weights = []
    node_mod_idx = []
    for i, m in enumerate(moduli):
        for r in range(m):
            nodes.append((i, r))
            node_to_modres.append((m, r))
            node_weights.append(Fraction(1, m))
            node_mod_idx.append(i)
    
    n_nodes = len(nodes)
    
    # Build conflict bitsets: for each node, which other nodes conflict with it
    # conflicts[node] is a bitset (Python int) where bit j is set if node conflicts with node j
    conflicts = [0] * n_nodes
    
    # Precompute: for each pair of moduli (i, j), the set of residues that conflict
    # For moduli i and j with gcd g, residues r_i and r_j conflict if r_i % g == r_j % g
    
    # Build conflict structure more efficiently
    # For each node a, determine which nodes b conflict with a
    for a in range(n_nodes):
        ma, ra = node_to_modres[a]
        ia = node_mod_idx[a]
        conflict_mask = 0
        for b in range(a + 1, n_nodes):
            mb, rb = node_to_modres[b]
            ib = node_mod_idx[b]
            # Same modulus: always conflict (can only pick one residue per modulus)
            if ia == ib:
                conflict_mask |= (1 << b)
                continue
            # Different moduli: check disjointness
            g = math.gcd(ma, mb)
            if (ra % g) == (rb % g):
                conflict_mask |= (1 << b)
        conflicts[a] = conflict_mask
        # Also set bit a in conflicts[b] for b > a
        for b in range(a + 1, n_nodes):
            if (conflict_mask >> b) & 1:
                conflicts[b] |= (1 << a)
    
    # Also: nodes with same modulus conflict with each other (already handled above)
    
    # Precompute adjacency for quick lookup
    # adj_mask[node] = bitset of all nodes that conflict with node
    
    # Upper bound: sum of weights of all remaining non-conflicting nodes
    # We'll use a simple bound: sum of weights of all available nodes
    
    # Branch and bound
    best_weight = Fraction(0, 1)
    best_solution = []  # list of node indices
    
    # Sort nodes by weight descending for better pruning
    # Actually, let's use a different approach: process moduli in order
    # and for each modulus, branch on which residue to pick (or skip)
    
    # Alternative: use a recursive search over moduli
    # State: current modulus index, current set of selected nodes
    # For each modulus, try all valid residues or skip
    
    selected_nodes = []
    used_moduli = set()
    
    def solve(idx, current_weight, current_nodes):
        nonlocal best_weight, best_solution
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound: current weight + sum of 1/m for all remaining moduli
        remaining_weight = Fraction(0, 1)
        for j in range(idx, n_moduli):
            remaining_weight += Fraction(1, moduli[j])
        
        if current_weight + remaining_weight <= best_weight:
            return
        
        if idx == n_moduli:
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = current_nodes[:]
            return
        
        m = moduli[idx]
        
        # Try each residue for this modulus
        for r in range(m):
            # Check if (m, r) is compatible with all currently selected nodes
            ok = True
            for (m2, r2) in current_nodes:
                if not _disjoint(m, r, m2, r2):
                    ok = False
                    break
            if ok:
                current_nodes.append((m, r))
                solve(idx + 1, current_weight + Fraction(1, m), current_nodes)
                current_nodes.pop()
        
        # Also try skipping this modulus
        solve(idx + 1, current_weight, current_nodes)
    
    # This is too slow for all moduli. Let's use a smarter approach.
    # Use the greedy solution as a lower bound, then try to improve.
    
    greedy = _greedy_construction()
    greedy_score = _score(greedy)
    best_weight = Fraction(0, 1)
    for m, r in greedy:
        best_weight += Fraction(1, m)
    best_solution = greedy[:]
    
    # Try to improve by local search and targeted branch-and-bound on subsets
    # Focus on the most promising improvements: adding small moduli that were skipped
    
    skipped_moduli = [m for m in range(20, 201) if m not in set(c[0] for c in greedy)]
    
    # For each skipped modulus, try to add it by adjusting existing residues
    for m_add in skipped_moduli:
        if time.time() > deadline:
            break
        
        # Try to find a residue for m_add that's compatible with current solution
        # If not possible, try swapping one existing modulus's residue or removing one
        for r_add in range(m_add):
            # Check if (m_add, r_add) is compatible with all current classes
            ok = True
            for (m2, r2) in best_solution:
                if not _disjoint(m_add, r_add, m2, r2):
                    ok = False
                    break
            if ok:
                # Can add directly
                new_solution = best_solution[:] + [[m_add, r_add]]
                new_score = best_weight + Fraction(1, m_add)
                if new_score > best_weight:
                    best_weight = new_score
                    best_solution = new_solution
                break
    
    # Try removing one modulus and adding two smaller ones
    for i, (m_remove, r_remove) in enumerate(list(best_solution)):
        if time.time() > deadline:
            break
        
        # Remove modulus m_remove
        remaining = [c for j, c in enumerate(best_solution) if j != i]
        remaining_weight = best_weight - Fraction(1, m_remove)
        
        # Try to add back two moduli that were skipped
        for m1_add in skipped_moduli:
            if m1_add == m_remove:
                continue
            r1 = _find_valid_residue(m1_add, remaining)
            if r1 is None:
                continue
            
            temp_remaining = remaining + [[m1_add, r1]]
            temp_weight = remaining_weight + Fraction(1, m1_add)
            
            for m2_add in skipped_moduli:
                if m2_add == m_remove or m2_add == m1_add:
                    continue
                r2 = _find_valid_residue(m2_add, temp_remaining)
                if r2 is None:
                    continue
                
                new_solution = temp_remaining + [[m2_add, r2]]
                new_weight = temp_weight + Fraction(1, m2_add)
                
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_solution = new_solution
    
    return best_solution

def _local_search_improve(classes, deadline, rng):
    """Try to improve classes by adding more moduli or swapping residues."""
    classes = [c[:] for c in classes]
    used_moduli = set(c[0] for c in classes)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding unselected small moduli
        candidates = [m for m in range(20, 201) if m not in used_moduli]
        candidates.sort()
        
        for m in candidates[:]:
            if time.time() > deadline:
                break
            r = _find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
                improved = True
        
        # Try local search: for each selected class, try different residues
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m_i, r_i = classes[i]
            for _ in range(5):
                if time.time() > deadline:
                    break
                alt_r = rng.randint(0, m_i - 1)
                if alt_r == r_i:
                    continue
                ok = True
                for j in range(len(classes)):
                    if i == j:
                        continue
                    m_j, r_j = classes[j]
                    if not _disjoint(m_i, alt_r, m_j, r_j):
                        ok = False
                        break
                if ok:
                    classes[i][1] = alt_r
                    improved = True
    
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    rng = random.Random(42)
    
    # Step 1: Greedy construction
    greedy = _greedy_construction()
    best_score = _score(greedy)
    best_classes = greedy[:]
    
    # Step 2: Local search improvement
    improved = _local_search_improve(best_classes, deadline - 30, rng)
    if _is_valid(improved):
        sc = _score(improved)
        if sc > best_score:
            best_score = sc
            best_classes = improved[:]
    
    # Step 3: Branch-and-bound style improvement
    if time.time() < deadline - 10:
        bbw = _branch_and_bound_mwis(list(range(20, 201)), deadline - 10, rng)
        if bbw:
            if _is_valid(bbw):
                sc = _score(bbw)
                if sc > best_score:
                    best_score = sc
                    best_classes = bbw[:]
                # Further local search on this result
                improved2 = _local_search_improve(best_classes, deadline - 5, rng)
                if _is_valid(improved2):
                    sc2 = _score(improved2)
                    if sc2 > best_score:
                        best_score = sc2
                        best_classes = improved2[:]
    
    # Step 4: Final local search pass
    if time.time() < deadline - 2:
        final = _local_search_improve(best_classes, deadline - 2, rng)
        if _is_valid(final):
            sc = _score(final)
            if sc > best_score:
                best_score = sc
                best_classes = final[:]
    
    # Ensure validity
    if not _is_valid(best_classes):
        # Fallback to a known valid solution
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
    
    return best_classes
