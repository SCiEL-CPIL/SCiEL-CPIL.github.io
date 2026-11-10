import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n = len(moduli)
    
    # Precompute gcds and weights
    # weight[i] = 1/m_i
    weights = [Fraction(1, m) for m in moduli]
    mod_list = moduli
    
    # For each modulus i, determine which residues r are "blocked" by each other modulus j
    # Two classes (m_i, r_i) and (m_j, r_j) conflict iff r_i ≡ r_j (mod gcd(m_i, m_j))
    # For a fixed pair (i, j), the conflict condition depends only on r_i mod g and r_j mod g
    # where g = gcd(m_i, m_j).
    
    # Precompute gcd matrix
    gcds = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(mod_list[i], mod_list[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # For a given assignment of residues to a subset, we need to check pairwise compatibility.
    # 
    # Strategy: Branch and Bound on which moduli to include.
    # For a partial assignment, we track for each unassigned modulus the set of "forbidden residue classes mod g" 
    # imposed by each assigned modulus. Actually, it's simpler to just track the assigned residues and check.
    
    # Better approach for speed: 
    # For each modulus i, precompute for each other modulus j the gcd g_ij.
    # When we assign residue r_i to modulus i, it forbids residues r_j ≡ r_i (mod g_ij) for modulus j.
    # So for each modulus j, we maintain a set of forbidden residue classes: a list of (g, r) pairs,
    # meaning residue r_j is forbidden if r_j ≡ r (mod g).
    # A residue r_j is valid if for ALL (g, r) in the forbidden list for j, r_j % g != r % g.
    
    # This is getting complex. Let me use a simpler but effective approach:
    # 1. Greedy warm-start with multiple random orderings
    # 2. Local search: try to add unselected moduli, try to swap residues
    # 3. If time permits, a limited branch-and-bound
    
    # Greedy construction
    def greedy(order, rng=None):
        """Greedy: process moduli in given order, pick first valid residue."""
        selected = []  # list of (index, residue)
        # For efficiency, track forbidden constraints per unselected modulus
        # For each modulus j, list of (g, r) that forbid residues
        forbidden = [[] for _ in range(n)]
        
        for idx in order:
            m = mod_list[idx]
            # Find a valid residue
            # Check each residue 0..m-1
            found = False
            for r in range(m):
                ok = True
                for (g, fr) in forbidden[idx]:
                    if r % g == fr % g:
                        ok = False
                        break
                if ok:
                    selected.append((idx, r))
                    # Update forbidden lists for all other moduli
                    for j in range(n):
                        if j == idx or (j, None) in [(s[0], None) for s in selected]:
                            continue
                        # Actually, we need to check if j is already selected
                        # Let's use a set for selected indices
                        pass
                    # Simpler: just iterate all j
                    for j in range(n):
                        if j == idx:
                            continue
                        g = gcds[idx][j]
                        forbidden[j].append((g, r))
                    found = True
                    break
        return selected
    
    # Better greedy with proper tracking
    def greedy_v2(order):
        selected_idx = set()
        selected_res = {}
        # forbidden[j] = list of (g, r) pairs
        forbidden = [[] for _ in range(n)]
        
        for idx in order:
            if idx in selected_idx:
                continue
            m = mod_list[idx]
            found = False
            for r in range(m):
                ok = True
                for (g, fr) in forbidden[idx]:
                    if r % g == fr % g:
                        ok = False
                        break
                if ok:
                    selected_idx.add(idx)
                    selected_res[idx] = r
                    # Update forbidden for all j not yet selected
                    for j in range(n):
                        if j == idx or j in selected_idx:
                            continue
                        g = gcds[idx][j]
                        forbidden[j].append((g, r))
                    found = True
                    break
        return [(mod_list[i], selected_res[i]) for i in sorted(selected_idx)]
    
    def score_of(classes):
        s = Fraction(0)
        for m, r in classes:
            s += Fraction(1, m)
        return s
    
    # Try multiple greedy orderings
    best_classes = None
    best_score = Fraction(0)
    
    # Deterministic orderings
    rng = random.Random(42)
    
    orderings = []
    # Natural order
    orderings.append(list(range(n)))
    # Reverse
    orderings.append(list(range(n-1, -1, -1)))
    # Random shuffles
    for _ in range(50):
        o = list(range(n))
        rng.shuffle(o)
        orderings.append(o)
    
    for order in orderings:
        if time.time() > deadline:
            break
        classes = greedy_v2(order)
        s = score_of(classes)
        if s > best_score:
            best_score = s
            best_classes = classes
    
    # Local search: try to add moduli, try residue changes
    def local_search(classes, max_iters=10000):
        """Try to improve by adding unselected moduli or changing residues."""
        selected_set = set()
        res_map = {}
        for m, r in classes:
            idx = mod_list.index(m)
            selected_set.add(idx)
            res_map[idx] = r
        
        # Build forbidden from current selection
        forbidden = [[] for _ in range(n)]
        for idx in selected_set:
            r = res_map[idx]
            for j in range(n):
                if j == idx or j in selected_set:
                    continue
                g = gcds[idx][j]
                forbidden[j].append((g, r))
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try to add an unselected modulus
            unselected = [j for j in range(n) if j not in selected_set]
            # Sort by 1/m descending (small m first)
            unselected.sort(key=lambda j: -1.0/mod_list[j])
            
            for j in unselected:
                if time.time() > deadline:
                    break
                m = mod_list[j]
                # Find a valid residue
                for r in range(m):
                    ok = True
                    for (g, fr) in forbidden[j]:
                        if r % g == fr % g:
                            ok = False
                            break
                    if ok:
                        # Add it
                        selected_set.add(j)
                        res_map[j] = r
                        for k in range(n):
                            if k == j or k in selected_set:
                                continue
                            g = gcds[j][k]
                            forbidden[k].append((g, r))
                        improved = True
                        break
                # Only add one at a time per iteration to keep it simple
                if improved:
                    break
            
            # Try to change residue of a selected modulus to free up space
            # (This is more complex; skip for now if time is tight)
        
        # Rebuild classes
        result = []
        for idx in sorted(selected_set):
            result.append([mod_list[idx], res_map[idx]])
        return result
    
    # Apply local search to best
    if best_classes is not None:
        improved = local_search(best_classes, max_iters=5000)
        s = score_of(improved)
        if s > best_score:
            best_score = s
            best_classes = improved
    
    # Try local search on other good solutions too
    # (Already done implicitly, but let's also try from a few more starting points)
    
    # Convert to required format
    if best_classes is None:
        return []
    
    return best_classes
