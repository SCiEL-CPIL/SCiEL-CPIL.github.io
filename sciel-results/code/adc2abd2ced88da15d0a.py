"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _max_clique_bitset(adj, n):
    """
    Find maximum clique in a graph represented by bitsets.
    adj[i] is a bitset of neighbors of i.
    Returns the size and the bitset of the clique.
    Uses a recursive branch-and-bound approach.
    """
    best_size = 0
    best_clique = 0
    
    # Order vertices by degree descending to improve pruning
    degrees = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map new index to old index
    old_to_new = {v: i for i, v in enumerate(order)}
    new_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i] & (1 << j):
                new_adj[i] |= (1 << old_to_new[j])
                
    # Re-map adj to new order
    adj_new = [0] * n
    for i in range(n):
        adj_new[i] = new_adj[i]
        
    # Standard Bron-Kerbosch with pivot, or simpler Tomita-style expansion
    # For N=127, a well-optimized recursive search is sufficient.
    
    def expand(clique, candidates):
        nonlocal best_size, best_clique
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
            return
        
        if len(clique) + candidates.bit_count() <= best_size:
            return
            
        # Choose pivot u from candidates U(N)
        # Pivot heuristic: pick u with max degree in candidates
        # Here we just pick the first one or one with high bit count in adj
        # Simple pivot: pick v in candidates that maximizes candidates & adj[v]
        pivot = -1
        max_common = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            common = candidates & adj_new[v]
            c = common.bit_count()
            if c > max_common:
                max_common = c
                pivot = v
            temp ^= lsb
            
        # Branch on candidates NOT adjacent to pivot
        # If candidates & ~adj[pivot] is empty, we can skip? No, we must check.
        # Actually, if pivot is in candidates, we branch on candidates \ N(pivot).
        # If no pivot (candidates empty), handled above.
        
        branch_set = candidates & ~adj_new[pivot] if pivot != -1 else candidates
        
        while branch_set:
            lsb = branch_set & -branch_set
            v = lsb.bit_length() - 1
            
            expand(clique | lsb, candidates & adj_new[v])
            candidates ^= lsb
            branch_set ^= lsb
            
    expand(0, (1 << n) - 1)
    
    # Convert best_clique bitset back to list of original moduli indices
    clique_mods = []
    temp = best_clique
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        clique_mods.append(order[v])
        temp ^= lsb
        
    return best_size, clique_mods

def _find_max_classes(max_mod=128, deadline=None):
    if deadline is None:
        deadline = time.time() + 50  # Safety margin
    
    all_moduli = list(range(2, max_mod+1))
    n = len(all_moduli)
    
    # 1. Build Non-Coprime Graph
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if gcd(all_moduli[i], all_moduli[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Find Maximum Clique
    max_size, clique_mods = _max_clique_bitset(adj, n)
    
    # 3. Solve CSP for Residues
    # Order moduli in clique by "constraint tightness" (smallest modulus first? or highest gcd degree?)
    # Smallest modulus has fewer residues, so it's a good start variable.
    clique_mods_sorted = sorted(clique_mods, key=lambda x: x)
    
    def _solve_csp(mods, deadline):
        """
        Assign residues to mods such that for all i<j, r_i != r_j mod gcd(m_i, m_j).
        Returns list of [r, m] or None.
        """
        n_c = len(mods)
        if n_c == 0:
            return []
        
        # Precompute constraints
        # constraints[i] is a dict: j -> list of forbidden residues mod gcd?
        # Actually, we just check on the fly.
        
        # Order variables: Most Constrained First.
        # A variable m_i is constrained by all m_j.
        # The number of forbidden residues for r_i depends on the gcds.
        # Smaller m_i has fewer total residues, so it's harder to satisfy.
        # Let's stick to sorted by modulus ascending.
        
        def backtrack(idx, assigned):
            if time.time() > deadline:
                return None
            if idx == n_c:
                return assigned[:]
            
            m = mods[idx]
            # Try residues 0..m-1
            # Optimization: Check against all previously assigned
            for r in range(m):
                valid = True
                for j in range(idx):
                    r_j = assigned[j][0]
                    m_j = mods[j]
                    g = gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        valid = False
                        break
                if valid:
                    res = backtrack(idx + 1, assigned + [[r, m]])
                    if res is not None:
                        return res
            return None
            
        return backtrack(0, [])

    solution = _solve_csp(clique_mods_sorted, deadline)
    
    if solution is not None:
        return solution
        
    # If CSP failed (unlikely for max clique in this domain, but possible),
    # we might need to find the next largest clique.
    # For robustness within time limits, we return an empty list or a smaller valid set found by greedy?
    # The task requires a nonempty list.
    # Let's do a quick greedy fallback if the max clique CSP fails.
    
    # Fallback: Greedy
    best_set = []
    used_mods = set()
    for m in sorted(all_moduli, reverse=True):
        can_add = True
        for _, m_old in best_set:
            if gcd(m, m_old) == 1:
                can_add = False
                break
        if not can_add:
            continue
        # Try to find residue
        found = False
        for r in range(m):
            ok = True
            for r_old, m_old in best_set:
                g = gcd(m, m_old)
                if (r - r_old) % g == 0:
                    ok = False
                    break
            if ok:
                best_set.append([r, m])
                found = True
                break
        if found:
            used_mods.add(m)
            
    return best_set if best_set else [[0, 2]]

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
