import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))
    n = len(moduli)
    
    # Map modulus to index
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Build coprime graph (edges if gcd == 1)
    # We want Max Independent Set in this graph.
    # This is equivalent to Max Clique in the complement (Non-Coprime Graph).
    # Let's build the Non-Coprime Graph (NCG) adjacency bitsets.
    # NCG[i] has bit j set if gcd(moduli[i], moduli[j]) > 1.
    
    ncg = [0] * n
    for i in range(n):
        mi = moduli[i]
        for j in range(i + 1, n):
            mj = moduli[j]
            if gcd(mi, mj) > 1:
                ncg[i] |= (1 << j)
                ncg[j] |= (1 << i)
                
    # Bron-Kerbosch with pivot for Max Clique in NCG.
    # This finds the largest set of moduli with pairwise gcd > 1.
    
    best_clique = 0
    best_clique_nodes = 0 # bitmask
    
    def bronk(R, P, X):
        nonlocal best_clique, best_clique_nodes
        if time.time() > deadline - 5.0:
            return
        if not P and not X:
            size = bin(R).count('1')
            if size > best_clique:
                best_clique = size
                best_clique_nodes = R
            return
        
        # Pivot selection: choose u from P ∪ X with max degree in P
        # To maximize pruning, pick u with max neighbors in P.
        u = 0
        max_deg = -1
        candidates = P | X
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            deg = (P & ncg[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                u = v
            candidates ^= lsb
            
        # Recurse on P \ N(u)
        # If u is in P, N(u) is ncg[u] & P.
        # If u is in X, N(u) is ncg[u] & P.
        # In both cases, neighbors in P are ncg[u] & P.
        neighbors_in_P = ncg[u] & P
        # We iterate over v in P \ N(u)
        candidates = P & ~neighbors_in_P
        
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            bronk(R | lsb, P & ncg[v], X & ncg[v])
            P &= ~lsb
            X |= lsb
            
    # Initial call
    # R = 0, P = all nodes, X = 0
    all_nodes_mask = (1 << n) - 1
    bronk(0, all_nodes_mask, 0)
    
    # Extract moduli from best_clique_nodes
    selected_moduli = []
    mask = best_clique_nodes
    while mask:
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        selected_moduli.append(moduli[idx])
        mask ^= lsb
        
    if not selected_moduli:
        return [[0, 2]]
        
    # Now we have a set of moduli with pairwise gcd > 1.
    # We need to assign residues r_i such that r_i != r_j mod gcd(m_i, m_j).
    # This is a CSP.
    
    # Sort by modulus size? Or by number of constraints?
    # Let's try to solve using backtracking.
    # Order: Most constrained first.
    # Constraint count for m_i is (k-1) for all i, but the modulus of the constraint is gcd(m_i, m_j).
    # Smaller gcd means stricter constraint? No, r mod g.
    # If g is large, fewer residues are forbidden?
    # Forbidden residues for m_i relative to m_j: r_i cannot be congruent to r_j mod g.
    # This forbids m_i/g residues out of m_i? No.
    # The condition is r_i % g != r_j % g.
    # For a fixed r_j, there are m_i/g values of r_i in [0, m_i) that satisfy r_i % g == r_j % g.
    # So it forbids m_i/g residues.
    # Larger m_i/g means more forbidden residues.
    # So moduli with large gcd with others are more constrained?
    # Actually, if gcd is small, m_i/g is large, so more residues are forbidden.
    # So pairs with small gcd are "tighter" constraints.
    
    # Let's just try a simple greedy with backtracking.
    # Sort by modulus descending?
    
    def solve_residues(mods):
        k = len(mods)
        if k == 0:
            return []
        
        # Precompute gcds
        gcds = [[1]*k for _ in range(k)]
        for i in range(k):
            for j in range(i+1, k):
                g = gcd(mods[i], mods[j])
                gcds[i][j] = g
                gcds[j][i] = g
                
        # Order variables
        # Heuristic: sort by modulus, or by degree in constraint graph?
        # Let's try sorting by modulus ascending (smaller moduli have fewer residues to try).
        order = sorted(range(k), key=lambda i: mods[i])
        
        assigned = [None] * k
        
        def backtrack(idx):
            if time.time() > deadline - 2.0:
                return False
            if idx == k:
                return True
            
            i = order[idx]
            m_i = mods[i]
            
            for r_i in range(m_i):
                ok = True
                for j in range(idx):
                    j_orig = order[j]
                    g = gcds[i][j_orig]
                    if (r_i % g) == (assigned[j_orig] % g):
                        ok = False
                        break
                if ok:
                    assigned[i] = r_i
                    if backtrack(idx + 1):
                        return True
                    assigned[i] = None
            return False
            
        if backtrack(0):
            return [[assigned[i], mods[i]] for i in range(k)]
        else:
            return None
            
    sol = solve_residues(selected_moduli)
    
    if sol:
        return sol
    else:
        # Fallback to current heuristic if B&B set fails residue assignment
        # (Unlikely, but possible)
        pass

    # Fallback: Use the existing greedy logic if the exact method fails or times out
    # Re-implement the fast greedy from the original code as fallback
    
    gcd_table = [[0] * (MOD_MAX + 1) for _ in range(MOD_MAX + 1)]
    for i in range(2, MOD_MAX + 1):
        for j in range(i, MOD_MAX + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def disjoint(r1, m1, r2, m2):
        g = gcd_table[m1][m2]
        if g <= 1:
            return False
        return (r1 % g) != (r2 % g)

    def valid_residues(m, selected):
        out = []
        for r in range(m):
            ok = True
            for r_k, m_k in selected:
                if not disjoint(r, m, r_k, m_k):
                    ok = False
                    break
            if ok:
                out.append(r)
        return out

    def most_constrained_greedy():
        remaining = set(moduli)
        selected = []
        while remaining:
            if time.time() > deadline:
                break
            best_m = None
            best_vr = None
            best_count = None
            for m in remaining:
                vr = valid_residues(m, selected)
                c = len(vr)
                if c == 0:
                    continue
                if best_count is None or c < best_count:
                    best_count = c
                    best_m = m
                    best_vr = vr
                    if c == 1:
                        break
            if best_m is None:
                break
            r = best_vr[0]
            selected.append((r, best_m))
            remaining.discard(best_m)
        return [[r, m] for r, m in selected]

    return most_constrained_greedy()
