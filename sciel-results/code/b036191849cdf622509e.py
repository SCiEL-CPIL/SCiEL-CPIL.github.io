import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcd_matrix = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    # For each modulus m, precompute which residues are compatible with each other modulus
    # compatible[m_idx][m2_idx] = set of residues r for modulus m that are disjoint from ALL residues of modulus m2
    # Actually, we need: for a given (r1, m1) and (r2, m2), they're disjoint iff (r1 - r2) % gcd(m1,m2) != 0
    
    # Let's build the conflict graph differently:
    # Each node is (modulus_idx, residue)
    # Two nodes conflict if they're the same modulus OR if their residue classes intersect
    
    # For branch and bound, we'll work with bitsets
    # Total nodes: sum(m for m in moduli) = sum(2..128) = (2+128)*127/2 = 8255
    # That's manageable with Python integers as bitsets
    
    # Index all nodes
    node_to_mod_idx = []
    node_to_residue = []
    mod_to_nodes = [list() for _ in range(n_moduli)]
    
    node_id = 0
    for m_idx, m in enumerate(moduli):
        for r in range(m):
            node_to_mod_idx.append(m_idx)
            node_to_residue.append(r)
            mod_to_nodes[m_idx].append(node_id)
            node_id += 1
    
    total_nodes = node_id
    
    # Build conflict bitsets
    # conflict[node] = bitmask of all nodes that conflict with this node
    # Two nodes conflict if:
    #   - same modulus, OR
    #   - different moduli m1, m2 with residues r1, r2 where (r1-r2) % gcd(m1,m2) == 0
    
    # Precompute for efficiency
    # For each pair of moduli (m1, m2), precompute which residue pairs conflict
    # Then build the full conflict bitset
    
    # This might be memory intensive: 8255 nodes, each with a bitset of 8255 bits
    # 8255 * 8255 / 8 bytes ≈ 8.5 MB, which is fine
    
    # But building it naively is O(n^2 * something), let's be smart
    
    # Alternative: for branch and bound, we don't need the full conflict graph upfront
    # We can compute conflicts on the fly, but that's slow
    
    # Let's use a different approach: maximum independent set on a graph where
    # we want to pick one node per modulus (or skip a modulus)
    
    # Actually, let's reconsider. The problem is:
    # - We have 127 moduli (2 through 128)
    # - For each modulus m, we can choose at most one residue r in [0, m)
    # - We want to maximize the number of moduli for which we choose a residue
    # - Subject to: for any two chosen (r1, m1) and (r2, m2), (r1-r2) % gcd(m1,m2) != 0
    
    # This is a maximum independent set problem on a specific type of graph.
    
    # Let's use a greedy + local search approach with better heuristics,
    # since a full branch-and-bound on 8255 nodes might be too slow.
    
    # Actually, let's try a smarter approach:
    # 1. Start with all primes (they're always mutually disjoint)
    # 2. Try to add composites using backtracking with pruning
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    
    def find_max_disjoint(moduli_list, deadline):
        """Find maximum set of pairwise disjoint residue classes with given moduli."""
        if not moduli_list:
            return []
        
        # Sort moduli by size (smaller first might be easier to place)
        sorted_mods = sorted(moduli_list)
        
        best = []
        
        def backtrack(idx, selected, remaining_budget):
            nonlocal best
            if time.time() > deadline:
                return
            if idx == len(sorted_mods):
                if len(selected) > len(best):
                    best = selected.copy()
                return
            
            m = sorted_mods[idx]
            
            # Pruning: even if we add all remaining, can we beat best?
            if len(selected) + (len(sorted_mods) - idx) <= len(best):
                return
            
            # Try to find a valid residue for m
            valid_residues = []
            for r in range(m):
                compatible = True
                for m_p, r_p in selected:
                    g = math.gcd(m, m_p)
                    if (r - r_p) % g == 0:
                        compatible = False
                        break
                if compatible:
                    valid_residues.append(r)
            
            # Try each valid residue
            for r in valid_residues:
                selected.append((m, r))
                backtrack(idx + 1, selected, remaining_budget)
                selected.pop()
            
            # Also try skipping this modulus
            backtrack(idx + 1, selected, remaining_budget)
        
        backtrack(0, [], len(sorted_mods))
        return best
    
    # Strategy: Use a two-phase approach
    # Phase 1: Greedy with multiple random restarts
    # Phase 2: Local search to improve
    
    best_solution = []
    
    # Phase 1: Greedy with various orderings
    def greedy_with_order(order, randomize_residue=False):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid_residues = []
            for r in range(m):
                compatible = True
                for m_p, r_p in selected:
                    g = math.gcd(m, m_p)
                    if (r - r_p) % g == 0:
                        compatible = False
                        break
                if compatible:
                    valid_residues.append(r)
            if valid_residues:
                if randomize_residue:
                    r = random.choice(valid_residues)
                else:
                    r = valid_residues[0]
                selected.append((m, r))
        return selected
    
    # Try different orderings
    all_orderings = []
    
    # Primes first
    all_orderings.append(primes + composites)
    
    # Random orderings
    for _ in range(200):
        mods = moduli.copy()
        random.shuffle(mods)
        all_orderings.append(mods)
    
    # Sort by various criteria
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    div_count = {m: count_divisors(m) for m in moduli}
    all_orderings.append(sorted(moduli, key=lambda x: div_count[x]))
    all_orderings.append(sorted(moduli, key=lambda x: -div_count[x]))
    
    # Sort by max gcd with others
    max_gcds = {m: 0 for m in moduli}
    for m in moduli:
        for m2 in moduli:
            if m != m2:
                g = math.gcd(m, m2)
                if g > max_gcds[m]:
                    max_gcds[m] = g
    all_orderings.append(sorted(moduli, key=lambda x: -max_gcds[x]))
    
    for order in all_orderings:
        if time.time() > deadline:
            break
        sol = greedy_with_order(order, randomize_residue=True)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Phase 2: Local search - try to add moduli not in best_solution
    # Or try to change residues to allow more moduli
    
    selected_mods = set(m for m, r in best_solution)
    unselected_mods = [m for m in moduli if m not in selected_mods]
    
    # Try to add each unselected modulus
    for m in unselected_mods:
        if time.time() > deadline:
            break
        # Try all residues for m
        for r in range(m):
            compatible = True
            for m_p, r_p in best_solution:
                g = math.gcd(m, m_p)
                if (r - r_p) % g == 0:
                    compatible = False
                    break
            if compatible:
                # Can we add (m, r)?
                new_solution = best_solution + [(m, r)]
                best_solution = new_solution
                break
    
    # Try local search: for each selected modulus, try changing its residue
    # to see if it allows adding an unselected modulus
    for _ in range(100):  # Limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        selected_mods = set(m for m, r in best_solution)
        unselected_mods = [m for m in moduli if m not in selected_mods]
        
        # Try changing each selected modulus's residue
        for i, (m, r) in enumerate(best_solution):
            for new_r in range(m):
                if new_r == r:
                    continue
                
                # Check if new_r is compatible with all other selected moduli
                compatible = True
                for j, (m_p, r_p) in enumerate(best_solution):
                    if j == i:
                        continue
                    g = math.gcd(m, m_p)
                    if (new_r - r_p) % g == 0:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Try to add an unselected modulus with the new configuration
                test_solution = best_solution.copy()
                test_solution[i] = (m, new_r)
                
                for m_add in unselected_mods:
                    for r_add in range(m_add):
                        add_compatible = True
                        for m_p, r_p in test_solution:
                            g = math.gcd(m_add, m_p)
                            if (r_add - r_p) % g == 0:
                                add_compatible = False
                                break
                        if add_compatible:
                            # We can add m_add
                            best_solution = test_solution + [(m_add, r_add)]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
