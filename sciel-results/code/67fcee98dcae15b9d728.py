import math
import time
import random
from functools import lru_cache

MAX_MOD = 128
DEADLINE = 90.0
MARGIN = 8.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    n_mods = len(all_mods)
    
    # Pre-compute gcds
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # For each pair of moduli, precompute the set of (r1, r2) pairs that are disjoint
    # But this is too large. Instead, for each modulus m and each residue r, 
    # and for each other modulus m', determine which residues of m' are compatible.
    
    # Structural approach: Maximum Independent Set via branch and bound
    # Nodes: (index into all_mods, residue)
    # Edge between two nodes if they are NOT disjoint
    # We want max independent set
    
    # But total nodes = sum(m for m in 2..128) = 128*129/2 - 1 = 8255
    # That's manageable with bitsets.
    
    # Build the conflict graph using bitsets
    # Node id: for modulus m at position i, residues 0..m-1 get node ids
    # Let's assign global node ids
    
    node_mod = []  # node_id -> modulus
    node_res = []  # node_id -> residue
    node_idx = []  # node_id -> index in all_mods
    mod_start = {}  # modulus -> starting node id
    
    node_id = 0
    for i, m in enumerate(all_mods):
        mod_start[m] = node_id
        for r in range(m):
            node_mod.append(m)
            node_res.append(r)
            node_idx.append(i)
            node_id += 1
    
    total_nodes = node_id
    print(f"Total nodes: {total_nodes}")
    
    # Build conflict bitsets: for each node, which other nodes conflict with it
    # Two nodes (r1,m1) and (r2,m2) conflict if (r2-r1) % gcd(m1,m2) == 0
    # i.e., they are NOT disjoint
    
    # Use Python integers as bitsets
    conflict = [0] * total_nodes
    
    for i in range(total_nodes):
        m1 = node_mod[i]
        r1 = node_res[i]
        for j in range(i+1, total_nodes):
            m2 = node_mod[j]
            r2 = node_res[j]
            g = math.gcd(m1, m2)
            if (r2 - r1) % g == 0:
                # They conflict
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    print("Conflict graph built")
    
    # Now solve Maximum Independent Set on this graph
    # Equivalent to Maximum Clique on the complement graph
    # But complement is dense, so let's work with the conflict graph directly
    
    # Branch and bound for Maximum Independent Set:
    # At each step, pick a vertex. Either include it (remove it and its neighbors) 
    # or exclude it (remove it from consideration).
    
    # Use a greedy coloring bound on the complement for upper bound.
    # Actually, for MIS, we can use a greedy coloring on the induced subgraph 
    # to get an upper bound on the size of the MIS.
    
    # Simpler: use a recursive branch and bound with good ordering.
    
    best_set = set()
    best_size = 0
    current_set = set()
    
    # Order vertices by degree (higher degree first for pruning)
    degrees = [bin(conflict[i]).count('1') for i in range(total_nodes)]
    
    # But total_nodes is 8255, and we can't do full BnB on that.
    # We need a smarter approach.
    
    # Alternative structural approach: 
    # Since we can pick at most ONE residue per modulus, the problem is:
    # Select a subset S of moduli, and for each m in S, pick a residue r_m,
    # such that for all pairs m1, m2 in S: (r_m1 - r_m2) % gcd(m1,m2) != 0.
    # Maximize |S|.
    
    # This is a constraint satisfaction problem. Let's use backtracking with 
    # forward checking, ordering moduli by "constraint tightness".
    
    # Key insight: moduli that are coprime to all others impose no constraints.
    # So prioritize moduli with many gcd>1 relationships.
    
    # Compute for each modulus, how many other moduli share gcd > 1
    constraint_count = [0] * n_mods
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j:
                if math.gcd(all_mods[i], all_mods[j]) > 1:
                    constraint_count[i] += 1
    
    # Order moduli by constraint_count descending
    order = sorted(range(n_mods), key=lambda i: -constraint_count[i])
    ordered_mods = [all_mods[i] for i in order]
    ordered_indices = order
    
    # Backtracking with forward checking
    # State: for each modulus in the ordered list, either unassigned, assigned to a residue, or skipped
    # We want to maximize the number of assigned moduli.
    
    # Use a smarter approach: iterate over subsets of moduli (in order of constraint)
    # and check feasibility via constraint propagation.
    
    # Actually, let's use a different structural formulation:
    # This is a Maximum Independent Set on a "modulus-level" graph where
    # edges represent "cannot both be selected with any residue assignment".
    # But that's not quite right either.
    
    # Let me try: backtracking search over moduli in order of constraint_count,
    # at each step try all residues or skip, with forward checking.
    
    # For efficiency, precompute for each pair of moduli (i,j) the set of 
    # (r_i, r_j) pairs that are valid (disjoint).
    
    # valid_pairs[i][j] = set of (r_i, r_j) that are disjoint
    # But storing sets is expensive. Instead, for a given r_i and m_i, 
    # and m_j, the invalid residues for m_j are those where r_j % g == r_i % g.
    
    def get_invalid_residues(r1, m1, m2):
        """Return the set of residues r2 in [0, m2) that conflict with r1 mod m1."""
        g = math.gcd(m1, m2)
        r1_mod_g = r1 % g
        return set(range(r1_mod_g, m2, g))
    
    # Backtracking with forward checking
    # Domain for each modulus: initially all residues 0..m-1
    # When we assign a residue to a modulus, we prune domains of future moduli.
    
    def backtrack(pos, domains, current_assignment):
        """
        pos: position in ordered_mods
        domains: list of sets, domains[i] = valid residues for ordered_mods[i]
        current_assignment: dict from pos to residue
        Returns: (best_count, best_assignment)
        """
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return (0, {})
        
        if pos == n_mods:
            count = len(current_assignment)
            if count > best_size:
                best_size = count
                best_set = dict(current_assignment)
            return (count, current_assignment)
        
        m = ordered_mods[pos]
        
        # Option 1: Skip this modulus
        skip_count, skip_assign = backtrack(pos + 1, domains, current_assignment)
        
        # Option 2: Try each residue in the domain
        best_count = skip_count
        best_assign = skip_assign
        
        for r in list(domains[pos]):
            # Check if r is still valid
            if r not in domains[pos]:
                continue
            
            # Forward check: prune domains of future moduli
            new_domains = [None] * n_mods
            feasible = True
            for j in range(pos + 1, n_mods):
                mj = ordered_mods[j]
                invalid = get_invalid_residues(r, m, mj)
                new_domains[j] = domains[j] - invalid
                if not new_domains[j]:
                    feasible = False
                    break
            # Copy remaining domains
            for j in range(pos + 1, n_mods):
                if new_domains[j] is None:
                    new_domains[j] = domains[j]
            
            if not feasible:
                continue
            
            current_assignment[pos] = r
            count, assign = backtrack(pos + 1, new_domains, current_assignment)
            del current_assignment[pos]
            
            if count > best_count:
                best_count = count
                best_assign = assign
        
        return (best_count, best_assign)
    
    # Initialize domains
    initial_domains = [set(range(m)) for m in ordered_mods]
    
    # This is still exponential. We need a much better approach.
    
    # STRUCTURAL CHANGE: Use a greedy + local search approach with a different 
    # representation. Instead of backtracking over all moduli, use a maximum 
    # independent set heuristic on the conflict graph.
    
    # Let's use a different approach: 
    # 1. Start with all moduli
    # 2. Greedily remove moduli that are "most constrained" until we have a 
    #    feasible set
    # 3. Then try to add back removed moduli
    
    # Actually, let me try a cleaner approach: 
    # Use a maximum independent set algorithm on the modulus-level graph.
    # Two moduli m1, m2 are "compatible" if there EXISTS r1, r2 such that 
    # they are disjoint. This is always true unless m1 | m2 and m2 | m1 
    # (i.e., m1 = m2) or... actually, for any two distinct moduli, there always 
    # exist disjoint residues! Because we can always pick r1 and r2 such that 
    # r1 % gcd(m1,m2) != r2 % gcd(m1,m2).
    
    # Wait, that means ANY subset of moduli is potentially feasible! The question 
    # is whether we can find a consistent assignment of residues.
    
    # So the problem is: what is the maximum size of a subset of moduli for which 
    # a consistent residue assignment exists?
    
    # This is a constraint satisfaction problem. Let me use a smarter search.
    
    # Key: use arc consistency / AC-3 to prune domains, then backtracking.
    
    # But with 127 moduli, this is still hard. Let me try a different angle.
    
    # Insight: The constraints are local (pairwise). This is a binary CSP.
    # We can use a weighted constraint satisfaction approach.
    
    # Let me try: start with all moduli, apply arc consistency, and if domains 
    # become empty, remove the most constrained modulus and retry.
    
    def apply_arc_consistency(moduli_set, domains):
        """Apply AC-3. Returns True if all domains non-empty, False otherwise."""
        queue = [(i, j) for i in range(len(moduli_set)) for j in range(len(moduli_set)) if i != j]
        changed = True
        while queue and changed:
            changed = False
            i, j = queue.pop(0)
            mi = moduli_set[i]
            mj = moduli_set[j]
            g = math.gcd(mi, mj)
            # For each value in domains[i], check if there's a supporting value in domains[j]
            new_di = set()
            for ri in domains[i]:
                ri_mod_g = ri % g
                # Is there an rj in domains[j] with rj % g != ri_mod_g?
                has_support = False
                for rj in domains[j]:
                    if rj % g != ri_mod_g:
                        has_support = True
                        break
                if has_support:
                    new_di.add(ri)
            if new_di != domains[i]:
                domains[i] = new_di
                changed = True
                if not domains[i]:
                    return False
                # Add neighbors of i back to queue
                for k in range(len(moduli_set)):
                    if k != i and k != j:
                        queue.append((k, i))
        return True
    
    # Greedy removal approach
    all_indices = list(range(n_mods))
    removed = set()
    current_set = set(all_indices)
    domains = {i: set(range(all_mods[i])) for i in all_indices}
    
    # Apply AC-3 on the full set
    moduli_list = [all_mods[i] for i in sorted(current_set)]
    idx_to_pos = {idx: pos for pos, idx in enumerate(sorted(current_set))}
    ac_domains = [set(range(all_mods[idx])) for idx in sorted(current_set)]
    
    # This is getting complicated. Let me try a simpler but effective approach.
    
    # APPROACH: Greedy with local search, but with a better initial ordering 
    # and smarter residue selection.
    
    # Order moduli by: number of prime factors (more factors = more constraints)
    def num_prime_factors(m):
        count = 0
        d = 2
        while d * d <= m:
            while m % d == 0:
                count += 1
                m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    # Order by number of distinct prime factors, then by value
    def distinct_prime_factors(m):
        factors = set()
        d = 2
        while d * d <= m:
            while m % d == 0:
                factors.add(d)
                m //= d
            d += 1
        if m > 1:
            factors.add(m)
        return len(factors)
    
    # Moduli with more distinct prime factors are more constrained
    # Sort by number of distinct prime factors descending, then by value
    order2 = sorted(range(n_mods), key=lambda i: (-distinct_prime_factors(all_mods[i]), all_mods[i]))
    
    # Greedy assignment with forward checking
    def greedy_with_fc(order, seed_val):
        random.seed(seed_val)
        assignment = {}
        domains = {i: set(range(all_mods[i])) for i in order}
        
        for pos, idx in enumerate(order):
            m = all_mods[idx]
            if not domains[idx]:
                continue
            
            # Pick a residue that minimizes the impact on future domains
            best_r = None
            best_score = -1
            residues = list(domains[idx])
            random.shuffle(residues)
            
            for r in residues:
                # Score: how many future moduli remain feasible
                score = 0
                for j in range(pos + 1, len(order)):
                    idx2 = order[j]
                    m2 = all_mods[idx2]
                    g = math.gcd(m, m2)
                    r_mod_g = r % g
                    # Count how many residues in domains[idx2] are NOT r_mod_g mod g
                    valid_count = sum(1 for r2 in domains[idx2] if r2 % g != r_mod_g)
                    score += valid_count
                if score > best_score:
                    best_score = score
                    best_r = r
            
            if best_r is not None:
                assignment[idx] = best_r
                # Prune future domains
                for j in range(pos + 1, len(order)):
                    idx2 = order[j]
                    m2 = all_mods[idx2]
                    g = math.gcd(m, m2)
                    r_mod_g = best_r % g
                    domains[idx2] = {r2 for r2 in domains[idx2] if r2 % g != r_mod_g}
        
        return assignment
    
    # Try multiple seeds and keep the best
    best_assignment = {}
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        
        assign = greedy_with_fc(order2, seed)
        count = len(assign)
        
        if count > best_count:
            best_count = count
            best_assignment = assign
    
    print(f"Best count after greedy: {best_count}")
    
    # Local search: try to add more moduli by reassigning residues
    def local_search_improve(assignment, order, t_limit):
        current_count = len(assignment)
        improved = True
        
        while improved and time.time() < t_limit:
            improved = False
            # Try to add each unassigned modulus
            unassigned = [i for i in range(n_mods) if i not in assignment]
            random.shuffle(unassigned)
            
            for idx in unassigned:
                m = all_mods[idx]
                # Try all residues
                found = False
                for r in range(m):
                    # Check if r is compatible with all assigned moduli
                    valid = True
                    for j, rj in assignment.items():
                        mj = all_mods[j]
                        g = math.gcd(m, mj)
                        if r % g == rj % g:
                            valid = False
                            break
                    if valid:
                        assignment[idx] = r
                        current_count += 1
                        improved = True
                        found = True
                        break
                
                if found:
                    continue
            
            if not improved:
                # Try to reassign some moduli to free up space
                assigned_list = list(assignment.keys())
                random.shuffle(assigned_list)
                
                for idx in assigned_list:
                    m = all_mods[idx]
                    old_r = assignment[idx]
                    del assignment[idx]
                    
                    # Try to add an unassigned modulus
                    unassigned2 = [i for i in range(n_mods) if i not in assignment]
                    random.shuffle(unassigned2)
                    
                    added = False
                    for uidx in unassigned2:
                        um = all_mods[uidx]
                        for ur in range(um):
                            # Check ur is compatible with all currently assigned
                            valid = True
                            for j, rj in assignment.items():
                                mj = all_mods[j]
                                g = math.gcd(um, mj)
                                if ur % g == rj % g:
                                    valid = False
                                    break
                            if valid:
                                # Now check if idx can be reassigned
                                for new_r in range(m):
                                    valid2 = True
                                    for j, rj in assignment.items():
                                        mj = all_mods[j]
                                        g = math.gcd(m, mj)
                                        if new_r % g == rj % g:
                                            valid2 = False
                                            break
                                    if valid2:
                                        assignment[uidx] = ur
                                        assignment[idx] = new_r
                                        added = True
                                        break
                                if added:
                                    break
                        if added:
                            break
                    
                    if not added:
                        # Put idx back
                        assignment[idx] = old_r
                    else:
                        improved = True
                        break
        
        return assignment
    
    if time.time() < deadline - 3.0:
        best_assignment = local_search_improve(best_assignment, order2, deadline - 2.0)
        best_count = len(best_assignment)
    
    print(f"Final count: {best_count}")
    
    result = []
    for idx in sorted(best_assignment.keys()):
        m = all_mods[idx]
        r = best_assignment[idx]
        result.append([r, m])
    
    if not result:
        return [[0, 2]]
    
    return result
