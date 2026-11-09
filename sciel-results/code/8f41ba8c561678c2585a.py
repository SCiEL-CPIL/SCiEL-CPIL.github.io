import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80  # 80s deadline to stay safely under 90s wall-clock

    # All candidate moduli from 2 to 128
    all_moduli = list(range(2, 129))
    n = len(all_moduli)

    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])

    # Determine which pairs of moduli are "compatible" — i.e., there exists at least one
    # pair of residues (r1, r2) such that (r1 - r2) % gcd(m1, m2) != 0.
    # Two residue classes {x ≡ r1 mod m1} and {x ≡ r2 mod m2} are disjoint iff
    # (r1 - r2) % gcd(m1, m2) != 0.
    # So for a given pair (m1, m2), we can pick residues to make them disjoint as long as
    # gcd(m1, m2) < min(m1, m2), which is always true unless one divides the other in a way that
    # forces congruence... actually, the condition is simply that we can choose r1, r2 with
    # (r1 - r2) % g != 0 where g = gcd(m1, m2). This is always possible as long as g < m1 and g < m2,
    # which is always true for m1 != m2. Wait, no — we need to check if there EXISTS a valid assignment
    # for the entire set, not just pairwise.
    
    # The real question is: given a subset S of moduli, does there exist an assignment of residues
    # r_i for each i in S such that for all i != j, (r_i - r_j) % gcd(m_i, m_j) != 0?
    
    # This is a constraint satisfaction problem. Let's use a smarter approach:
    # 1. Build a graph where nodes are moduli, and an edge between i and j means they CAN coexist
    #    (i.e., there exists residues making them disjoint). Actually, ALL pairs can coexist pairwise.
    # 2. The hard part is finding a subset that admits a GLOBAL assignment.
    
    # Better approach: Use backtracking with strong pruning.
    # Key insight: Process moduli in order of increasing "constraint tightness".
    # A modulus m is more constrained if it shares large gcds with many other moduli.
    
    # Precompute for each modulus, the list of other moduli and their gcds
    # Sort moduli by a heuristic: moduli with more "conflicting" partners first
    
    # For each modulus i, compute a "constraint score": sum of gcd(i, j) for all j
    constraint_scores = []
    for i in range(n):
        score = 0
        for j in range(n):
            if i != j:
                score += gcd_mat[i][j]
        constraint_scores.append(score)
    
    # Sort indices by descending constraint score (most constrained first)
    order = sorted(range(n), key=lambda i: -constraint_scores[i])
    
    # Now do backtracking in this order, trying to build the largest valid set
    # We'll use a recursive backtracking that tries to include each modulus in order,
    # and prunes when no valid residue exists for the current modulus given previous assignments.
    
    best_result = []
    best_size = 0
    node_count = 0
    MAX_NODES = 2000000
    
    # We'll use a different strategy: greedy construction with multiple random restarts,
    # followed by local improvement. This is often faster than full backtracking.
    
    import random
    
    def try_greedy(moduli_subset):
        """Try to assign residues to all moduli in moduli_subset using greedy with backtracking."""
        if not moduli_subset:
            return {}
        
        n_sub = len(moduli_subset)
        # Map from sub-index to global index
        global_indices = [moduli_subset[i] for i in range(n_sub)]
        
        # Precompute local gcd matrix
        local_gcd = [[0] * n_sub for _ in range(n_sub)]
        for i in range(n_sub):
            for j in range(n_sub):
                local_gcd[i][j] = gcd_mat[global_indices[i]][global_indices[j]]
        
        # Greedy with backtracking
        assignment = {}
        
        def backtrack(idx):
            nonlocal node_count
            if node_count > MAX_NODES:
                return False
            node_count += 1
            if time.time() > deadline:
                return False
            
            if idx == n_sub:
                return True
            
            gi = global_indices[idx]
            m = all_moduli[gi]
            
            # Find valid residues for this modulus given current assignments
            valid_rs = []
            for r in range(m):
                ok = True
                for j in range(idx):
                    gj = global_indices[j]
                    g = local_gcd[idx][j]
                    if (r - assignment[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            
            if not valid_rs:
                return False
            
            # Try each valid residue
            for r in valid_rs:
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                del assignment[idx]
            
            return False
        
        # Try in the order given
        if backtrack(0):
            return {global_indices[i]: assignment[i] for i in range(n_sub)}
        return None
    
    # Strategy: Start with all moduli, try to find a valid assignment.
    # If that fails, remove the most problematic modulus and retry.
    # Use a priority queue or greedy removal.
    
    # Actually, let's use a different structural approach:
    # 1. First, find a large set of moduli that are "pairwise coprime" or have small gcds.
    # 2. Then use backtracking to find valid residues.
    
    # Better: Use the insight that if we pick moduli that are powers of 2, they have large gcds
    # with each other. So we should AVOID picking many even moduli together.
    
    # Let's try a different structural approach:
    # Group moduli by their largest prime factor or by their "type".
    # Actually, let's just do a smart backtracking with MRV (Minimum Remaining Values) heuristic.
    
    def solve_with_mrv(moduli_indices):
        """Solve the CSP using MRV heuristic."""
        nonlocal node_count, best_result, best_size
        
        n_sub = len(moduli_indices)
        if n_sub == 0:
            return {}
        
        # Map sub-index to global index
        gi_list = [moduli_indices[i] for i in range(n_sub)]
        
        local_gcd = [[0] * n_sub for _ in range(n_sub)]
        for i in range(n_sub):
            for j in range(n_sub):
                local_gcd[i][j] = gcd_mat[gi_list[i]][gi_list[j]]
        
        assignment = {}
        
        def backtrack(assigned_set):
            nonlocal node_count
            if node_count > MAX_NODES or time.time() > deadline:
                return None
            node_count += 1
            
            if len(assigned_set) == n_sub:
                return {gi_list[i]: assignment[i] for i in range(n_sub)}
            
            # MRV: find unassigned variable with fewest valid residues
            unassigned = [i for i in range(n_sub) if i not in assigned_set]
            if not unassigned:
                return {gi_list[i]: assignment[i] for i in range(n_sub)}
            
            best_idx = -1
            best_valid = None
            best_count = float('inf')
            
            for idx in unassigned:
                m = all_moduli[gi_list[idx]]
                valid_rs = []
                for r in range(m):
                    ok = True
                    for j in assigned_set:
                        g = local_gcd[idx][j]
                        if (r - assignment[j]) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid_rs.append(r)
                
                if not valid_rs:
                    return None  # Dead end
                
                if len(valid_rs) < best_count:
                    best_count = len(valid_rs)
                    best_idx = idx
                    best_valid = valid_rs
                    if best_count == 1:
                        break
            
            if best_idx == -1:
                return None
            
            assigned_set.add(best_idx)
            
            for r in best_valid:
                assignment[best_idx] = r
                result = backtrack(assigned_set)
                if result is not None:
                    return result
                del assignment[best_idx]
            
            assigned_set.remove(best_idx)
            return None
        
        return backtrack(set())
    
    # Phase 1: Try to solve for all moduli
    full_solution = solve_with_mrv(list(range(n)))
    if full_solution:
        result = [[full_solution[i], all_moduli[i]] for i in full_solution]
        return result
    
    # Phase 2: If full set fails, use greedy removal to find the largest subset
    # Start with all moduli, and iteratively remove the one that causes the most conflicts
    
    current_set = list(range(n))
    
    # Try solving for the current set
    while current_set and time.time() < deadline:
        solution = solve_with_mrv(current_set)
        if solution:
            result = [[solution[i], all_moduli[i]] for i in solution]
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
            break
        
        # Remove the most constrained modulus from the current set
        # Heuristic: remove the one with highest constraint score among remaining
        if len(current_set) > 1:
            # Find the modulus with highest sum of gcds with others in the set
            max_score = -1
            max_idx = current_set[0]
            for i in current_set:
                score = 0
                for j in current_set:
                    if i != j:
                        score += gcd_mat[i][j]
                if score > max_score:
                    max_score = score
                    max_idx = i
            current_set.remove(max_idx)
        else:
            break
    
    if best_result:
        return best_result
    
    # Fallback: try a few random subsets
    for _ in range(100):
        if time.time() > deadline:
            break
        random.shuffle(order)
        subset = order[:random.randint(10, 40)]
        solution = solve_with_mrv(subset)
        if solution:
            result = [[solution[i], all_moduli[i]] for i in solution]
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
    
    if best_result:
        return best_result
    
    return [[0, 2]]
