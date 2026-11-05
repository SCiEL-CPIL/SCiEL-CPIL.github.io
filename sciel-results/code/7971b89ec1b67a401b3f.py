"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)

    m = len(all_masks)  # C(16,4) = 1820

    # Precompute pairwise intersections
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            inter[i][j] = mi & all_masks[j]
            inter[j][i] = inter[i][j]

    # For each pair (i,j) with intersection value v, precompute the set of k such that
    # inter[i][k] == v and inter[j][k] == v (excluding i, j themselves)
    # This allows O(1) violation check when adding a new element:
    # When adding new_idx, we check if for any pair (a,b) in family,
    # inter[a][b] == inter[a][new_idx] == inter[b][new_idx]
    # i.e., new_idx is in the "conflict set" of pair (a,b)

    # conflict[a][b] = set of indices c (c != a, c != b) such that
    # inter[a][b] == inter[a][c] == inter[b][c]
    # If new_idx is in conflict[a][b] for any pair (a,b) in family, adding new_idx creates a sunflower

    # Precompute for each pair (a,b) the list of c that form a sunflower with a,b
    # Since m=1820, number of pairs is ~1.65M, and for each we'd need to check all c...
    # That's 1820^3 ~ 6 billion, too slow.
    
    # Alternative: for each new_idx, precompute which pairs (a,b) would be violated.
    # For a fixed new_idx, the violated pairs are those (a,b) where:
    # inter[a][b] == inter[a][new_idx] == inter[b][new_idx]
    # Let v_a = inter[a][new_idx], v_b = inter[b][new_idx]
    # We need inter[a][b] == v_a == v_b
    # So v_a must equal v_b, and inter[a][b] must equal that value.
    
    # For each new_idx, build a dict: for each value v, collect all a such that inter[a][new_idx] == v
    # Then for each v, all pairs (a,b) in that group where inter[a][b] == v form violations.
    
    # But this is still expensive. Let's use a different approach:
    # For the greedy algorithm, we maintain for each pair in the family whether it's "active"
    # and check in O(|family|) per addition using a smarter data structure.

    # Actually, let's use the following approach:
    # Maintain a 2D array conflict_count[a][b] = number of c in family such that
    # inter[a][b] == inter[a][c] == inter[b][c]
    # When adding new_idx, for each pair (a,b) in family, check if inter[a][b] == inter[a][new_idx] == inter[b][new_idx]
    # This is O(|family|^2) per addition, which for |family|~200 is 40000 ops per addition, 1820 additions = 72M ops
    # That might be tight in Python. Let's optimize.

    # Better: for each new_idx, precompute a bitmask of which existing family members would conflict.
    # But the family changes, so we need to recompute.

    # Let's just use the straightforward O(|family|^2) check but optimize with NumPy.

    import numpy as np

    # Convert inter to a numpy array for faster operations
    inter_np = np.array(inter, dtype=np.int32)

    def check_violation_idx(family_indices, new_idx):
        """
        Check if adding new_idx to family creates a sunflower violation.
        Uses numpy for speed.
        """
        nf = len(family_indices)
        if nf < 2:
            return False
        if nf == 2:
            a, b = family_indices
            v = inter[a][b]
            return inter[a][new_idx] == v and inter[b][new_idx] == v
        
        farr = np.array(family_indices, dtype=np.int32)
        # Get intersections of new_idx with all family members
        new_inter = inter_np[new_idx, farr]  # shape (nf,)
        # Get pairwise intersections among family members
        # We need to check: for all pairs (i,j) in family, inter[f[i]][f[j]] == new_inter[i] == new_inter[j]
        # i.e., new_inter[i] == new_inter[j] == inter[f[i]][f[j]]
        
        # Create a matrix of new_inter values
        ni_i = new_inter[:, np.newaxis]  # (nf, 1)
        ni_j = new_inter[np.newaxis, :]  # (1, nf)
        
        # Get pairwise inter matrix for family
        fam_inter = inter_np[np.ix_(farr, farr)]  # (nf, nf)
        
        # Check where all three are equal
        # We need ni_i == ni_j AND fam_inter == ni_i (which implies == ni_j)
        # But we only need to check upper triangle (i < j)
        mask_upper = np.triu(np.ones((nf, nf), dtype=bool), k=1)
        
        violation = (ni_i == ni_j) & (fam_inter == ni_i) & mask_upper
        return np.any(violation)

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    best_size = 0
    best_family_idx = []

    # Strategy 1: Greedy in natural order
    family = []
    for idx in range(m):
        if time.time() > deadline:
            break
        if not check_violation_idx(family, idx):
            family.append(idx)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles
    random.seed(12345)
    max_attempts = 50

    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break

        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local improvement with swaps
    def local_improve(family_indices, time_budget):
        """Try to improve family by additions and swaps."""
        improved = family_indices[:]
        in_family = set(improved)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)
        
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            
            # Try additions
            for idx in candidates:
                if time.time() > time_budget:
                    break
                if not check_violation_idx(improved, idx):
                    improved.append(idx)
                    in_family.add(idx)
                    candidates.remove(idx)
                    changed = True
            
            # Try swaps: remove one, add one
            if time.time() > time_budget - 1:
                break
            nf = len(improved)
            for i in range(nf):
                if time.time() > time_budget:
                    break
                removed = improved[i]
                # Temporarily remove
                improved[i] = -1  # placeholder
                # Check which candidates can replace it
                for idx in candidates:
                    if time.time() > time_budget:
                        break
                    # Build new family without removed, with idx
                    # Check if valid
                    new_family = improved[:i] + improved[i+1:] + [idx]
                    # Check for violations in new_family
                    valid = True
                    nnf = len(new_family)
                    if nnf >= 3:
                        nfa = np.array(new_family, dtype=np.int32)
                        ni_matrix = inter_np[np.ix_(nfa, nfa)]
                        # For each triple, check if all pairwise intersections are equal
                        # This is O(n^3) but n is small
                        for a in range(nnf):
                            for b in range(a+1, nnf):
                                v = ni_matrix[a, b]
                                for c in range(b+1, nnf):
                                    if ni_matrix[a, c] == v and ni_matrix[b, c] == v:
                                        valid = False
                                        break
                            if not valid:
                                break
                    if valid:
                        improved[i] = idx
                        in_family.remove(removed)
                        in_family.add(idx)
                        candidates.remove(idx)
                        changed = True
                        break  # Restart after a successful swap
                if not changed or improved[i] == -1:
                    improved[i] = removed  # Restore if no swap happened
        
        return improved

    # Apply local improvement to best so far
    time_budget = deadline - 5
    improved = local_improve(best_family_idx, time_budget)
    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 4: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579, 2468, 1357, 9182, 7530]:
        if time.time() > deadline - 8:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Final local improvement on the best
    if time.time() < deadline - 2:
        improved = local_improve(best_family_idx, deadline - 1)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
