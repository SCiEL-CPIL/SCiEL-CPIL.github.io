import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start = time.time()
    DEADLINE = start + 55.0
    
    # All 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)
    
    def can_add_fast(C, F, intersection_map):
        """
        Fast check using precomputed intersection map.
        intersection_map maps each mask to list of indices in F that have this mask.
        For C to be valid, for every pair (i,j) in F, we need NOT (F[i]&C == F[j]&C == F[i]&F[j]).
        Equivalently: for each possible intersection value v, if there are >=2 elements in F 
        that both intersect C in exactly v, AND those elements also intersect each other in v,
        then C is blocked.
        
        Actually, let's think differently: C is blocked if there exist i,j such that
        F[i]&C = F[j]&C = F[i]&F[j].
        
        For a given C, let's compute for each element f in F: inter = f & C.
        Then for each pair (i,j), check if inter_i == inter_j == (F[i] & F[j]).
        
        Alternative: Group elements by their intersection with C.
        For each group G_v = {i : F[i] & C = v}, check if any pair in G_v has F[i]&F[j] == v.
        """
        # Build mapping: intersection value -> list of indices
        inter_groups = defaultdict(list)
        for idx, f in enumerate(F):
            v = f & C
            inter_groups[v].append(idx)
        
        for v, indices in inter_groups.items():
            if len(indices) < 2:
                continue
            # Check all pairs in this group
            for ii in range(len(indices)):
                i = indices[ii]
                for jj in range(ii+1, len(indices)):
                    j = indices[jj]
                    if (F[i] & F[j]) == v:
                        return False, (F[i], F[j])
        return True, None

    def greedy_pass(order):
        family = []
        # For speed, we can use the slower can_add initially since family is small
        # But let's just use can_add_fast with an empty map initially
        for C in order:
            # For greedy, family is growing, so we need to check against all existing
            # Using the simple O(N^2) check is fine for moderate N
            valid = True
            for i in range(len(family)):
                for j in range(i+1, len(family)):
                    v = family[i] & family[j]
                    if (family[i] & C) == v and (family[j] & C) == v:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                family.append(C)
        return family

    best_family = []
    best_size = 0
    
    # We will use a loop that alternates between greedy restarts and local search
    # to avoid getting stuck in a single local optimum.
    
    # Initial phase: Get a decent family
    # The "Inspiration 1" showed that deterministic/lexicographic order works well.
    # Let's try a few deterministic-ish orders and random ones.
    
    # Order 1: Natural (Lexicographic)
    fam1 = greedy_pass(all_sets)
    if len(fam1) > best_size:
        best_family = fam1
        best_size = len(fam1)
        
    # Order 2: Reverse Lexicographic
    fam2 = greedy_pass(all_sets[::-1])
    if len(fam2) > best_size:
        best_family = fam2
        best_size = len(fam2)
        
    # Order 3: Random
    for _ in range(5):
        if time.time() > DEADLINE: break
        order = all_sets[:]
        random.shuffle(order)
        fam3 = greedy_pass(order)
        if len(fam3) > best_size:
            best_family = fam3
            best_size = len(fam3)
            
    # Local Search Phase
    # We will run local search on the best family found so far.
    # If we have time, we might restart local search with a new greedy base.
    
    def local_search(family, deadline):
        family = list(family)
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        
        # Shuffle candidates to ensure variety
        random.shuffle(candidates)
        
        no_improve_count = 0
        max_no_improve = 500  # Reduced to allow more restarts
        
        # Precompute pair intersections for faster checks
        def compute_pair_intersections(F):
            """Returns dict mapping (i,j) -> F[i]&F[j] for i<j"""
            pairs = {}
            for i in range(len(F)):
                for j in range(i+1, len(F)):
                    pairs[(i,j)] = F[i] & F[j]
            return pairs
        
        pair_inter = compute_pair_intersections(family)
        
        def can_add_with_pairs(C, F, pair_inter):
            """Check if C can be added, using precomputed pair intersections."""
            # For each pair (i,j), check if (F[i]&C) == (F[j]&C) == pair_inter[(i,j)]
            for (i,j), v in pair_inter.items():
                if (F[i] & C) == v and (F[j] & C) == v:
                    return False, (F[i], F[j])
            return True, None
        
        def can_add_except_with_pairs(C, F, pair_inter, excluded_idx):
            """Check if C can be added when excluding element at excluded_idx."""
            for (i,j), v in pair_inter.items():
                if i == excluded_idx or j == excluded_idx:
                    continue
                if (F[i] & C) == v and (F[j] & C) == v:
                    return False, (F[i], F[j])
            return True, None
        
        while time.time() < deadline:
            if not candidates:
                break
                
            # Pick a candidate
            idx = random.randint(0, len(candidates) - 1)
            C = candidates[idx]
            
            valid, block_pair = can_add_with_pairs(C, family, pair_inter)
            
            if valid:
                # Add it
                old_len = len(family)
                family.append(C)
                in_family.add(C)
                # Update pair intersections: new pairs involving the new element
                for i in range(old_len):
                    pair_inter[(i, old_len)] = family[i] & C
                    pair_inter[(old_len, i)] = family[i] & C  # For symmetry, though we only use i<j
                candidates.pop(idx)
                no_improve_count = 0
            else:
                # Try to resolve the conflict
                A, B = block_pair
                # Find indices of A and B in family
                A_idx = family.index(A)
                B_idx = family.index(B)
                
                improved = False
                
                # Try swapping A with C
                valid_swap, _ = can_add_except_with_pairs(C, family, pair_inter, A_idx)
                if valid_swap:
                    # Perform swap
                    family[A_idx] = C
                    in_family.discard(A)
                    in_family.add(C)
                    # Update pair intersections
                    for i in range(len(family)):
                        if i == A_idx:
                            continue
                        if i < A_idx:
                            pair_inter[(i, A_idx)] = family[i] & C
                            pair_inter[(A_idx, i)] = family[i] & C
                        else:
                            pair_inter[(A_idx, i)] = family[i] & C
                            pair_inter[(i, A_idx)] = family[i] & C
                    candidates.pop(idx)
                    candidates.append(A)
                    improved = True
                    no_improve_count = 0
                else:
                    # Try swapping B with C
                    valid_swap, _ = can_add_except_with_pairs(C, family, pair_inter, B_idx)
                    if valid_swap:
                        family[B_idx] = C
                        in_family.discard(B)
                        in_family.add(C)
                        # Update pair intersections
                        for i in range(len(family)):
                            if i == B_idx:
                                continue
                            if i < B_idx:
                                pair_inter[(i, B_idx)] = family[i] & C
                                pair_inter[(B_idx, i)] = family[i] & C
                            else:
                                pair_inter[(B_idx, i)] = family[i] & C
                                pair_inter[(i, B_idx)] = family[i] & C
                        candidates.pop(idx)
                        candidates.append(B)
                        improved = True
                        no_improve_count = 0
                    else:
                        # Try a random swap with any member
                        if no_improve_count % 5 == 0:
                            if family:
                                rand_idx = random.randint(0, len(family)-1)
                                if rand_idx != A_idx and rand_idx != B_idx:
                                    R = family[rand_idx]
                                    valid_swap, _ = can_add_except_with_pairs(C, family, pair_inter, rand_idx)
                                    if valid_swap:
                                        family[rand_idx] = C
                                        in_family.discard(R)
                                        in_family.add(C)
                                        # Update pair intersections
                                        for i in range(len(family)):
                                            if i == rand_idx:
                                                continue
                                            if i < rand_idx:
                                                pair_inter[(i, rand_idx)] = family[i] & C
                                                pair_inter[(rand_idx, i)] = family[i] & C
                                            else:
                                                pair_inter[(rand_idx, i)] = family[i] & C
                                                pair_inter[(i, rand_idx)] = family[i] & C
                                        candidates.pop(idx)
                                        candidates.append(R)
                                        improved = True
                                        no_improve_count = 0
                        if not improved:
                            no_improve_count += 1
                            if no_improve_count > max_no_improve:
                                break
                                
        return family

    # Run local search on the best family
    if best_family:
        improved_family = local_search(best_family, DEADLINE - 2.0)
        if len(improved_family) > len(best_family):
            best_family = improved_family

    # If we still have time, do multiple rounds of greedy + local search
    if time.time() < DEADLINE - 5.0:
        for _ in range(5):
            if time.time() > DEADLINE - 5.0: break
            order = all_sets[:]
            random.shuffle(order)
            fam = greedy_pass(order)
            if len(fam) > len(best_family):
                best_family = fam
            
            # Try to improve this new family
            if time.time() < DEADLINE - 2.0:
                improved_fam = local_search(best_family, DEADLINE - 2.0)
                if len(improved_fam) > len(best_family):
                    best_family = improved_fam

    # Final polish: try to add any remaining candidates
    if time.time() < DEADLINE - 1.0 and best_family:
        in_family = set(best_family)
        remaining = [s for s in all_sets if s not in in_family]
        random.shuffle(remaining)
        for C in remaining:
            if time.time() > DEADLINE - 0.5:
                break
            valid = True
            for i in range(len(best_family)):
                for j in range(i+1, len(best_family)):
                    v = best_family[i] & best_family[j]
                    if (best_family[i] & C) == v and (best_family[j] & C) == v:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_family.append(C)

    return [mask_to_list(m) for m in best_family]
