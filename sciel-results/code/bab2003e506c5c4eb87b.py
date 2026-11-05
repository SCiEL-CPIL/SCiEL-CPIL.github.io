import time
import random
import math
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin

    # STS(15) construction via GF(2)^4 non-zero vectors
    # Points 0..14 correspond to non-zero vectors in GF(2)^4
    # Triple: {i, j, i^j} for i != j, i^j != 0
    sts_triples = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            k = i ^ j
            if k > j:
                t = tuple(sorted([i - 1, j - 1, k - 1]))
                sts_triples.append(t)
    
    n_sts = len(sts_triples)  # Should be 35
    
    # Precompute pairs for each STS triple
    sts_triple_pairs = []
    for t in sts_triples:
        sts_triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
    
    # All pairs on points 0..14
    all_pairs_15 = list(itertools.combinations(range(15), 2))
    pair_set = set(all_pairs_15)
    
    # Precompute: for each pair, which STS triples contain it
    pair_to_sts = {p: [] for p in all_pairs_15}
    for idx, t in enumerate(sts_triples):
        for p in sts_triple_pairs[idx]:
            pair_to_sts[p].append(idx)
    
    # All 8-subsets of 16 points for alpha evaluation
    subsets_8 = list(itertools.combinations(range(16), 8))
    n_subsets_8 = len(subsets_8)
    
    # Precompute for each 8-subset: which STS triples are fully contained
    subset_sts_indices = []
    for s in subsets_8:
        s_set = set(s)
        sts_in_s = []
        for idx, t in enumerate(sts_triples):
            if t[0] in s_set and t[1] in s_set and t[2] in s_set:
                sts_in_s.append(idx)
        subset_sts_indices.append(sts_in_s)
    
    # Also precompute subsets of other sizes for full alpha computation
    # We need to check sizes 8, 7, 6, ... to find exact alpha
    subsets_by_size = {}
    for k in range(16, -1, -1):
        subsets_by_size[k] = list(itertools.combinations(range(16), k))
    
    # For efficient alpha check, precompute masks
    max_mask = 1 << 16
    subset_masks = {}
    for k in range(16, -1, -1):
        for s in subsets_by_size[k]:
            m = 0
            for p in s:
                m |= (1 << p)
            subset_masks[s] = m
    
    # Precompute STS triple masks
    sts_masks = []
    for t in sts_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        sts_masks.append(m)
    
    def compute_alpha(triples):
        """Compute independence number exactly."""
        if not triples:
            return 16
        
        # Build bad_count: for each subset, count how many triples are contained in it
        # Use a dictionary for efficiency since we only need to check specific subsets
        # Actually, let's use the bitmask approach but only for relevant subsets
        
        # A subset S is independent iff no triple from `triples` is fully contained in S
        # For each size k from 16 down to 1, check if any k-subset is independent
        
        # Build a set of triple masks for fast lookup
        triple_mask_set = set()
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_mask_set.add(m)
        
        for k in range(16, 0, -1):
            for s in subsets_by_size[k]:
                s_mask = subset_masks[s]
                # Check if any triple is contained in s
                # A triple (a,b,c) is contained in s iff {a,b,c} ⊆ s
                # Which means s_mask has bits a, b, c set
                # Equivalent: for some triple mask tm, (tm & s_mask) == tm
                contained = False
                for tm in triple_mask_set:
                    if (tm & s_mask) == tm:
                        contained = True
                        break
                if not contained:
                    return k
        return 0
    
    def compute_alpha_fast(triples, triple_masks_list):
        """Faster alpha computation using precomputed triple masks."""
        if not triples:
            return 16
        
        triple_mask_set = set(triple_masks_list)
        
        for k in range(16, 0, -1):
            for s in subsets_by_size[k]:
                s_mask = subset_masks[s]
                contained = False
                for tm in triple_mask_set:
                    if (tm & s_mask) == tm:
                        contained = True
                        break
                if not contained:
                    return k
        return 0
    
    def get_final_triples_from_matching(M):
        """Given a matching M on points 0..14, return the final triple system."""
        M_set = set(M)
        final = []
        final_masks = []
        
        # Add new triples with point 15
        for u, v in M:
            t = (15, min(u, v), max(u, v))
            t = tuple(sorted(t))
            final.append(list(t))
            m = (1 << 15) | (1 << min(u, v)) | (1 << max(u, v))
            final_masks.append(m)
        
        # Add kept STS triples (those not having a pair in M)
        for idx, t in enumerate(sts_triples):
            keep = True
            for p in sts_triple_pairs[idx]:
                if p in M_set:
                    keep = False
                    break
            if keep:
                final.append(list(t))
                final_masks.append(sts_masks[idx])
        
        return final, final_masks
    
    def is_valid_packing(triples):
        """Check if no two triples share more than one point."""
        pair_count = {}
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    p = tuple(sorted((t[i], t[j])))
                    if p in pair_count:
                        return False
                    pair_count[p] = 1
        return True
    
    # Search over matchings using local search
    best_alpha = 16
    best_triples = []
    
    def random_matching():
        pts = list(range(15))
        random.shuffle(pts)
        M = []
        for i in range(0, 14, 2):
            M.append(tuple(sorted((pts[i], pts[i + 1]))))
        return M
    
    def matching_neighbor(M):
        """Generate a random neighbor of matching M by swapping one pair."""
        M = M[:]
        if not M:
            return M
        
        idx_remove = random.randint(0, len(M) - 1)
        removed_pair = M.pop(idx_remove)
        
        used_points = set()
        for u, v in M:
            used_points.add(u)
            used_points.add(v)
        
        available_points = [p for p in range(15) if p not in used_points]
        
        if len(available_points) >= 2:
            pair_candidates = list(itertools.combinations(available_points, 2))
            new_pair = tuple(sorted(pair_candidates[random.randint(0, len(pair_candidates) - 1)]))
            M.append(new_pair)
        
        return M
    
    # Multiple restarts with local search
    n_restarts = 50
    sa_steps_per_restart = 2000
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        M = random_matching()
        triples, triple_masks = get_final_triples_from_matching(M)
        
        if not is_valid_packing(triples):
            continue
        
        alpha = compute_alpha_fast(triples, triple_masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            if best_alpha <= 7:
                break
        
        # Local search / SA
        T = 1.0
        T_min = 0.01
        alpha_decay = 0.999
        
        for step in range(sa_steps_per_restart):
            if time.time() > deadline - 2:
                break
            
            new_M = matching_neighbor(M)
            new_triples, new_masks = get_final_triples_from_matching(new_M)
            
            if not is_valid_packing(new_triples):
                continue
            
            new_alpha = compute_alpha_fast(new_triples, new_masks)
            
            delta = new_alpha - alpha
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                # Accept
                M = new_M
                triples = new_triples
                triple_masks = new_masks
                alpha = new_alpha
                
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_triples = triples[:]
                    if best_alpha <= 7:
                        break

            
            T *= alpha_decay
            if T < T_min:
                T = T_min
        
        if best_alpha <= 7:
            break
    
    # If we haven't found something good, try a more direct SA on arbitrary packings
    # using the bitmask bad_count approach from the original
    if best_alpha > 8:
        # Fallback: use the original SA approach but with better initialization
        points = list(range(16))
        all_triples = list(itertools.combinations(points, 3))
        n_triples = len(all_triples)
        
        triple_pairs = []
        for t in all_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            triple_pairs.append(pairs)
        
        triple_masks_arr = []
        for t in all_triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks_arr.append(mask)
        
        # Precompute supersets for each triple
        triple_supersets = []
        for m in triple_masks_arr:
            supersets = []
            comp = (max_mask - 1) ^ m
            sub = comp
            while True:
                s = m | sub
                supersets.append(s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
            triple_supersets.append(supersets)
        
        # Precompute subsets by popcount
        subsets_by_popcount = [[] for _ in range(17)]
        for k in range(17):
            for s in itertools.combinations(range(16), k):
                m = 0
                for p in s:
                    m |= (1 << p)
                subsets_by_popcount[k].append(m)
        
        # Initialize with STS-based solution
        used_pairs = set()
        current_triples = []
        bad_count = [0] * max_mask
        current_alpha = 16
        
        # Start from the best matching solution found
        if best_triples:
            for t in best_triples:
                for i in range(3):
                    for j in range(i + 1, 3):
                        used_pairs.add(tuple(sorted((t[i], t[j]))))
            current_triples = best_triples[:]
            for t in current_triples:
                idx = all_triples.index(tuple(t))
                for s in triple_supersets[idx]:
                    bad_count[s] += 1
            current_alpha = best_alpha
        else:
            # Random greedy init
            indices = list(range(n_triples))
            random.shuffle(indices)
            for idx in indices:
                if all(p not in used_pairs for p in triple_pairs[idx]):
                    current_triples.append(all_triples[idx])
                    for p in triple_pairs[idx]:
                        used_pairs.add(p)
                    for s in triple_supersets[idx]:
                        bad_count[s] += 1
            # Compute alpha
            for k in range(16, 0, -1):
                found = False
                for m in subsets_by_popcount[k]:
                    if bad_count[m] == 0:
                        found = True
                        break
                if found:
                    current_alpha = k
                    break
        
        # SA
        T = 1.0
        T_min = 0.01
        alpha_decay = 0.9995
        
        while time.time() < deadline - 1:
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                candidates = []
                for idx in range(n_triples):
                    if all(p not in used_pairs for p in triple_pairs[idx]):
                        candidates.append(idx)
                
                if not candidates:
                    continue
                
                idx = random.choice(candidates)
                old_alpha = current_alpha
                
                current_triples.append(all_triples[idx])
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                for s in triple_supersets[idx]:
                    bad_count[s] += 1
                
                # Compute new alpha
                new_alpha = current_alpha
                found = False
                for m in subsets_by_popcount[current_alpha]:
                    if bad_count[m] == 0:
                        found = True
                        break
                if not found:
                    for k in range(current_alpha - 1, 0, -1):
                        for m in subsets_by_popcount[k]:
                            if bad_count[m] == 0:
                                new_alpha = k
                                found = True
                                break
                        if found:
                            break
                            
                delta = new_alpha - current_alpha
                current_alpha = new_alpha
                
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                else:
                    current_triples.pop()
                    for p in triple_pairs[idx]:
                        used_pairs.discard(p)
                    for s in triple_supersets[idx]:
                        bad_count[s] -= 1
                    current_alpha = old_alpha
                    
            elif move_type < 0.7:
                # Remove
                if not current_triples:
                    continue
                
                rem_pos = random.randrange(len(current_triples))
                t_rem = current_triples[rem_pos]
                idx = all_triples.index(tuple(t_rem))
                old_alpha = current_alpha
                
                current_triples.pop(rem_pos)
                for p in triple_pairs[idx]:
                    used_pairs.discard(p)
                for s in triple_supersets[idx]:
                    bad_count[s] -= 1
                
                new_alpha = current_alpha
                for k in range(current_alpha + 1, 17):
                    found = False
                    for m in subsets_by_popcount[k]:
                        if bad_count[m] == 0:
                            found = True
                            break
                    if found:
                        new_alpha = k
                        break
                        
                delta = new_alpha - current_alpha
                current_alpha = new_alpha
                
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                else:
                    current_triples.insert(rem_pos, t_rem)
                    for p in triple_pairs[idx]:
                        used_pairs.add(p)
                    for s in triple_supersets[idx]:
                        bad_count[s] += 1
                    current_alpha = old_alpha
                    
            else:
                # Swap
                if not current_triples:
                    continue
                
                rem_pos = random.randrange(len(current_triples))
                t_rem = current_triples[rem_pos]
                idx_rem = all_triples.index(tuple(t_rem))
                
                current_triples.pop(rem_pos)
                for p in triple_pairs[idx_rem]:
                    used_pairs.discard(p)
                for s in triple_supersets[idx_rem]:
                    bad_count[s] -= 1
                
                candidates = []
                for idx_add in range(n_triples):
                    if idx_add == idx_rem:
                        continue
                    if all(p not in used_pairs for p in triple_pairs[idx_add]):
                        candidates.append(idx_add)
                        
                if not candidates:
                    current_triples.insert(rem_pos, t_rem)
                    for p in triple_pairs[idx_rem]:
                        used_pairs.add(p)
                    for s in triple_supersets[idx_rem]:
                        bad_count[s] += 1
                    continue
                    
                idx_add = random.choice(candidates)
                old_alpha = current_alpha
                
                current_triples.append(all_triples[idx_add])
                for p in triple_pairs[idx_add]:
                    used_pairs.add(p)
                for s in triple_supersets[idx_add]:
                    bad_count[s] += 1
                
                new_alpha = current_alpha
                found = False
                for m in subsets_by_popcount[current_alpha]:
                    if bad_count[m] == 0:
                        found = True
                        break
                if not found:
                    for k in range(current_alpha - 1, 0, -1):
                        for m in subsets_by_popcount[k]:
                            if bad_count[m] == 0:
                                new_alpha = k
                                found = True
                                break
                        if found:
                            break
                            
                delta = new_alpha - current_alpha
                current_alpha = new_alpha
                
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                else:
                    current_triples.pop()
                    for p in triple_pairs[idx_add]:
                        used_pairs.discard(p)
                    for s in triple_supersets[idx_add]:
                        bad_count[s] -= 1
                    current_triples.insert(rem_pos, t_rem)
                    for p in triple_pairs[idx_rem]:
                        used_pairs.add(p)
                    for s in triple_supersets[idx_rem]:
                        bad_count[s] += 1
                    current_alpha = old_alpha
                    
            T *= alpha_decay
            if T < T_min:
                T = T_min
    
    # Final validation
    if best_triples:
        if not is_valid_packing(best_triples):
            # Fallback to a simple valid solution
            best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best_triples
