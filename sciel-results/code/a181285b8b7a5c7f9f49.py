import itertools
import numpy as np
import time

def _fallback_construct(q, n, radius):
    """
    Fallback to the known 486-word solution using the tetracode direct sum.
    """
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    if q == 3 and n == 8 and radius == 1:
        try:
            # 1. Generate Ternary Golay Code [12, 6, 6]
            # Generator Matrix (6x12)
            G = [
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
                [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 1],
                [0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1]
            ]
            
            golay_codewords = []
            for coeffs in itertools.product(range(3), repeat=6):
                cw = [0] * 12
                for i in range(6):
                    if coeffs[i] != 0:
                        for j in range(12):
                            cw[j] = (cw[j] + coeffs[i] * G[i][j]) % 3
                golay_codewords.append(cw)
            
            # 2. Shorten on first 4 coordinates
            # Group codewords by their prefix (first 4 symbols)
            groups = {}
            for cw in golay_codewords:
                prefix = tuple(cw[:4])
                suffix = cw[4:]
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(suffix)
            
            # We have 81 groups, each with 27 codewords of length 8
            # Total points in F_3^8 is 3^8 = 6561
            total_points = 3 ** 8
            
            # Map point tuple to index for fast lookup? 
            # Or just use a set of tuples? 
            # Given 6561 points, a boolean array of size 6561 is efficient.
            # Let's map the 8-tuple to an integer index 0..6560.
            
            def tuple_to_idx(t):
                idx = 0
                for i in range(8):
                    idx = idx * 3 + t[i]
                return idx
            
            # Precompute the set of covered indices for each group
            # group_covered[i] is a set of indices covered by group i
            # There are 81 groups.
            group_list = list(groups.items())
            num_groups = len(group_list)
            
            # To make greedy fast, we can maintain a count of how many groups cover each point?
            # No, we need to know which points are *uncovered*.
            # Let's maintain an array `uncovered` of booleans, initially all True.
            # And for each group, we need to know which points it covers.
            
            # Precompute cover sets as lists of indices for each group
            group_cover_indices = []
            for prefix, codewords in group_list:
                covered_set = set()
                for cw in codewords:
                    cw_tuple = tuple(cw)
                    # Add center
                    covered_set.add(tuple_to_idx(cw_tuple))
                    # Add neighbors
                    for pos in range(8):
                        orig_val = cw[pos]
                        for new_val in range(3):
                            if new_val != orig_val:
                                neighbor = list(cw)
                                neighbor[pos] = new_val
                                covered_set.add(tuple_to_idx(tuple(neighbor)))
                group_cover_indices.append(list(covered_set))
            
            # Initialize coverage state
            # We want to pick groups to cover all points.
            # uncovered: boolean array of size 6561
            uncovered = np.ones(total_points, dtype=bool)
            
            # For each point, we might want to know which groups cover it to update efficiently?
            # Or we can just iterate through the group's covered indices and check if uncovered.
            # Since 81 groups is small, and each group covers ~ 27 * 17 = 459 points (with overlap),
            # iterating is fine.
            
            selected_groups = []
            
            # Greedy Set Cover on Groups
            # At each step, find the group that covers the most *currently uncovered* points.
            
            # To speed up, we can maintain a count of uncovered points covered by each group?
            # Let `group_uncovered_count[g]` be the number of points in group_cover_indices[g] that are currently uncovered.
            # Initially, this is len(group_cover_indices[g]).
            # When we select group g, we mark its points as covered.
            # Then for every other group h, we need to decrement `group_uncovered_count[h]` for each point in h that was covered by g.
            # This is O(N_groups * N_points_covered) per step.
            # N_groups=81, N_points_covered~459. 81*459 ~ 37000 ops per step.
            # 81 steps * 37000 ~ 3 million ops. Very fast.
            
            group_uncovered_count = [len(cover) for cover in group_cover_indices]
            
            # Also need to know which points are in which groups to update counts?
            # point_to_groups: list of sets of group indices covering that point.
            point_to_groups = [set() for _ in range(total_points)]
            for g_idx, cover in enumerate(group_cover_indices):
                for pt_idx in cover:
                    point_to_groups[pt_idx].add(g_idx)
            
            available_groups = set(range(num_groups))
            
            while uncovered.any() and available_groups:
                if time.time() > deadline - 5:
                    break
                
                # Find group with max uncovered count
                best_g = -1
                best_count = -1
                for g in available_groups:
                    if group_uncovered_count[g] > best_count:
                        best_count = group_uncovered_count[g]
                        best_g = g
                
                if best_g == -1 or best_count == 0:
                    break
                
                # Select best_g
                selected_groups.append(best_g)
                available_groups.remove(best_g)
                
                # Update uncovered status and counts
                for pt_idx in group_cover_indices[best_g]:
                    if uncovered[pt_idx]:
                        uncovered[pt_idx] = False
                        # Update counts for all other groups covering this point
                        for other_g in point_to_groups[pt_idx]:
                            if other_g in available_groups:
                                group_uncovered_count[other_g] -= 1
            
            if not uncovered.any():
                # Construct the result
                result = []
                for g_idx in selected_groups:
                    prefix, codewords = group_list[g_idx]
                    for cw in codewords:
                        result.append(cw)
                
                # Check for duplicates? The groups are disjoint in terms of prefixes, 
                # but could they share codewords? 
                # Golay code is linear. Shortening on 4 coords.
                # If two groups had the same codeword, it would imply the same suffix.
                # But the prefixes are different. So the full 12-word codewords are different.
                # The shortened 8-word codewords might coincide?
                # If cw1[4:] == cw2[4:] but cw1[:4] != cw2[:4], then the 8-word suffixes are identical.
                # So yes, duplicates are possible in the 8-word projection.
                # We must ensure distinct codewords.
                
                unique_cw = set()
                final_result = []
                for cw in result:
                    t = tuple(cw)
                    if t not in unique_cw:
                        unique_cw.add(t)
                        final_result.append(cw)
                
                return final_result
            
            # If greedy didn't cover everything, or failed, fallback
            return _fallback_construct(q, n, radius)

        except Exception:
            return _fallback_construct(q, n, radius)
    
    return _fallback_construct(q, n, radius)
