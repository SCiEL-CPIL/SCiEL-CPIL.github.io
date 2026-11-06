import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Generate all 4-subsets of [1, 16] as bitmasks and store original tuples
    candidates = []
    candidate_tuples = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        candidate_tuples.append(comb)
    
    N = len(candidates)  # 1820

    # Precompute pairwise intersections
    # pair_int[i][j] = candidates[i] & candidates[j]
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    # Precompute for each pair (i, j) the set of k such that {i, j, k} forms a sunflower.
    # A sunflower with core C requires:
    #   A ∩ B = C
    #   A ∩ C = C
    #   B ∩ C = C
    # Here A=candidates[i], B=candidates[j], C=candidates[k].
    # So we need:
    #   pair_int[i][j] == pair_int[i][k] == pair_int[j][k]
    
    # For efficiency, we can precompute for each pair (i, j) the intersection value v = pair_int[i][j].
    # Then for a candidate k to form a sunflower with i and j, we must have:
    #   pair_int[i][k] == v AND pair_int[j][k] == v.
    
    # Let's build a structure to quickly check conflicts.
    # For a given pair (i, j) with intersection v, we want to know if any k in the current family
    # has pair_int[i][k] == v and pair_int[j][k] == v.
    
    # This is still O(N) per check if we just scan.
    # However, we can precompute for each (i, v) the list of k such that pair_int[i][k] == v.
    # Then for a pair (i, j) with v, we check intersection of list[i][v] and list[j][v] against the current family.
    
    # Actually, simpler approach for the "add" check:
    # When adding k to family F:
    # For every pair (i, j) in F, check if (i, j, k) is a sunflower.
    # This is O(|F|^2) per addition. Total O(N * |F|^2) which is ~ 1820 * 250^2 = 113M ops.
    # In Python, this might be too slow for 55 seconds if done naively with nested loops and list lookups.
    
    # Let's optimize the check.
    # We can maintain a set of "blocked" pairs or something? No, the conflict depends on the triple.
    
    # Alternative:
    # For each k, precompute the set of pairs (i, j) that form a sunflower with k.
    # Let conflicts[k] be a set of frozensets {i, j} or just a set of pairs (min(i,j), max(i,j)).
    # Then when adding k, we check if any pair in conflicts[k] is fully contained in the current family.
    
    # How to build conflicts[k]?
    # For a fixed k, a pair (i, j) forms a sunflower with k if:
    #   pair_int[i][j] == pair_int[i][k] == pair_int[j][k]
    # Let v_ik = pair_int[i][k], v_jk = pair_int[j][k], v_ij = pair_int[i][j].
    # Condition: v_ik == v_jk == v_ij.
    
    # We can iterate i, j for each k? That's N^3 = 1820^3 ~ 6e9. Too slow.
    
    # Better:
    # For each pair (i, j), let v = pair_int[i][j].
    # We want to find all k such that pair_int[i][k] == v and pair_int[j][k] == v.
    # Let S_i_v = {k | pair_int[i][k] == v}
    # Let S_j_v = {k | pair_int[j][k] == v}
    # Then the set of k's is S_i_v ∩ S_j_v.
    # For each such k, we add the pair (i, j) to conflicts[k].
    
    # Precomputation cost:
    # Iterate all pairs (i, j): N^2 / 2 ~ 1.6M pairs.
    # For each pair, compute v.
    # Look up S_i_v and S_j_v.
    # Intersect them.
    # For each k in intersection, add (i, j) to conflicts[k].
    
    # How to represent S_i_v?
    # For each i, we can have a dict: mask -> list of k's.
    # Let's build:
    # mask_to_ks[i] = { mask: [k1, k2, ...] }
    
    mask_to_ks = [{} for _ in range(N)]
    for i in range(N):
        for k in range(N):
            if i == k:
                continue
            v = pair_int[i][k]
            if v not in mask_to_ks[i]:
                mask_to_ks[i][v] = []
            mask_to_ks[i][v].append(k)
    
    # Now build conflicts
    # conflicts[k] = set of pairs (i, j) such that (i, j, k) is a sunflower.
    # We can use a list of sets, or a list of lists and then convert to set for fast lookup?
    # Since we need to check "is pair (i, j) in conflicts[k]", a set of tuples is good.
    
    conflicts = [set() for _ in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            v = pair_int[i][j]
            # Get lists of k's
            list_ik = mask_to_ks[i].get(v, [])
            list_jk = mask_to_ks[j].get(v, [])
            
            if not list_ik or not list_jk:
                continue
            
            # Intersect the two lists
            # Since they are lists, converting to set is O(len).
            # The lists are likely small (intersection of 4-sets is 0,1,2,3 elements).
            # Number of k's with specific intersection with i is small.
            set_ik = set(list_ik)
            common_ks = set_ik.intersection(list_jk)
            
            for k in common_ks:
                # (i, j, k) is a sunflower
                # Add (i, j) to conflicts[k]
                # Also add (i, k) to conflicts[j] and (j, k) to conflicts[i]?
                # No, the conflict is symmetric. If we are adding k, we check if any pair in F conflicts with k.
                # The pair is (i, j).
                conflicts[k].add((i, j))
                
                # Wait, if we add i, we need to know if (j, k) is a conflict for i.
                # So we should add all 3 pairs to the respective conflict sets.
                # conflicts[i].add((j, k))
                # conflicts[j].add((i, k))
                # conflicts[k].add((i, j))
                
                # Let's do that to be safe and symmetric.
                conflicts[i].add((j, k))
                conflicts[j].add((i, k))
                conflicts[k].add((i, j))

    # Now the check is:
    # To add k to family F (represented as a set of indices):
    # For each pair (i, j) in conflicts[k]:
    #   if i in F and j in F: return False
    # return True
    
    # This is O(|conflicts[k]|) per check.
    # How large is |conflicts[k]|?
    # It's the number of pairs (i, j) that form a sunflower with k.
    # In the worst case, this could be large, but hopefully manageable.
    # Total edges in conflict graph is sum |conflicts[k]|.
    # This precomputation might be heavy but let's try.
    
    # Optimization:
    # We can precompute the list of pairs for each k as a list, and also maybe index them?
    # But checking "i in F and j in F" is O(1) if F is a set.
    
    # Let's implement the greedy with this check.
    
    best_family = []
    best_size = 0
    
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        random.seed(attempts)
        
        # Randomized greedy
        # Order candidates randomly
        order = list(range(N))
        random.shuffle(order)
        
        family = []
        fam_set = set()
        
        for idx in order:
            # Check if idx can be added
            can_add = True
            for pair in conflicts[idx]:
                i, j = pair
                if i in fam_set and j in fam_set:
                    can_add = False
                    break
            
            if can_add:
                family.append(idx)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # Local search / Second pass
        if time.time() < deadline - 5.0:
            current_best = best_family.copy()
            current_best_set = set(current_best)
            remaining = [i for i in range(N) if i not in current_best_set]
            random.shuffle(remaining)
            
            improved = False
            for idx in remaining:
                can_add = True
                for pair in conflicts[idx]:
                    i, j = pair
                    if i in current_best_set and j in current_best_set:
                        can_add = False
                        break
                
                if can_add:
                    current_best.append(idx)
                    current_best_set.add(idx)
                    improved = True
            
            if improved and len(current_best) > best_size:
                best_size = len(current_best)
                best_family = current_best

    # Convert best_family (list of indices) to a list of tuples
    result = []
    for idx in best_family:
        result.append(candidate_tuples[idx])
        
    return result

# Run construct to verify it works and doesn't crash immediately
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
