import time
import random
import math
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

# Precompute all K(2,2,2) configurations
# A K(2,2,2) is defined by 6 distinct vertices partitioned into 3 pairs
# The 8 transversal triples are all combinations picking one from each pair
K222_LIST = []  # List of sets of 8 triple indices
_seen_k222 = set()
for part1 in combinations(range(V), 2):
    for part2 in combinations(range(V), 2):
        if any(x in part1 for x in part2):
            continue
        for part3 in combinations(range(V), 2):
            if any(x in part1 or x in part2 for x in part3):
                continue
            # Generate the 8 transversal triples
            triples_in_k = set()
            for a in part1:
                for b in part2:
                    for c in part3:
                        triples_in_k.add(_tidx(a, b, c))
            key = frozenset(triples_in_k)
            if key not in _seen_k222:
                _seen_k222.add(key)
                K222_LIST.append(list(triples_in_k))

N_K222 = len(K222_LIST)

# For each triple, list which K(2,2,2)s it belongs to
triple_to_k222 = [[] for _ in range(N_TRIPLES)]
for ki, k222 in enumerate(K222_LIST):
    for ti in k222:
        triple_to_k222[ti].append(ki)

# For each K(2,2,2), we need to track how many of its 8 triples are present
# We'll use a counter array

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = 0  # bitmask of present triples
    best_common = [[0] * V for _ in range(V)]
    
    # Weights for triples (for guided search)
    weights = [1.0] * N_TRIPLES
    
    def check_conflict_fast(presence, common, a, b, c):
        """
        Check if adding {a,b,c} creates a K(2,2,2).
        Returns True if conflict, False otherwise.
        """
        mask_bc = common[b][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        mask_ac = common[a][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        mask_ab = common[a][b] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        
        a_p_mask = mask_bc
        while a_p_mask:
            a_p_bit = a_p_mask & (-a_p_mask)
            a_p = a_p_bit.bit_length() - 1
            a_p_mask ^= a_p_bit
            
            b_p_mask = mask_ac & ~(1 << a_p)
            while b_p_mask:
                b_p_bit = b_p_mask & (-b_p_mask)
                b_p = b_p_bit.bit_length() - 1
                b_p_mask ^= b_p_bit
                
                if not (common[a_p][b_p] & (1 << c)):
                    continue
                
                c_cand = (common[a][b] & 
                          common[a][b_p] & 
                          common[a_p][b] & 
                          common[a_p][b_p])
                c_cand &= ~(1 << a)
                c_cand &= ~(1 << b)
                c_cand &= ~(1 << c)
                c_cand &= ~(1 << a_p)
                c_cand &= ~(1 << b_p)
                
                if c_cand:
                    return True
        
        return False
    
    def add_triple(common, idx):
        a, b, c = ALL_TRIPLES[idx]
        common[a][b] |= (1 << c)
        common[b][a] |= (1 << c)
        common[a][c] |= (1 << b)
        common[c][a] |= (1 << b)
        common[b][c] |= (1 << a)
        common[c][b] |= (1 << a)
    
    def remove_triple(common, idx):
        a, b, c = ALL_TRIPLES[idx]
        common[a][b] &= ~(1 << c)
        common[b][a] &= ~(1 << c)
        common[a][c] &= ~(1 << b)
        common[c][a] &= ~(1 << b)
        common[b][c] &= ~(1 << a)
        common[c][b] &= ~(1 << a)
    
    def greedy_init():
        """Weighted greedy initialization."""
        common = [[0] * V for _ in range(V)]
        presence = 0
        count = 0
        remaining = list(range(N_TRIPLES))
        random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline - 15:
                break
            a, b, c = ALL_TRIPLES[idx]
            if not check_conflict_fast(presence, common, a, b, c):
                presence |= (1 << idx)
                add_triple(common, idx)
                count += 1
        
        return presence, common, count
    
    def local_search(presence, common, count):
        """Multi-round hill climbing with perturbations."""
        present_list = [i for i in range(N_TRIPLES) if (presence >> i) & 1]
        missing = [i for i in range(N_TRIPLES) if not (presence >> i) & 1]
        
        best_local_count = count
        best_local_presence = presence
        best_local_common = [row[:] for row in common]
        
        # Multi-round hill climbing
        for round_num in range(10):
            if time.time() > deadline - 8:
                break
            
            # Try to add missing triples
            added = True
            while added:
                added = False
                for idx in missing[:]:
                    if time.time() > deadline - 8:
                        break
                    a, b, c = ALL_TRIPLES[idx]
                    if not check_conflict_fast(presence, common, a, b, c):
                        presence |= (1 << idx)
                        add_triple(common, idx)
                        count += 1
                        missing.remove(idx)
                        present_list.append(idx)
                        added = True
                        if count > best_local_count:
                            best_local_count = count
                            best_local_presence = presence
                            best_local_common = [row[:] for row in common]
            
            # Try swaps: remove one present, try to add multiple missing
            improved = True
            while improved:
                improved = False
                if not present_list or not missing:
                    break
                
                # Try random swaps
                for _ in range(200):
                    if time.time() > deadline - 8:
                        break
                    
                    idx1 = random.choice(present_list)
                    idx2 = random.choice(missing)
                    
                    # Remove idx1
                    remove_triple(common, idx1)
                    presence &= ~(1 << idx1)
                    count -= 1
                    
                    # Try to add idx2 and any other missing that become possible
                    added_any = False
                    for idx3 in missing[:]:
                        a3, b3, c3 = ALL_TRIPLES[idx3]
                        if not check_conflict_fast(presence, common, a3, b3, c3):
                            presence |= (1 << idx3)
                            add_triple(common, idx3)
                            count += 1
                            missing.remove(idx3)
                            present_list.append(idx3)
                            added_any = True
                    
                    if count > best_local_count:
                        best_local_count = count
                        best_local_presence = presence
                        best_local_common = [row[:] for row in common]
                        improved = True
                    else:
                        # Revert
                        # Remove all newly added
                        for idx3 in present_list[:]:
                            if idx3 != idx1 and not (best_local_presence >> idx3) & 1:
                                remove_triple(common, idx3)
                                presence &= ~(1 << idx3)
                                count -= 1
                                present_list.remove(idx3)
                                missing.append(idx3)
                        # Re-add idx1
                        add_triple(common, idx1)
                        presence |= (1 << idx1)
                        count += 1
                        present_list.append(idx1)
                        missing.remove(idx1) if idx1 in missing else None
            
            # Perturbation: remove some random triples to escape local optimum
            if round_num > 0 and round_num % 3 == 0:
                n_remove = random.randint(5, 15)
                for _ in range(n_remove):
                    if present_list:
                        idx = random.choice(present_list)
                        remove_triple(common, idx)
                        presence &= ~(1 << idx)
                        count -= 1
                        present_list.remove(idx)
                        missing.append(idx)
        
        return best_local_presence, best_local_common, best_local_count
    
    # Main search loop with restarts
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Greedy initialization
        presence, common, count = greedy_init()
        
        # Local search
        presence, common, count = local_search(presence, common, count)
        
        if count > best_count:
            best_count = count
            best_presence = presence
            best_common = [row[:] for row in common]
        
        # Update weights based on what was found
        for idx in range(N_TRIPLES):
            if (presence >> idx) & 1:
                weights[idx] *= 1.02
            else:
                weights[idx] *= 0.98
        
        # Normalize weights
        max_w = max(weights) if weights else 1.0
        if max_w > 0:
            weights = [w / max_w for w in weights]
    
    # Final improvement on best solution
    if best_count > 0 and time.time() < deadline - 5:
        common = best_common
        presence = best_presence
        count = best_count
        
        present_list = [i for i in range(N_TRIPLES) if (presence >> i) & 1]
        missing = [i for i in range(N_TRIPLES) if not (presence >> i) & 1]
        
        # Try to add any missing triples
        for idx in missing[:]:
            if time.time() > deadline - 3:
                break
            a, b, c = ALL_TRIPLES[idx]
            if not check_conflict_fast(presence, common, a, b, c):
                presence |= (1 << idx)
                add_triple(common, idx)
                count += 1
                missing.remove(idx)
                present_list.append(idx)
        
        if count > best_count:
            best_count = count
            best_presence = presence
            best_common = [row[:] for row in common]
    
    # Build result
    result = []
    for i in range(N_TRIPLES):
        if (best_presence >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

random.seed(42)
