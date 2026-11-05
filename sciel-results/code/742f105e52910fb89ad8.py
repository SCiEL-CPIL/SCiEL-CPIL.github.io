import itertools
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using cyclic construction.
    Assumes v is prime (19) to use the cyclic group Z_v.
    """
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy if parameters change, though task is fixed
        return _fallback_greedy(v, k, t)

    # 1. Generate all t-subsets and group them into orbits under +1 mod v
    # Orbit representative: rotate so min element is 0
    def canonicalize(subset):
        s = sorted(subset)
        # Find rotation that makes min 0? No, for cyclic group, 
        # we can just rotate the list so that 0 is in it? 
        # Actually, since v is prime, any subset not containing 0 can be rotated to contain 0.
        # But subsets might contain 0 already.
        # Canonical form: rotate so that the smallest element is 0.
        # Since it's a set on Z_v, "smallest" is well-defined in 0..v-1.
        # But the orbit is under addition. 
        # Two subsets are in the same orbit if one is a rotation of the other.
        # To find a representative, we can try all v rotations and pick the lexicographically smallest tuple.
        # v=19, so 19 checks per subset.
        best = None
        for shift in range(v):
            rotated = tuple((x + shift) % v for x in sorted(subset))
            # Sort the rotated tuple to have a canonical order? 
            # No, the rotation preserves relative order in the cycle, but as a set, 
            # we want a canonical representation.
            # Standard way: sort the elements, then rotate the *sequence*?
            # No, a subset is a set. The orbit consists of sets.
            # To compare sets, we sort their elements.
            # So, for a given subset S, generate all rotations R_i.
            # For each R_i, sort it to get a tuple T_i.
            # The canonical representative is min(T_i).
            sorted_rot = tuple(sorted((x + shift) % v for x in subset))
            if best is None or sorted_rot < best:
                best = sorted_rot
        return best

    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_orbits = {}
    for sub in all_t_subs:
        rep = canonicalize(sub)
        if rep not in t_orbits:
            t_orbits[rep] = []
        t_orbits[rep].append(sub)
    
    t_orbit_list = list(t_orbits.keys())
    t_orbit_index = {rep: i for i, rep in enumerate(t_orbit_list)}
    n_t_orbits = len(t_orbit_list)
    
    # 2. Generate all k-subsets and group them into orbits
    all_k_subs = list(itertools.combinations(range(v), k))
    k_orbits = {}
    for sub in all_k_subs:
        rep = canonicalize(sub)
        if rep not in k_orbits:
            k_orbits[rep] = []
        k_orbits[rep].append(sub)
        
    k_orbit_list = list(k_orbits.keys())
    n_k_orbits = len(k_orbit_list)
    
    # 3. For each k-orbit, determine which t-orbits it covers
    # block_masks[i] is a bitmask of t-orbits covered by k-orbit i
    block_masks = [0] * n_k_orbits
    
    # Precompute t-subsets for each k-subset to speed up?
    # We just need to know if a t-subset from t_orbit j is contained in any rotation of k_orbit i.
    # Actually, if a t-subset S is contained in a k-subset K, then the orbit of S is covered by the orbit of K.
    # So for each k-orbit rep, pick one representative K.
    # Find all t-subsets contained in K.
    # For each such t-subset, get its orbit rep.
    # Set the bit for that t-orbit.
    
    for i, k_rep in enumerate(k_orbit_list):
        # k_rep is a sorted tuple of k elements
        mask = 0
        # Get all t-subsets of this k_rep
        for t_sub in itertools.combinations(k_rep, t):
            t_rep = canonicalize(t_sub)
            t_idx = t_orbit_index[t_rep]
            mask |= (1 << t_idx)
        block_masks[i] = mask

    # 4. Greedy Set Cover on the orbit space
    uncovered_mask = (1 << n_t_orbits) - 1
    selected_base_blocks = []
    used_k_orbits = set()
    
    # To allow for better heuristics, we can randomize tie-breaking or use a more advanced greedy
    # But standard greedy is a good start.
    
    while uncovered_mask:
        best_idx = -1
        max_covered = 0
        
        for i in range(n_k_orbits):
            if i in used_k_orbits:
                continue
            covered = (uncovered_mask & block_masks[i]).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_idx = i
            elif covered == max_covered and covered > 0:
                # Tie-breaking: prefer blocks that cover "harder" items?
                # Or just random?
                # Let's stick to first found for determinism, or random for variety.
                # For reproducibility, deterministic is better unless we run multiple times.
                # We can run multiple times with random tie-breaking and keep the best.
                pass
                
        if max_covered == 0:
            break # Should not happen
            
        selected_base_blocks.append(k_orbit_list[best_idx])
        used_k_orbits.add(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 5. Expand base blocks to full blocks
    final_blocks = []
    for base in selected_base_blocks:
        base_set = set(base)
        for shift in range(v):
            block = tuple((x + shift) % v for x in base_set)
            # Ensure distinctness and sorting for consistency
            # Since base blocks are distinct orbits, these generated blocks are distinct.
            final_blocks.append(sorted(block))
            
    # Verify validity (optional but good for safety)
    # Check if all t-subsets are covered
    # This step can be expensive if done naively, but v=19 is small.
    # We can skip exhaustive check if we trust the orbit logic.
    # The orbit logic guarantees that if the t-orbit is covered, all its members are covered.
    # So if uncovered_mask is 0, we are good.
    
    return final_blocks

def _fallback_greedy(v, k, t):
    import itertools
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    block_masks = [0] * n_k
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    uncovered_mask = (1 << n_t) - 1
    blocks = []
    while uncovered_mask:
        best_block_idx = -1
        max_covered = 0
        for i in range(n_k):
            covered = (uncovered_mask & block_masks[i]).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_block_idx = i
        if max_covered == 0:
            break
        blocks.append(list(k_subsets[best_block_idx]))
        uncovered_mask &= ~block_masks[best_block_idx]
    return blocks
