"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools


def _precompute_neighbors(q, n, radius=1):
    """Precompute neighbor indices for all words in {0..q-1}^n within Hamming distance radius."""
    total = q ** n
    if radius == 1:
        num_neighbors = 1 + n * (q - 1)
        neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
        q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        for w in range(total):
            digits = np.array([w // q_powers[i] % q for i in range(n)], dtype=np.int64)
            neighbors[w, 0] = w
            idx = 1
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[pos] = val
                    neighbors[w, idx] = int(np.dot(digits, q_powers))
                    idx += 1
                digits[pos] = orig
        return neighbors
    else:
        raise NotImplementedError("Only radius 1 supported")


def _tetracode_initial(q, n):
    """Generate tetracode-based initial solution."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _word_to_idx(w, q_powers):
    return int(np.dot(np.array(w, dtype=np.int64), q_powers))


def _idx_to_word(idx, q, n):
    result = []
    for i in range(n - 1, -1, -1):
        result.append(idx % q)
        idx //= q
    result.reverse()
    return result


def _greedy_set_cover(q, n, radius, time_limit=30.0):
    """
    Structural change: Set-cover greedy with vectorized marginal gain.
    
    Instead of tetracode algebraic construction, we build the initial solution
    by repeatedly selecting the center that maximizes newly-covered words.
    This is a fundamentally different initialization family.
    """
    start = time.time()
    total = q ** n
    neighbors = _precompute_neighbors(q, n, radius)
    
    covered = np.zeros(total, dtype=np.int8)
    selected = []
    
    while True:
        if time.time() - start > time_limit:
            break
        
        # Vectorized: gain[c] = number of currently uncovered words in ball(c)
        # = sum over j in neighbors[c] of (1 - covered[j])
        uncovered = 1 - covered  # int8 array
        gains = neighbors @ uncovered  # (total,) int32 array
        
        if gains.max() == 0:
            break
        
        best_c = int(np.argmax(gains))
        selected.append(best_c)
        covered[neighbors[best_c]] = 1
    
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    return [_idx_to_word(c, q, n) for c in selected]


def _pairwise_swap_local_search(q, n, radius, initial_codewords, time_limit=90.0):
    """
    Structural change: Deterministic pairwise-swap local search (2-opt style).
    
    Instead of stochastic simulated annealing, we perform:
    1. Redundancy elimination (remove centers whose ball is fully covered by others)
    2. Exhaustive pairwise swap: for each pair (i, j) of selected centers,
       try replacing center i with an unselected center k, and accept if
       the solution remains valid and we can then remove more redundancies.
    
    This is a different optimization paradigm: deterministic, structured
    neighborhood exploration vs. random-walk with temperature.
    """
    start = time.time()
    total = q ** n
    neighbors = _precompute_neighbors(q, n, radius)
    num_neighbors = neighbors.shape[1]
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    # Convert initial codewords to indices
    initial_indices = [_word_to_idx(w, q_powers) for w in initial_codewords]
    
    # count[word] = number of selected centers whose ball contains word
    count = np.zeros(total, dtype=np.int32)
    for ci in initial_indices:
        count[neighbors[ci]] += 1
    
    selected_set = set(initial_indices)
    selected_list = list(initial_indices)
    
    best_size = len(selected_list)
    best_set = set(selected_list)
    
    def try_remove_redundant(selected_list, selected_set, count):
        """Remove all redundant centers. Returns new list/set/count."""
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list) - 1, -1, -1):
                ci = selected_list[i]
                if count[neighbors[ci]].min() > 1:
                    count[neighbors[ci]] -= 1
                    selected_set.discard(ci)
                    selected_list.pop(i)
                    changed = True
        return selected_list, selected_set, count
    
    # Phase 1: Eliminate redundancies
    selected_list, selected_set, count = try_remove_redundant(selected_list, selected_set, count)
    if len(selected_list) < best_size:
        best_size = len(selected_list)
        best_set = set(selected_list)
    
    # Phase 2: Pairwise swap local search
    # For efficiency, we iterate through selected centers and try swapping each
    # with a candidate from a precomputed "promising" set.
    
    # Precompute: for each word, how many selected centers cover it
    # We use count array for this.
    
    # Build a list of all centers not in selected_set for candidate swaps
    all_centers = set(range(total))
    
    # Strategy: for each selected center, try replacing it with each center
    # that covers some of the same words. To limit search, we focus on centers
    # that are "near" the current center (Hamming distance <= 2).
    
    # Precompute centers within Hamming distance 2 of each selected center
    # For efficiency, only compute on demand
    
    def get_nearby_centers(ci, max_dist=2):
        """Get all centers within Hamming distance max_dist of ci."""
        if max_dist == 1:
            return list(neighbors[ci])
        elif max_dist == 2:
            # Centers at distance 0, 1, or 2
            seen = set(neighbors[ci].tolist())
            # For each neighbor at distance 1, get its neighbors at distance 1
            for nb in neighbors[ci, 1:]:  # skip self
                for nb2 in neighbors[nb]:
                    seen.add(int(nb2))
            return list(seen)
        else:
            return list(range(total))
    
    improved = True
    max_passes = 50
    pass_num = 0
    
    while improved and time.time() - start < time_limit and pass_num < max_passes:
        improved = False
        pass_num += 1
        
        # Try removing each selected center and replacing with a better one
        for i in range(len(selected_list)):
            if time.time() - start > time_limit:
                break
            
            ci = selected_list[i]
            
            # Try removing ci and see if we can cover its uniquely-covered words
            # with a single new center
            count[neighbors[ci]] -= 1
            selected_set.discard(ci)
            selected_list.pop(i)
            
            # Find words now uncovered
            uncovered_words = np.where(count == 0)[0]
            
            if len(uncovered_words) == 0:
                # ci was redundant, keep it removed
                improved = True
                if len(selected_list) < best_size:
                    best_size = len(selected_list)
                    best_set = set(selected_list)
                continue
            
            # Try to find a center that covers all uncovered words
            # gain[k] = number of uncovered words in ball(k)
            # We need gain[k] == len(uncovered_words)
            uncovered_mask = np.zeros(total, dtype=np.int8)
            uncovered_mask[uncovered_words] = 1
            gains = neighbors @ uncovered_mask
            
            best_gain = gains.max()
            
            if best_gain == len(uncovered_words):
                # Found a center that covers all uncovered words
                new_c = int(np.argmax(gains))
                count[neighbors[new_c]] += 1
                selected_set.add(new_c)
                selected_list.append(new_c)
                improved = True
                
                # Try to remove more redundancies
                selected_list, selected_set, count = try_remove_redundant(selected_list, selected_set, count)
                if len(selected_list) < best_size:
                    best_size = len(selected_list)
                    best_set = set(selected_list)
            else:
                # Revert: put ci back
                count[neighbors[ci]] += 1
                selected_set.add(ci)
                selected_list.insert(i, ci)
    
    # Final cleanup
    selected_list, selected_set, count = try_remove_redundant(selected_list, selected_set, count)
    
    # Use best found
    if len(selected_list) > best_size:
        selected_list = list(best_set)
    
    # Verify coverage
    final_count = np.zeros(total, dtype=np.int32)
    for ci in selected_list:
        final_count[neighbors[ci]] += 1
    if final_count.min() < 1:
        selected_list = list(best_set)
    
    return [_idx_to_word(ci, q, n) for ci in selected_list]


def _hybrid_search(q, n, radius, time_limit=120.0):
    """
    Structural change: Multi-phase hybrid search combining different families.
    
    Phase 1: Set-cover greedy (different from tetracode)
    Phase 2: Pairwise-swap local search (different from SA)
    Phase 3: Random restarts with perturbation if time remains
    """
    start = time.time()
    deadline = start + time_limit
    
    # Phase 1: Greedy set cover
    greedy_time = min(30.0, time_limit * 0.3)
    initial = _greedy_set_cover(q, n, radius, time_limit=greedy_time)
    
    # Also try tetracode as alternative initial
    tetracode = _tetracode_initial(q, n)
    
    # Pick the smaller initial
    if len(tetracode) < len(initial):
        initial = tetracode
    
    best_result = initial
    best_size = len(initial)
    
    # Phase 2: Local search
    ls_time = min(80.0, deadline - time.time() - 10)
    if ls_time > 5:
        result = _pairwise_swap_local_search(q, n, radius, initial, time_limit=ls_time)
        
        # Verify
        total = q ** n
        q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        neighbors = _precompute_neighbors(q, n, radius)
        count = np.zeros(total, dtype=np.int32)
        for w in result:
            ci = _word_to_idx(w, q_powers)
            count[neighbors[ci]] += 1
        
        if count.min() >= 1 and len(result) < best_size:
            best_result = result
            best_size = len(result)
    
    # Phase 3: If time remains, try a few random perturbations + local search
    remaining = deadline - time.time() - 5
    if remaining > 15:
        rng = random.Random(123)
        total = q ** n
        q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        neighbors = _precompute_neighbors(q, n, radius)
        
        # Convert best_result to indices
        best_indices = [_word_to_idx(w, q_powers) for w in best_result]
        
        num_restarts = min(3, int(remaining // 15))
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Perturb: remove 5-10 random centers, then re-greedy
            perturbed = list(best_indices)
            num_remove = min(10, max(5, len(perturbed) // 20))
            for _ in range(num_remove):
                if not perturbed:
                    break
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
            
            # Re-greedy to fill gaps
            count = np.zeros(total, dtype=np.int32)
            for ci in perturbed:
                count[neighbors[ci]] += 1
            
            # Greedy fill
            while True:
                if time.time() > deadline - 5:
                    break
                uncovered = (count == 0)
                if not uncovered.any():
                    break
                uncovered_mask = uncovered.astype(np.int8)
                gains = neighbors @ uncovered_mask
                if gains.max() == 0:
                    break
                best_c = int(np.argmax(gains))
                if best_c in perturbed:
                    # Mark as covered but don't add (already there)
                    count[neighbors[best_c]] += 1
                    continue
                perturbed.append(best_c)
                count[neighbors[best_c]] += 1
            
            # Local search on perturbed
            if time.time() < deadline - 10:
                ls_t = min(10.0, deadline - time.time() - 5)
                perturbed_words = [_idx_to_word(ci, q, n) for ci in perturbed]
                result = _pairwise_swap_local_search(q, n, radius, perturbed_words, time_limit=ls_t)
                
                # Verify
                count2 = np.zeros(total, dtype=np.int32)
                for w in result:
                    ci = _word_to_idx(w, q_powers)
                    count2[neighbors[ci]] += 1
                
                if count2.min() >= 1 and len(result) < best_size:
                    best_result = result
                    best_size = len(result)
    
    return best_result


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # Leave 10s margin for safety
    
    # Try hybrid search
    try:
        result = _hybrid_search(q, n, radius, time_limit=deadline - start - 5)
        
        # Final verification
        total = q ** n
        q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        neighbors = _precompute_neighbors(q, n, radius)
        count = np.zeros(total, dtype=np.int32)
        for w in result:
            ci = _word_to_idx(w, q_powers)
            count[neighbors[ci]] += 1
        
        if count.min() >= 1:
            return result
    except Exception:
        pass
    
    # Fallback: tetracode
    return _tetracode_initial(q, n)
# EVOLVE-BLOCK-END
