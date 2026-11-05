"""Combined program: NumPy Greedy Set Cover + Iterative Redundancy Pruning."""
import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: finish by 110s, limit is 120s
    
    total_words = q ** n
    
    # Helper: convert word (tuple/list) to index
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Helper: convert index to word (tuple)
    def index_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))

    # Helper: get neighbors (indices) of a word index within radius 1
    # Optimized for radius=1
    def get_neighbors(idx):
        neighbors = [idx]
        for pos in range(n):
            pow_q = q ** (n - 1 - pos)
            current_val = (idx // pow_q) % q
            for new_val in range(q):
                if new_val != current_val:
                    neighbors.append(idx + (new_val - current_val) * pow_q)
        return neighbors

    # If the space is small enough for a fast greedy, use it.
    if total_words <= 100000:
        
        if radius == 1:
            sphere_size = 1 + n * (q - 1)
            # Build neighbor index table: neighbors[i] = list of indices within dist 1 of i
            # Using a flat array for speed in NumPy operations
            neighbors = np.empty((total_words, sphere_size), dtype=np.int32)
            for i in range(total_words):
                nbrs = get_neighbors(i)
                # Ensure length matches sphere_size (it should for radius 1)
                if len(nbrs) < sphere_size:
                    # Pad if necessary (should not happen for radius 1)
                    nbrs = nbrs + [i] * (sphere_size - len(nbrs))
                neighbors[i] = nbrs[:sphere_size]

            # --- Phase 1: NumPy Greedy Set Cover ---
            covered = np.zeros(total_words, dtype=bool)
            code_indices = []

            # We will run the greedy until no more gains or time limit
            # To improve quality, we can use a "lazy" evaluation or restart
            # but for now, standard greedy is fast.
            
            while time.time() < deadline - 20.0:  # Leave 20s for pruning
                # Gains: how many uncovered words each candidate covers
                gains = (~covered)[neighbors].sum(axis=1)
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                code_indices.append(best_idx)
                covered[neighbors[best_idx]] = True

            # Convert indices to codewords
            result_greedy = []
            for idx in code_indices:
                result_greedy.append(list(index_to_word(idx)))

            # --- Phase 2: Direct Sum Construction Baseline ---
            # This provides a structurally strong alternative
            if q == 3 and n >= 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                base_n = 4
                base_code = tetracode
            elif q >= 3 and n >= 4:
                base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
                base_n = 4
            else:
                base_n = n
                base_code = [list(range(n))] # Fallback, weak

            if n >= base_n:
                tails = list(itertools.product(range(q), repeat=n - base_n))
                result_direct = [h + list(tail) for h in base_code for tail in tails]
            else:
                result_direct = [list(range(n))]

            # Choose the better starting point for pruning
            # Usually greedy is better, but direct sum might be better for some topologies
            if len(result_greedy) <= len(result_direct):
                initial_code = result_greedy
            else:
                initial_code = result_direct

            # --- Phase 3: Iterative Redundancy Pruning ---
            # This is the key refinement from Inspiration 1
            # We try to remove codewords that are redundant.
            
            # Precompute coverage for the initial code
            # covered_by[wi] = list of codeword indices that cover word wi
            # codeword_covers[ci] = set of word indices covered by codeword ci
            
            total_words_int = total_words
            covered_by = [[] for _ in range(total_words_int)]
            codeword_covers = []
            
            for ci, c in enumerate(initial_code):
                c_idx = word_to_index(tuple(c))
                covers = set(get_neighbors(c_idx))
                codeword_covers.append(covers)
                for wi in covers:
                    covered_by[wi].append(ci)
            
            active = [True] * len(initial_code)
            
            # Pruning loop
            # We iterate until no more removals or time runs out
            max_prune_iterations = 50
            for iteration in range(max_prune_iterations):
                if time.time() > deadline - 5.0:
                    break
                
                removed_any = False
                # To improve pruning efficiency, we can randomize the order
                # or use a heuristic. Let's try a few random orders if time permits.
                # For now, deterministic order, but we can restart with random order.
                
                # Try deterministic pass first
                for ci in range(len(initial_code)):
                    if time.time() > deadline - 5.0:
                        break
                    if not active[ci]:
                        continue
                    
                    is_redundant = True
                    for wi in codeword_covers[ci]:
                        found_other = False
                        for cj in covered_by[wi]:
                            if cj != ci and active[cj]:
                                found_other = True
                                break
                        if not found_other:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        active[ci] = False
                        removed_any = True
                
                if not removed_any:
                    break

            # Build final code from active flags
            final_code = [initial_code[i] for i in range(len(initial_code)) if active[i]]
            
            # --- Phase 4: Randomized Restart Pruning (if time allows) ---
            # If we have time, try to see if a different random removal order yields better results
            if time.time() < deadline - 10.0 and len(final_code) > 70:
                best_code = final_code
                best_len = len(final_code)
                
                # Number of random trials based on time remaining
                time_remaining = deadline - 10.0 - time.time()
                num_trials = int(time_remaining / 0.5) # 0.5s per trial approx
                num_trials = min(num_trials, 50)
                
                for trial in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break
                    
                    active2 = [True] * len(initial_code)
                    order = list(range(len(initial_code)))
                    random.shuffle(order)
                    
                    for _ in range(max_prune_iterations):
                        if time.time() > deadline - 5.0:
                            break
                        removed_any = False
                        for ci in order:
                            if time.time() > deadline - 5.0:
                                break
                            if not active2[ci]:
                                continue
                            
                            is_redundant = True
                            for wi in codeword_covers[ci]:
                                found_other = False
                                for cj in covered_by[wi]:
                                    if cj != ci and active2[cj]:
                                        found_other = True
                                        break
                                if not found_other:
                                    is_redundant = False
                                    break
                            
                            if is_redundant:
                                active2[ci] = False
                                removed_any = True
                        
                        if not removed_any:
                            break
                    
                    trial_code = [initial_code[i] for i in range(len(initial_code)) if active2[i]]
                    if len(trial_code) < best_len:
                        best_len = len(trial_code)
                        best_code = trial_code

                final_code = best_code

            # Final Verification
            if _verify_coverage(final_code, q, n, radius):
                return final_code
            else:
                # Fallback to direct sum if verification fails (shouldn't happen)
                if _verify_coverage(result_direct, q, n, radius):
                    return result_direct
                else:
                    return result_greedy # Last resort

        else:
            # Generic radius: use a simpler greedy with itertools for small spaces
            if total_words <= 10000:
                return _greedy_generic(q, n, radius, deadline)
            # Otherwise fall through to direct sum

    # --- Fallback: Direct sum construction ---
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    elif q >= 3 and n >= 4 and radius == 1:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4
    else:
        base_n = min(n, 4)
        base_code = []
        for tail in itertools.product(range(q), repeat=base_n):
            base_code.append(list(tail))

    if n < base_n:
        return [list(range(n))]

    tails = list(itertools.product(range(q), repeat=n - base_n))
    return [h + list(tail) for h in base_code for tail in tails]


def _verify_coverage(codewords, q, n, radius):
    """Check that every word is within Hamming distance radius of some codeword."""
    total_words = q ** n
    covered = set()
    for cw in codewords:
        idx = 0
        for i, v in enumerate(cw):
            idx += v * (q ** (n - 1 - i))
        if radius == 1:
            covered.add(idx)
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                cur = (idx // pow_q) % q
                for nv in range(q):
                    if nv != cur:
                        covered.add(idx + (nv - cur) * pow_q)
        else:
            for delta in _all_deltas(n, q, radius):
                covered.add(idx + delta)
    return len(covered) == total_words


def _all_deltas(n, q, radius):
    """Yield index deltas for all words within Hamming distance radius of the zero word."""
    deltas = set()
    deltas.add(0)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            base_options = []
            for pos in positions:
                pow_q = q ** (n - 1 - pos)
                base_options.append([d * pow_q for d in range(1, q)])
            for combo in itertools.product(*base_options):
                deltas.add(sum(combo))
    return deltas


def _greedy_generic(q, n, radius, deadline):
    """Generic greedy for small spaces and any radius."""
    total_words = q ** n
    neighbors_list = []
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        nbrs = _neighbors_of_word(w, q, n, radius)
        idx_nbrs = []
        for nb in nbrs:
            idx = 0
            for j, v in enumerate(nb):
                idx += v * (q ** (n - 1 - j))
            idx_nbrs.append(idx)
        neighbors_list.append(idx_nbrs)

    covered = [False] * total_words
    code_indices = []

    while time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            gain = sum(1 for nb in neighbors_list[i] if not covered[nb])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code_indices.append(best_idx)
        for nb in neighbors_list[best_idx]:
            covered[nb] = True

    result = []
    for idx in code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result


def _neighbors_of_word(word, q, n, radius):
    """Return all words within Hamming distance <= radius of the given word."""
    result = [list(word)]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            base = list(word)
            for combo in itertools.product(*[range(q)] * r):
                for pos, val in zip(positions, combo):
                    if val != base[pos]:
                        base[pos] = val
                if base != list(word):
                    result.append(list(base))
            base = list(word)
    return result
