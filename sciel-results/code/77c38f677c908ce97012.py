import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    N = q ** n
    
    # Precompute powers for fast index conversion
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * q_powers[n-1-i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbor lists and masks
    neighbor_lists = [[] for _ in range(N)]
    neighbor_masks = [0] * N
    
    if radius == 1:
        for c_idx in range(N):
            mask = 1 << c_idx
            neighbor_lists[c_idx].append(c_idx)
            for i in range(n):
                pow_q = q_powers[n-1-i]
                orig_digit = (c_idx // pow_q) % q
                for new_val in range(q):
                    if new_val != orig_digit:
                        w_idx = c_idx + (new_val - orig_digit) * pow_q
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
            neighbor_masks[c_idx] = mask
    else:
        all_words = [idx_to_word(i) for i in range(N)]
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            neighbor_lists[c_idx].append(c_idx)
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
            neighbor_masks[c_idx] = mask
    
    # Precompute coverers: for each word, list of codewords that cover it
    coverers = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbor_lists[c]:
            coverers[w].append(c)
    
    def get_coders_for_word(w_idx):
        """Return list of codeword indices that cover w_idx"""
        return coverers[w_idx]
    
    def tetracode_init():
        """Generate structured initialization using tetracode (optimal ternary code of length 4, size 9)"""
        # Tetracode: all words (a, b, a+b, a+2b) mod 3
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        # Extend to length 8 by direct product with all words of length 4
        code = []
        for tc in tetracode:
            for tail in itertools.product(range(3), repeat=4):
                code.append(tc + list(tail))
        
        # Convert to indices
        return [word_to_idx(w) for w in code]
    
    def greedy_set_cover():
        """Greedy set cover using heap for efficiency"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # Initial gains: all codewords cover ball_size words
        ball_size = 1 + (q - 1) * n if radius == 1 else len(neighbor_lists[0])
        gains = [ball_size] * N
        
        # Use a simple approach: repeatedly pick codeword with max new coverage
        while uncovered:
            if time.time() > deadline - 5:
                break
            
            best_c = -1
            best_gain = -1
            
            # Sample-based greedy for speed
            sample_size = min(100, N)
            candidates = random.sample(range(N), sample_size) if N > sample_size else list(range(N))
            
            for c in candidates:
                if c in code_set:
                    continue
                gain = (uncovered & neighbor_masks[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                # Fallback: full search
                for c in range(N):
                    if c in code_set:
                        continue
                    gain = (uncovered & neighbor_masks[c]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
        
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        code_set = set(code)
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    code.pop(i)
                    code_set.discard(c)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def two_opt_local_search(code, rng):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                c_remove = code[i]
                
                newly_uncovered = []
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    code.pop(i)
                    code_set.discard(c_remove)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                
                if len(newly_uncovered) > 3:
                    continue
                
                # Find candidates that cover all newly_uncovered
                candidate_codewords = set(coverers[newly_uncovered[0]])
                for w in newly_uncovered[1:]:
                    candidate_codewords.intersection_update(set(coverers[w]))
                candidate_codewords -= code_set
                
                if len(candidate_codewords) > 100:
                    candidates_list = list(candidate_codewords)
                    rng.shuffle(candidates_list)
                    candidate_codewords = set(candidates_list[:100])
                
                best_add = -1
                best_gain = -1
                
                for c_add in candidate_codewords:
                    mask_add = neighbor_masks[c_add]
                    all_covered = True
                    for w in newly_uncovered:
                        if (mask_add >> w) & 1 == 0:
                            all_covered = False
                            break
                    if not all_covered:
                        continue
                    
                    new_coverage = 0
                    for w in neighbor_lists[c_add]:
                        if cover_counts[w] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_add = c_add
                
                if best_add != -1:
                    code.pop(i)
                    code_set.discard(c_remove)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    
                    code.append(best_add)
                    code_set.add(best_add)
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1
                    
                    improved = True
                    break
        
        return code
    
    def two_swap_local_search(code, rng):
        """2-swap: try to swap two codewords to enable removal of a third"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                c1 = code[i]
                
                newly_uncovered = []
                for w in neighbor_lists[c1]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if len(newly_uncovered) > 4:
                    continue
                
                # Find candidates to replace c1
                candidates = set(coverers[newly_uncovered[0]]) if newly_uncovered else set()
                for w in newly_uncovered[1:]:
                    candidates.intersection_update(set(coverers[w]))
                candidates -= code_set
                candidates.discard(c1)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for c2 in cand_list[:50]:
                    mask2 = neighbor_masks[c2]
                    all_covered = all((mask2 >> w) & 1 for w in newly_uncovered)
                    if not all_covered:
                        continue
                    
                    # Check if adding c2 allows removing another codeword
                    best_remove = -1
                    for j in range(len(code)):
                        if j == i:
                            continue
                        cj = code[j]
                        can_remove = True
                        for w in neighbor_lists[cj]:
                            if cover_counts[w] == 1 and w not in newly_uncovered:
                                if not ((mask2 >> w) & 1):
                                    can_remove = False
                                    break
                        if can_remove:
                            best_remove = j
                            break
                    
                    if best_remove != -1:
                        # Perform the swap and removal
                        code.pop(i)
                        code_set.discard(c1)
                        for w in neighbor_lists[c1]:
                            cover_counts[w] -= 1
                        
                        cj = code[best_remove]
                        code.pop(best_remove)
                        code_set.discard(cj)
                        for w in neighbor_lists[cj]:
                            cover_counts[w] -= 1
                        
                        code.append(c2)
                        code_set.add(c2)
                        for w in neighbor_lists[c2]:
                            cover_counts[w] += 1
                        
                        improved = True
                        break
                
                if improved:
                    break
        
        return code
    
    def perturb(code, rng, strength=0.05):
        """Randomly swap some codewords to escape local minima"""
        code = list(code)
        if not code:
            return code
        
        code_set = set(code)
        n_swaps = max(1, int(len(code) * strength))
        for _ in range(n_swaps):
            if len(code) < 2:
                break
            i = rng.randint(0, len(code) - 1)
            old = code[i]
            code_set.discard(old)
            new_c = rng.randint(0, N - 1)
            while new_c in code_set:
                new_c = rng.randint(0, N - 1)
            code[i] = new_c
            code_set.add(new_c)
        
        return code
    
    def full_pipeline(seed, use_tetracode=True):
        """Run the full optimization pipeline with a given seed"""
        rng = random.Random(seed)
        
        if use_tetracode:
            code = tetracode_init()
        else:
            code = greedy_set_cover()
        
        code = local_search_remove(code)
        code = two_opt_local_search(code, rng)
        code = two_swap_local_search(code, rng)
        return code
    
    # Initial solution with tetracode
    best_code = full_pipeline(42, use_tetracode=True)
    best_size = len(best_code)
    
    # Also try greedy for comparison
    greedy_code = full_pipeline(123, use_tetracode=False)
    if len(greedy_code) < best_size:
        best_code = greedy_code
        best_size = len(best_code)
    
    # ILS: Iterated Local Search with perturbation
    rng_ils = random.Random(12345)
    current_code = list(best_code)
    current_size = best_size
    
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        
        # Perturb current solution
        perturbed = perturb(current_code, rng_ils, strength=0.03 + 0.02 * (iteration % 5) / 5)
        
        # Run local search on perturbed solution
        improved = local_search_remove(perturbed)
        improved = two_opt_local_search(improved, rng_ils)
        improved = two_swap_local_search(improved, rng_ils)
        
        # Accept if better or equal (to allow exploration)
        if len(improved) < current_size:
            current_code = improved
            current_size = len(improved)
            if current_size < best_size:
                best_code = list(current_code)
                best_size = current_size
        elif len(improved) == current_size and rng_ils.random() < 0.3:
            current_code = improved
        
        # Occasionally restart with tetracode or greedy
        if iteration % 6 == 0 and time.time() < deadline - 20:
            fresh = full_pipeline(rng_ils.randint(1, 10000), use_tetracode=(iteration % 2 == 0))
            if len(fresh) < current_size:
                current_code = fresh
                current_size = len(fresh)
                if current_size < best_size:
                    best_code = list(current_code)
                    best_size = current_size
    
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    # Convert to words
    all_words = [idx_to_word(i) for i in range(N)]
    result = [all_words[i] for i in best_code]
    return result
