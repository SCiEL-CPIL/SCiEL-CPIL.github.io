import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112  # 8s buffer for safety
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        N = q ** n
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n-1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        all_words = [idx_to_word(i) for i in range(N)]
        neighbor_masks = [0] * N
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            if radius == 1:
                for i in range(n):
                    orig = c_word[i]
                    for new_val in range(q):
                        if new_val != orig:
                            new_word = c_word.copy()
                            new_word[i] = new_val
                            w_idx = word_to_idx(new_word)
                            mask |= (1 << w_idx)
            neighbor_masks[c_idx] = mask
        uncovered = (1 << N) - 1
        code = []
        while uncovered and time.time() < deadline:
            best_c = -1
            best_count = -1
            for c in range(N):
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            uncovered &= ~neighbor_masks[best_c]
        return [all_words[i] for i in code]
    
    # Main path: q=3, n=8, radius=1
    N = 3 ** 8  # 6561
    
    # Use numpy arrays for fast neighbor representation
    # Each codeword covers itself + 8*2 = 17 words
    # Precompute neighbor lists
    neighbor_lists = []
    for c in range(N):
        word = []
        temp = c
        for i in range(8):
            word.append(temp % 3)
            temp //= 3
        word = word[::-1]  # word[0] is most significant
        
        neighbors = [c]
        for pos in range(8):
            orig = word[pos]
            for delta in [1, 2]:
                new_val = (orig + delta) % 3
                new_word = word[:]
                new_word[pos] = new_val
                # Convert back to index
                idx = 0
                for j in range(8):
                    idx = idx * 3 + new_word[j]
                neighbors.append(idx)
        neighbor_lists.append(neighbors)
    
    # Convert to numpy for fast operations
    neighbor_arrays = [np.array(nl, dtype=np.int32) for nl in neighbor_lists]
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    def build_cover_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        return counts
    
    # Algebraic initialization:
    # The ternary space GF(3)^8 has a natural structure.
    # We use a grid-based approach: partition into 4+4 and use product codes.
    
    def algebraic_init():
        """
        Build an initial covering using the algebraic structure.
        
        Key insight: GF(3)^8 = GF(3)^4 × GF(3)^4
        A word w = (a, b) where a, b ∈ GF(3)^4.
        Distance 1 means either a or b changes in one coordinate.
        
        Strategy: Use a covering of GF(3)^4 with small radius,
        then combine appropriately.
        
        Actually, for radius 1 in 8D, each codeword covers 1 + 8*2 = 17 words.
        6561 / 17 ≈ 386. The known answer is 486.
        
        Better approach: use the fact that GF(3)^8 has a group structure.
        Pick a random subgroup or coset structure to seed the greedy.
        """
        # Grid initialization: divide 8 positions into two groups of 4
        # For each "first half" pattern, pick codewords that cover the "second half"
        
        # Actually, let's use a smarter algebraic seed:
        # The automorphism group of the Hamming graph includes coordinate permutations
        # and digit permutations within each coordinate.
        
        # Seed with a structured set: use all words where the sum of digits
        # in certain positions satisfies a condition.
        
        # Simple structured seed: take a random subset of size ~400-500
        # that has good spread, then let greedy fill in.
        
        # Use a lattice-like structure:
        # Map GF(3)^8 to integers 0..6560
        # Take every k-th element with some perturbation
        
        random.seed(42)
        
        # Start with a rough estimate: 6561/17 ≈ 386, but we know 486 is optimal
        # Start with 450 random points spread out
        seed_size = 450
        candidates = list(range(N))
        random.shuffle(candidates)
        
        # Pick well-spread candidates: use a greedy on a subspace
        # Actually, just pick random and let the greedy complete
        code = candidates[:seed_size]
        code_set = set(code)
        
        # Complete the covering
        counts = build_cover_counts(code)
        uncovered_mask = np.where(counts == 0)[0]
        
        while len(uncovered_mask) > 0 and time_left() > 5:
            # Find the codeword that covers the most uncovered words
            best_c = -1
            best_count = 0
            for c in range(N):
                if c in code_set:
                    continue
                covered = np.sum(counts[neighbor_arrays[c]] == 0)
                if covered > best_count:
                    best_count = covered
                    best_c = c
            
            if best_c == -1 or best_count == 0:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            counts[neighbor_arrays[best_c]] += 1
            uncovered_mask = np.where(counts == 0)[0]
        
        return code
    
    def greedy_complete(code):
        """Complete a partial covering with greedy."""
        code_set = set(code)
        counts = build_cover_counts(code)
        
        while True:
            if time_left() < 2:
                break
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            best_c = -1
            best_count = 0
            for c in range(N):
                if c in code_set:
                    continue
                covered = np.sum(counts[neighbor_arrays[c]] == 0)
                if covered > best_count:
                    best_count = covered
                    best_c = c
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            counts[neighbor_arrays[best_c]] += 1
        
        return code
    
    def reduce_code(code):
        """Remove redundant codewords and try swaps to reduce size."""
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 3:
            improved = False
            
            # Try removing each codeword
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any word is uniquely covered by c
                unique = np.any(counts[neighbor_arrays[c]] == 1)
                if not unique:
                    # Remove it
                    code.pop(i)
                    code_set.discard(c)
                    counts[neighbor_arrays[c]] -= 1
                    improved = True
                    break
        
        # Try 2-opt swaps: remove one, add one, then try to remove more
        improved = True
        while improved and time_left() > 5:
            improved = False
            
            for i in range(len(code)):
                if time_left() < 3:
                    break
                
                c_remove = code[i]
                # Words that would become uncovered
                newly_uncovered = neighbor_arrays[c_remove][counts[neighbor_arrays[c_remove]] == 1]
                
                if len(newly_uncovered) == 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    improved = True
                    break
                
                # Find a codeword that covers all newly_uncovered
                best_add = -1
                best_gain = -1
                
                for c_add in range(N):
                    if c_add in code_set:
                        continue
                    
                    # Check if c_add covers all newly_uncovered
                    if not np.all(np.isin(newly_uncovered, neighbor_arrays[c_add])):
                        continue
                    
                    # Count how many currently uncovered words c_add would cover
                    gain = np.sum(counts[neighbor_arrays[c_add]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = c_add
                
                if best_add != -1 and best_gain > 0:
                    # Perform the swap
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    
                    code.append(best_add)
                    code_set.add(best_add)
                    counts[neighbor_arrays[best_add]] += 1
                    
                    improved = True
                    break
        
        return code
    
    def simulated_annealing(code, max_iters=500):
        """
        Simulated annealing with moves: remove, add, swap.
        Objective: minimize |code| + penalty * uncovered_count
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        uncovered_count = np.sum(counts == 0)
        penalty = max(2 * len(code), 10)
        
        current_obj = len(code) + penalty * uncovered_count
        best_code = list(code)
        best_obj = current_obj
        
        temp = 20.0
        cooling = 0.98
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            if temp < 0.1:
                break
            
            move_type = random.random()
            
            if move_type < 0.35 and len(code) > 10:
                # Remove a random codeword
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                
                newly_uncovered = np.sum(counts[neighbor_arrays[c_remove]] == 1)
                delta_obj = -1 + penalty * newly_uncovered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.pop(idx)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    current_obj = new_obj
                    uncovered_count += newly_uncovered
                    
            elif move_type < 0.65:
                # Add a random codeword
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                
                newly_covered = np.sum(counts[neighbor_arrays[c_add]] == 0)
                delta_obj = 1 - penalty * newly_covered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.append(c_add)
                    code_set.add(c_add)
                    counts[neighbor_arrays[c_add]] += 1
                    current_obj = new_obj
                    uncovered_count -= newly_covered
                    
            else:
                # Swap
                if len(code) == 0:
                    continue
                
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                
                # Compute coverage change
                rem_uncovered = np.sum(counts[neighbor_arrays[c_remove]] == 1)
                add_covered = np.sum(counts[neighbor_arrays[c_add]] == 0)
                uncovered_delta = rem_uncovered - add_covered
                delta_obj = penalty * uncovered_delta
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code[idx] = c_add
                    code_set.discard(c_remove)
                    code_set.add(c_add)
                    counts[neighbor_arrays[c_remove]] -= 1
                    counts[neighbor_arrays[c_add]] += 1
                    current_obj = new_obj
                    uncovered_count += uncovered_delta
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(code)
            
            temp *= cooling
        
        # Ensure full coverage
        counts_best = build_cover_counts(best_code)
        if np.any(counts_best == 0):
            best_code = greedy_complete(best_code)
        
        return best_code
    
    def symmetry_breaking_restarts(code, num_restarts=5):
        """
        Apply symmetry-breaking restarts: perturb the code by applying
        random automorphisms (coordinate permutations + digit permutations)
        and re-optimizing.
        """
        best_code = code
        best_size = len(code)
        
        for r in range(num_restarts):
            if time_left() < 10:
                break
            
            # Apply a random automorphism
            perm = list(range(8))
            random.shuffle(perm)
            digit_perms = [list(range(3)) for _ in range(8)]
            for dp in digit_perms:
                random.shuffle(dp)
            
            # Transform the code
            transformed = []
            seen = set()
            for c in code:
                word = []
                temp = c
                for i in range(8):
                    word.append(temp % 3)
                    temp //= 3
                word = word[::-1]
                
                new_word = [0] * 8
                for pos in range(8):
                    new_word[perm[pos]] = digit_perms[pos][word[pos]]
                
                idx = 0
                for j in range(8):
                    idx = idx * 3 + new_word[j]
                
                if idx not in seen:
                    transformed.append(idx)
                    seen.add(idx)
            
            # Complete and reduce
            transformed = greedy_complete(transformed)
            transformed = reduce_code(transformed)
            transformed = simulated_annealing(transformed, max_iters=100)
            transformed = reduce_code(transformed)
            
            if len(transformed) < best_size:
                best_code = transformed
                best_size = len(transformed)
        
        return best_code
    
    # Main algorithm
    # 1. Algebraic initialization
    code = algebraic_init()
    code = greedy_complete(code)
    code = reduce_code(code)
    
    best_code = code
    best_size = len(code)
    
    # 2. Simulated annealing
    sa_code = simulated_annealing(code, max_iters=300)
    sa_code = reduce_code(sa_code)
    if len(sa_code) < best_size:
        best_code = sa_code
        best_size = len(sa_code)
    
    # 3. Symmetry-breaking restarts
    sb_code = symmetry_breaking_restarts(best_code, num_restarts=4)
    if len(sb_code) < best_size:
        best_code = sb_code
        best_size = len(sb_code)
    
    # 4. Final reduction pass
    best_code = reduce_code(best_code)
    best_code = simulated_annealing(best_code, max_iters=150)
    best_code = reduce_code(best_code)
    
    # Verify full coverage
    final_counts = build_cover_counts(best_code)
    if np.any(final_counts == 0):
        best_code = greedy_complete(best_code)
        best_code = reduce_code(best_code)
    
    # Convert indices to words
    result = []
    for c in best_code:
        word = []
        temp = c
        for i in range(8):
            word.append(temp % 3)
            temp //= 3
        word = word[::-1]
        result.append(word)
    
    return result
