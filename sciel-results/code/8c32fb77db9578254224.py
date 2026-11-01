import itertools
import time
import random
import numpy as np

def _get_neighbors_indices(c, q, n):
    """Return list of indices of neighbors at Hamming distance <= 1 from c (tuple)."""
    pass

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if n == 0:
        return [[]]
    
    # Handle small cases or specific known perfect codes
    if q == 3 and radius == 1 and n == 4:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words as lists of indices
    # neighbors[i] is a list of indices j such that dist(all_words[i], all_words[j]) <= radius
    # For radius=1, this is just the star neighborhood.
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[pos] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
    
    # Function to check if a set of codeword indices covers all words
    def is_valid(code_indices):
        covered = [0] * total_words
        for c_idx in code_indices:
            for nb_idx in neighbors[c_idx]:
                covered[nb_idx] += 1
        return all(c > 0 for c in covered)

    # Function to get coverage counts
    def get_coverage(code_indices):
        covered = [0] * total_words
        for c_idx in code_indices:
            for nb_idx in neighbors[c_idx]:
                covered[nb_idx] += 1
        return covered

    # Precompute for each word, which codewords cover it (reverse neighbor list)
    # covered_by[i] = list of j such that i is in neighbors[j]
    covered_by = [[] for _ in range(total_words)]
    for j in range(total_words):
        for i in neighbors[j]:
            covered_by[i].append(j)

    # Greedy Initialization
    def greedy_init(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(total_words))
        code_indices = []
        # To speed up, we can shuffle the order of candidates or just iterate
        # A simple greedy: pick word covering most uncovered
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            # Optimization: Instead of checking all words, we can be smarter, but for 729 words it's fast
            for i in range(total_words):
                if i in code_indices:
                    continue
                cnt = sum(1 for nb in neighbors[i] if nb in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            for nb in neighbors[best_idx]:
                uncovered.discard(nb)
                
        return code_indices

    # Direct Sum Initialization (for q=3, n=6, radius=1)
    def direct_sum_init():
        if q == 3 and n == 6 and radius == 1:
            tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
            code = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    c = h + tail
                    code.append(word_to_idx[c])
            return code
        return greedy_init(0)

    # Local Search with improved swap and 2-swap operations
    def local_search(init_code, seed=0):
        rng = random.Random(seed)
        active = list(init_code)
        active_set = set(active)
        coverage = get_coverage(active)
        
        # Ensure valid
        if not is_valid(active):
            # Fix by adding words
            uncovered = [i for i in range(total_words) if coverage[i] == 0]
            while uncovered and time.time() < deadline:
                best_idx = -1
                best_count = -1
                for i in range(total_words):
                    if i in active_set:
                        continue
                    cnt = sum(1 for nb in neighbors[i] if coverage[nb] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_idx == -1:
                    break
                active.append(best_idx)
                active_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    coverage[nb] += 1
                uncovered = [i for i in uncovered if coverage[i] == 0]

        best_size = len(active)
        best_code = list(active)
        
        # Helper: find best swap candidate for removing codeword c
        def find_swap_candidate(c, active_set, coverage):
            # Find words uniquely covered by c
            unique_words = [nb for nb in neighbors[c] if coverage[nb] == 1]
            if not unique_words:
                return None, 0  # Can remove, handled by removal phase
            
            best_candidate = -1
            best_score = -1
            
            for c_prime in range(total_words):
                if c_prime in active_set:
                    continue
                
                # Check if c_prime covers all unique_words
                covers_all = all(w in neighbors[c_prime] for w in unique_words)
                if not covers_all:
                    continue
                
                # Score: prefer candidates that cover many already-well-covered words
                score = sum(1 for nb in neighbors[c_prime] if coverage[nb] > 1)
                
                if score > best_score:
                    best_score = score
                    best_candidate = c_prime
            
            return best_candidate, best_score
        
        # Helper: perform 2-swap (remove two, add one or two)
        def try_2_swap(active, active_set, coverage, rng):
            # Try to find a pair (c1, c2) to remove and a single c3 to add
            # such that all words remain covered
            if len(active) < 2:
                return False
            
            # Get all codewords
            codes = list(active)
            rng.shuffle(codes)
            
            # Try limited pairs to keep it fast
            max_pairs = min(50, len(codes) * (len(codes) - 1) // 2)
            tried = 0
            
            for i in range(len(codes)):
                if tried >= max_pairs:
                    break
                c1 = codes[i]
                for j in range(i + 1, len(codes)):
                    if tried >= max_pairs:
                        break
                    tried += 1
                    c2 = codes[j]
                    
                    # Find words uniquely covered by c1 or c2 (after removing both)
                    # A word w is uniquely covered if coverage[w] == 1 and the only codeword covering it is c1 or c2
                    # Or coverage[w] == 2 and the only codewords covering it are c1 and c2
                    
                    unique_after_removal = []
                    for nb1 in neighbors[c1]:
                        if coverage[nb1] == 1:
                            # Only c1 covers this
                            unique_after_removal.append(nb1)
                        elif coverage[nb1] == 2:
                            # c1 and one other cover this; if that other is c2, then after removal it's uncovered
                            # Check if c2 is the only other
                            others = [x for x in covered_by[nb1] if x not in (c1, c2)]
                            if not others:
                                unique_after_removal.append(nb1)
                    
                    for nb2 in neighbors[c2]:
                        if nb2 not in neighbors[c1]:  # Avoid double counting
                            if coverage[nb2] == 1:
                                unique_after_removal.append(nb2)
                            elif coverage[nb2] == 2:
                                others = [x for x in covered_by[nb2] if x not in (c1, c2)]
                                if not others:
                                    unique_after_removal.append(nb2)
                    
                    if not unique_after_removal:
                        # Can remove both! This is a 2-removal
                        for nb in neighbors[c1]:
                            coverage[nb] -= 1
                        for nb in neighbors[c2]:
                            coverage[nb] -= 1
                        active.remove(c1)
                        active.remove(c2)
                        active_set.discard(c1)
                        active_set.discard(c2)
                        return True
                    
                    # Find a single c3 that covers all unique_after_removal
                    best_c3 = -1
                    best_score = -1
                    
                    for c3 in range(total_words):
                        if c3 in active_set:
                            continue
                        
                        covers_all = all(w in neighbors[c3] for w in unique_after_removal)
                        if not covers_all:
                            continue
                        
                        score = sum(1 for nb in neighbors[c3] if coverage[nb] > 1)
                        if score > best_score:
                            best_score = score
                            best_c3 = c3
                    
                    if best_c3 != -1:
                        # Perform 2-swap: remove c1, c2; add c3
                        for nb in neighbors[c1]:
                            coverage[nb] -= 1
                        for nb in neighbors[c2]:
                            coverage[nb] -= 1
                        active.remove(c1)
                        active.remove(c2)
                        active_set.discard(c1)
                        active_set.discard(c2)
                        
                        active.append(best_c3)
                        active_set.add(best_c3)
                        for nb in neighbors[best_c3]:
                            coverage[nb] += 1
                        return True
            
            return False
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Removal
            removal_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                can_remove = all(coverage[nb] > 1 for nb in neighbors[c])
                if can_remove:
                    for nb in neighbors[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set.discard(c)
                    improved = True
                    removal_found = True
                    break
            
            if removal_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                continue

            # Phase 2: 1-swap
            swap_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                candidate, score = find_swap_candidate(c, active_set, coverage)
                
                if candidate is not None and candidate != -1:
                    # Perform swap
                    for nb in neighbors[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set.discard(c)
                    
                    active.append(candidate)
                    active_set.add(candidate)
                    for nb in neighbors[candidate]:
                        coverage[nb] += 1
                    
                    improved = True
                    swap_found = True
                    break

            if swap_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                continue

            # Phase 3: 2-swap
            if try_2_swap(active, active_set, coverage, rng):
                improved = True
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                continue

            # Phase 4: Perturbation
            if time.time() < deadline - 5:
                perturb_count = rng.randint(1, 3)
                removed = []
                for _ in range(perturb_count):
                    if not active:
                        break
                    idx = rng.randint(0, len(active) - 1)
                    c = active.pop(idx)
                    active_set.discard(c)
                    for nb in neighbors[c]:
                        coverage[nb] -= 1
                    removed.append(c)
                
                # Repair
                uncovered = [i for i in range(total_words) if coverage[i] == 0]
                while uncovered and time.time() < deadline:
                    best_idx = -1
                    best_count = -1
                    for i in range(total_words):
                        if i in active_set:
                            continue
                        cnt = sum(1 for nb in neighbors[i] if coverage[nb] == 0)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = i
                    if best_idx == -1:
                        # Revert
                        for c in removed:
                            if c not in active_set:
                                active.append(c)
                                active_set.add(c)
                                for nb in neighbors[c]:
                                    coverage[nb] += 1
                        removed = []
                        break
                    active.append(best_idx)
                    active_set.add(best_idx)
                    for nb in neighbors[best_idx]:
                        coverage[nb] += 1
                    uncovered = [i for i in uncovered if coverage[i] == 0]
                
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
            
            if len(active) < best_size:
                best_size = len(active)
                best_code = list(active)

        return best_code

    # Main Logic: Try multiple strategies with time budgeting
    best_code = None
    best_size = float('inf')
    
    strategies = []
    
    # 1. Direct Sum (known good starting point for q=3, n=6, r=1)
    if q == 3 and n == 6 and radius == 1:
        strategies.append(('direct_sum', direct_sum_init()))
    
    # 2. Multiple greedy inits with different seeds
    for seed in [123, 456, 789, 1024, 2048]:
        strategies.append((f'greedy_{seed}', greedy_init(seed)))
    
    # Time budget per strategy: spend at least 15s on the first, then split remaining
    num_strategies = len(strategies)
    time_per_strategy = max(10, (deadline - time.time()) / num_strategies - 5)
    
    for name, init in strategies:
        if time.time() > deadline - 10:
            break
        
        # Run local search with a time limit
        local_deadline = min(deadline - 5, time.time() + time_per_strategy)
        old_deadline = deadline
        deadline = local_deadline
        result = local_search(init, seed=hash(name) % 1000)
        deadline = old_deadline
        
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    if best_code is None:
        best_code = greedy_init(0)

    # Final polish: run local search on the best code with remaining time
    if time.time() < deadline - 5:
        final_deadline = deadline
        old_deadline = deadline
        deadline = final_deadline
        polished = local_search(best_code, seed=9999)
        deadline = old_deadline
        if len(polished) < best_size:
            best_size = len(polished)
            best_code = polished

    # Convert indices back to lists of ints
    return [list(all_words[idx]) for idx in best_code]
