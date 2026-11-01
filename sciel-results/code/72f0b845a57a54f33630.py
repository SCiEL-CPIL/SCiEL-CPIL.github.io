import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 0:
        return [[]]
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Generic fallback
    return _generic_construct(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q, n, r = 3, 6, 1
    total = q ** n  # 729

    # Precompute all words as indices
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor lists (word + all words at Hamming distance <= 1)
    neighbors_list = [None] * total
    neighbors_set = [None] * total
    for i, w in enumerate(all_words):
        nb_set = {i}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nb_set.add(idx_map[tuple(nw)])
        neighbors_set[i] = nb_set
        neighbors_list[i] = list(nb_set)

    # Block structure: last 2 coordinates define 9 blocks of 81 words
    # Block b contains words with last 2 coords = (b//3, b%3)
    block_of = [0] * total
    block_words = [[] for _ in range(9)]
    for i, w in enumerate(all_words):
        b = w[4] * 3 + w[5]
        block_of[i] = b
        block_words[b].append(i)

    # Start with tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = set()
    for head in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = head + list(tail)
            code.add(idx_map[tuple(word)])

    # Compute cover count
    cover_count = [0] * total
    for c in code:
        for nb in neighbors_list[c]:
            cover_count[nb] += 1

    # Convert to numpy for vectorized operations
    cc_np = np.array(cover_count, dtype=np.int32)
    # Precompute neighbor matrix as bitsets (729 bits per word, use uint64 array of 12 uint64s)
    # Actually, let's use a different approach: precompute for each word, the list of neighbors
    # For greedy repair, we want to find which candidate covers the most uncovered words

    def is_valid():
        return all(cover_count[i] > 0 for i in range(total))

    def try_remove(c, code_set, cc):
        """Try removing codeword c. Return True if successful (code still valid)."""
        for nb in neighbors_list[c]:
            if cc[nb] == 1:
                return False
        code_set.remove(c)
        for nb in neighbors_list[c]:
            cc[nb] -= 1
        return True

    def try_swap(c, new_c, code_set, cc):
        """Try replacing c with new_c. Return True if successful."""
        if new_c in code_set:
            return False
        unique_to_c = []
        for nb in neighbors_list[c]:
            if cc[nb] == 1:
                unique_to_c.append(nb)

        for u in unique_to_c:
            if u != new_c and u not in neighbors_set[new_c]:
                return False

        code_set.remove(c)
        for nb in neighbors_list[c]:
            cc[nb] -= 1
        code_set.add(new_c)
        for nb in neighbors_list[new_c]:
            cc[nb] += 1
        return True

    def local_search(code_set, cc, deadline):
        """Iteratively try to reduce code size via removals and swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Pass 1: try removing each codeword
            for c in list(code_set):
                if time.time() > deadline:
                    break
                if try_remove(c, code_set, cc):
                    improved = True

            if time.time() > deadline:
                break

            # Pass 2: try swaps - more aggressive
            code_list = list(code_set)
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_to_c = []
                for nb in neighbors_list[c]:
                    if cc[nb] == 1:
                        unique_to_c.append(nb)

                if not unique_to_c:
                    continue

                # Generate candidates: neighbors of unique words, excluding current code
                candidates = set()
                for u in unique_to_c:
                    candidates.update(neighbors_list[u])
                candidates -= code_set

                # Try all candidates (not just first)
                for new_c in candidates:
                    if time.time() > deadline:
                        break
                    if try_swap(c, new_c, code_set, cc):
                        improved = True
                        # Don't break - keep trying more swaps
                        break

            # Pass 3: try pair swaps (remove 2, add 1)
            if time.time() < deadline:
                code_list = list(code_set)
                random.shuffle(code_list)
                for i in range(len(code_list) - 1):
                    if time.time() > deadline:
                        break
                    c1, c2 = code_list[i], code_list[i+1]
                    if c2 not in code_set:
                        continue
                    # Check if removing both leaves any word uncovered
                    # Words that would become uncovered
                    uncovered_if_removed = set()
                    for nb in neighbors_list[c1]:
                        if cc[nb] == 1:
                            uncovered_if_removed.add(nb)
                    for nb in neighbors_list[c2]:
                        if cc[nb] == 1:
                            uncovered_if_removed.add(nb)
                        elif cc[nb] == 2:
                            # Only c1 covers it? Check
                            pass
                    # More careful: compute coverage without c1 and c2
                    # This is expensive, so only try for small sets
                    if len(uncovered_if_removed) > 0:
                        # Find a word that covers all of uncovered_if_removed
                        for candidate in range(total):
                            if candidate in code_set:
                                continue
                            covers_all = True
                            for u in uncovered_if_removed:
                                if u not in neighbors_set[candidate]:
                                    covers_all = False
                                    break
                            if covers_all:
                                # Perform the swap: remove c1, c2, add candidate
                                code_set.remove(c1)
                                for nb in neighbors_list[c1]:
                                    cc[nb] -= 1
                                code_set.remove(c2)
                                for nb in neighbors_list[c2]:
                                    cc[nb] -= 1
                                code_set.add(candidate)
                                for nb in neighbors_list[candidate]:
                                    cc[nb] += 1
                                improved = True
                                break

    def greedy_repair(code_set, cc, deadline):
        """Add codewords to cover all uncovered words using numpy acceleration."""
        while not is_valid() and time.time() < deadline:
            uncovered = [i for i in range(total) if cc[i] == 0]
            if not uncovered:
                break

            # Use numpy to find best candidate
            uncovered_set = set(uncovered)
            best_c = -1
            best_cnt = -1

            # For each candidate not in code, count how many uncovered words it covers
            # Vectorize: for each candidate, check intersection with uncovered
            # Since total=729 is small, do it efficiently
            for c in range(total):
                if c in code_set:
                    continue
                # Count how many of c's neighbors are uncovered
                cnt = 0
                for nb in neighbors_list[c]:
                    if nb in uncovered_set:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c

            if best_c == -1 or best_cnt == 0:
                break

            code_set.add(best_c)
            for nb in neighbors_list[best_c]:
                cc[nb] += 1

    # Initial local search
    local_search(code, cover_count, deadline)
    greedy_repair(code, cover_count, deadline)

    best_size = len(code)
    best_code = set(code)
    best_cc = list(cover_count)

    # ILS: perturb (remove random words), repair, local search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        # Adaptive perturbation: start small, increase if stuck
        base_k = 2 + (iteration % 3)
        k = random.randint(base_k, min(base_k + 4, max(2, len(code) // 8)))
        code_list = list(code)
        random.shuffle(code_list)
        removed = code_list[:k]
        for c in removed:
            code.remove(c)
            for nb in neighbors_list[c]:
                cover_count[nb] -= 1

        # Repair
        greedy_repair(code, cover_count, deadline)

        # Local search
        local_search(code, cover_count, deadline)

        # Track best
        if len(code) < best_size:
            best_size = len(code)
            best_code = set(code)
            best_cc = list(cover_count)

        # If we reached the known optimum, stop
        if best_size <= 73:
            break

    # Use best found
    code = best_code
    cover_count = best_cc

    # Final intensive local search pass
    local_search(code, cover_count, deadline)
    greedy_repair(code, cover_count, deadline)

    # One more round of local search if time permits
    if time.time() < deadline - 2.0:
        local_search(code, cover_count, deadline)
        greedy_repair(code, cover_count, deadline)

    return [list(all_words[c]) for c in sorted(code)]


def _generic_construct(q, n, radius, deadline):
    """Simple greedy for other cases."""
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        for r2 in range(radius + 1):
            for pos in itertools.combinations(range(n), r2):
                for vals in itertools.product(range(q), repeat=r2):
                    nw = list(w)
                    for p, v in zip(pos, vals):
                        nw[p] = v
                    neighbors[i].add(idx_map[tuple(nw)])
    
    neighbors_list = [list(s) for s in neighbors]
    
    code = set()
    cover_count = [0] * total
    
    while time.time() < deadline:
        uncovered = [i for i in range(total) if cover_count[i] == 0]
        if not uncovered:
            break
        
        best_c = -1
        best_cnt = -1
        for c in range(total):
            if c in code:
                continue
            cnt = sum(1 for u in uncovered if u in neighbors[c])
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        
        if best_c == -1 or best_cnt == 0:
            break
        
        code.add(best_c)
        for nb in neighbors_list[best_c]:
            cover_count[nb] += 1
    
    return [list(all_words[c]) for c in sorted(code)]
