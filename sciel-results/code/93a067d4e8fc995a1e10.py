import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def to_int(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val

    def to_digits(val):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = val % q
            val //= q
        return digits

    # Precompute balls
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = to_digits(w)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv != orig:
                    digits[pos] = nv
                    ball.append(to_int(digits))
            digits[pos] = orig
        balls[w] = ball

    # Precompute inverse index: for each word, which centers cover it
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = set()
            centers_covering[w].add(i)

    def structured_init():
        """Tetracode product initialization for q=3, n=8."""
        if q != 3 or n != 8:
            return greedy_init()

        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list(set(to_int(c) for c in product_code))

        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1

        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            gains = uncovered[balls].sum(axis=1)
            code_set = set(code_ints)
            for c in code_set:
                gains[c] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best = int(np.argmax(gains))
            code_ints.append(best)
            cover_count[balls[best]] += 1
            uncovered = (cover_count == 0)

        return code_ints, cover_count

    def greedy_init():
        """Standard greedy set cover."""
        uncovered = np.ones(total, dtype=bool)
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            best = int(np.argmax(gains))
            code.append(best)
            uncovered[balls[best]] = False
            cover_count[balls[best]] += 1
        return code, cover_count

    def trim_code(code, cover_count):
        """Remove all redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.all(cover_count[balls[c]] > 1):
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iters=50000, allow_2swap=True):
        """Targeted local search using inverse index for fast candidate finding, with optional 2-swap moves."""
        code_set = set(code)
        improved = True
        iter_count = 0
        no_improve_count = 0
        max_no_improve = 50 if allow_2swap else 200

        while improved and time.time() < deadline - 5 and iter_count < max_iters:
            improved = False
            iter_count += 1
            rng = random.Random(iter_count)
            code_indices = list(range(len(code)))
            rng.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline - 5:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    no_improve_count = 0
                    break

                # Find candidates covering all only_by_i using inverse index
                candidate_set = None
                for w in only_by_i:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                # Exclude current code
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Score candidates: how many uniquely-covered points they help
                best_cand = None
                best_score = -1
                only_by_i_set = set(only_by_i.tolist())
                for cand in candidate_set:
                    bp_cand = balls[cand]
                    bp_cand_set = set(bp_cand.tolist())
                    help_count = len(only_by_i_set & bp_cand_set)
                    if help_count > best_score:
                        best_score = help_count
                        best_cand = cand

                if best_cand is not None and best_score > 0:
                    # Perform swap
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    code[ci] = best_cand
                    code_set.add(best_cand)
                    cover_count[balls[best_cand]] += 1
                    improved = True
                    no_improve_count = 0
                    break

            # If no 1-swap improvement, try 2-swap moves to escape local minima
            if not improved and allow_2swap:
                no_improve_count += 1
                if no_improve_count <= max_no_improve and time.time() < deadline - 5:
                    # Try a 2-swap: remove two codewords, add two new ones
                    if len(code) >= 2:
                        # Pick two codewords to potentially remove
                        idx1 = rng.randint(0, len(code) - 1)
                        idx2 = rng.randint(0, len(code) - 1)
                        while idx2 == idx1:
                            idx2 = rng.randint(0, len(code) - 1)
                        
                        c1 = code[idx1]
                        c2 = code[idx2]
                        
                        # Compute what would be uncovered if we remove both
                        temp_cover = cover_count.copy()
                        temp_cover[balls[c1]] -= 1
                        temp_cover[balls[c2]] -= 1
                        
                        uncovered_after_removal = temp_cover == 0
                        n_uncovered = uncovered_after_removal.sum()
                        
                        if n_uncovered == 0:
                            # Both can be removed without uncovering anything
                            # Find best replacement pair
                            gains = uncovered_after_removal.astype(np.int32)
                            # For each candidate, count how many of the uniquely-covered-by-either points it covers
                            unique_mask = (temp_cover == 1) & (cover_count == 1)
                            # Points that are only covered by c1 or c2
                            c1_unique = (cover_count == 1) & (np.isin(np.arange(total), balls[c1]))
                            c2_unique = (cover_count == 1) & (np.isin(np.arange(total), balls[c2]))
                            either_unique = c1_unique | c2_unique
                            
                            if either_unique.sum() > 0:
                                # Find best single replacement
                                cand_gains = either_unique[balls].sum(axis=1)
                                cand_gains[list(code)] = -1
                                best_new = int(np.argmax(cand_gains))
                                if cand_gains[best_new] > 0:
                                    # Remove both, add one
                                    code.pop(idx2)
                                    code.pop(idx1)
                                    cover_count[balls[c1]] -= 1
                                    cover_count[balls[c2]] -= 1
                                    code.append(best_new)
                                    cover_count[balls[best_new]] += 1
                                    code_set.discard(c1)
                                    code_set.discard(c2)
                                    code_set.add(best_new)
                                    improved = True
                                    no_improve_count = 0
                                    continue
                        elif n_uncovered <= 4:
                            # Try to find two replacements that cover all uncovered
                            uncov_list = np.where(uncovered_after_removal)[0].tolist()
                            # Find candidates that cover most of the uncovered
                            cand_scores = {}
                            for w in uncov_list:
                                for c in centers_covering[w]:
                                    if c not in code_set:
                                        cand_scores[c] = cand_scores.get(c, 0) + 1
                            
                            if cand_scores:
                                sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
                                # Try top candidates as pair
                                top_cands = [c for c, s in sorted_cands[:20]]
                                best_pair = None
                                best_pair_score = -1
                                for a_idx in range(min(len(top_cands), 15)):
                                    for b_idx in range(a_idx + 1, min(len(top_cands), 15)):
                                        a, b = top_cands[a_idx], top_cands[b_idx]
                                        # Check if {a, b} covers all uncovered
                                        covered = set(balls[a].tolist()) | set(balls[b].tolist())
                                        if all(w in covered for w in uncov_list):
                                            # Score: total coverage gain
                                            score = len(covered & set(balls[c1].tolist() + balls[c2].tolist()))
                                            if score > best_pair_score:
                                                best_pair_score = score
                                                best_pair = (a, b)
                                
                                if best_pair:
                                    a, b = best_pair
                                    code.pop(idx2)
                                    code.pop(idx1)
                                    cover_count[balls[c1]] -= 1
                                    cover_count[balls[c2]] -= 1
                                    code.append(a)
                                    code.append(b)
                                    cover_count[balls[a]] += 1
                                    cover_count[balls[b]] += 1
                                    code_set.discard(c1)
                                    code_set.discard(c2)
                                    code_set.add(a)
                                    code_set.add(b)
                                    improved = True
                                    no_improve_count = 0
                                    continue

        return code, cover_count

    def genetic_algorithm(code, cover_count, pop_size=20, generations=50):
        """
        Genetic algorithm on binary genomes representing codeword presence.
        Fitness: minimize codeword count, penalize coverage gaps.
        """
        code_set = set(code)
        best_code = list(code)
        best_cover = cover_count.copy()
        best_fitness = len(code)

        # Create initial population
        population = []
        for c in best_code:
            cover_count[balls[c]] += 1

        # Population: list of (code_list, cover_count, fitness)
        pop = [(list(best_code), best_cover.copy(), len(best_code))]

        # Add some perturbed versions
        rng = random.Random(42)
        for _ in range(pop_size - 1):
            perturbed = list(best_code)
            # Randomly remove a few
            for _ in range(min(3, len(perturbed))):
                idx = rng.randint(0, len(perturbed) - 1)
                c = perturbed.pop(idx)
            # Randomly add a few
            for _ in range(min(3, total)):
                c = rng.randint(0, total - 1)
                if c not in perturbed:
                    perturbed.append(c)
            # Compute cover count
            pc = np.zeros(total, dtype=np.int32)
            for c in perturbed:
                pc[balls[c]] += 1
            uncovered = (pc == 0).sum()
            fitness = len(perturbed) + uncovered * 10
            pop.append((perturbed, pc, fitness))

        for gen in range(generations):
            if time.time() > deadline - 10:
                break

            # Sort by fitness
            pop.sort(key=lambda x: x[2])

            # Keep best
            if pop[0][2] < best_fitness:
                best_fitness = pop[0][2]
                best_code = list(pop[0][0])
                best_cover = pop[0][1].copy()

            # Create next generation
            new_pop = []

            # Elitism: keep top half
            elite_count = max(1, len(pop) // 2)
            new_pop.extend(pop[:elite_count])

            # Crossover and mutation
            while len(new_pop) < pop_size:
                if time.time() > deadline - 10:
                    break

                # Select parents (tournament)
                p1_idx = rng.randint(0, len(pop) - 1)
                p2_idx = rng.randint(0, len(pop) - 1)
                # Tournament selection
                for _ in range(3):
                    c1 = rng.randint(0, len(pop) - 1)
                    if pop[c1][2] < pop[p1_idx][2]:
                        p1_idx = c1
                for _ in range(3):
                    c2 = rng.randint(0, len(pop) - 1)
                    if pop[c2][2] < pop[p2_idx][2]:
                        p2_idx = c2

                parent1 = pop[p1_idx][0]
                parent2 = pop[p2_idx][0]

                # Crossover: uniform crossover on codeword sets
                child_set = set(parent1) | set(parent2)
                # Randomly drop some
                child_list = list(child_set)
                rng.shuffle(child_list)
                # Keep a random subset
                keep_count = max(1, len(child_list) - rng.randint(0, 5))
                child_list = child_list[:keep_count]

                # Mutation: add/remove random codewords
                for _ in range(rng.randint(0, 3)):
                    if child_list and rng.random() < 0.5:
                        idx = rng.randint(0, len(child_list) - 1)
                        child_list.pop(idx)
                    else:
                        c = rng.randint(0, total - 1)
                        if c not in child_list:
                            child_list.append(c)

                # Compute fitness
                pc = np.zeros(total, dtype=np.int32)
                for c in child_list:
                    pc[balls[c]] += 1
                uncovered = (pc == 0).sum()
                fitness = len(child_list) + uncovered * 10
                new_pop.append((child_list, pc, fitness))

            pop = new_pop

        # Final trim
        best_code, best_cover = trim_code(best_code, best_cover)
        return best_code, best_cover

    # Main execution
    # Try structured init first
    code, cover_count = structured_init()
    code, cover_count = trim_code(code, cover_count)
    code, cover_count = targeted_local_search(code, cover_count, max_iters=50000, allow_2swap=True)

    if time.time() < deadline - 25:
        # Try genetic algorithm
        code, cover_count = genetic_algorithm(code, cover_count, pop_size=15, generations=30)
        code, cover_count = trim_code(code, cover_count)
        code, cover_count = targeted_local_search(code, cover_count, max_iters=40000, allow_2swap=True)

    # Final cleanup
    code, cover_count = trim_code(code, cover_count)
    # One more pass of local search with 2-swaps on the trimmed code
    if time.time() < deadline - 10:
        code, cover_count = targeted_local_search(code, cover_count, max_iters=20000, allow_2swap=True)
        code, cover_count = trim_code(code, cover_count)

    # Convert to word lists
    result = [to_digits(w) for w in code]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
