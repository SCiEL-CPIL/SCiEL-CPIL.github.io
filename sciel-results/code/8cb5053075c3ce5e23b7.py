import itertools
import random
import time

def _popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def _local_search(S, coverage, total_words, S_set, union_mask):
    """Improve set S via removal and swap operations. Returns (S, S_set, union_mask)."""
    improved = True
    while improved:
        improved = False
        for i in S:
            # Check if i is redundant (all its coverage is covered by others)
            others_mask = union_mask & ~coverage[i]
            unique_i = coverage[i] & ~others_mask
            
            if unique_i == 0:
                # Remove redundant codeword
                S.remove(i)
                S_set.discard(i)
                union_mask &= ~coverage[i]
                improved = True
                break
            
            # Try to swap i with j that covers unique_i and adds more new coverage
            best_j = -1
            best_gain = -1
            for j in range(total_words):
                if j in S_set:
                    continue
                if (unique_i & ~coverage[j]) != 0:
                    continue
                gain = _popcount(coverage[j] & ~union_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            
            if best_j != -1 and best_gain > 0:
                S.remove(i)
                S_set.discard(i)
                S.append(best_j)
                S_set.add(best_j)
                union_mask = (union_mask & ~coverage[i]) | coverage[best_j]
                improved = True
                break
    return S, S_set, union_mask

def _greedy_cover(coverage, total_words, random_order):
    """Greedy set cover. Returns list of chosen indices."""
    covered_mask = 0
    full_mask = (1 << total_words) - 1
    chosen = []
    for idx in random_order:
        if covered_mask == full_mask:
            break
        new_cov = coverage[idx] & ~covered_mask
        if new_cov:
            chosen.append(idx)
            covered_mask |= coverage[idx]
    return chosen

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute coverage bitsets
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = 729
    
    coverage = [0] * total_words
    for i, c in enumerate(all_words):
        mask = 1 << i  # covers itself (word i)
        for pos in range(6):
            for val in range(3):
                if val == c[pos]:
                    continue
                new_c = list(c)
                new_c[pos] = val
                mask |= 1 << word_to_idx[tuple(new_c)]
        coverage[i] = mask

    best_code = None
    best_size = total_words
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 40s margin

    # --- Phase 1: Greedy + Local Search with multiple restarts ---
    r = 0
    while time.time() < deadline:
        random.seed(r)
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        S = _greedy_cover(coverage, total_words, candidates)
        S_set = set(S)
        union_mask = 0
        for i in S:
            union_mask |= coverage[i]
        
        S, S_set, union_mask = _local_search(S, coverage, total_words, S_set, union_mask)
        
        if len(S) < best_size:
            best_size = len(S)
            best_code = [all_words[i] for i in S]
        r += 1

    # --- Phase 2: Genetic Algorithm with sphere-preserving crossover ---
    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Initialize population from best solution found
    population = [list(best_code)]
    
    # Add some diverse individuals
    random.seed(42)
    for _ in range(10):
        cand = list(range(total_words))
        random.shuffle(cand)
        S = _greedy_cover(coverage, total_words, cand)
        S_set = set(S)
        union_mask = 0
        for i in S:
            union_mask |= coverage[i]
        S, S_set, union_mask = _local_search(S, coverage, total_words, S_set, union_mask)
        population.append([all_words[i] for i in S])
    
    # GA loop
    gen = 0
    pop_size = len(population)
    while time.time() < deadline:
        # Evaluate fitness: number of codewords (lower is better), but must be valid
        def fitness(ind):
            # Check validity
            covered = 0
            for word in ind:
                covered |= coverage[word_to_idx[tuple(word)]]
            if covered != (1 << total_words) - 1:
                return (len(ind), 1)  # Invalid, penalize
            return (len(ind), 0)
        
        # Sort by fitness
        population.sort(key=fitness)
        best_ind = population[0]
        best_fit = fitness(best_ind)
        
        if best_fit[1] == 0 and best_fit[0] < best_size:
            best_size = best_fit[0]
            best_code = best_ind
        
        # Selection: keep top half
        elite = population[:pop_size // 2]
        
        # Crossover and mutation
        new_population = list(elite)
        
        while len(new_population) < pop_size and time.time() < deadline:
            # Pick parents
            p1 = random.choice(elite)
            p2 = random.choice(population)
            
            # Sphere-preserving crossover: combine codewords that don't overlap too much
            child = []
            used_positions = set()
            
            # Try to take codewords from both parents, preferring those that add coverage
            covered = 0
            for word in p1:
                cov = coverage[word_to_idx[tuple(word)]]
                if cov & ~covered:
                    child.append(word)
                    covered |= cov
            
            for word in p2:
                if len(child) >= len(p1) + len(p2) - len(elite[0]):
                    break
                cov = coverage[word_to_idx[tuple(word)]]
                if cov & ~covered:
                    child.append(word)
                    covered |= cov
            
            # Mutation: randomly add/remove codewords
            if random.random() < 0.3 and child:
                idx = random.randint(0, len(child) - 1)
                child.pop(idx)
            
            if random.random() < 0.3:
                new_word = random.choice(all_words)
                if new_word not in child:
                    child.append(new_word)
            
            new_population.append(child)
        
        # Local search on some offspring
        for i in range(min(5, len(new_population))):
            S = [word_to_idx[tuple(w)] for w in new_population[i]]
            S_set = set(S)
            union_mask = 0
            for j in S:
                union_mask |= coverage[j]
            S, S_set, union_mask = _local_search(S, coverage, total_words, S_set, union_mask)
            new_population[i] = [all_words[j] for j in S]
        
        population = new_population
        gen += 1

    return [list(c) for c in best_code]

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# We place it inside the block as requested by the format, though the prompt says "Rewrite the program from scratch".
# The seed program has EVOLVE-BLOCK markers. I will provide the full file content.
# EVOLVE-BLOCK-END
