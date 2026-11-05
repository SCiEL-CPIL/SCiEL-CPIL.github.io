import itertools
import time
import sys
import random
import numpy as np

def _precompute_overlap(n):
    """
    Precompute the overlap matrix for all pairs of permutations of 1..n-1.
    overlap[i][j] = largest k such that suffix(perm_i, k) == prefix(perm_j, k).
    """
    perms = list(itertools.permutations(range(1, n)))
    M = len(perms)
    # Map perm to index
    idx_map = {p: i for i, p in enumerate(perms)}
    
    # overlap matrix
    overlap = np.zeros((M, M), dtype=np.int8)
    
    # For each pair, compute overlap
    # To speed up, we can iterate by length k
    # But M is small (720 for n=7), so O(M^2 * n) is fine.
    
    for i in range(M):
        pi = perms[i]
        for j in range(M):
            pj = perms[j]
            # Max possible overlap is n-1
            # Check k from n-1 down to 0
            for k in range(n-1, -1, -1):
                if k == 0:
                    overlap[i][j] = 0
                    break
                if pi[n-1-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    return perms, overlap

def _eval_order(order, overlap, M, block_len):
    """
    Evaluate the length of the superpermutation for a given order.
    order: list of indices into the perms list.
    """
    # L = M * block_len - sum(overlap[order[i-1]][order[i]])
    # Note: order is a permutation of 0..M-1
    # We need to sum overlaps between consecutive elements in order
    # order is a list of ints (indices)
    
    # Efficient sum using numpy if order is an array
    # But order changes frequently in GA.
    # Simple loop is fast enough for M=720.
    
    total_overlap = 0
    for i in range(1, M):
        total_overlap += overlap[order[i-1]][order[i]]
        
    return M * block_len - total_overlap

def _standard_order(n):
    """Generate the standard recursive order of permutations of 1..n-1."""
    if n == 2:
        return [0] # Only 1 perm of 1
    if n == 3:
        # Perms of 1,2: (1,2), (2,1)
        # Standard superperm for n=3: 1 2 3 1 2 1 3 2 1
        # Blocks: 
        # P1=(1,2) -> 1 2 3 1 2
        # P2=(2,1) -> 2 1 3 2 1
        # Overlap: suffix(1,2) = (1,2) vs prefix(2,1) = (2,1)? No.
        # Let's just generate the standard word and extract order.
        pass
    
    # Recursive construction of the order
    # For n=2, perms of 1: [(1,)]
    # For n=3, perms of 1,2: [(1,2), (2,1)]
    # Standard construction:
    # Start with order for n-1.
    # For each perm p in order:
    #   Insert p, then p with n appended? No.
    # The standard construction for the WORD is:
    # S_n = S_{n-1}
    # For each (n-1)-perm p in S_{n-1} (in order of appearance):
    #   Append p, n, p
    
    # This defines the order of blocks.
    # Let's generate the order directly.
    
    def gen_order(k):
        if k == 1:
            return [(1,)]
        prev = gen_order(k-1)
        # We need to interleave?
        # Standard algorithm:
        # Let P be the list of permutations of 1..k-1 in the order they appear in S_{k-1}.
        # S_k starts with S_{k-1}.
        # Then for each p in P:
        #   Append p, k, p
        # Wait, the "p" in "append p, k, p" is the permutation itself.
        # And the order of these blocks is the same as P.
        # So the order of k-perms (which are blocks of k-1 perms) is just P.
        # But we need the order of (k-1)-perms for the next step.
        
        # Actually, the order of (n-1)-perms in the standard construction is:
        # Take the order of (n-2)-perms from S_{n-2}.
        # For each (n-2)-perm q in that order:
        #   The corresponding (n-1)-perm is q with (n-1) inserted?
        
        # Let's just build the word and extract.
        def build_word(k):
            if k == 1:
                return [1]
            if k == 2:
                return [1, 2, 1]
            prev = build_word(k-1)
            seen = set()
            order = []
            # Extract (k-1)-perms from prev
            for i in range(len(prev) - (k-1) + 1):
                w = tuple(prev[i:i+k-1])
                if w not in seen:
                    seen.add(w)
                    if sorted(w) == list(range(1, k)):
                        order.append(w)
            
            new = []
            for p in order:
                block = list(p) + [k] + list(p)
                # Overlap
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new)-o:] == block[:o]:
                        ov = o
                        break
                new.extend(block[ov:])
            return new
        
        word = build_word(k)
        # Extract order of (k-1) perms
        seen = set()
        res = []
        for i in range(len(word) - (k-1) + 1):
            w = tuple(word[i:i+k-1])
            if w not in seen:
                seen.add(w)
                if sorted(w) == list(range(1, k)):
                    res.append(w)
        return res

    perms_tuples = gen_order(n-1)
    return perms_tuples

def _tuples_to_indices(perms_tuples, idx_map):
    return [idx_map[p] for p in perms_tuples]

def _crossover(parent1, parent2, M):
    """Order Crossover."""
    child = [0] * M
    # Pick two random cut points
    c1, c2 = sorted(random.sample(range(M), 2))
    
    # Copy segment from parent1
    for i in range(c1, c2+1):
        child[i] = parent1[i]
        
    # Fill rest with parent2 order, skipping those already in child
    in_child = set(child[c1:c2+1])
    j = 0
    for i in range(M):
        if i >= c1 and i <= c2:
            continue
        val = parent2[j]
        while val in in_child:
            j += 1
            val = parent2[j]
        child[i] = val
        in_child.add(val)
        j += 1
        
    return child

def _mutate(ind, M, rate=0.1):
    """Swap and Insert mutations."""
    ind = list(ind)
    # Swap
    if random.random() < rate:
        i, j = random.sample(range(M), 2)
        ind[i], ind[j] = ind[j], ind[i]
    # Insert
    if random.random() < rate:
        i = random.randrange(M)
        val = ind.pop(i)
        j = random.randrange(M)
        ind.insert(j, val)
    return ind

def _local_search_2opt(order, overlap, M, block_len, deadline):
    """
    2-opt local search: reverse a segment.
    """
    best_order = list(order)
    best_len = _eval_order(best_order, overlap, M, block_len)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(M-1):
            if time.time() > deadline:
                break
            for j in range(i+1, M):
                if time.time() > deadline:
                    break
                # Reverse segment [i, j]
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_len = _eval_order(new_order, overlap, M, block_len)
                if new_len < best_len:
                    best_order = new_order
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
    return best_order, best_len

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    if n <= 5:
        # Standard construction is optimal or close enough
        def _standard_superperm(k):
            if k == 1:
                return [1]
            if k == 2:
                return [1, 2, 1]
            if k == 3:
                return [1, 2, 3, 1, 2, 1, 3, 2, 1]
            prev = _standard_superperm(k - 1)
            seen = set()
            order = []
            for i in range(len(prev) - (k - 1) + 1):
                window = tuple(prev[i:i + k - 1])
                if window not in seen:
                    seen.add(window)
                    if sorted(window) == list(range(1, k)):
                        order.append(window)
            new = []
            for perm in order:
                block = list(perm) + [k] + list(perm)
                max_ov = min(len(new), len(block))
                overlap = 0
                for ov in range(max_ov, -1, -1):
                    if ov == 0 or new[len(new) - ov:] == block[:ov]:
                        overlap = ov
                        break
                new.extend(block[overlap:])
            return new
        return "".join(map(str, _standard_superperm(n)))

    # n >= 6
    # Precompute overlap
    perms, overlap = _precompute_overlap(n)
    M = len(perms)
    block_len = 2 * n - 1
    
    # Standard order
    std_perms = _standard_order(n)
    idx_map = {p: i for i, p in enumerate(perms)}
    std_order = _tuples_to_indices(std_perms, idx_map)
    
    # Initial population
    pop_size = 20
    population = []
    population.append(std_order)
    
    # Add random and locally optimized individuals
    for _ in range(pop_size - 1):
        ind = list(range(M))
        random.shuffle(ind)
        # Quick local search
        ind, _ = _local_search_2opt(ind, overlap, M, block_len, time.time() + 0.5)
        population.append(ind)
    
    # Evaluate initial population
    fitnesses = [_eval_order(ind, overlap, M, block_len) for ind in population]
    
    best_order = population[fitnesses.index(min(fitnesses))]
    best_len = min(fitnesses)
    
    # GA Loop
    while time.time() < deadline - 10:
        # Selection and Crossover
        new_pop = []
        # Elitism: keep best
        best_idx = fitnesses.index(min(fitnesses))
        new_pop.append(population[best_idx])
        
        for _ in range(pop_size - 1):
            # Tournament selection
            p1_idx = random.randrange(pop_size)
            p2_idx = random.randrange(pop_size)
            if fitnesses[p1_idx] > fitnesses[p2_idx]:
                p1_idx, p2_idx = p2_idx, p1_idx
            
            child = _crossover(population[p1_idx], population[p2_idx], M)
            child = _mutate(child, M)
            
            # Local search on child (limited time)
            child, _ = _local_search_2opt(child, overlap, M, block_len, time.time() + 0.2)
            
            new_pop.append(child)
            
        population = new_pop
        fitnesses = [_eval_order(ind, overlap, M, block_len) for ind in population]
        
        cur_best_idx = fitnesses.index(min(fitnesses))
        if fitnesses[cur_best_idx] < best_len:
            best_len = fitnesses[cur_best_idx]
            best_order = population[cur_best_idx]
            
    # Final polish
    if time.time() < deadline - 2:
        best_order, best_len = _local_search_2opt(best_order, overlap, M, block_len, deadline - 2)
        
    # Construct word
    word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_ov = min(len(word), len(block))
        ov = 0
        for o in range(max_ov, -1, -1):
            if o == 0 or word[len(word)-o:] == block[:o]:
                ov = o
                break
        word.extend(block[ov:])
        
    return "".join(map(str, word))

if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        print(f"n={n}, len={len(w)}")
    # Test n=7
    start = time.time()
    w = construct(7)
    elapsed = time.time() - start
    print(f"n=7, len={len(w)}, time={elapsed:.2f}s")
