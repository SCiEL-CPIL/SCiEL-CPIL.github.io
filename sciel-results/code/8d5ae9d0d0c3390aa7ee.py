import numpy as np
import itertools
import time
import random
import heapq

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: ball[i] is list of indices within Hamming distance 1 of word i
balls = []
for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices.append(idx)
    balls.append(np.array(ball_indices, dtype=np.int32))

# Convert to list of lists for faster access
balls_list = [list(b) for b in balls]

# Precompute the distance-2 graph adjacency for independent set initialization
# Two codewords at distance 2 cannot both be in a minimal cover (one can replace the other's coverage)
# Actually, for covering codes, codewords at distance 2 overlap in exactly 4 positions (ball size 19, intersection = 19+19-37=1? No)
# Ball size = 1 + 6*2 = 13
# Two words at distance d: intersection of balls has size depending on d
# For d=1: intersection = 1 + 5*2 = 11 (share center? No, different centers. Share words at distance 1 from both)
# Let me think differently. The independent set idea: if two codewords are at distance 2, their balls overlap significantly.
# Actually the key insight from inspiration 3 is that a greedy independent set on the distance-2 graph gives better initial cover.

# Let's precompute which pairs are at distance 2
dist2_adj = [[] for _ in range(NUM_WORDS)]
for i in range(NUM_WORDS):
    for j in range(i + 1, NUM_WORDS):
        d = np.sum(all_words[i] != all_words[j])
        if d == 2:
            dist2_adj[i].append(j)
            dist2_adj[j].append(i)

def greedy_independent_set_init():
    """Initialize using greedy independent set on distance-2 graph, then cover remaining."""
    random.shuffle(all_words[0])  # dummy to seed RNG differently
    # Greedy: pick codeword, remove it and its distance-2 neighbors from consideration
    # But we still need to cover everything, so this is just for ordering
    # Actually, let's do: greedy set cover but prefer codewords that are "independent" (not at dist 2 from already chosen)
    
    covered = np.zeros(NUM_WORDS, dtype=bool)
    code = []
    chosen_set = set()
    
    # Precompute for each codeword, which other codewords are at distance 2
    # Use a boolean mask for speed
    dist2_mask = np.zeros((NUM_WORDS, NUM_WORDS), dtype=bool)
    for i in range(NUM_WORDS):
        for j in dist2_adj[i]:
            dist2_mask[i, j] = True
    
    while not np.all(covered):
        scores = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in range(NUM_WORDS):
            if c in chosen_set:
                continue
            scores[c] = np.sum(~covered[balls[c]])
        
        max_score = np.max(scores)
        if max_score == 0:
            break
        
        # Among candidates with max score, prefer those not at distance 2 from any chosen codeword
        cands = np.where(scores == max_score)[0]
        
        # Score candidates by independence
        best_cand = None
        best_indep_score = -1
        for c in cands:
            # Count how many chosen codewords are at distance 2 from c
            indep_score = 0
            for chosen in chosen_set:
                if dist2_mask[c, chosen]:
                    indep_score -= 1
            # Also add small random tiebreaker
            indep_score += random.random() * 0.01
            if indep_score > best_indep_score:
                best_indep_score = indep_score
                best_cand = c
        
        code.append(best_cand)
        chosen_set.add(best_cand)
        covered[balls[best_cand]] = True
    
    return code

def local_search_pq(code):
    """Aggressive local search using priority queue to remove redundant codewords."""
    code = list(code)
    if not code:
        return code
    
    # Compute coverage counts
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        for w in balls[c]:
            cov_counts[w] += 1
    
    # Build priority queue of potentially removable codewords
    # A codeword is removable if all words in its ball have cov_count > 1
    # Use a heap ordered by "removability score" (min cov_count in its ball, higher = more removable)
    
    def removability_score(c):
        """Higher score means more likely to be removable."""
        return np.min(cov_counts[balls[c]])
    
    # Initial heap
    heap = []
    for i, c in enumerate(code):
        s = removability_score(c)
        heapq.heappush(heap, (-s, i))  # min-heap, so negate
    
    # Active indices set
    active = set(range(len(code)))
    
    while heap:
        neg_s, i = heapq.heappop(heap)
        if i not in active:
            continue
        s = -neg_s
        # Recompute score (it may have changed)
        actual_s = removability_score(code[i])
        if actual_s != s:
            # Re-push with updated score
            heapq.heappush(heap, (-actual_s, i))
            continue
        
        if actual_s > 1:
            # Codeword i is removable
            c = code[i]
            for w in balls[c]:
                cov_counts[w] -= 1
            active.discard(i)
            # Remove from code list (mark as inactive, compact later)
            code[i] = -1  # mark as removed
        
        # After removal, neighbors' scores may change, re-push them
        # For efficiency, just re-push all active codewords that share coverage with c
        # But for small problem, just re-push all active
        for j in active:
            if j != i and code[j] >= 0:
                new_s = removability_score(code[j])
                heapq.heappush(heap, (-new_s, j))
    
    # Compact
    result = [c for c in code if c >= 0]
    return result

def k_swap_perturb(code, k):
    """Remove k random codewords and add k new ones to cover gaps."""
    code = list(code)
    if len(code) <= k:
        return code
    
    # Remove k random codewords
    to_remove = random.sample(range(len(code)), k)
    to_remove_set = set(to_remove)
    new_code = [c for i, c in enumerate(code) if i not in to_remove_set]
    
    # Recompute coverage
    covered = np.zeros(NUM_WORDS, dtype=bool)
    for c in new_code:
        covered[balls[c]] = True
    
    # Add k new codewords to cover gaps
    for _ in range(k):
        if np.all(covered):
            break
        scores = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in range(NUM_WORDS):
            if c in new_code:
                continue
            scores[c] = np.sum(~covered[balls[c]])
        
        max_score = np.max(scores)
        if max_score == 0:
            break
        
        cands = np.where(scores == max_score)[0]
        c = cands[random.randint(0, len(cands) - 1)]
        new_code.append(c)
        covered[balls[c]] = True
    
    return new_code

def greedy_set_cover():
    """Standard greedy set cover."""
    covered = np.zeros(NUM_WORDS, dtype=bool)
    code = []
    
    while not np.all(covered):
        scores = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in range(NUM_WORDS):
            scores[c] = np.sum(~covered[balls[c]])
        
        max_score = np.max(scores)
        if max_score == 0:
            break
        
        cands = np.where(scores == max_score)[0]
        c = cands[random.randint(0, len(cands) - 1)]
        code.append(c)
        covered[balls[c]] = True
    
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        w2i = {tuple(w): i for i, w in enumerate(all_w)}
        bl = []
        for i in range(len(all_w)):
            c = all_w[i]
            bi = [i]
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bi.append(w2i[tuple(nb)])
            bl.append(np.array(bi, dtype=np.int32))
        
        covered = np.zeros(len(all_w), dtype=bool)
        code = []
        while not np.all(covered):
            scores = np.zeros(len(all_w), dtype=np.int32)
            for c in range(len(all_w)):
                scores[c] = np.sum(~covered[bl[c]])
            max_s = np.max(scores)
            if max_s == 0:
                break
            cands = np.where(scores == max_s)[0]
            c = cands[random.randint(0, len(cands) - 1)]
            code.append(c)
            covered[bl[c]] = True
        return [list(all_w[c]) for c in code]
    
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (150s limit - 35s buffer)
    
    # Initial best from independent set initialization
    best_code = greedy_independent_set_init()
    best_size = len(best_code)
    
    # ILS loop
    current_code = list(best_code)
    current_size = best_size
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Perturbation: k-swap
        k = random.choice([1, 2, 3])
        perturbed = k_swap_perturb(current_code, k)
        
        # Local search
        improved = local_search_pq(perturbed)
        
        current_size = len(improved)
        
        # Accept if better or equal (with some probability for diversity)
        if current_size < best_size:
            best_size = current_size
            best_code = list(improved)
            current_code = list(improved)
        elif current_size == best_size and random.random() < 0.3:
            current_code = list(improved)
        elif random.random() < 0.05:
            # Occasionally accept worse solutions to escape local optima
            current_code = list(improved)
    
    # Final local search on best
    final_code = local_search_pq(best_code)
    if len(final_code) < best_size:
        best_code = final_code
        best_size = len(final_code)
    
    return [list(all_words[c]) for c in best_code]
