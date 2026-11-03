import numpy as np
import itertools
import time
import random

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: ball_matrix[c, w] = 1 if w is within distance 1 of c
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
ball_indices = []  # list of arrays, each containing indices in the ball

for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

# Convert to list of arrays for fast indexing
ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]

# Precompute for WalkSAT: for each word w, list of codewords whose ball covers w
# This is the transpose of ball_matrix, stored as sparse lists
word_to_codewords = [[] for _ in range(NUM_WORDS)]
for c in range(NUM_WORDS):
    for w in ball_indices[c]:
        word_to_codewords[w].append(c)
word_to_codewords = [np.array(lst, dtype=np.int32) for lst in word_to_codewords]

# Precompute ball_counts: number of codewords covering each word (used for quick checks)
# Actually we need per-word: which codewords cover it -> word_to_codewords
# And per-codeword: which words it covers -> ball_indices
# For WalkSAT flip evaluation:
#   If we select codeword c: gain = number of newly covered words (uncovered words in ball[c])
#   If we deselect codeword c: loss = number of words that become uncovered (words in ball[c] 
#     that are only covered by c)
# We track coverage counts dynamically.

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def _walksat_local_search(selected_set, time_budget, max_iters=50000):
    """
    WalkSAT-style local search on the hitting set formulation.
    selected_set: set of codeword indices currently selected.
    Objective: minimize |selected_set| subject to all words covered.
    We use a smooth surrogate: minimize |selected_set| + lambda * (uncovered_words)
    where uncovered_words = number of words not covered by any selected codeword.
    
    Flips: add a codeword (if it covers new words) or remove a codeword (if no word
    becomes uncovered). WalkSAT randomly picks flips with probability proportional
    to how "helpful" they are, with some noise to escape local minima.
    """
    start_t = time.time()
    selected = set(selected_set)
    
    # Compute initial coverage
    cov_count = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in selected:
        cov_count[ball_indices[c]] += 1
    
    num_uncovered = int(np.sum(cov_count == 0))
    current_cost = len(selected) + 100 * num_uncovered  # heavy penalty for uncovered
    
    best_selected = set(selected)
    best_cost = current_cost
    
    np.random.seed(random.randint(0, 2**31))
    
    for it in range(max_iters):
        if time.time() - start_t > time_budget:
            break
        
        # Compute scores for all possible flips
        # For adding codeword c: gain = number of uncovered words in ball[c]
        # For removing codeword c: loss = number of words in ball[c] with cov_count == 1
        #   (these would become uncovered)
        
        # Vectorized: for each codeword, compute
        # add_gain[c] = sum of (cov_count == 0) over ball_indices[c]
        # remove_loss[c] = sum of (cov_count == 1) over ball_indices[c]
        
        # We can compute this using matrix multiplication or direct indexing
        # add_gain: for each c, count uncovered in its ball
        # remove_loss: for each c, count singly-covered in its ball
        
        uncovered_mask = (cov_count == 0)
        singly_mask = (cov_count == 1)
        
        # add_gain[c] = ball_matrix[c].dot(uncovered_mask)
        # remove_loss[c] = ball_matrix[c].dot(singly_mask)
        add_gain = ball_matrix.dot(uncovered_mask).astype(np.int32)
        remove_loss = ball_matrix.dot(singly_mask).astype(np.int32)
        
        # For codewords NOT in selected: consider adding
        # For codewords IN selected: consider removing
        
        # Build candidate list with scores
        candidates = []
        for c in range(NUM_WORDS):
            if c in selected:
                # Removing c: cost change = -1 + 100 * remove_loss[c]
                # (we lose 1 codeword, but gain 100*remove_loss[c] uncovered words)
                score = -1 + 100 * remove_loss[c]
                if score < 0:  # only consider if it reduces cost
                    candidates.append((score, c, 'remove'))
            else:
                # Adding c: cost change = +1 - 100 * add_gain[c]
                score = 1 - 100 * add_gain[c]
                if score < 0:  # only consider if it reduces cost
                    candidates.append((score, c, 'add'))
        
        if not candidates:
            # No improving move; try random walk
            # Pick a random codeword to flip
            if selected:
                c = random.choice(list(selected))
                if remove_loss[c] == 0:
                    # Safe to remove
                    selected.remove(c)
                    cov_count[ball_indices[c]] -= 1
                    num_uncovered = int(np.sum(cov_count == 0))
                    current_cost = len(selected) + 100 * num_uncovered
                else:
                    # Walk: remove anyway (may increase uncovered)
                    selected.remove(c)
                    cov_count[ball_indices[c]] -= 1
                    num_uncovered = int(np.sum(cov_count == 0))
                    current_cost = len(selected) + 100 * num_uncovered
            else:
                # Add a random codeword
                c = random.randint(0, NUM_WORDS - 1)
                selected.add(c)
                cov_count[ball_indices[c]] += 1
                num_uncovered = int(np.sum(cov_count == 0))
                current_cost = len(selected) + 100 * num_uncovered
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_selected = set(selected)
            continue
        
        # Pick best candidate with some randomness (WalkSAT style)
        # Sort by score and pick from top few
        candidates.sort(key=lambda x: x[0])
        top_k = min(5, len(candidates))
        chosen = random.choice(candidates[:top_k])
        score, c, action = chosen
        
        # Apply flip
        if action == 'add':
            selected.add(c)
            cov_count[ball_indices[c]] += 1
        else:  # remove
            selected.remove(c)
            cov_count[ball_indices[c]] -= 1
        
        num_uncovered = int(np.sum(cov_count == 0))
        current_cost = len(selected) + 100 * num_uncovered
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_selected = set(selected)
        
        # If fully covered and we've been improving, continue to reduce size
        # If not fully covered, the penalty drives us to cover first
    
    # Return the best found
    return best_selected

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy for general case
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (150s limit)
    
    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Phase 1: Vectorized greedy set cover with randomization
    # ball_matrix is (NUM_WORDS, NUM_WORDS), int8
    # covered is (NUM_WORDS,) bool
    # score[c] = sum of uncovered in ball[c] = ball_matrix[c].dot(~covered)
    
    np.random.seed(42)
    
    for restart in range(100000):
        if time.time() > deadline:
            break
        
        # Allocate remaining time budget for this restart
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Split time: 30% for greedy, 70% for WalkSAT local search
        greedy_budget = remaining * 0.3
        walksat_budget = remaining * 0.7
        
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase
        greedy_start = time.time()
        while not np.all(covered):
            if time.time() - greedy_start > greedy_budget:
                break
            # Vectorized score calculation
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            # Randomized selection from top candidates
            # Pick from candidates with score >= max_score * 0.85 to add diversity
            threshold = max_score * 0.85
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[ball_indices[c]] = True
        
        # If greedy didn't finish, complete with standard greedy (no randomization)
        while not np.all(covered):
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
            c = int(np.argmax(scores))
            code.append(c)
            covered[ball_indices[c]] = True
        
        # Reduction Phase: remove redundant codewords
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
        
        # WalkSAT Local Search Phase
        if walksat_budget > 1:
            selected_set = set(code)
            improved_set = _walksat_local_search(selected_set, walksat_budget, max_iters=100000)
            
            # Verify coverage of improved_set
            if improved_set:
                test_cov = np.zeros(NUM_WORDS, dtype=bool)
                for c in improved_set:
                    test_cov[ball_indices[c]] = True
                if np.all(test_cov):
                    code = list(improved_set)
        
        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                # We found the best known. Continue trying to improve.
                pass

    return best_code
