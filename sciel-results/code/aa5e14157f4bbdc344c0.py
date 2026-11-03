"""Criticality-based greedy + strict 2-opt local search for K_q(n, radius)."""
import time
import itertools
import numpy as np
import random

_DEADLINE = 150.0


def _word_to_idx(word, q, n):
    """Convert a word (list of ints) to a linear index in [0, q**n)."""
    idx = 0
    for x in word:
        idx = idx * q + x
    return idx


def _idx_to_word(idx, q, n):
    """Convert a linear index to a word (list of ints)."""
    word = [0] * n
    for c in range(n - 1, -1, -1):
        word[c] = idx % q
        idx //= q
    return word


def _get_neighbors(cw_idx, q, n, radius):
    """Return list of indices of all words within Hamming distance <= radius of cw_idx."""
    if radius == 1:
        neighbors = [cw_idx]
        word = _idx_to_word(cw_idx, q, n)
        for pos in range(n):
            offset = q ** (n - 1 - pos)
            cur = word[pos]
            for v in range(q):
                if v != cur:
                    neighbors.append(cw_idx + (v - cur) * offset)
        return neighbors
    else:
        total = q ** n
        word = _idx_to_word(cw_idx, q, n)
        neighbors = []
        for i in range(total):
            w = _idx_to_word(i, q, n)
            d = sum(1 for a, b in zip(word, w) if a != b)
            if d <= radius:
                neighbors.append(i)
        return neighbors


def _precompute_neighbors(q, n, radius):
    """Precompute neighbor lists for all words as a list of arrays."""
    total = q ** n
    neighbors = [None] * total
    for i in range(total):
        neighbors[i] = np.array(_get_neighbors(i, q, n, radius), dtype=np.int32)
    return neighbors


def _criticality_greedy(q, n, radius, deadline, seed=42):
    """
    Criticality-based greedy: prioritize words that are hardest to cover.
    At each step, pick the codeword that covers the most 'critical' uncovered words,
    where criticality is weighted by how few other codewords could cover them.
    Returns a list of codeword indices.
    """
    rng = random.Random(seed)
    total = q ** n
    all_neighbors = _precompute_neighbors(q, n, radius)
    
    uncovered = np.ones(total, dtype=bool)
    coverage_count = np.zeros(total, dtype=np.int32)
    codewords = []
    
    # Precompute for each word, which codewords can cover it (inverse neighbor map)
    # This helps estimate criticality
    # For radius=1, each word is covered by itself and its 2*(q-1)*n neighbors
    # We'll use a simpler heuristic: criticality = 1/coverage_count if covered, else large
    
    while uncovered.any() and time.time() < deadline:
        # Find the most critical uncovered word
        # Criticality: words with fewer potential covering codewords are more critical
        # For now, use a simple score: prefer codewords that cover many uncovered words
        # but weight by how "isolated" those uncovered words are
        
        # Compute marginal gain for each candidate codeword
        # Only consider uncovered words as primary candidates, plus some random covered ones
        uncovered_idx = np.where(uncovered)[0]
        if len(uncovered_idx) == 0:
            break
        
        # Candidate set: all uncovered words + sample of covered words
        covered_idx = np.where(~uncovered)[0]
        if len(covered_idx) > 500:
            extra_sample = rng.sample(list(covered_idx), 500)
        else:
            extra_sample = list(covered_idx)
        
        candidates = list(uncovered_idx) + extra_sample
        candidates = list(set(candidates))
        
        best_cw = -1
        best_score = -1
        
        for cw in candidates:
            if time.time() > deadline:
                break
            nbs = all_neighbors[cw]
            # Count uncovered words this covers
            cov_mask = uncovered[nbs]
            cnt = cov_mask.sum()
            if cnt == 0:
                continue
            
            # Score: number of uncovered words covered, with a small bonus for covering
            # words that are "harder" to cover (lower coverage_count)
            # Use average inverse coverage as a tiebreaker
            hard_words = nbs[cov_mask]
            if len(hard_words) > 0:
                # Words with coverage_count 0 are hardest
                inv_cov = np.where(coverage_count[hard_words] == 0, 2.0, 1.0)
                score = cnt * 10.0 + inv_cov.mean()
            else:
                score = cnt
            
            if score > best_score:
                best_score = score
                best_cw = cw
        
        if best_cw == -1:
            best_cw = int(uncovered_idx[0])
        
        codewords.append(best_cw)
        nbs = all_neighbors[best_cw]
        uncovered[nbs] = False
        coverage_count[nbs] += 1
    
    return codewords


def _strict_2opt(codewords, q, n, radius, deadline, seed=42):
    """
    Strict 2-opt local search: only accept moves that strictly reduce codeword count
    or maintain count while increasing minimum coverage multiplicity.
    """
    rng = random.Random(seed)
    total = q ** n
    all_neighbors = _precompute_neighbors(q, n, radius)
    
    coverage_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        coverage_count[all_neighbors[cw]] += 1
    
    cw_set = set(codewords)
    best_min_cov = coverage_count.min()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try removals
        for i, cw in enumerate(list(codewords)):
            if time.time() > deadline:
                break
            nbs = all_neighbors[cw]
            if (coverage_count[nbs] >= 2).all():
                coverage_count[nbs] -= 1
                codewords.pop(i)
                cw_set.discard(cw)
                improved = True
                best_min_cov = coverage_count.min()
                break
        
        if improved:
            continue
        
        # Phase 2: Try 2-opt swaps (remove one, add one)
        # Only accept if it increases minimum coverage multiplicity
        n_cw = len(codewords)
        if n_cw <= 1:
            break
        
        # Sample removal candidates
        n_samples = min(n_cw, 30)
        remove_candidates = rng.sample(codewords, n_samples)
        
        for rem_cw in remove_candidates:
            if time.time() > deadline:
                break
            
            nbs_rem = all_neighbors[rem_cw]
            new_cov = coverage_count.copy()
            new_cov[nbs_rem] -= 1
            
            # Words that would become uncovered
            uncovered_after = np.where(new_cov == 0)[0]
            if len(uncovered_after) == 0:
                # Can just remove - but we already tried that in Phase 1
                # This shouldn't happen if Phase 1 is working
                continue
            
            # Find a replacement that covers all uncovered_after
            # Candidates: words within distance 1 of any uncovered word
            add_candidates = set()
            for w in uncovered_after[:20]:  # Limit for speed
                word = _idx_to_word(int(w), q, n)
                for pos in range(n):
                    offset = q ** (n - 1 - pos)
                    cur = word[pos]
                    for v in range(q):
                        if v != cur:
                            add_candidates.add(int(w) + (v - cur) * offset)
                add_candidates.add(int(w))
            
            add_candidates -= cw_set
            if not add_candidates:
                continue
            
            best_add = -1
            best_min_cov_new = -1
            
            for add_cw in list(add_candidates)[:40]:
                nbs_add = all_neighbors[add_cw]
                test_cov = new_cov.copy()
                test_cov[nbs_add] += 1
                
                if (test_cov >= 1).all():
                    min_cov = test_cov.min()
                    if min_cov > best_min_cov_new:
                        best_min_cov_new = min_cov
                        best_add = add_cw
                    if min_cov > best_min_cov:
                        break
            
            if best_add != -1 and best_min_cov_new > best_min_cov:
                # Accept: improves minimum coverage
                coverage_count = new_cov
                coverage_count[all_neighbors[best_add]] += 1
                codewords.remove(rem_cw)
                cw_set.discard(rem_cw)
                codewords.append(best_add)
                cw_set.add(best_add)
                best_min_cov = best_min_cov_new
                improved = True
                break
    
    # Final cleanup
    for i, cw in enumerate(list(codewords)):
        if time.time() > deadline:
            break
        nbs = all_neighbors[cw]
        if (coverage_count[nbs] >= 2).all():
            coverage_count[nbs] -= 1
            codewords.pop(i)
            cw_set.discard(cw)
    
    return codewords


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _DEADLINE - 10.0  # safety margin
    
    total = q ** n
    
    # Phase 1: Criticality-based greedy
    greedy_result = _criticality_greedy(q, n, radius, deadline, seed=42)
    
    # Phase 2: Strict 2-opt local search
    result = _strict_2opt(greedy_result, q, n, radius, deadline, seed=42)
    
    # Verify coverage
    all_neighbors = _precompute_neighbors(q, n, radius)
    covered = np.zeros(total, dtype=bool)
    for cw in result:
        covered[all_neighbors[cw]] = True
    
    if not covered.all():
        # Fallback: tetracode direct sum
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            out = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=n - 4):
                    out.append(h + list(tail))
            result = [_word_to_idx(w, q, n) for w in out]
        else:
            result = list(range(total))
    
    # Convert back to list of words
    return [_idx_to_word(idx, q, n) for idx in result]
