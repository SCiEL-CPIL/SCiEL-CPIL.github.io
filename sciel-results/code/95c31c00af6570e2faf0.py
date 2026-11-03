"""Football pool: ternary covering code K_3(6,1) via bit-parallel set cover + local search."""
import time
import random
import itertools

_DEADLINE = 150.0  # wall-clock seconds


def _all_words(q, n):
    """Generate all q^n words as tuples."""
    return list(itertools.product(range(q), repeat=n))


def _hamming(a, b):
    """Hamming distance between two tuples."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_masks(words, q, n, radius):
    """For each potential codeword, build a bitmask of which space words it covers."""
    m = len(words)
    # Precompute index for each word
    idx = {w: i for i, w in enumerate(words)}
    masks = []
    for cw in words:
        mask = 0
        for w in words:
            if _hamming(cw, w) <= radius:
                mask |= (1 << idx[w])
        masks.append(mask)
    return masks


def _greedy_cover(masks, m):
    """Greedy set cover: repeatedly pick the mask covering most uncovered words."""
    uncovered = (1 << m) - 1
    chosen = []
    while uncovered:
        best = -1
        best_count = -1
        for i, mk in enumerate(masks):
            cnt = bin(uncovered & mk).count('1')
            if cnt > best_count:
                best_count = cnt
                best = i
        if best_count == 0:
            break
        chosen.append(best)
        uncovered &= ~masks[best]
    return chosen


def _is_covered(chosen, masks, m):
    """Check if chosen masks cover all m words."""
    union = 0
    for i in chosen:
        union |= masks[i]
    return union == (1 << m) - 1


def _local_search(chosen, masks, m, deadline):
    """Try to remove redundant codewords, then try swaps to reduce count."""
    chosen = list(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each codeword
        for i in range(len(chosen)):
            test = chosen[:i] + chosen[i+1:]
            if _is_covered(test, masks, m):
                chosen = test
                improved = True
                break
        if time.time() >= deadline:
            break
        # Try swapping one codeword for another not in the set
        for i in range(len(chosen)):
            for j in range(m):
                if j in chosen:
                    continue
                test = chosen[:i] + [j] + chosen[i+1:]
                if _is_covered(test, masks, m):
                    # Only accept if it doesn't increase count (it doesn't) 
                    # and we haven't already tried this
                    chosen = test
                    improved = True
                    break
            if improved:
                break
    return chosen


def _perturb(chosen, masks, m, deadline, rng):
    """Perturb: remove a random codeword, then greedily repair."""
    if not chosen:
        return chosen
    new_chosen = list(chosen)
    # Remove 1-3 random codewords
    k = min(rng.randint(1, 3), len(new_chosen))
    for _ in range(k):
        if new_chosen:
            idx = rng.randint(0, len(new_chosen) - 1)
            new_chosen.pop(idx)
    # Greedily repair
    uncovered = (1 << m) - 1
    for i in new_chosen:
        uncovered &= ~masks[i]
    while uncovered and time.time() < deadline:
        best = -1
        best_count = -1
        for i in range(m):
            if i in new_chosen:
                continue
            cnt = bin(uncovered & masks[i]).count('1')
            if cnt > best_count:
                best_count = cnt
                best = i
        if best_count == 0:
            break
        new_chosen.append(best)
        uncovered &= ~masks[best]
    return new_chosen


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5.0  # safety margin
    
    words = _all_words(q, n)
    m = len(words)
    
    # Build coverage masks
    masks = _build_masks(words, q, n, radius)
    
    # Initial greedy cover
    chosen = _greedy_cover(masks, m)
    
    # Local search to remove redundancies
    chosen = _local_search(chosen, masks, m, deadline)
    
    best = list(chosen)
    best_len = len(best)
    
    # Multi-start perturbation
    rng = random.Random(42)
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        perturbed = _perturb(best, masks, m, deadline, rng)
        perturbed = _local_search(perturbed, masks, m, deadline)
        
        if len(perturbed) < best_len and _is_covered(perturbed, masks, m):
            best = perturbed
            best_len = len(best)
        
        # If we've reached a good solution, try more aggressive perturbations
        if best_len <= 73 and time.time() < deadline:
            # Try removing more codewords
            for _ in range(5):
                if time.time() >= deadline:
                    break
                cand = list(best)
                k = rng.randint(2, 5)
                for _ in range(k):
                    if cand:
                        idx = rng.randint(0, len(cand) - 1)
                        cand.pop(idx)
                # Repair
                uncovered = (1 << m) - 1
                for i in cand:
                    uncovered &= ~masks[i]
                while uncovered and time.time() < deadline:
                    best_idx = -1
                    best_cnt = -1
                    for i in range(m):
                        if i in cand:
                            continue
                        cnt = bin(uncovered & masks[i]).count('1')
                        if cnt > best_cnt:
                            best_cnt = cnt
                            best_idx = i
                    if best_cnt == 0:
                        break
                    cand.append(best_idx)
                    uncovered &= ~masks[best_idx]
                cand = _local_search(cand, masks, m, deadline)
                if len(cand) < best_len and _is_covered(cand, masks, m):
                    best = cand
                    best_len = len(best)
    
    # Convert indices back to words
    result = [list(words[i]) for i in best]
    return result
