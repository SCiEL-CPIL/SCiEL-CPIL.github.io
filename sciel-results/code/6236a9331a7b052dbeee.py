"""Fresh design: maximize comparable pairs via chain-like structure + local search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
N = 12
M = 40
FULL_MASK = (1 << N) - 1
DEADLINE_MARGIN = 5.0  # seconds before the 90s wall-clock limit

def _count_pairs(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    cnt = 0
    n = len(masks)
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            if mi & mj == mi and mi != mj:
                cnt += 1
            elif mi & mj == mj and mi != mj:
                cnt += 1
    return cnt

def _build_chain():
    """Build 40 distinct sets forming a long chain (every pair comparable)."""
    # A full chain of length 12 gives 66 pairs. We need 40 distinct sets.
    # Use the 40 largest "rank-ordered" subsets? Better: take a Hamiltonian-like
    # chain in the subset lattice by adding/removing elements to keep distinctness.
    # Simplest valid chain: all subsets of size 1..11 that form a nested sequence.
    # Construct a nested sequence of 40 distinct subsets.
    seq = []
    cur = 0
    # We need 40 distinct masks, strictly increasing by inclusion.
    # Greedy: start empty, repeatedly add one element; when full, we can't.
    # Instead use: for k in 0..12, include all subsets of size k? That's not a chain.
    # A true chain: S1 ⊂ S2 ⊂ ... ⊂ S40. Max length is 13 (0..12 elements).
    # So a pure chain of 40 is impossible! Max chain length = 13.
    # Therefore we need a family that is "almost" a chain with branches.
    # Strategy: take a maximal chain (13 sets) and add sets that are comparable
    # to as many as possible. Sets comparable to the whole chain are those that
    # lie on a chain containing it — essentially subsets/supersets of chain members.
    # Better: use a "ladder" — multiple parallel chains sharing endpoints.
    # Design: pick 13-set chain C. Then add sets that are subsets of some C_i
    # and supersets of some C_j. To maximize comparability, add sets that fit
    # into the chain's gaps.
    # Alternative strong structure: take all subsets of a fixed 8-element set?
    # No. Let's use: 40 sets = union of several chains.
    # Simplest high-score: take a 13-chain, then for each of the remaining 27
    # sets, pick a set that is comparable to as many existing sets as possible.
    # We'll do greedy insertion.
    # First build the 13-chain: sizes 0,1,2,...,12 with nested elements.
    chain = []
    cur = 0
    chain.append(cur)
    for i in range(1, N + 1):
        cur = cur | (1 << (i - 1))
        chain.append(cur)
    # chain has 13 masks: sizes 0..12
    used = set(chain)
    result = list(chain)
    # Greedily add 27 more sets, each maximizing comparability with current set.
    remaining = M - len(result)
    for _ in range(remaining):
        best_mask = None
        best_score = -1
        # Candidate masks: all masks not yet used (4096 total, minus used)
        # To speed up, consider masks that are subsets or supersets of some chain member.
        # Actually just iterate all 4096 masks; 27 * 4096 * (pair checks) is fine.
        for mask in range(1 << N):
            if mask in used:
                continue
            # Count comparability with current result
            sc = 0
            for r in result:
                if (mask & r) == mask and mask != r:
                    sc += 1
                elif (mask & r) == r and mask != r:
                    sc += 1
            if sc > best_score:
                best_score = sc
                best_mask = mask
        if best_mask is None:
            break
        result.append(best_mask)
        used.add(best_mask)
    return result

def _local_search(masks, max_iter=2000):
    """Local search: replace one set with a better one if it increases total pairs."""
    current_score = _count_pairs(masks)
    used = set(masks)
    best = list(masks)
    best_score = current_score
    for _ in range(max_iter):
        improved = False
        for idx in range(M):
            old = best[idx]
            # Try replacing best[idx] with some other mask
            # Sample candidate masks
            for _ in range(50):
                cand = random.randint(0, (1 << N) - 1)
                if cand in used or cand == old:
                    continue
                # Compute new score contribution
                # Remove old, add cand
                old_comp = 0
                for j in range(M):
                    if j == idx:
                        continue
                    r = best[j]
                    if (old & r) == old and old != r:
                        old_comp += 1
                    elif (old & r) == r and old != r:
                        old_comp += 1
                    elif (r & old) == r and r != old:
                        old_comp += 1
                # Wait, need to be careful: old_comp counts pairs involving old
                # Let me recount properly
                old_comp = 0
                for j in range(M):
                    if j == idx:
                        continue
                    r = best[j]
                    if old != r:
                        if (old & r) == old:  # old subset of r
                            old_comp += 1
                        elif (old & r) == r:  # r subset of old
                            old_comp += 1
                cand_comp = 0
                for j in range(M):
                    if j == idx:
                        continue
                    r = best[j]
                    if cand != r:
                        if (cand & r) == cand:
                            cand_comp += 1
                        elif (cand & r) == r:
                            cand_comp += 1
                new_score = best_score - old_comp + cand_comp
                if new_score > best_score:
                    used.discard(old)
                    used.add(cand)
                    best[idx] = cand
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN

    # Start with chain-based construction
    best_masks = _build_chain()
    best_score = _count_pairs(best_masks)

    # Local search with multiple random restarts
    random.seed(42)
    while time.time() < deadline:
        # Local search on current best
        candidate, score = _local_search(list(best_masks), max_iter=500)
        if score > best_score:
            best_masks = candidate
            best_score = score
        # Check deadline
        if time.time() >= deadline:
            break
        # Random restart: perturb and search
        if time.time() > start_time + 30:
            # Reduce perturbation as we approach deadline
            perturb = min(3, int((deadline - time.time()) / 10) + 1)
        else:
            perturb = 5
        perturbed = list(best_masks)
        used = set(perturbed)
        for _ in range(perturb):
            idx = random.randint(0, M - 1)
            old = perturbed[idx]
            # Try to find a replacement
            for _ in range(100):
                cand = random.randint(0, (1 << N) - 1)
                if cand not in used:
                    perturbed[idx] = cand
                    used.discard(old)
                    used.add(cand)
                    break
        candidate, score = _local_search(perturbed, max_iter=300)
        if score > best_score:
            best_masks = candidate
            best_score = score

    # Convert to list of lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(N) if (mask >> b) & 1])
    return result
# EVOLVE-BLOCK-END
