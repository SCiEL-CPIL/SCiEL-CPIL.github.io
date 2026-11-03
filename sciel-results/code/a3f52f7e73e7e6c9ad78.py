import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    # Precompute balls as numpy array
    def build_balls():
        balls = np.empty((total, ball_size), dtype=np.int32)
        for w in range(total):
            digits = []
            tmp = w
            for i in range(n):
                digits.append(tmp % q)
                tmp //= q
            digits = digits[::-1]

            ball = [w]
            for pos in range(n):
                orig = digits[pos]
                for nv in range(q):
                    if nv != orig:
                        new_word = digits[:]
                        new_word[pos] = nv
                        val = 0
                        for d in new_word:
                            val = val * q + d
                        ball.append(val)
            balls[w] = ball
        return balls

    balls = build_balls()

    # Build inverse index: for each word, which centers cover it
    # This is critical for fast candidate enumeration in local search
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # Tetracode product initialization
    def structured_init():
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        
        def digits_to_int(digits, base):
            num = 0
            for d in reversed(digits):
                num = num * base + d
            return num

        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        
        code_ints = [digits_to_int(c, q) for c in product_code]
        code_ints = list(set(code_ints))
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        # Greedy extension
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            gains = uncovered[balls].sum(axis=1)
            # Exclude already chosen
            for c in code_ints:
                gains[c] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        
        return code_ints, cover_count

    # Greedy set cover (standard)
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)

        while covered_count < total:
            if time.time() > deadline - 5:
                break

            covered_in_balls = covered[balls]
            covered_per_ball = covered_in_balls.sum(axis=1)
            gains = ball_size - covered_per_ball

            best_word = int(np.argmax(gains))
            best_new = int(gains[best_word])
            if best_new == 0:
                break

            code.append(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
                cover_count[bp] += 1

        return code, cover_count

    # Try structured init first, fall back to pure greedy
    code, cover_count = structured_init()
    if not code:
        code, cover_count = greedy_cover()

    code_set = set(code)

    # Aggressive trimming of redundant codewords
    def trim(code, cover_count):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if time.time() > deadline - 10:
                    break
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True
        return code, cover_count

    code, cover_count = trim(code, cover_count)

    # Local search: targeted swap using inverse index (optimized)
    def local_search(code, cover_count, time_budget=100):
        code_set = set(code)
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(12345)

        while improved and time.time() < end_time:
            improved = False
            # Shuffle for diversity
            indices = list(range(len(code)))
            rng.shuffle(indices)

            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                # Find words uniquely covered by i
                unique_mask = cover_count[bp_i] == 1
                unique_pts = bp_i[unique_mask]

                if len(unique_pts) == 0:
                    # Remove redundant
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates that cover ALL unique_pts using inverse index
                # Intersection of centers_covering[w] for w in unique_pts
                candidate_set = None
                for w in unique_pts:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                # Filter out already in code
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue

                # Score each candidate: minimize waste
                # A candidate is valid iff it covers all unique_pts (guaranteed by construction)
                # AND after removing i, no point becomes uncovered.
                # Points that become uncovered after removing i: those in bp_i with cover_count==1 (the unique_pts)
                # So we need cand to cover all unique_pts -- already ensured.
                # Score: number of points in bp_cand that have cover_count >= 2 (waste after swap)
                # More precisely: after removing i, eff_count[p] = cover_count[p] - 1 if p in bp_i else cover_count[p]
                # cand adds 1 to bp_cand. Net change for p in bp_cand: +1
                # Net change for p in bp_i \ bp_cand: -1
                # For validity: all p in bp_i must have eff_count[p] >= 1 after removal,
                #   i.e., cover_count[p] - 1 >= 1 for p in bp_i not in bp_cand => cover_count[p] >= 2
                #   and for p in bp_i ∩ bp_cand: cover_count[p] - 1 + 1 = cover_count[p] >= 1 (always true)
                # So validity check: for all p in bp_i, if p not in bp_cand then cover_count[p] >= 2
                # Equivalent: all p in bp_i with cover_count[p] == 1 must be in bp_cand
                # That is exactly: unique_pts ⊆ bp_cand, which is guaranteed by our candidate selection.
                # So all candidates from the intersection are valid!
                # Score: minimize total redundancy added = sum over p in bp_cand of (1 if eff_count[p] >= 1 else 0)
                # where eff_count[p] = cover_count[p] - (1 if p in bp_i else 0)
                # waste = count of p in bp_cand where cover_count[p] - (1 if p in bp_i else 0) >= 1
                #       = count of p in bp_cand where cover_count[p] >= 2, minus count of p in bp_cand ∩ bp_i where cover_count[p] == 1
                #       = (cover_count[bp_cand] >= 2).sum() - (unique_mask[bp_cand]).sum()
                # Since unique_pts ⊆ bp_cand, (unique_mask[bp_cand]).sum() = len(unique_pts)
                
                best_cand = None
                best_score = float('inf')
                n_unique = len(unique_pts)

                # Precompute mask for intersection with bp_i for exact scoring
                in_bp_i = np.zeros(total, dtype=np.int8)
                in_bp_i[bp_i] = 1

                for cand in candidates:
                    bp_cand = balls[cand]
                    # Exact score: minimize sum(cover_count[bp_cand]) - sum(cover_count[bp_i ∩ bp_cand])
                    # This is equivalent to minimizing the increase in total redundancy.
                    # sum(cover_count[bp_cand]) is the total coverage count in the candidate ball.
                    # sum(cover_count[bp_i ∩ bp_cand]) is the overlap part.
                    # The net change in sum(cover_count) is:
                    # +1 for each p in bp_cand
                    # -1 for each p in bp_i
                    # Total change = ball_size - ball_size = 0? No.
                    # Let's re-derive.
                    # Objective: Minimize Sum(cover_count).
                    # Change = Sum_{p in bp_cand} 1 - Sum_{p in bp_i} 1 = 0.
                    # So minimizing Sum(cover_count) is constant?
                    # No, we want to minimize the *redundancy* which is Sum(cover_count) - total.
                    # Since total is constant, minimizing Sum(cover_count) is the goal.
                    # But the change in Sum(cover_count) is always 0 for a swap (remove 1, add 1).
                    # Wait. The "waste" or "redundancy" is not just the sum.
                    # The sum of cover counts is always (K * ball_size). If K is constant, sum is constant.
                    # So minimizing K is the goal. We are swapping, so K is constant.
                    # Then what are we optimizing?
                    # We are not changing K. We are just swapping.
                    # So why swap?
                    # Because the *distribution* of cover counts matters for *future* swaps.
                    # A "better" code is one where cover counts are more "balanced" or have less "waste" in the sense that it allows further reductions?
                    # Actually, if K is constant, we are just exploring the space of codes of size K.
                    # We hope to find a code of size K that allows a *removal* (which reduces K).
                    # So the goal of the swap is to create a state where some codeword has 0 unique points.
                    # How to measure "closeness to having a removable codeword"?
                    # Maybe: Minimize the number of "unique points" for the *new* codeword?
                    # Or: Minimize the number of unique points for *any* codeword?
                    # If we swap i with c, i is gone. c is added.
                    # The unique points of c in the new state are those p in bp_c such that new_cover_count[p] == 1.
                    # new_cover_count[p] = cover_count[p] - (1 if p in bp_i else 0) + (1 if p in bp_c else 0).
                    # We want to minimize the number of such p for c?
                    # Or maybe we want to maximize the chance that *some* other codeword becomes removable?
                    # A simpler heuristic: Minimize the "overlap" of c with the existing code, to keep the code "sparse"?
                    # Or: Maximize the number of points in bp_c that are *already* covered by >= 2 (so they don't contribute to uniqueness)?
                    # Let's stick to the previous "waste" heuristic but make it exact.
                    # Previous: waste = ball_size - free_points.
                    # free_points = count of p in bp_c with cover_count[p] == 1.
                    # This approximates the number of points that will *not* be unique to c?
                    # If cover_count[p] == 1 and p in bp_c, and p NOT in bp_i, then new_cover_count[p] = 1 + 1 = 2. Not unique.
                    # If cover_count[p] == 1 and p in bp_c AND p in bp_i, then new_cover_count[p] = 1 - 1 + 1 = 1. Unique!
                    # So free_points overestimates the non-unique points.
                    # Exact non-unique points for c:
                    # p in bp_c such that new_cover_count[p] >= 2.
                    # new_cover_count[p] = cover_count[p] + (1 if p in bp_c else 0) - (1 if p in bp_i else 0).
                    # If p in bp_c: new_count = cover_count[p] + 1 - (1 if p in bp_i else 0).
                    # We want new_count >= 2 => cover_count[p] + 1 - (1 if p in bp_i else 0) >= 2
                    # => cover_count[p] - (1 if p in bp_i else 0) >= 1
                    # => cover_count[p] >= 1 + (1 if p in bp_i else 0)
                    # If p not in bp_i: cover_count[p] >= 1 (always true since covered).
                    # If p in bp_i: cover_count[p] >= 2.
                    
                    # So non_unique_count = (count of p in bp_c \ bp_i) + (count of p in bp_c ∩ bp_i with cover_count[p] >= 2)
                    # unique_count = ball_size - non_unique_count
                    # We want to minimize unique_count (to make c potentially removable later).
                    # unique_count = (count of p in bp_c ∩ bp_i with cover_count[p] == 1)
                    # Because for p in bp_c \ bp_i, new_count >= 2 is always true (since cover_count >= 1).
                    # Wait, is cover_count[p] >= 1 always true? Yes, all points are covered.
                    # So for p in bp_c \ bp_i, new_count = cover_count[p] + 1 >= 2.
                    # So p in bp_c \ bp_i is NEVER unique to c.
                    # Only p in bp_c ∩ bp_i can be unique to c.
                    # And p in bp_c ∩ bp_i is unique to c iff new_count[p] == 1.
                    # new_count[p] = cover_count[p] - 1 + 1 = cover_count[p].
                    # So p is unique to c iff cover_count[p] == 1.
                    
                    # So unique_count_c = count of p in bp_c ∩ bp_i with cover_count[p] == 1.
                    # We want to minimize this.
                    # This is a very cheap metric!
                    
                    # Let's compute this.
                    # intersection_mask = in_bp_i[bp_cand]
                    # unique_count_c = (cover_count[bp_cand] == 1).sum() * intersection_mask? No.
                    # unique_count_c = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                    
                    score = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                    
                    if score < best_score:
                        best_score = score
                        best_cand = cand

                if best_cand is not None:
                    # Perform swap
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    
                    bp_new = balls[best_cand]
                    cover_count[bp_new] += 1
                    code.append(best_cand)
                    code_set.add(best_cand)
                    improved = True

        return code, cover_count

    code, cover_count = local_search(code, cover_count, time_budget=90)

    # Final trim
    code, cover_count = trim(code, cover_count)

    # High-precision polish: try aggressive single-removal swaps with exact scoring
    def polish_pass(code, cover_count, time_budget=15):
        code_set = set(code)
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(99999)

        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)

            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                unique_mask = cover_count[bp_i] == 1
                unique_pts = bp_i[unique_mask]

                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates covering all unique_pts
                candidate_set = None
                for w in unique_pts:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue

                # Same heuristic as local_search: minimize unique points of the new codeword
                # unique_count_c = count of p in bp_c ∩ bp_i with cover_count[p] == 1
                best_cand = None
                best_score = float('inf')
                
                in_bp_i = np.zeros(total, dtype=np.int8)
                in_bp_i[bp_i] = 1

                for cand in candidates:
                    bp_cand = balls[cand]
                    score = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                    
                    if score < best_score:
                        best_score = score
                        best_cand = cand

                if best_cand is not None:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    
                    bp_new = balls[best_cand]
                    cover_count[bp_new] += 1
                    code.append(best_cand)
                    code_set.add(best_cand)
                    improved = True

        return code, cover_count

    code, cover_count = polish_pass(code, cover_count, time_budget=15)

    # Final final trim
    code, cover_count = trim(code, cover_count)

    # Convert to word lists
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    result = [to_word(w) for w in code]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
