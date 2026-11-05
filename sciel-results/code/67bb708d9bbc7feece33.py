"""Stronger algorithm for K_3(7,1): greedy construction + 1-for-0 deletion + 2-for-1 replacement."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_TIME_BUDGET = 120.0
_SAFETY_MARGIN = 2.0


def construct(q, n, radius):
    start = time.time()
    deadline = start + _TIME_BUDGET - _SAFETY_MARGIN

    # --- Precompute: every word and its ball (radius r) as tuples of indices ---
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Build balls: ball[i] = set of indices within radius r of word i
    # For efficiency, precompute neighbors by position
    # For each word and each position, the q-1 alternatives at that position
    # ball(word) = {word} union all single-position variants

    # We'll represent counts as a dict idx -> int
    # and use a set for the code

    def time_left():
        return deadline - time.time()

    def build_balls():
        """Return a list of frozensets (or lists) of indices within radius of each word."""
        balls = []
        for w in all_words:
            ball = [word_to_idx[w]]
            for pos in range(n):
                for alt in range(q):
                    if alt == w[pos]:
                        continue
                    v = list(w)
                    v[pos] = alt
                    ball.append(word_to_idx[tuple(v)])
            balls.append(frozenset(ball))
        return balls

    balls = build_balls()
    # Also build inverse: for each index, which codeword balls cover it is implicit
    # We need: for a candidate codeword c, the set of words it covers = balls[c]
    # For a word w, which codewords cover it = those c where w in balls[c]
    # Precompute coverers: coverers[w] = set of c indices such that w in balls[c]
    # This is expensive for large spaces. For q=3,n=7: 2187 words, each ball size = 1 + 7*2 = 15
    # So total edges = 2187 * 15 = 32805, manageable.

    coverers = [set() for _ in range(total)]
    for c_idx in range(total):
        for w_idx in balls[c_idx]:
            coverers[w_idx].add(c_idx)

    def greedy_cover():
        """Greedy set-cover: repeatedly pick the word that covers the most uncovered words."""
        uncovered = set(range(total))
        code = []
        # Count how many uncovered words each candidate covers
        # Initialize
        cover_count = [len(balls[i] & uncovered) for i in range(total)]
        while uncovered and time_left() > 0.5:
            best = -1
            best_count = 0
            for i in range(total):
                if cover_count[i] > best_count:
                    best_count = cover_count[i]
                    best = i
            if best == -1:
                break
            code.append(best)
            covered_now = balls[best] & uncovered
            uncovered -= covered_now
            # Update cover counts: for each uncovered word, decrement if best was a coverer
            for w in covered_now:
                for c in coverers[w]:
                    if c != best:
                        cover_count[c] -= 1
        return code

    def try_deletion(code_set):
        """Try to remove codewords one by one if their ball is fully covered by others.
        Returns a boolean if any deletion was made."""
        # Build coverage count: for each word, how many codewords cover it
        cov_count = defaultdict(int)
        for c in code_set:
            for w in balls[c]:
                cov_count[w] += 1
        # A codeword can be removed if every word it covers has cov_count >= 2
        removable = []
        for c in list(code_set):
            ok = True
            for w in balls[c]:
                if cov_count[w] <= 1:
                    ok = False
                    break
            if ok:
                removable.append(c)
        if not removable:
            return False
        # Remove one at a time, updating counts
        for c in removable:
            # Re-check: after removing others, is c still removable?
            # Simpler: just remove all that were removable initially, but do it safely
            pass
        # Do removals in order, updating
        removed = False
        for c in removable:
            if c not in code_set:
                continue
            # Check again
            ok = True
            for w in balls[c]:
                if cov_count[w] <= 1:
                    ok = False
                    break
            if ok:
                code_set.remove(c)
                for w in balls[c]:
                    cov_count[w] -= 1
                removed = True
        return removed

    def try_2for1(code_set):
        """Try to replace two codewords with one. 
        If two codewords c1, c2 have a word w in their uncovered-by-others union
        that is covered by some single codeword c3, then remove c1,c2 and add c3.
        Actually: if there exists a word w such that w is covered by c1 or c2,
        and the set of words only covered by c1 or c2 (not by any other codeword)
        is a subset of balls[c3] for some c3 not in {c1,c2}, then we can do the swap.
        
        Simpler heuristic: for each pair (c1,c2), find the "exclusive" words:
        words covered by c1 or c2 but by no other codeword. If this set is empty,
        we can remove both. If not, check if there's a c3 that covers all exclusive words.
        """
        if len(code_set) < 2:
            return False

        # Build cov_count
        cov_count = defaultdict(int)
        for c in code_set:
            for w in balls[c]:
                cov_count[w] += 1

        code_list = list(code_set)
        # For efficiency, only try a subset of pairs
        random.shuffle(code_list)
        limit = min(len(code_list), 80)
        for i in range(limit):
            if time_left() < 1.0:
                break
            c1 = code_list[i]
            # Find exclusive words for c1: words w in balls[c1] where cov_count[w] <= 1
            # Actually we need words covered by c1 or c2 but by no other.
            # Let's first find words that c1 uniquely covers (cov_count[w]==1 and w in balls[c1])
            # But that's not quite right for 2-for-1.
            # 
            # Better approach: for each c2 in code_list[i+1:], compute the union of
            # exclusive coverage. This is expensive. Let's limit further.
            for j in range(i + 1, limit):
                c2 = code_list[j]
                # Find words covered by c1 or c2 but by no other codeword
                exclusive = set()
                for w in balls[c1]:
                    if cov_count[w] == 1:  # only c1 covers it (since c1 is in code_set)
                        exclusive.add(w)
                for w in balls[c2]:
                    if cov_count[w] == 1:
                        exclusive.add(w)
                # But wait: a word might be covered by both c1 and c2, and cov_count[w]==2
                # In that case it's not exclusive to the pair alone.
                # The exclusive set for the pair {c1,c2} is:
                # {w : w in balls[c1] or w in balls[c2]} and cov_count[w] <= 2
                # and no other codeword covers w.
                # Actually cov_count[w] counts all codewords covering w.
                # So exclusive_to_pair = {w : w in (balls[c1] | balls[c2]) and cov_count[w] <= 2}
                # But if cov_count[w] == 2, it could be covered by c1 and some other c3, not c2.
                # We need: words covered by c1 or c2, and NOT covered by any codeword other than c1, c2.
                # That means: w in (balls[c1] | balls[c2]) and cov_count[w] - (1 if w in balls[c1] else 0) - (1 if w in balls[c2] else 0) == 0
                # i.e., the number of other codewords covering w is 0.
                other_cover = 0
                exclusive_words = []
                union_c12 = balls[c1] | balls[c2]
                for w in union_c12:
                    # Count how many codewords in code_set other than c1,c2 cover w
                    cnt = 0
                    for c in coverers[w]:
                        if c not in (c1, c2) and c in code_set:
                            cnt += 1
                    if cnt == 0:
                        exclusive_words.append(w)
                if not exclusive_words:
                    # Can remove both
                    code_set.discard(c1)
                    code_set.discard(c2)
                    return True
                # Check if there exists a c3 (not in code_set, or in code_set but not c1,c2)
                # such that all exclusive_words are in balls[c3]
                # Candidate c3 must be in the intersection of coverers of all exclusive_words
                # But that intersection might be empty. Let's find common coverers.
                common = None
                for w in exclusive_words[:3]:  # Limit to first 3 for speed
                    cands = coverers[w] - {c1, c2}
                    if common is None:
                        common = cands
                    else:
                        common &= cands
                    if not common:
                        break
                if common:
                    # Pick one
                    c3 = next(iter(common))
                    code_set.discard(c1)
                    code_set.discard(c2)
                    code_set.add(c3)
                    return True
        return False

    def local_search(code_set):
        improved = True
        while improved and time_left() > 1.0:
            improved = False
            if try_deletion(code_set):
                improved = True
            if time_left() > 1.0:
                if try_2for1(code_set):
                    improved = True

    # Main algorithm
    code = greedy_cover()
    code_set = set(code)
    local_search(code_set)

    # Try a few random restarts of greedy if time permits
    while time_left() > 5.0:
        # Perturb: remove a random codeword and re-greedy fill
        if len(code_set) <= 0:
            break
        removed = random.choice(list(code_set))
        code_set.discard(removed)
        # Re-cover: find uncovered words
        uncovered = set(range(total))
        for c in code_set:
            uncovered -= balls[c]
        # Greedy fill
        while uncovered and time_left() > 2.0:
            best = -1
            best_count = 0
            for i in range(total):
                if i in code_set:
                    continue
                cnt = len(balls[i] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best = i
            if best == -1 or best_count == 0:
                break
            code_set.add(best)
            covered_now = balls[best] & uncovered
            uncovered -= covered_now
        local_search(code_set)

    return [list(all_words[c]) for c in sorted(code_set)]
# EVOLVE-BLOCK-END
