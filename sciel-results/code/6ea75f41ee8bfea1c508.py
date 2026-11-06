import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 145.0

    # Try algebraic construction via M12 group action
    try:
        a = [1,2,3,4,5,6,7,8,9,10,11,0]
        b = [1,0,3,2,5,4,7,6,9,8,11,10]

        def compose(p, q):
            return tuple(p[q[i]] for i in range(12))

        def invert(p):
            inv = [0]*12
            for i in range(12):
                inv[p[i]] = i
            return tuple(inv)

        a_tup = tuple(a)
        b_tup = tuple(b)
        identity = tuple(range(12))

        gens = [a_tup, b_tup, invert(a_tup), invert(b_tup)]
        seen = {identity}
        queue = [identity]
        group = [identity]

        while queue:
            if time.time() - start_time > timeout - 5:
                raise TimeoutError()
            g = queue.pop(0)
            for gen in gens:
                new_g = compose(g, gen)
                if new_g not in seen:
                    seen.add(new_g)
                    queue.append(new_g)
                    group.append(new_g)

        base_block = frozenset([0, 1, 2, 3, 4, 5])
        blocks_set = set()
        for g in group:
            new_block = frozenset([g[i] for i in base_block])
            blocks_set.add(new_block)

        # Verify: no 5-subset in two blocks
        five_count = {}
        valid = True
        for blk in blocks_set:
            for s in combinations(sorted(blk), 5):
                if s in five_count:
                    valid = False
                    break
                five_count[s] = 1
            if not valid:
                break

        if valid and len(blocks_set) == 132:
            return [list(blk) for blk in blocks_set]
    except Exception:
        pass

    # Fallback: backtracking
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}

    all_6subsets = list(combinations(points, 6))
    num_blocks = len(all_6subsets)

    block_subs = []
    candidates = [[] for _ in range(num_5subsets)]

    for b_idx, b in enumerate(all_6subsets):
        subs = []
        for s in combinations(b, 5):
            s_idx = subset_to_idx[s]
            subs.append(s_idx)
            candidates[s_idx].append(b_idx)
        block_subs.append(subs)

    conflicts = [set() for _ in range(num_blocks)]
    for b_idx, subs in enumerate(block_subs):
        for s_idx in subs:
            for other_b in candidates[s_idx]:
                if other_b != b_idx:
                    conflicts[b_idx].add(other_b)

    conflicts = [list(s) for s in conflicts]

    valid = [True] * num_blocks
    best_solution = []

    def backtrack(uncovered, current_blocks):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False

        if not uncovered:
            best_solution = current_blocks[:]
            return True

        min_subset = -1
        min_count = float('inf')

        for s_idx in uncovered:
            count = 0
            for b_idx in candidates[s_idx]:
                if valid[b_idx]:
                    count += 1
            if count < min_count:
                min_count = count
                min_subset = s_idx
            if min_count == 0:
                return False
            if min_count == 1:
                break

        for b_idx in candidates[min_subset]:
            if not valid[b_idx]:
                continue

            undo_stack = []
            valid[b_idx] = False
            undo_stack.append(b_idx)

            for c in conflicts[b_idx]:
                if valid[c]:
                    valid[c] = False
                    undo_stack.append(c)

            new_uncovered = set(uncovered)
            for s_idx in block_subs[b_idx]:
                new_uncovered.discard(s_idx)

            if backtrack(new_uncovered, current_blocks + [b_idx]):
                for idx in reversed(undo_stack):
                    valid[idx] = True
                return True

            for idx in reversed(undo_stack):
                valid[idx] = True

        return False

    initial_uncovered = set(range(num_5subsets))

    if backtrack(initial_uncovered, []):
        return [list(all_6subsets[i]) for i in best_solution]
    else:
        covered = set()
        blocks = []
        for b in all_6subsets:
            subsets = tuple(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
