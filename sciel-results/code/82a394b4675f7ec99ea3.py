import time
import random
import itertools

def construct(v, k, t):
    deadline = time.time() + 145.0
    best_blocks = None
    best_count = float('inf')

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute for each k-subset (block) the set of t-subsets it covers
    # Store as a frozenset of indices into all_t_subsets for fast ops
    # Actually, store as a set of the t-subsets themselves (tuples)
    block_t_sets = []
    for blk in all_k_subsets:
        block_t_sets.append(frozenset(itertools.combinations(blk, t)))

    def build_initial_greedy(seed):
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            # Pick a random uncovered subset
            start = rng.choice(list(uncovered))
            block = list(start)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_pt = -1
                candidates = [x for x in range(v) if x not in block_set]
                if len(block) >= t - 1:
                    block_subs = list(itertools.combinations(block, t - 1))
                else:
                    block_subs = []
                for x in candidates:
                    cnt = 0
                    for sub in block_subs:
                        new_t = tuple(sorted(sub + (x,)))
                        if new_t in uncovered:
                            cnt += 1
                    if cnt > best_gain:
                        best_gain = cnt
                        best_pt = x
                if best_pt == -1 or best_gain <= 0:
                    best_pt = rng.choice(candidates)
                block.append(best_pt)
                block_set.add(best_pt)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        return blocks

    def local_remove(blocks):
        cover_count = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        changed = True
        cur = list(blocks)
        while changed:
            changed = False
            for i in range(len(cur)):
                redundant = True
                for sub in itertools.combinations(cur[i], t):
                    if cover_count.get(sub, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(cur[i], t):
                        cover_count[sub] -= 1
                    cur.pop(i)
                    changed = True
                    break
        return cur

    def simulated_annealing(blocks, seed, time_budget):
        rng = random.Random(seed)
        end_time = time.time() + time_budget
        cur_blocks = list(blocks)
        cur_count = len(cur_blocks)
        
        # Build coverage count
        cover_count = {}
        for b in cur_blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        best_local = list(cur_blocks)
        best_local_count = cur_count
        temperature = 0.5
        cooling = 0.9995
        
        while time.time() < end_time:
            # Pick a random block to remove
            i = rng.randrange(len(cur_blocks))
            remove_block = cur_blocks[i]
            
            # Check which t-subsets become uncovered
            newly_uncovered = []
            for sub in itertools.combinations(remove_block, t):
                if cover_count.get(sub, 0) <= 1:
                    newly_uncovered.append(sub)
            
            if not newly_uncovered:
                # Safe to remove
                for sub in itertools.combinations(remove_block, t):
                    cover_count[sub] -= 1
                cur_blocks.pop(i)
                cur_count -= 1
                if cur_count < best_local_count:
                    best_local_count = cur_count
                    best_local = list(cur_blocks)
                temperature *= cooling
                continue
            
            # Need to find a replacement block that covers all newly_uncovered
            # and ideally covers other uncovered ones too
            # Sample some random blocks
            found = False
            for _ in range(50):
                new_block = rng.choice(all_k_subsets)
                if new_block in cur_blocks:
                    continue
                new_block_t = set(itertools.combinations(new_block, t))
                # Check if it covers all newly_uncovered
                if all(sub in new_block_t for sub in newly_uncovered):
                    # Accept
                    # Remove old block coverage
                    for sub in itertools.combinations(remove_block, t):
                        cover_count[sub] -= 1
                    # Add new block coverage
                    for sub in new_block_t:
                        cover_count[sub] = cover_count.get(sub, 0) + 1
                    cur_blocks[i] = new_block
                    found = True
                    break
            
            if not found:
                # Try to accept with probability (for now, just skip)
                pass
            
            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01
        
        return best_local

    # Phase 1: Get a good starting point
    start_blocks = build_initial_greedy(42)
    start_blocks = local_remove(start_blocks)
    best_blocks = start_blocks
    best_count = len(start_blocks)

    # Phase 2: Simulated annealing with multiple restarts
    seed_counter = 0
    while time.time() < deadline:
        # Determine time budget for this SA run
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_budget = min(10.0, remaining / 2.0)
        
        # Restart from best or from a fresh greedy
        if seed_counter % 3 == 0:
            init_blocks = list(best_blocks)
        else:
            init_blocks = build_initial_greedy(seed_counter)
            init_blocks = local_remove(init_blocks)
        
        sa_blocks = simulated_annealing(init_blocks, seed_counter, time_budget)
        
        if len(sa_blocks) < best_count:
            # Verify validity
            covered = set()
            for b in sa_blocks:
                for sub in itertools.combinations(b, t):
                    covered.add(sub)
            if len(covered) == n_t:
                best_blocks = sa_blocks
                best_count = len(sa_blocks)
        
        seed_counter += 1

    return best_blocks
