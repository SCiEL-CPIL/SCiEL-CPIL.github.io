import itertools
import random
import time

V = 24
K = 7
T = 3
POINTS = range(V)
ALL_TRIPLES = list(itertools.combinations(POINTS, T))
NUM_TRIPLES = len(ALL_TRIPLES)
TRIPLE_TO_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}
ALL_BLOCKS = list(itertools.combinations(POINTS, K))
NUM_BLOCKS = len(ALL_BLOCKS)

block_triples = [[] for _ in range(NUM_BLOCKS)]
triple_blocks = [[] for _ in range(NUM_TRIPLES)]

for b_idx, b in enumerate(ALL_BLOCKS):
    for t in itertools.combinations(b, T):
        t_idx = TRIPLE_TO_IDX[t]
        block_triples[b_idx].append(t_idx)
        triple_blocks[t_idx].append(b_idx)

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters (not expected)
        return []

    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0

    triple_counts = [0] * NUM_TRIPLES
    uncovered_count = NUM_TRIPLES
    uncovered_set = set(range(NUM_TRIPLES))
    active_blocks = set()

    def add_block(b_idx):
        nonlocal uncovered_count
        active_blocks.add(b_idx)
        for t_idx in block_triples[b_idx]:
            if triple_counts[t_idx] == 0:
                uncovered_count -= 1
                uncovered_set.discard(t_idx)
            triple_counts[t_idx] += 1

    def remove_block(b_idx):
        nonlocal uncovered_count
        active_blocks.remove(b_idx)
        for t_idx in block_triples[b_idx]:
            triple_counts[t_idx] -= 1
            if triple_counts[t_idx] == 0:
                uncovered_count += 1
                uncovered_set.add(t_idx)

    def local_search():
        improved = True
        while improved:
            improved = False
            blocks_to_check = list(active_blocks)
            rng.shuffle(blocks_to_check)
            for b_idx in blocks_to_check:
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if triple_counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(b_idx)
                    improved = True
                    break

    # 1. Greedy Initialization
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
        # Pick a random uncovered triple
        # Convert set to list for random choice
        # This is O(U). U is small during repair, large during init.
        # To optimize init, we can just pick a random index from 0..NUM_TRIPLES and check if it's uncovered.
        # Expected tries = NUM_TRIPLES / uncovered_count.
        # At start, 1 try. At end, many tries.
        # But U decreases, so tries increase.
        # However, we only do this ~70 times.
        # Let's stick to rng.choice(list(uncovered_set)) as it's simpler and robust.
        
        t_idx = rng.choice(list(uncovered_set))
        
        candidates = triple_blocks[t_idx]
        best_block = None
        best_score = -1
        
        for b_idx in candidates:
            score = 0
            for tt in block_triples[b_idx]:
                if triple_counts[tt] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_block = b_idx
                
        if best_block is not None:
            add_block(best_block)

    local_search()
    best_blocks = list(active_blocks)
    best_size = len(best_blocks)

    # 2. VNS
    m = 1
    max_m = 5

    while time.time() - start_time < time_limit:
        local_search()
        
        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_blocks = list(active_blocks)
            m = 1

        if time.time() - start_time > time_limit - 5:
            break

        # Shake
        blocks_list = list(active_blocks)
        if len(blocks_list) >= m:
            removed = rng.sample(blocks_list, m)
            for b_idx in removed:
                remove_block(b_idx)
            
            # Repair
            while uncovered_count > 0:
                if time.time() - start_time > time_limit - 2:
                    break
                t_idx = rng.choice(list(uncovered_set))
                candidates = triple_blocks[t_idx]
                best_block = None
                best_score = -1
                
                for b_idx in candidates:
                    if b_idx in active_blocks:
                        continue
                    score = 0
                    for tt in block_triples[b_idx]:
                        if triple_counts[tt] == 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_block = b_idx
                        
                if best_block is None:
                    break
                add_block(best_block)
            
            local_search()
            
            if len(active_blocks) < best_size:
                best_size = len(active_blocks)
                best_blocks = list(active_blocks)
                m = 1
            else:
                m += 1
                if m > max_m:
                    m = 1

    return [list(ALL_BLOCKS[b_idx]) for b_idx in best_blocks]
