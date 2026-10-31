import itertools
import numpy as np
import time
import random
import scipy.sparse as sp

def construct(v, k, t):
    # Parameters are fixed for this task: v=19, k=7, t=4
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy for other params
        return _greedy_generic(v, k, t)

    # Precomputation specific to v=19, k=7, t=4
    # map4 is a 4D array of size 19*19*19*19
    map4 = np.full((19, 19, 19, 19), -1, dtype=np.int32)
    id_counter = 0
    all_4s = []
    for comb in itertools.combinations(range(19), 4):
        a, b, c, d = comb
        map4[a, b, c, d] = id_counter
        all_4s.append(comb)
        id_counter += 1
    n_4s = id_counter # 3876
    
    # Precompute all blocks and their 4-subsets
    all_blocks = list(itertools.combinations(range(19), 7))
    n_blocks = len(all_blocks)
    
    # block_4s: list of lists of 4-subset IDs for each block
    # M: sparse matrix (n_blocks x n_4s) where M[i, j] = 1 if block i covers 4-subset j
    block_4s = []
    M_rows = []
    M_cols = []
    M_data = []
    
    # tsub_to_blocks: list of lists of block indices
    tsub_to_blocks = [[] for _ in range(n_4s)]
    
    for i, block in enumerate(all_blocks):
        ids = []
        for sub in itertools.combinations(block, 4):
            ts_id = map4[sub[0], sub[1], sub[2], sub[3]]
            ids.append(ts_id)
            M_rows.append(i)
            M_cols.append(ts_id)
            M_data.append(1)
            tsub_to_blocks[ts_id].append(i)
        block_4s.append(np.array(ids, dtype=np.int32))
        
    M = sp.csr_matrix((M_data, (M_rows, M_cols)), shape=(n_blocks, n_4s))
    
    # State
    counts = np.zeros(n_4s, dtype=np.int32)
    blocks = [] # List of block indices
    block_set = set()
    
    best_blocks = []
    best_size = float('inf')
    
    def update_best():
        nonlocal best_blocks, best_size
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
    def add_block(idx):
        blocks.append(idx)
        block_set.add(idx)
        counts[block_4s[idx]] += 1
        
    def remove_block(idx):
        blocks.remove(idx)
        block_set.remove(idx)
        counts[block_4s[idx]] -= 1

    def is_valid():
        return np.all(counts > 0)
        
    def remove_redundant():
        removed_any = False
        changed = True
        while changed:
            changed = False
            # Iterate in reverse
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                is_red = True
                for ts_id in block_4s[b_idx]:
                    if counts[ts_id] == 1:
                        is_red = False
                        break
                if is_red:
                    remove_block(b_idx)
                    removed_any = True
                    changed = True
        return removed_any

    def greedy_init():
        nonlocal counts, blocks, block_set
        counts = np.zeros(n_4s, dtype=np.int32)
        blocks = []
        block_set = set()
        
        while True:
            if time.time() - start_time > time_limit:
                break
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
            
            # Find block that maximizes coverage of uncovered 4s
            # scores = M @ uncovered_mask
            scores = (M @ uncovered_mask).toarray().flatten()
            
            # Mask out already selected blocks
            for b_idx in block_set:
                scores[b_idx] = 0
                
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            add_block(best_idx)
            
    def repair():
        # Greedily add blocks to cover uncovered 4s
        while True:
            if time.time() - start_time > time_limit:
                return
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
            
            scores = (M @ uncovered_mask).toarray().flatten()
            for b_idx in block_set:
                scores[b_idx] = 0
                
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            add_block(best_idx)

    def try_swap():
        # Try to swap a random block with a candidate that covers its critical sets
        if not blocks:
            return False
            
        # Pick a random block
        b_idx = random.choice(blocks)
        
        # Find critical 4-subsets (covered only by b_idx)
        crits = [ts_id for ts_id in block_4s[b_idx] if counts[ts_id] == 1]
        
        if not crits:
            # Redundant, should be handled by remove_redundant
            return False
            
        # Find candidates covering all crits
        # Intersection of tsub_to_blocks[c] for c in crits
        candidates = set(tsub_to_blocks[crits[0]])
        for c in crits[1:]:
            candidates.intersection_update(tsub_to_blocks[c])
            
        candidates -= block_set
        
        if not candidates:
            return False
            
        new_b_idx = random.choice(list(candidates))
        
        # Perform swap
        remove_block(b_idx)
        add_block(new_b_idx)
        
        return True

    def perturb():
        # Remove a random block and repair
        if not blocks:
            return
        b_idx = random.choice(blocks)
        remove_block(b_idx)
        repair()

    # 1. Greedy Initialization
    greedy_init()
    remove_redundant()
    update_best()
    
    # 2. Local Search
    stagnant_count = 0
    
    while time.time() - start_time < time_limit - 2:
        # Try to improve
        improved = remove_redundant()
        if improved:
            update_best()
            stagnant_count = 0
            continue
        
        # Try swap
        if try_swap():
            # After swap, try to remove redundants again
            remove_redundant()
            update_best()
            stagnant_count = 0
            continue
            
        # Stagnation
        stagnant_count += 1
        
        if stagnant_count > 20:
            # Perturb
            perturb()
            remove_redundant()
            update_best()
            stagnant_count = 0
        else:
            # Small random move: add a random block that covers some uncovered?
            # Or just wait?
            # Let's try a random add that is not redundant immediately?
            # Actually, just continue to perturb if stagnant too long.
            pass

    # Final Cleanup
    blocks = list(best_blocks)
    block_set = set(blocks)
    counts = np.zeros(n_4s, dtype=np.int32)
    for idx in blocks:
        counts[block_4s[idx]] += 1
        
    # Final Repair just in case
    repair()
    
    # Convert indices back to lists of points
    result = [list(all_blocks[i]) for i in blocks]
    return result

def _greedy_generic(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
