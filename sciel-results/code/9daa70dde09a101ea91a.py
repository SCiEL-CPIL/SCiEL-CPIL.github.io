import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145

    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    
    # Precompute triplet IDs in a 3D array for O(1) lookup
    # tid_map[a, b, c] = id of triplet {a, b, c}
    # Since a,b,c are distinct, order doesn't matter for the set, but for array index we need to handle permutations?
    # No, we can just fill all 6 permutations for each triplet.
    tid_map = np.full((v, v, v), -1, dtype=np.int32)
    for i, triplet in enumerate(all_triplets):
        a, b, c = triplet
        tid_map[a, b, c] = i
        tid_map[a, c, b] = i
        tid_map[b, a, c] = i
        tid_map[b, c, a] = i
        tid_map[c, a, b] = i
        tid_map[c, b, a] = i

    def get_tid(a, b, c):
        return tid_map[a, b, c]

    def greedy_construct(seed=0, diversity=0.0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        pos = np.arange(n_triplets)
        blocks = []
        
        while uncovered_list:
            if time.time() > deadline - 20:
                break
                
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                candidates = [p for p in range(v) if p not in block_set]
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                gains = []
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        a, b = pair
                        tid = get_tid(a, b, p)
                        if is_uncovered[tid]:
                            gain += 1
                    gains.append(gain)
                
                # Select best point
                # If diversity > 0, pick randomly among top candidates
                max_gain = max(gains)
                if diversity > 0 and rng.random() < diversity:
                    # Pick among those with gain >= max_gain * 0.8
                    top_candidates = [candidates[i] for i, g in enumerate(gains) if g >= max_gain * 0.8]
                    best_point = rng.choice(top_candidates)
                else:
                    best_point = candidates[gains.index(max_gain)]
                    # Tie breaking
                    if gains.count(max_gain) > 1:
                        tied = [candidates[i] for i, g in enumerate(gains) if g == max_gain]
                        best_point = rng.choice(tied)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                a, b, c = triplet
                tid = get_tid(a, b, c)
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()
                    pos[tid] = -1

        return blocks

    def local_search(blocks):
        count = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for triplet in itertools.combinations(b, t):
                a, b_, c = triplet
                count[get_tid(a, b_, c)] += 1
        
        # Iterative deletion
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                redundant = True
                for triplet in itertools.combinations(b, t):
                    a, b_, c = triplet
                    if count[get_tid(a, b_, c)] == 1:
                        redundant = False
                        break
                if redundant:
                    for triplet in itertools.combinations(b, t):
                        a, b_, c = triplet
                        count[get_tid(a, b_, c)] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple seeds with varied diversity
    num_seeds = 150
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        
        diversity = (seed % 5) * 0.1
        blocks = greedy_construct(seed, diversity=diversity)
        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            
    if best_blocks is None:
        best_blocks = local_search(greedy_construct(0))
        
    return best_blocks

if __name__ == "__main__":
    pass
