import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the bitset of t-subset indices it covers
    k_sub_bitsets = [0] * n_k
    for i, k_sub in enumerate(all_k_subs):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        k_sub_bitsets[i] = mask

    # For each t-subset, list of k-subset indices covering it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, mask in enumerate(k_sub_bitsets):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            t_sub_to_k_idxs[idx].append(i)
            m ^= lsb

    def get_set_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        return bits

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        # We can speed up greedy by not checking all blocks if we maintain a list of "active" blocks?
        # But all blocks are active initially.
        
        for _ in range(n_k):
            if uncovered_mask == 0:
                break
            
            best_idx = -1
            best_gain = 0
            
            # Add small noise to break ties and explore
            noise = rng.random() * 0.01 if seed > 0 else 0.0
            
            for i in range(n_k):
                block_mask = k_sub_bitsets[i]
                common = block_mask & uncovered_mask
                gain = common.bit_count()
                
                if gain + noise > best_gain:
                    best_gain = gain + noise
                    best_idx = i
                    if gain == 56: # Max possible
                        break
                
            if best_idx == -1 or best_gain < 1.0: # gain is int, so <1 means 0
                break
                
            selected.append(best_idx)
            uncovered_mask &= ~k_sub_bitsets[best_idx]
            
        return selected

    def reduce_blocks(blocks_idx):
        if not blocks_idx:
            return blocks_idx
        
        cover_count = [0] * n_t
        for b_idx in blocks_idx:
            bits = get_set_bits(k_sub_bitsets[b_idx])
            for t_idx in bits:
                cover_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                bits = get_set_bits(k_sub_bitsets[b_idx])
                is_redundant = True
                for t_idx in bits:
                    if cover_count[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for t_idx in bits:
                        cover_count[t_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
            
        return blocks_idx

    def local_search(initial_blocks, max_time):
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = [0] * n_t
        block_set = set(current_blocks)
        
        for b_idx in current_blocks:
            bits = get_set_bits(k_sub_bitsets[b_idx])
            for t_idx in bits:
                cover_count[t_idx] += 1
        
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(123)
        T = 30.0
        cooling = 0.995
        min_T = 0.05
        
        while time.time() < max_time and T > min_T:
            # Try moves
            move_type = rng.random()
            
            if move_type < 0.5 and current_blocks:
                # Remove
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                bits = get_set_bits(k_sub_bitsets[b_idx])
                
                min_count = min(cover_count[t_idx] for t_idx in bits)
                
                if min_count > 1:
                    for t_idx in bits:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    block_set.discard(b_idx)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    # Invalid
                    # Calculate penalty
                    uncovered_count = sum(1 for t_idx in bits if cover_count[t_idx] == 1)
                    if uncovered_count > 0:
                        delta_E = 1.0 + 10.0 * uncovered_count
                        if rng.random() < np.exp(-delta_E / T):
                            for t_idx in bits:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(i)
                            block_set.discard(b_idx)
                            # Energy increased, don't update best yet

            elif move_type < 0.9 and current_blocks:
                # Swap
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                old_bits = get_set_bits(k_sub_bitsets[old_b])
                
                critical = [t_idx for t_idx in old_bits if cover_count[t_idx] == 1]
                
                if not critical:
                    # Swap with random
                    new_b = rng.randint(0, n_k - 1)
                    if new_b in block_set:
                        continue
                    
                    for t_idx in old_bits:
                        cover_count[t_idx] -= 1
                    new_bits = get_set_bits(k_sub_bitsets[new_b])
                    for t_idx in new_bits:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    block_set.discard(old_b)
                    block_set.add(new_b)
                    
                else:
                    if len(critical) > 4:
                        continue
                        
                    candidates = set(t_sub_to_k_idxs[critical[0]])
                    for c in critical[1:]:
                        candidates &= set(t_sub_to_k_idxs[c])
                        if not candidates:
                            break
                    
                    candidates -= block_set
                    if not candidates:
                        continue
                        
                    new_b = rng.choice(list(candidates))
                    
                    for t_idx in old_bits:
                        cover_count[t_idx] -= 1
                    new_bits = get_set_bits(k_sub_bitsets[new_b])
                    for t_idx in new_bits:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    block_set.discard(old_b)
                    block_set.add(new_b)
                    
            else:
                # 2-for-1 or just skip
                pass

            T *= cooling

        # Final reduction
        current_blocks = reduce_blocks(current_blocks)
        
        if len(current_blocks) < len(best_blocks):
            return current_blocks
        return best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Randomized restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]
        for seed in seeds:
            if time.time() >= deadline - 5.0:
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Local search
    if best_blocks and time.time() < deadline - 5.0:
        ls_result = local_search(best_blocks, deadline - 5.0)
        if len(ls_result) < best_len:
            best_blocks = ls_result
            best_len = len(best_blocks)

    # Final reduction
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
