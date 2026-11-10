# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using bitset-accelerated greedy
    initialization, simulated annealing for local search, and redundant-block removal.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute a Python int bitmask of which t-subsets it covers
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        block_masks.append(mask)

    # For each t-subset, list of block indices that contain it
    blocks_containing = [[] for _ in range(n_t)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            blocks_containing[idx].append(b_idx)
            m ^= lsb

    def greedy_bitset(rng, deadline):
        """
        Bitset-accelerated greedy set cover.
        """
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = 0
            best_mask = 0

            order = list(range(n_blocks))
            rng.shuffle(order)

            for b_idx in order:
                if time.time() > deadline:
                    break
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_mask = block_masks[b_idx]
                elif gain == best_gain and gain > 0 and rng.random() < 0.3:
                    best_block = b_idx
                    best_mask = block_masks[b_idx]

            if best_block == -1 or best_gain <= 0:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~best_mask
            remaining = uncovered.bit_count()

        if remaining > 0:
            return None
        return blocks

    def remove_redundant(blocks, deadline):
        """
        Remove redundant blocks iteratively.
        """
        if not blocks:
            return blocks

        n = len(blocks)
        # Map block tuples to indices
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}

        # Compute coverage count per t-subset
        cov_count = [0] * n_t
        for b in blocks:
            b_idx = block_to_idx[b]
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lsb

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks

                b = blocks[i]
                b_idx = block_to_idx[b]
                mask = block_masks[b_idx]

                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb

                if redundant:
                    blocks.pop(i)
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    changed = True

        return blocks

    def simulated_annealing(initial_blocks, deadline, seed):
        """
        Simulated annealing to reduce block count.
        State: set of selected block indices.
        Moves: remove a block, add a block, swap a block.
        Cost: number of blocks (minimize).
        Constraint: every t-subset must be covered.
        """
        rng = random.Random(seed)
        
        # Convert blocks to indices
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        current_set = set(block_to_idx[b] for b in initial_blocks)
        current_cost = len(current_set)
        
        # Coverage count for current state
        cov_count = [0] * n_t
        for b_idx in current_set:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lsb
        
        best_set = set(current_set)
        best_cost = current_cost
        
        # SA parameters
        T_start = 1.0
        T_end = 0.001
        alpha = 0.995  # cooling rate
        T = T_start
        
        max_iter = 50000
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Choose a random move
            move_type = rng.random()
            new_set = set(current_set)
            new_cost = current_cost
            
            if move_type < 0.4 and len(current_set) > 1:
                # Remove a random block
                to_remove = rng.choice(list(current_set))
                mask = block_masks[to_remove]
                # Check if all t-subsets in this block are covered by others
                m = mask
                can_remove = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        can_remove = False
                        break
                    m ^= lsb
                
                if can_remove:
                    new_set.discard(to_remove)
                    new_cost -= 1
                    # Update coverage
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                else:
                    # Invalid move, skip
                    continue
                    
            elif move_type < 0.7:
                # Add a random block
                candidates = [i for i in range(n_blocks) if i not in current_set]
                if not candidates:
                    continue
                to_add = rng.choice(candidates)
                new_set.add(to_add)
                new_cost += 1
                # Update coverage
                mask = block_masks[to_add]
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cov_count[idx] += 1
                    m ^= lsb
                    
            else:
                # Swap: remove one, add one
                if len(current_set) < 1:
                    continue
                to_remove = rng.choice(list(current_set))
                candidates = [i for i in range(n_blocks) if i not in current_set]
                if not candidates:
                    continue
                to_add = rng.choice(candidates)
                
                # Check if removing to_remove is valid after adding to_add
                mask_remove = block_masks[to_remove]
                mask_add = block_masks[to_add]
                
                # Temporary: what would coverage look like?
                # For each t-subset in mask_remove, check if cov_count[idx] >= 2 or it's in mask_add
                m = mask_remove
                can_swap = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1 and not (mask_add & lsb):
                        can_swap = False
                        break
                    m ^= lsb
                
                if can_swap:
                    new_set.discard(to_remove)
                    new_set.add(to_add)
                    # new_cost stays same
                    # Update coverage
                    m = mask_remove
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    m = mask_add
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] += 1
                        m ^= lsb
                else:
                    continue
            
            # Evaluate new state
            # Check validity: all cov_count[idx] >= 1
            valid = True
            for idx in range(n_t):
                if cov_count[idx] < 1:
                    valid = False
                    break
            
            if not valid:
                # Roll back
                # This is complex; let's just reject
                # Roll back coverage changes
                # For simplicity, we'll just revert by recomputing if needed
                # But that's expensive. Let's track changes properly.
                # Actually, let's just reject invalid moves.
                # Roll back coverage:
                if move_type < 0.4:
                    # Was a removal
                    mask = block_masks[to_remove]
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] += 1
                        m ^= lsb
                elif move_type < 0.7:
                    # Was an addition
                    mask = block_masks[to_add]
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                else:
                    # Was a swap
                    mask_remove = block_masks[to_remove]
                    mask_add = block_masks[to_add]
                    m = mask_remove
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] += 1
                        m ^= lsb
                    m = mask_add
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                continue
            
            # Accept/reject
            delta = new_cost - current_cost
            if delta <= 0:
                # Better or equal
                current_set = new_set
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_set = set(current_set)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Worse, accept with probability exp(-delta/T)
                if T > 0 and rng.random() < math.exp(-delta / T):
                    current_set = new_set
                    current_cost = new_cost
                    no_improve += 1
                else:
                    # Roll back
                    if move_type < 0.4:
                        mask = block_masks[to_remove]
                        m = mask
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            cov_count[idx] += 1
                            m ^= lsb
                    elif move_type < 0.7:
                        mask = block_masks[to_add]
                        m = mask
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            cov_count[idx] -= 1
                            m ^= lsb
                    else:
                        mask_remove = block_masks[to_remove]
                        mask_add = block_masks[to_add]
                        m = mask_remove
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            cov_count[idx] += 1
                            m ^= lsb
                        m = mask_add
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            cov_count[idx] -= 1
                            m ^= lsb
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
            
            if no_improve > max_no_improve:
                break
        
        # Convert best_set back to blocks
        best_blocks = [all_blocks[i] for i in best_set]
        return best_blocks

    # Main loop: multiple randomized restarts with SA
    best_blocks = None
    best_len = float('inf')

    restart_count = 0
    while time.time() < deadline - 30.0:
        restart_count += 1
        rng = random.Random(42 + restart_count)

        blocks = greedy_bitset(rng, deadline)

        if blocks is not None:
            blocks = remove_redundant(blocks, deadline)

            # Apply simulated annealing
            if time.time() < deadline - 25.0:
                sa_blocks = simulated_annealing(blocks, deadline, 1000 + restart_count)
                if sa_blocks:
                    sa_blocks = remove_redundant(sa_blocks, deadline)
                    if len(sa_blocks) < best_len:
                        best_len = len(sa_blocks)
                        best_blocks = sa_blocks
                else:
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = blocks
            else:
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks

        if time.time() > deadline - 35.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1 or best_gain <= 0:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~block_masks[best_block]
            remaining = uncovered.bit_count()

        if remaining > 0:
            return blocks

        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
